"""Training helpers, sampling, validation, and checkpoint loading."""
import os
import csv
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from dataclasses import asdict
from model import SharedVAEEncoder, ConditionalGenerator, DiffusionSchedule

def reparameterize(mu, logvar):
    return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)


def kl_divergence(mu, logvar):
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()


def set_requires_grad(model, value):
    for parameter in model.parameters():
        parameter.requires_grad = value


def gradient_penalty(critic, real, fake, labels):
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=real.device)
    mixed = epsilon * real + (1 - epsilon) * fake
    mixed.requires_grad_(True)
    score = critic(mixed, labels)
    gradients = torch.autograd.grad(
        outputs=score,
        inputs=mixed,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0].flatten(1)
    return (gradients.norm(2, dim=1) - 1).pow(2).mean()

def image_psnr(pred, target):
    pred = ((pred + 1) / 2).clamp(0, 1)
    target = ((target + 1) / 2).clamp(0, 1)
    mse = F.mse_loss(pred, target, reduction="none").flatten(1).mean(1)
    return (-10 * torch.log10(mse.clamp_min(1e-12))).mean().item()


def append_csv(path, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def validate(encoder, generator, loader, device, max_batches=20):
    encoder.eval()
    generator.eval()
    sums = {"l1_mu": 0.0, "l1_sample": 0.0, "psnr_mu": 0.0,
            "kl": 0.0, "mu_mean": 0.0, "mu_std": 0.0}
    count = 0

    for batch_index, (real, labels) in enumerate(loader):
        if batch_index >= max_batches:
            break
        real = real.to(device)
        labels = labels.to(device, dtype=torch.long)
        mu, logvar = encoder(real)
        sampled_z = reparameterize(mu, logvar)
        recon_mu = generator(mu, labels)
        recon_sample = generator(sampled_z, labels)

        sums["l1_mu"] += F.l1_loss(recon_mu, real).item()
        sums["l1_sample"] += F.l1_loss(recon_sample, real).item()
        sums["psnr_mu"] += image_psnr(recon_mu, real)
        sums["kl"] += kl_divergence(mu, logvar).item()
        sums["mu_mean"] += mu.mean().item()
        sums["mu_std"] += mu.std().item()
        count += 1

    return {f"val_{k}": v / max(1, count) for k, v in sums.items()}


@torch.no_grad()
def save_examples(encoder, generator, fixed_e, fixed_d, label_e, label_d,
                  random_z, save_root, epoch):
    encoder.eval()
    generator.eval()

    mu_e, logvar_e = encoder(fixed_e)
    mu_d, logvar_d = encoder(fixed_d)

    outputs = {
        f"EuroSAT_mu_ep{epoch}.png": generator(mu_e, label_e),
        f"DeepGlobe_mu_ep{epoch}.png": generator(mu_d, label_d),
        f"EuroSAT_sample_ep{epoch}.png":
            generator(reparameterize(mu_e, logvar_e), label_e),
        f"DeepGlobe_sample_ep{epoch}.png":
            generator(reparameterize(mu_d, logvar_d), label_d),
        f"random_EuroSAT_ep{epoch}.png": generator(random_z, label_e),
        f"random_DeepGlobe_ep{epoch}.png": generator(random_z, label_d),
    }
    for filename, images in outputs.items():
        save_image((images + 1) / 2, os.path.join(save_root, filename), nrow=8)


def save_checkpoint(path, epoch, encoder, generator, critic,
                    opt_e, opt_g, opt_d, config, validation):
    torch.save(
        {
            "epoch": epoch,
            "encoder": encoder.state_dict(),
            "generator": generator.state_dict(),
            "critic": critic.state_dict(),
            "optimizer_encoder": opt_e.state_dict(),
            "optimizer_generator": opt_g.state_dict(),
            "optimizer_critic": opt_d.state_dict(),
            "config": asdict(config),
            "validation": validation,
        },
        path,
    )

