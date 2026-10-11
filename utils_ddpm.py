"""Training helpers, sampling, validation, and checkpoint loading."""
import os
import csv
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from model import SharedVAEEncoder, ConditionalGenerator, DiffusionSchedule

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_requires_grad(model: nn.Module, value: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = value


def append_csv(path: str, row: dict) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def add_awgn_by_snr(signal: torch.Tensor, snr_db: float) -> torch.Tensor:
    if math.isinf(snr_db):
        return signal
    signal_power = signal.pow(2).mean(dim=1, keepdim=True).clamp_min(1e-12)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    return signal + torch.randn_like(signal) * torch.sqrt(noise_power)


def latent_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    mse = F.mse_loss(pred, target).item()
    cosine = F.cosine_similarity(pred, target, dim=1).mean().item()
    return {"mse": mse, "cosine": cosine}


@torch.no_grad()
def sample_ddpm(
    model: nn.Module,
    schedule: DiffusionSchedule,
    cond: torch.Tensor,
    labels: torch.Tensor,
    z_dim: int,
    clip_x0: float | None = None,
) -> torch.Tensor:
    model.eval()
    x = torch.randn(cond.size(0), z_dim, device=cond.device)

    for step in reversed(range(schedule.timesteps)):
        t = torch.full((cond.size(0),), step, device=cond.device, dtype=torch.long)
        pred_eps = model(x, t, labels, cond)
        pred_x0 = schedule.predict_x0_from_eps(x, t, pred_eps)
        if clip_x0 is not None:
            pred_x0 = pred_x0.clamp(-clip_x0, clip_x0)

        beta_t = schedule.extract(schedule.betas, t, x.shape)
        alpha_t = schedule.extract(schedule.alphas, t, x.shape)
        alpha_bar_t = schedule.extract(schedule.alpha_bars, t, x.shape)
        alpha_bar_prev = schedule.extract(schedule.alpha_bars_prev, t, x.shape)

        coef_x0 = beta_t * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar_t)
        coef_xt = (1.0 - alpha_bar_prev) * torch.sqrt(alpha_t) / (1.0 - alpha_bar_t)
        mean = coef_x0 * pred_x0 + coef_xt * x

        if step > 0:
            variance = schedule.extract(schedule.posterior_variance, t, x.shape)
            x = mean + torch.sqrt(variance) * torch.randn_like(x)
        else:
            x = mean

    return x


@torch.no_grad()
def validate(
    model: nn.Module,
    schedule: DiffusionSchedule,
    encoder: nn.Module,
    generator: nn.Module,
    fixed_e: torch.Tensor,
    fixed_d: torch.Tensor,
    epoch: int,
    save_root: str,
    z_dim: int,
    validation_snr_db: float,
) -> dict:
    model.eval()
    encoder.eval()
    generator.eval()

    label_e = torch.zeros(fixed_e.size(0), dtype=torch.long, device=fixed_e.device)
    label_d = torch.ones(fixed_d.size(0), dtype=torch.long, device=fixed_d.device)

    z_e, _ = encoder(fixed_e)
    z_d, _ = encoder(fixed_d)
    z_mix_clean = z_e + z_d
    z_rx = add_awgn_by_snr(z_mix_clean, validation_snr_db)

    pred_e = sample_ddpm(model, schedule, z_rx, label_e, z_dim)
    pred_d = sample_ddpm(model, schedule, z_rx, label_d, z_dim)

    recon_e = generator(pred_e, label_e)
    recon_d = generator(pred_d, label_d)
    oracle_e = generator(z_e, label_e)
    oracle_d = generator(z_d, label_d)

    # Zero-condition test: reveals label-only generation.
    zero_cond = torch.zeros_like(z_rx)
    zero_e = sample_ddpm(model, schedule, zero_cond, label_e, z_dim)
    zero_d = sample_ddpm(model, schedule, zero_cond, label_d, z_dim)
    zero_recon_e = generator(zero_e, label_e)
    zero_recon_d = generator(zero_d, label_d)

    # Partner-swap test: keep EuroSAT target, shuffle DeepGlobe partner.
    permutation = torch.randperm(z_d.size(0), device=z_d.device)
    swapped_mix = z_e + z_d[permutation]
    swapped_e = sample_ddpm(model, schedule, swapped_mix, label_e, z_dim)
    swapped_recon_e = generator(swapped_e, label_e)

    def image_l1(a: torch.Tensor, b: torch.Tensor) -> float:
        return F.l1_loss(a, b).item()

    metrics_e = latent_metrics(pred_e, z_e)
    metrics_d = latent_metrics(pred_d, z_d)
    metrics = {
        "latent_mse_e": metrics_e["mse"],
        "latent_mse_d": metrics_d["mse"],
        "latent_cos_e": metrics_e["cosine"],
        "latent_cos_d": metrics_d["cosine"],
        "image_l1_e_to_real": image_l1(recon_e, fixed_e),
        "image_l1_d_to_real": image_l1(recon_d, fixed_d),
        "image_l1_e_to_oracle": image_l1(recon_e, oracle_e),
        "image_l1_d_to_oracle": image_l1(recon_d, oracle_d),
        "zero_cond_l1_e_to_real": image_l1(zero_recon_e, fixed_e),
        "zero_cond_l1_d_to_real": image_l1(zero_recon_d, fixed_d),
        "partner_swap_e_change": image_l1(swapped_recon_e, recon_e),
    }

    count = fixed_e.size(0)
    nrow = max(1, int(math.sqrt(count)))
    save_image((fixed_e + 1.0) / 2.0, os.path.join(save_root, f"val_real_E_ep{epoch}.png"), nrow=nrow)
    save_image((fixed_d + 1.0) / 2.0, os.path.join(save_root, f"val_real_D_ep{epoch}.png"), nrow=nrow)
    save_image((oracle_e + 1.0) / 2.0, os.path.join(save_root, f"val_oracle_E_ep{epoch}.png"), nrow=nrow)
    save_image((oracle_d + 1.0) / 2.0, os.path.join(save_root, f"val_oracle_D_ep{epoch}.png"), nrow=nrow)
    save_image((recon_e + 1.0) / 2.0, os.path.join(save_root, f"val_sep_E_ep{epoch}.png"), nrow=nrow)
    save_image((recon_d + 1.0) / 2.0, os.path.join(save_root, f"val_sep_D_ep{epoch}.png"), nrow=nrow)
    save_image((zero_recon_e + 1.0) / 2.0, os.path.join(save_root, f"val_zero_cond_E_ep{epoch}.png"), nrow=nrow)
    save_image((zero_recon_d + 1.0) / 2.0, os.path.join(save_root, f"val_zero_cond_D_ep{epoch}.png"), nrow=nrow)
    save_image((swapped_recon_e + 1.0) / 2.0, os.path.join(save_root, f"val_partner_swap_E_ep{epoch}.png"), nrow=nrow)

    return metrics


def load_vae_gan_models(
    joint_checkpoint_path: str,
    device: str,
    z_dim: int,
    label_dim: int,
    num_classes: int,
):
    if not os.path.exists(joint_checkpoint_path):
        raise FileNotFoundError(
            f"Cannot find from-scratch VAE-GAN checkpoint:\n{joint_checkpoint_path}"
        )

    checkpoint = torch.load(joint_checkpoint_path, map_location=device)
    if "encoder" not in checkpoint or "generator" not in checkpoint:
        raise KeyError(
            "The checkpoint must contain 'encoder' and 'generator' state dictionaries."
        )

    encoder = SharedVAEEncoder(z_dim=z_dim).to(device)
    generator = ConditionalGenerator(
        z_dim=z_dim,
        num_classes=num_classes,
        label_dim=label_dim,
    ).to(device)

    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    generator.load_state_dict(checkpoint["generator"], strict=True)

    encoder.eval()
    generator.eval()
    set_requires_grad(encoder, False)
    set_requires_grad(generator, False)

    print(f"Loaded VAE-GAN checkpoint epoch = {checkpoint.get('epoch', 'unknown')}")
    print(f"Loaded VAE-GAN checkpoint stage = {checkpoint.get('stage', 'unknown')}")
    return encoder, generator
