import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset, random_split
from torchvision import datasets, transforms
from torchvision.utils import save_image

from model import SharedVAEEncoder,ConditionalGenerator,ConditionalCritic
from dataset import SatelliteDataset,DomainLabelDataset
from config import Config
from utils import *

def main():
    config = Config()
    device = config.device if torch.cuda.is_available() else "cpu"

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    os.makedirs(config.save_root, exist_ok=True)
    os.makedirs(config.model_root, exist_ok=True)
    history_path = os.path.join(config.save_root, "training_history.csv")
    if os.path.exists(history_path):
        os.remove(history_path)

    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    euro_base = datasets.ImageFolder(config.eurosat_root, transform=transform)
    euro_base = Subset(
        euro_base,
        range(min(config.max_images_per_domain, len(euro_base))),
    )
    eurosat = DomainLabelDataset(euro_base, 0)

    deepglobe = SatelliteDataset(config.deepglobe_root, transform, 1)
    deepglobe = Subset(
        deepglobe,
        range(min(config.max_images_per_domain, len(deepglobe))),
    )

    full_dataset = ConcatDataset([eurosat, deepglobe])
    val_size = max(1, int(len(full_dataset) * config.validation_ratio))
    train_size = len(full_dataset) - val_size
    split_gen = torch.Generator().manual_seed(config.seed)
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size], generator=split_gen
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(config.num_workers > 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(config.num_workers > 0),
    )

    fixed_count = min(64, len(eurosat), len(deepglobe))
    fixed_e = torch.stack([eurosat[i][0] for i in range(fixed_count)]).to(device)
    fixed_d = torch.stack([deepglobe[i][0] for i in range(fixed_count)]).to(device)
    label_e = torch.zeros(fixed_count, dtype=torch.long, device=device)
    label_d = torch.ones(fixed_count, dtype=torch.long, device=device)
    random_z = torch.randn(fixed_count, config.z_dim, device=device)

    save_image((fixed_e + 1) / 2, os.path.join(config.save_root, "real_EuroSAT.png"), nrow=8)
    save_image((fixed_d + 1) / 2, os.path.join(config.save_root, "real_DeepGlobe.png"), nrow=8)

    encoder = SharedVAEEncoder(config.z_dim).to(device)
    generator = ConditionalGenerator(
        config.z_dim, config.num_classes, config.label_dim
    ).to(device)
    critic = ConditionalCritic(
        config.num_classes, config.critic_feature_dim
    ).to(device)

    opt_e = torch.optim.Adam(encoder.parameters(), lr=config.lr_encoder, betas=(0.9, 0.999))
    opt_g = torch.optim.Adam(generator.parameters(), lr=config.lr_generator, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(critic.parameters(), lr=config.lr_critic, betas=(0.0, 0.9))

    print(f"Device={device}")
    print(f"EuroSAT={len(eurosat)} | DeepGlobe={len(deepglobe)}")
    print("Encoder, Generator and Critic are initialized from scratch.")

    best_l1 = float("inf")
    global_step = 0

    for epoch in range(1, config.epochs + 1):
        encoder.train()
        generator.train()
        critic.train()

        beta = config.beta_max * min(
            1.0, epoch / max(1, config.kl_warmup_epochs)
        )
        sums = {
            "encoder_total": 0.0, "generator_total": 0.0,
            "critic_total": 0.0, "pixel": 0.0, "feature": 0.0,
            "kl": 0.0, "recon_adv": 0.0, "random_adv": 0.0,
            "gp": 0.0, "mu_mean": 0.0, "mu_std": 0.0,
        }
        batches = 0

        for real, labels in train_loader:
            real = real.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            batch_now = real.size(0)

            # Critic
            set_requires_grad(critic, True)
            set_requires_grad(encoder, False)
            set_requires_grad(generator, False)
            d_accum = 0.0
            gp_accum = 0.0

            for _ in range(config.n_critic):
                with torch.no_grad():
                    mu_d, logvar_d = encoder(real)
                    recon_d = generator(reparameterize(mu_d, logvar_d), labels)
                    random_fake_d = generator(
                        torch.randn(batch_now, config.z_dim, device=device), labels
                    )

                fake_all = torch.cat([recon_d, random_fake_d], dim=0)
                fake_labels = torch.cat([labels, labels], dim=0)
                d_real = critic(real, labels)
                d_fake = critic(fake_all, fake_labels)

                fake_for_gp = recon_d if random.random() < 0.5 else random_fake_d
                gp = gradient_penalty(critic, real, fake_for_gp, labels)
                loss_d = d_fake.mean() - d_real.mean() + config.lambda_gp * gp

                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                opt_d.step()
                d_accum += loss_d.item()
                gp_accum += gp.item()

            d_value = d_accum / config.n_critic
            gp_value = gp_accum / config.n_critic

            # Encoder
            set_requires_grad(critic, False)
            set_requires_grad(encoder, True)
            set_requires_grad(generator, False)

            mu_e, logvar_e = encoder(real)
            recon_e = generator(reparameterize(mu_e, logvar_e), labels)
            pixel_e = F.l1_loss(recon_e, real)

            with torch.no_grad():
                _, real_feat_e = critic(real, labels, return_features=True)
            _, recon_feat_e = critic(recon_e, labels, return_features=True)

            feature_e = F.mse_loss(recon_feat_e, real_feat_e)
            kl_e = kl_divergence(mu_e, logvar_e)
            loss_e = (
                config.lambda_pixel * pixel_e
                + config.lambda_feature * feature_e
                + beta * kl_e
            )

            opt_e.zero_grad(set_to_none=True)
            loss_e.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.grad_clip)
            opt_e.step()

            # Generator
            set_requires_grad(critic, False)
            set_requires_grad(encoder, False)
            set_requires_grad(generator, True)

            with torch.no_grad():
                mu_g, logvar_g = encoder(real)
                z_g = reparameterize(mu_g, logvar_g)
                _, real_feat_g = critic(real, labels, return_features=True)

            recon_g = generator(z_g, labels)
            score_g, recon_feat_g = critic(recon_g, labels, return_features=True)

            pixel_g = F.l1_loss(recon_g, real)
            feature_g = F.mse_loss(recon_feat_g, real_feat_g)
            recon_adv = -score_g.mean()

            random_fake_g = generator(
                torch.randn(batch_now, config.z_dim, device=device), labels
            )
            random_adv = -critic(random_fake_g, labels).mean()

            loss_g = (
                config.lambda_pixel * pixel_g
                + config.lambda_feature * feature_g
                + config.lambda_recon_adv * recon_adv
                + config.lambda_random_adv * random_adv
            )

            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), config.grad_clip)
            opt_g.step()

            sums["encoder_total"] += loss_e.item()
            sums["generator_total"] += loss_g.item()
            sums["critic_total"] += d_value
            sums["pixel"] += pixel_g.item()
            sums["feature"] += feature_g.item()
            sums["kl"] += kl_e.item()
            sums["recon_adv"] += recon_adv.item()
            sums["random_adv"] += random_adv.item()
            sums["gp"] += gp_value
            sums["mu_mean"] += mu_e.detach().mean().item()
            sums["mu_std"] += mu_e.detach().std().item()
            batches += 1

            if global_step % 200 == 0:
                print(
                    f"Epoch {epoch}/{config.epochs} | Step={global_step} | "
                    f"E={loss_e.item():.4f} | G={loss_g.item():.4f} | "
                    f"D={d_value:.4f} | Pixel={pixel_g.item():.4f} | "
                    f"Feature={feature_g.item():.4f} | KL={kl_e.item():.4f} | "
                    f"beta={beta:.6f} | mu_std={mu_e.std().item():.4f}"
                )
            global_step += 1

        averages = {key: value / max(1, batches) for key, value in sums.items()}
        validation = validate(encoder, generator, val_loader, device)
        row = {"epoch": epoch, "beta": beta, **averages, **validation}
        append_csv(history_path, row)

        print("=" * 110)
        print(f"Epoch {epoch}/{config.epochs} | {averages}")
        print("Validation:", validation)
        print("=" * 110)

        latest = os.path.join(config.model_root, "vae_gan_from_scratch_latest.pth")
        save_checkpoint(
            latest, epoch, encoder, generator, critic,
            opt_e, opt_g, opt_d, config, validation
        )

        if epoch % config.save_every == 0 or epoch == 1 or epoch == config.epochs:
            save_examples(
                encoder, generator, fixed_e, fixed_d, label_e, label_d,
                random_z, config.save_root, epoch
            )
            save_checkpoint(
                os.path.join(
                    config.model_root, f"vae_gan_from_scratch_ep{epoch}.pth"
                ),
                epoch, encoder, generator, critic,
                opt_e, opt_g, opt_d, config, validation
            )

        if validation["val_l1_mu"] < best_l1:
            best_l1 = validation["val_l1_mu"]
            save_checkpoint(
                os.path.join(config.model_root, "vae_gan_from_scratch_best.pth"),
                epoch, encoder, generator, critic,
                opt_e, opt_g, opt_d, config, validation
            )
            print(f"Saved best model | val_l1_mu={best_l1:.6f}")

    torch.save(
        encoder.state_dict(),
        os.path.join(config.model_root, "vae_gan_scratch_encoder.pth"),
    )
    torch.save(
        generator.state_dict(),
        os.path.join(config.model_root, "vae_gan_scratch_generator.pth"),
    )
    torch.save(
        critic.state_dict(),
        os.path.join(config.model_root, "vae_gan_scratch_critic.pth"),
    )

    print("Training completed.")
    print("For DDPM training, use encoder output mu, not sampled z.")


if __name__ == "__main__":
    main()
