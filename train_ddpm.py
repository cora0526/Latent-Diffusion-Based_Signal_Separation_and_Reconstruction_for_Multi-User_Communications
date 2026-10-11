"""Train the conditional latent DDPM separator. Run: python train.py"""
import os
import math
import random
import torch
import torch.nn.functional as F
import config as cfg
from dataset import build_dataloaders
from model import LatentConditionalDDPM, DiffusionSchedule
from utils_ddpm import set_seed, add_awgn_by_snr, append_csv, load_vae_gan_models, validate


def endless_batches(loader):
    """Cycle through reshuffled epochs without caching DataLoader batches in RAM."""
    while True:
        yield from loader


def main():
    # Read all adjustable settings from config.py.
    seed = cfg.seed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z_dim = cfg.z_dim
    label_dim = cfg.label_dim
    num_classes = cfg.num_classes
    epochs = cfg.epochs
    steps_per_epoch = cfg.steps_per_epoch
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    learning_rate = cfg.learning_rate
    grad_clip = cfg.grad_clip
    diffusion_steps = cfg.diffusion_steps
    lambda_x0 = cfg.lambda_x0
    clean_epochs = cfg.clean_epochs
    min_train_snr_db = cfg.min_train_snr_db
    max_train_snr_db = cfg.max_train_snr_db
    validation_snr_db = cfg.validation_snr_db
    validate_every = cfg.validate_every
    fixed_count = cfg.fixed_count
    joint_checkpoint_path = cfg.joint_checkpoint_path
    save_root = cfg.save_root
    model_root = cfg.model_root
    latest_model_path = os.path.join(model_root, "DDPM_vaegan_using_VAE_recipe_latest.pth")
    history_path = os.path.join(save_root, "training_history.csv")
    validation_history_path = os.path.join(save_root, "validation_history.csv")
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(model_root, exist_ok=True)
    set_seed(seed)

    # Datasets and loaders.
    dataset_e, dataset_d, loader_e, loader_d, fixed_e, fixed_d = build_dataloaders(
        cfg.data_root_e, cfg.data_root_d, batch_size, num_workers, device, fixed_count
    )

    # ---------------------------
    # Frozen pretrained VAE-GAN
    # ---------------------------
    encoder, generator = load_vae_gan_models(
        joint_checkpoint_path=joint_checkpoint_path,
        device=device,
        z_dim=z_dim,
        label_dim=label_dim,
        num_classes=num_classes,
    )

    # ---------------------------
    # DDPM and optimizer
    # ---------------------------
    ddpm = LatentConditionalDDPM(
        z_dim=z_dim,
        num_classes=num_classes,
        time_dim=128,
        label_dim=64,
        hidden_dim=768,
        num_blocks=4,
        dropout=0.0,
    ).to(device)
    schedule = DiffusionSchedule(timesteps=diffusion_steps).to(device)
    optimizer = torch.optim.AdamW(ddpm.parameters(), lr=learning_rate, weight_decay=1e-4)

    print(f"Device             = {device}")
    print(f"EuroSAT images     = {len(dataset_e)}")
    print(f"DeepGlobe images   = {len(dataset_d)}")
    print(f"Steps per epoch    = {steps_per_epoch}")
    print(f"Diffusion steps    = {diffusion_steps}")

    iter_e = endless_batches(loader_e)
    iter_d = endless_batches(loader_d)
    global_step = 0
    best_validation_mse = float("inf")

    for epoch in range(1, epochs + 1):
        ddpm.train()
        epoch_total = 0.0
        epoch_eps_e = 0.0
        epoch_eps_d = 0.0
        epoch_x0_e = 0.0
        epoch_x0_d = 0.0
        epoch_latent_std_e = 0.0
        epoch_latent_std_d = 0.0

        for step in range(steps_per_epoch):
            image_e, _ = next(iter_e)
            image_d, _ = next(iter_d)
            image_e = image_e.to(device, non_blocking=True)
            image_d = image_d.to(device, non_blocking=True)

            with torch.no_grad():
                # Deterministic transmitted representation: use mu, not sampled z.
                z_e, _ = encoder(image_e)
                z_d, _ = encoder(image_d)

                z_mix = z_e + z_d
                if epoch <= clean_epochs:
                    train_snr_db = math.inf
                    z_rx = z_mix
                else:
                    train_snr_db = random.uniform(min_train_snr_db, max_train_snr_db)
                    z_rx = add_awgn_by_snr(z_mix, train_snr_db)

            batch_now = z_e.size(0)
            t_e = torch.randint(0, diffusion_steps, (batch_now,), device=device)
            t_d = torch.randint(0, diffusion_steps, (batch_now,), device=device)
            noise_e = torch.randn_like(z_e)
            noise_d = torch.randn_like(z_d)
            z_e_t = schedule.q_sample(z_e, t_e, noise_e)
            z_d_t = schedule.q_sample(z_d, t_d, noise_d)

            label_e = torch.zeros(batch_now, dtype=torch.long, device=device)
            label_d = torch.ones(batch_now, dtype=torch.long, device=device)

            pred_noise_e = ddpm(z_e_t, t_e, label_e, cond=z_rx)
            pred_noise_d = ddpm(z_d_t, t_d, label_d, cond=z_rx)

            loss_eps_e = F.mse_loss(pred_noise_e, noise_e)
            loss_eps_d = F.mse_loss(pred_noise_d, noise_d)

            pred_z0_e = schedule.predict_x0_from_eps(z_e_t, t_e, pred_noise_e)
            pred_z0_d = schedule.predict_x0_from_eps(z_d_t, t_d, pred_noise_d)
            loss_x0_e = F.mse_loss(pred_z0_e, z_e)
            loss_x0_d = F.mse_loss(pred_z0_d, z_d)

            loss_eps = 0.5 * (loss_eps_e + loss_eps_d)
            loss_x0 = 0.5 * (loss_x0_e + loss_x0_d)
            loss = loss_eps + lambda_x0 * loss_x0

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), grad_clip)
            optimizer.step()

            epoch_total += loss.item()
            epoch_eps_e += loss_eps_e.item()
            epoch_eps_d += loss_eps_d.item()
            epoch_x0_e += loss_x0_e.item()
            epoch_x0_d += loss_x0_d.item()
            epoch_latent_std_e += z_e.std().item()
            epoch_latent_std_d += z_d.std().item()

            if global_step % 200 == 0:
                snr_text = "clean" if math.isinf(train_snr_db) else f"{train_snr_db:.1f} dB"
                print(
                    f"Epoch {epoch}/{epochs} | Step {step}/{steps_per_epoch} | "
                    f"SNR={snr_text} | Total={loss.item():.4f} | "
                    f"eps_E={loss_eps_e.item():.4f} | eps_D={loss_eps_d.item():.4f} | "
                    f"x0_E={loss_x0_e.item():.4f} | x0_D={loss_x0_d.item():.4f}"
                )
            global_step += 1

        averages = {
            "epoch": epoch,
            "total_loss": epoch_total / steps_per_epoch,
            "eps_loss_e": epoch_eps_e / steps_per_epoch,
            "eps_loss_d": epoch_eps_d / steps_per_epoch,
            "x0_loss_e": epoch_x0_e / steps_per_epoch,
            "x0_loss_d": epoch_x0_d / steps_per_epoch,
            "latent_std_e": epoch_latent_std_e / steps_per_epoch,
            "latent_std_d": epoch_latent_std_d / steps_per_epoch,
        }
        append_csv(history_path, averages)

        print("=" * 110)
        print(
            f"Epoch {epoch}/{epochs} | Total={averages['total_loss']:.4f} | "
            f"eps_E={averages['eps_loss_e']:.4f} | eps_D={averages['eps_loss_d']:.4f} | "
            f"x0_E={averages['x0_loss_e']:.4f} | x0_D={averages['x0_loss_d']:.4f} | "
            f"std_E={averages['latent_std_e']:.4f} | std_D={averages['latent_std_d']:.4f}"
        )
        print("=" * 110)

        checkpoint = {
            "epoch": epoch,
            "model": ddpm.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": {
                "z_dim": z_dim,
                "label_dim": label_dim,
                "num_classes": num_classes,
                "diffusion_steps": diffusion_steps,
                "lambda_x0": lambda_x0,
                "joint_checkpoint_path": joint_checkpoint_path,
                "domain_labels": {"EuroSAT": 0, "DeepGlobe": 1},
            },
        }
        torch.save(checkpoint, latest_model_path)

        if epoch % validate_every == 0 or epoch == 1 or epoch == epochs:
            validation_metrics = validate(
                model=ddpm,
                schedule=schedule,
                encoder=encoder,
                generator=generator,
                fixed_e=fixed_e,
                fixed_d=fixed_d,
                epoch=epoch,
                save_root=save_root,
                z_dim=z_dim,
                validation_snr_db=validation_snr_db,
            )
            append_csv(validation_history_path, {"epoch": epoch, **validation_metrics})
            print("Validation:", validation_metrics)
            checkpoint["validation"] = validation_metrics
            torch.save(
                checkpoint,
                os.path.join(model_root, f"DDPM_vaegan_using_VAE_recipe_ep{epoch}.pth"),
            )

            avg_latent_mse = 0.5 * (
                validation_metrics["latent_mse_e"]
                + validation_metrics["latent_mse_d"]
            )
            if avg_latent_mse < best_validation_mse:
                best_validation_mse = avg_latent_mse
                torch.save(
                    checkpoint,
                    os.path.join(
                        model_root,
                        "DDPM_vaegan_scratch_latent_separator_best.pth",
                    ),
                )
                print(
                    f"Saved best DDPM | avg latent MSE={best_validation_mse:.6f}"
                )

    print("Training completed.")
    print(f"Latest DDPM checkpoint: {latest_model_path}")
    print(f"Validation images:      {save_root}")

if __name__ == "__main__":
    main()
