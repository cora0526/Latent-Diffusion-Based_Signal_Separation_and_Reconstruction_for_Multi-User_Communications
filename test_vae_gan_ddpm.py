"""Evaluate a pretrained VAE-GAN + conditional latent DDPM separator.

Uses the SAME validate() inference path as train.py, so reverse diffusion,
latent scaling, decoding and output images stay consistent with your project.

Example:
    python test_vae_gan_ddpm.py --snrs 0 5 10 15 20 25 30 --fixed-count 10
"""

import argparse
import csv
import json
import math
import os
import random

import torch

import config as cfg
from dataset import build_dataloaders
from model import LatentConditionalDDPM, DiffusionSchedule
from utils_ddpm import set_seed, load_vae_gan_models, validate


def parse_args():
    p = argparse.ArgumentParser(description="Test VAE-GAN + DDPM separation at multiple SNRs")
    p.add_argument("--checkpoint", default=os.path.join(cfg.model_root, "DDPM_vaegan_scratch_latent_separator_best.pth"),
                   help="Trained DDPM checkpoint")
    p.add_argument("--vae-checkpoint", default=cfg.joint_checkpoint_path,
                   help="Pretrained joint VAE-GAN checkpoint")
    p.add_argument("--output", default=os.path.join(cfg.save_root, "test_results"))
    p.add_argument("--snrs", nargs="+", type=float, default=[0, 5, 10, 15, 20, 25, 30],
                   help="AWGN SNR values in dB")
    p.add_argument("--fixed-count", type=int, default=10,
                   help="Number of fixed images selected from EACH dataset by build_dataloaders")
    p.add_argument("--batch-size", type=int, default=cfg.batch_size)
    p.add_argument("--num-workers", type=int, default=cfg.num_workers)
    p.add_argument("--seed", type=int, default=cfg.seed)
    return p.parse_args()


def main():
    args = parse_args()
    if args.fixed_count < 1:
        raise ValueError("--fixed-count must be >= 1")
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"DDPM checkpoint not found: {args.checkpoint}")
    if not os.path.isfile(args.vae_checkpoint):
        raise FileNotFoundError(f"VAE-GAN checkpoint not found: {args.vae_checkpoint}")

    os.makedirs(args.output, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Exactly the dataset interface used in train.py.
    dataset_e, dataset_d, _, _, fixed_e, fixed_d = build_dataloaders(
        cfg.data_root_e, cfg.data_root_d,
        args.batch_size, args.num_workers, device, args.fixed_count
    )
    print(f"EuroSAT images available: {len(dataset_e)}")
    print(f"DeepGlobe images available: {len(dataset_d)}")

    encoder, generator = load_vae_gan_models(
        joint_checkpoint_path=args.vae_checkpoint,
        device=device, z_dim=cfg.z_dim,
        label_dim=cfg.label_dim, num_classes=cfg.num_classes,
    )
    encoder.eval()
    generator.eval()

    # These model dimensions must match train.py. If you change train.py,
    # make the same changes here and retrain before loading the checkpoint.
    ddpm = LatentConditionalDDPM(
        z_dim=cfg.z_dim, num_classes=cfg.num_classes,
        time_dim=128, label_dim=64, hidden_dim=768,
        num_blocks=4, dropout=0.0,
    ).to(device)

    # Trusted local training checkpoint. A state-dict-only checkpoint is also supported.
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        saved_config = ckpt.get("config", {})
        for key in ("z_dim", "num_classes", "diffusion_steps"):
            if key in saved_config and saved_config[key] != getattr(cfg, key):
                raise ValueError(f"Checkpoint {key}={saved_config[key]} but config.py has {getattr(cfg, key)}")
        epoch = int(ckpt.get("epoch", 0))
    else:
        state_dict = ckpt
        epoch = 0
    ddpm.load_state_dict(state_dict, strict=True)
    ddpm.eval()
    schedule = DiffusionSchedule(timesteps=cfg.diffusion_steps).to(device)
    print(f"Loaded DDPM: {args.checkpoint} (epoch {epoch})")

    rows = []
    for snr in args.snrs:
        if not math.isfinite(snr):
            raise ValueError("All --snrs must be finite dB values")
        # Different subfolders avoid overwriting the validation images.
        tag = f"snr_{snr:g}dB".replace("-", "minus_").replace(".", "p")
        snr_dir = os.path.join(args.output, tag)
        os.makedirs(snr_dir, exist_ok=True)
        # Compare SNRs on the same fixed samples with a reproducible RNG.
        set_seed(args.seed)
        print(f"\nTesting SNR={snr:g} dB -> {snr_dir}")
        with torch.no_grad():
            metrics = validate(
                model=ddpm, schedule=schedule,
                encoder=encoder, generator=generator,
                fixed_e=fixed_e, fixed_d=fixed_d,
                epoch=epoch, save_root=snr_dir,
                z_dim=cfg.z_dim, validation_snr_db=snr,
            )
        if not isinstance(metrics, dict):
            raise TypeError(f"validate() must return a dict, got {type(metrics).__name__}")
        row = {"snr_db": snr, **metrics}
        rows.append(row)
        print("Metrics:", row)
        with open(os.path.join(snr_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(row, f, ensure_ascii=False, indent=2, default=float)

    # Union of metric names; validate() decides which metrics are available.
    columns = ["snr_db"] + sorted({key for row in rows for key in row if key != "snr_db"})
    csv_path = os.path.join(args.output, "test_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Results CSV: {csv_path}")
    print(f"Saved images (if implemented by validate): {args.output}")
    print("Note: these are fixed-sample evaluations, not necessarily a held-out test set.")


if __name__ == "__main__":
    main()
