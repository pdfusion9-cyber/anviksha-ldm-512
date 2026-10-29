"""Train the latent diffusion model.

Two stages, both resumable:

  1. `--stage cache`   encode every training image to a 64x64x4 latent with the
                       frozen VAE and store it as float16. Runs once.
  2. `--stage train`   train the UNet on cached latents. Cheap: no VAE in the
                       loop, no JPEG decoding, no resizing.

    python src/train.py --stage cache --data data/images --out runs/ldm
    python src/train.py --stage train --out runs/ldm --epochs 30 --time-budget 18000

The time budget is in seconds and exists because cloud runners are killed at a
fixed limit. The checkpoint stores model, optimiser and epoch, so the next run
continues without a loss spike.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diffusion import DiffusionSchedule, training_loss  # noqa: E402
from unet import LatentUNet  # noqa: E402
from vae_io import cache_fingerprint, encode_images, list_images, load_vae  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["cache", "train"], required=True)
    parser.add_argument("--data", default="data/images", type=Path)
    parser.add_argument("--out", default="runs/ldm", type=Path)
    parser.add_argument("--image-size", default=512, type=int)
    parser.add_argument("--latent-size", default=64, type=int)
    parser.add_argument("--base", default=64, type=int)
    parser.add_argument("--batch", default=4, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--threads", default=2, type=int)
    parser.add_argument("--time-budget", default=0.0, type=float)
    return parser.parse_args()


def run_cache(args: argparse.Namespace) -> Path:
    paths = list_images(args.data)
    if not paths:
        raise SystemExit(f"no images under {args.data}")
    print(f"encoding {len(paths)} images to latents")

    vae = load_vae()
    latent_path = args.out / "latents.npy"
    latents = encode_images(
        vae, paths, latent_path, size=args.image_size, batch=max(args.batch, 2)
    )
    (args.out / "latents.json").write_text(
        json.dumps(
            {
                "count": int(len(latents)),
                "shape": list(latents.shape),
                "image_size": args.image_size,
                "fingerprint": cache_fingerprint(paths, args.image_size),
            },
            indent=2,
        )
    )
    print(f"cached {latents.shape} -> {latent_path}")
    return latent_path


def run_train(args: argparse.Namespace) -> None:
    latent_path = args.out / "latents.npy"
    if not latent_path.exists():
        raise SystemExit("no latent cache; run --stage cache first")

    latents = torch.from_numpy(np.load(latent_path).astype(np.float32))
    print(f"loaded latents {tuple(latents.shape)}")

    model = LatentUNet(base=args.base)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"unet parameters: {parameters:,}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule = DiffusionSchedule()

    checkpoint_path = args.out / "ldm.pt"
    start_epoch = 0
    history: list[dict] = []
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimiser.load_state_dict(checkpoint["optimiser"])
        start_epoch = checkpoint["epoch"]
        history = checkpoint["history"]
        print(f"resumed at epoch {start_epoch}")

    started = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        order = np.random.default_rng(epoch).permutation(len(latents))
        total = 0.0
        steps = 0
        for start in range(0, len(order), args.batch):
            index = order[start : start + args.batch]
            if len(index) < 2:
                continue
            batch = latents[index]
            optimiser.zero_grad()
            loss = training_loss(model, schedule, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total += loss.item()
            steps += 1

        mean_loss = total / max(steps, 1)
        history.append({"epoch": epoch + 1, "loss": mean_loss})
        print(
            f"epoch {epoch + 1:3d}/{args.epochs}  loss {mean_loss:.5f}  "
            f"({time.time() - started:.0f}s)",
            flush=True,
        )

        torch.save(
            {
                "model": model.state_dict(),
                "optimiser": optimiser.state_dict(),
                "epoch": epoch + 1,
                "history": history,
                "base": args.base,
                "latent_size": args.latent_size,
            },
            checkpoint_path,
        )
        (args.out / "metrics.json").write_text(
            json.dumps(
                {
                    "parameters": parameters,
                    "epochs_completed": len(history),
                    "history": history,
                },
                indent=2,
            )
        )

        if args.time_budget and time.time() - started > args.time_budget:
            print("time budget reached; rerun to resume")
            break


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == "cache":
        run_cache(args)
    else:
        run_train(args)


if __name__ == "__main__":
    main()
