"""Generate 512x512 images and video from a trained latent diffusion model.

    python src/sample.py --checkpoint runs/ldm/ldm.pt --count 4 --out samples
    python src/sample.py --checkpoint runs/ldm/ldm.pt --video --frames 32
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diffusion import DiffusionSchedule, ddim_sample, interpolate_latents  # noqa: E402
from unet import LatentUNet  # noqa: E402
from vae_io import decode_latents, load_vae  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/ldm/ldm.pt", type=Path)
    parser.add_argument("--out", default="samples", type=Path)
    parser.add_argument("--count", default=4, type=int)
    parser.add_argument("--steps", default=50, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--frames", default=32, type=int)
    parser.add_argument("--fps", default=12, type=int)
    parser.add_argument("--threads", default=2, type=int)
    return parser.parse_args()


def load_model(path: Path) -> tuple[LatentUNet, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = LatentUNet(base=checkpoint.get("base", 64))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint.get("latent_size", 64)


def save_grid(images: np.ndarray, path: Path, columns: int = 2) -> None:
    from PIL import Image

    count, height, width, _ = images.shape
    rows = (count + columns - 1) // columns
    canvas = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        canvas[row * height : (row + 1) * height, column * width : (column + 1) * width] = image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)


def write_video(frames: list[np.ndarray], path: Path, fps: int) -> Path:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)
    return path


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)

    model, latent_size = load_model(args.checkpoint)
    schedule = DiffusionSchedule()
    vae = load_vae()
    generator = torch.Generator().manual_seed(args.seed)

    latents = ddim_sample(
        model,
        schedule,
        shape=(args.count, 4, latent_size, latent_size),
        steps=args.steps,
        generator=generator,
    )
    images = decode_latents(vae, latents)
    save_grid(images, args.out / f"grid_seed{args.seed}.png")
    print(f"wrote {args.out / f'grid_seed{args.seed}.png'} ({args.count} images, 512x512)")

    if args.video:
        start = latents[:1]
        end = latents[1:2] if args.count > 1 else latents[:1]
        walk = interpolate_latents(start, end, args.frames)
        decoded = decode_latents(vae, torch.cat(walk, dim=0))
        path = write_video(list(decoded), args.out / f"walk_seed{args.seed}.mp4", args.fps)
        print(f"wrote {path} ({args.frames} frames)")


if __name__ == "__main__":
    main()
