"""Latent-space plumbing: a frozen pretrained VAE plus a latent cache.

Why a pretrained VAE and not one trained here
---------------------------------------------
512x512 generation does not happen in pixel space. It happens in a 64x64x4
latent space produced by a variational autoencoder, which is what makes the
compute affordable at all. Training a VAE good enough for that from scratch
needs far more data and compute than this project has, so `stabilityai/sd-vae-ft-mse`
is used as a fixed, frozen compressor.

That distinction matters and is stated plainly in the README: the *compressor*
is pretrained, the *generator* (the diffusion model in latent space) is trained
from scratch here.

The cache step encodes the whole dataset once and stores float16 latents, so
training epochs never touch the VAE again. On a 2-core runner this turns a
~2 s/image cost into a one-time cost.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

VAE_ID = "stabilityai/sd-vae-ft-mse"
SCALING_FACTOR = 0.18215
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


def load_vae(model_id: str = VAE_ID, device: str = "cpu"):
    """Load the frozen VAE. Returns the model in eval mode with grads disabled."""
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(model_id, torch_dtype=torch.float32)
    vae.eval().to(device)
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    return vae


def list_images(root: Path) -> list[Path]:
    return [p for p in sorted(root.rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES]


@torch.no_grad()
def encode_images(
    vae,
    paths: list[Path],
    out_path: Path,
    size: int = 512,
    batch: int = 4,
    device: str = "cpu",
) -> np.ndarray:
    """Encode every image to a scaled latent and save it as float16 .npy.

    A sidecar `.json` records the source paths and a fingerprint of the settings,
    so a stale cache is detected instead of silently reused.
    """
    from PIL import Image

    latents: list[np.ndarray] = []
    for start in range(0, len(paths), batch):
        chunk = paths[start : start + batch]
        pixels = np.stack(
            [
                np.asarray(
                    Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC),
                    dtype=np.float32,
                )
                for path in chunk
            ]
        )
        tensor = torch.from_numpy(pixels.transpose(0, 3, 1, 2) / 127.5 - 1.0).to(device)
        latent = vae.encode(tensor).latent_dist.sample() * SCALING_FACTOR
        latents.append(latent.cpu().numpy().astype(np.float16))
        if (start // batch) % 25 == 0:
            print(f"  encoded {min(start + batch, len(paths))}/{len(paths)}", flush=True)

    stacked = np.concatenate(latents, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, stacked)
    return stacked


def cache_fingerprint(paths: list[Path], size: int) -> str:
    digest = hashlib.sha256()
    digest.update(str(size).encode())
    for path in paths:
        digest.update(path.name.encode())
        digest.update(str(path.stat().st_size).encode())
    return digest.hexdigest()[:16]


@torch.no_grad()
def decode_latents(vae, latents: torch.Tensor, device: str = "cpu") -> np.ndarray:
    """Turn scaled latents back into uint8 HWC images."""
    pixels = vae.decode(latents.to(device) / SCALING_FACTOR).sample
    pixels = ((pixels.clamp(-1, 1) + 1) / 2 * 255).byte().cpu().numpy()
    return pixels.transpose(0, 2, 3, 1)
