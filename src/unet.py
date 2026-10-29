"""A small UNet that predicts noise in a 64x64x4 latent space.

Why a UNet and not a transformer
--------------------------------
The obvious modern answer is a diffusion transformer (DiT). It is not the right
choice here, and the numbers say so:

  - A 64x64 latent with patch size 2 is 1,024 tokens. Self-attention at that
    length is 1,024^2 scores per head per block, and on two CPU cores attention
    has no accelerated kernel: PyTorch falls back to a matmul chain, while
    convolutions go through oneDNN and are several times faster per FLOP.
  - Attention also scales worse with resolution. Doubling to 128x128 latents
    would quadruple the token count and square the attention cost, while a
    convolution simply strides.
  - DiT's advantage — better scaling with data and parameters — only pays off
    in the regime where it was validated: hundreds of millions of parameters and
    GPU hours. At 10 M parameters on 1,500 images, it would train slower and
    converge no better.

So the efficient choice on this hardware is a convolutional UNet with attention
only at the lowest resolution (16x16, 256 tokens), which is where global
structure actually matters. That is the architecture below. A DiT variant lives
in `side/` as a note, not as a dependency of this project.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(steps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(half, device=steps.device) / half
    )
    arguments = steps[:, None].float() * frequencies[None]
    return torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=1)


class TimeBlock(nn.Module):
    """Residual conv block with the timestep embedding injected additively."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(F.silu(time_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Single-head self-attention used only at the coarsest resolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        tokens = height * width
        residual = x
        h = self.norm(x)
        q = self.query(h).view(batch, channels, tokens).transpose(1, 2)
        k = self.key(h).view(batch, channels, tokens)
        v = self.value(h).view(batch, channels, tokens).transpose(1, 2)
        attention = torch.softmax(q @ k / math.sqrt(channels), dim=-1)
        out = (attention @ v).transpose(1, 2).view(batch, channels, height, width)
        return residual + self.proj(out)


class LatentUNet(nn.Module):
    """Predicts the noise added to a (4, 64, 64) latent at a given timestep."""

    def __init__(
        self,
        in_channels: int = 4,
        base: int = 64,
        time_dim: int = 256,
        blocks: int = 2,
    ):
        super().__init__()
        widths = (base, base * 2, base * 4)  # 64 -> 128 -> 256
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )

        self.stem = nn.Conv2d(in_channels, widths[0], 3, padding=1)

        self.down1 = nn.ModuleList(
            [TimeBlock(widths[0], widths[0], time_dim) for _ in range(blocks)]
        )
        self.pool1 = nn.Conv2d(widths[0], widths[0], 4, 2, 1)

        self.down2 = nn.ModuleList(
            [TimeBlock(widths[0], widths[1], time_dim)]
            + [TimeBlock(widths[1], widths[1], time_dim) for _ in range(blocks - 1)]
        )
        self.pool2 = nn.Conv2d(widths[1], widths[1], 4, 2, 1)

        self.mid = nn.ModuleList(
            [
                TimeBlock(widths[1], widths[2], time_dim),
                SelfAttention(widths[2]),
                TimeBlock(widths[2], widths[2], time_dim),
            ]
        )

        # only the first block of each up stage receives the concatenated skip,
        # the rest run at the stage width
        self.up2 = nn.ModuleList(
            [TimeBlock(widths[2] + widths[1], widths[1], time_dim)]
            + [TimeBlock(widths[1], widths[1], time_dim) for _ in range(blocks - 1)]
        )
        self.up1 = nn.ModuleList(
            [TimeBlock(widths[1] + widths[0], widths[0], time_dim)]
            + [TimeBlock(widths[0], widths[0], time_dim) for _ in range(blocks - 1)]
        )
        self.head = nn.Sequential(
            nn.GroupNorm(8, widths[0]), nn.SiLU(), nn.Conv2d(widths[0], in_channels, 3, padding=1)
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_mlp(timestep_embedding(timesteps, self.time_dim))
        h1 = self.stem(x)
        for block in self.down1:
            h1 = block(h1, time_emb)
        h2 = self.pool1(h1)
        for block in self.down2:
            h2 = block(h2, time_emb)
        h3 = self.pool2(h2)

        for block in self.mid:
            h3 = block(h3) if isinstance(block, SelfAttention) else block(h3, time_emb)

        h3 = F.interpolate(h3, size=h2.shape[-2:], mode="nearest")
        h2 = torch.cat([h2, h3], dim=1)
        for block in self.up2:
            h2 = block(h2, time_emb)

        h2 = F.interpolate(h2, size=h1.shape[-2:], mode="nearest")
        h1 = torch.cat([h1, h2], dim=1)
        for block in self.up1:
            h1 = block(h1, time_emb)
        return self.head(h1)
