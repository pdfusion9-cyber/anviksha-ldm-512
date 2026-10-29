"""Denoising diffusion maths: schedule, training loss, DDIM sampling."""
from __future__ import annotations

import torch


class DiffusionSchedule:
    """Linear beta schedule with the derived quantities cached as tensors."""

    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4,
                 beta_end: float = 2e-2, device: str = "cpu"):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus = torch.sqrt(1.0 - self.alpha_bar)

    def to(self, device: str) -> "DiffusionSchedule":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.sqrt_alpha_bar = self.sqrt_alpha_bar.to(device)
        self.sqrt_one_minus = self.sqrt_one_minus.to(device)
        return self

    def q_sample(self, clean: torch.Tensor, timesteps: torch.Tensor,
                 noise: torch.Tensor) -> torch.Tensor:
        """Forward process: add noise to a clean latent."""
        a = self.sqrt_alpha_bar[timesteps][:, None, None, None]
        b = self.sqrt_one_minus[timesteps][:, None, None, None]
        return a * clean + b * noise


def training_loss(model, schedule: DiffusionSchedule, batch: torch.Tensor) -> torch.Tensor:
    """Standard epsilon-prediction objective."""
    noise = torch.randn_like(batch)
    timesteps = torch.randint(
        0, schedule.timesteps, (len(batch),), device=batch.device
    ).long()
    noisy = schedule.q_sample(batch, timesteps, noise)
    predicted = model(noisy, timesteps.float())
    return torch.nn.functional.mse_loss(predicted, noise)


@torch.no_grad()
def ddim_sample(
    model,
    schedule: DiffusionSchedule,
    shape: tuple[int, int, int, int],
    steps: int = 50,
    eta: float = 0.0,
    generator: torch.Generator | None = None,
    device: str = "cpu",
    progress=None,
) -> torch.Tensor:
    """Deterministic DDIM sampling (eta=0). Returns scaled latents."""
    sample = torch.randn(shape, generator=generator, device=device)
    # evenly spaced subset of the training timesteps, descending
    sequence = torch.linspace(
        schedule.timesteps - 1, 0, steps, device=device
    ).round().long()

    for index, timestep in enumerate(sequence):
        t = torch.full((shape[0],), int(timestep), device=device, dtype=torch.long)
        predicted_noise = model(sample, t.float())

        alpha_bar_t = schedule.alpha_bar[timestep]
        previous = sequence[index + 1] if index + 1 < len(sequence) else None
        alpha_bar_prev = (
            schedule.alpha_bar[previous] if previous is not None
            else torch.tensor(1.0, device=device)
        )

        # estimate the clean latent and step towards it
        clean = (sample - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(
            alpha_bar_t
        )
        clean = clean.clamp(-6, 6)
        direction = torch.sqrt(1 - alpha_bar_prev) * predicted_noise
        sample = torch.sqrt(alpha_bar_prev) * clean + direction

        if progress is not None and index % 10 == 0:
            progress(index / len(sequence), desc=f"sampling {index}/{len(sequence)}")

    return sample


def interpolate_latents(
    start: torch.Tensor,
    end: torch.Tensor,
    frames: int,
) -> list[torch.Tensor]:
    """Slerp between two clean latents and return one latent per frame.

    The endpoints come from real denoising runs, so both sit on the data
    manifold; a spherical path between them stays closer to it than a straight
    line does. Frames are decoded straight away — no diffusion is re-run, which
    is why this is fast and why it is motion in latent space rather than a
    learned temporal model.
    """
    return [
        _slerp(start, end, frame / max(frames - 1, 1))
        for frame in range(frames)
    ]


def _slerp(a: torch.Tensor, b: torch.Tensor, weight: float) -> torch.Tensor:
    a_flat, b_flat = a.flatten(1), b.flatten(1)
    a_norm = a_flat / a_flat.norm(dim=1, keepdim=True).clamp_min(1e-8)
    b_norm = b_flat / b_flat.norm(dim=1, keepdim=True).clamp_min(1e-8)
    cosine = (a_norm * b_norm).sum(dim=1, keepdim=True).clamp(-1, 1)
    omega = torch.acos(cosine)
    sine = torch.sin(omega)
    safe = torch.where(sine.abs() < 1e-6, torch.ones_like(sine), sine)
    out = (
        torch.sin((1 - weight) * omega) / safe * a_flat
        + torch.sin(weight * omega) / safe * b_flat
    )
    return out.view_as(a)
