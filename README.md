# Anviksha LDM-512 — 512×512 image and video generation on CPU

Latent diffusion at 512×512, trained on free GitHub Actions CPU runners.

The honest summary first: the **compressor is pretrained** (a frozen Stable
Diffusion VAE), the **generator is trained from scratch here** (a 1.4 M
parameter UNet in latent space). Training a 512×512 diffusion model entirely
from scratch — VAE included — needs GPU-months and millions of images. This
project does the part that is actually achievable on CPU, and says so.

## How 512×512 fits on a CPU

The trick is that nothing is generated at 512×512. The VAE compresses each image
to a **64×64×4 latent** — a 48× reduction in spatial size and 192× fewer values.
Diffusion runs entirely in that latent space, and the frozen VAE decodes the
result back to 512×512.

| Stage | Resolution | What happens |
|---|---|---|
| Encode | 512×512×3 → 64×64×4 | frozen VAE, once per image, cached to disk |
| Train | 64×64×4 | UNet predicts noise, ~2 s/step at batch 4 on 2 cores |
| Sample | 64×64×4 → 512×512×3 | DDIM 50 steps (~10 s/image), then decode |

## Measured cost on two CPU cores

| Operation | Time |
|---|---|
| One training step (batch 4) | 2.09 s |
| One epoch (1,542 images) | 13.4 min |
| 50-step DDIM sample, one 512×512 image | ~10 s |
| VAE encode of the whole dataset | ~1–1.5 h (one time) |

A 6-hour GitHub Actions job therefore completes the latent cache and roughly
20 training epochs.

## Architecture

```
src/vae_io.py     frozen VAE, latent encode/decode, disk cache
src/unet.py       1.4 M parameter UNet, attention only at 16x16
src/diffusion.py  DDPM schedule, epsilon objective, DDIM sampler, slerp
src/train.py      --stage cache | train, both resumable
src/sample.py     512x512 images and latent-walk video
app.py            Gradio UI (optional)
side/             exploratory notes, not wired into the project
```

## Why a UNet and not a diffusion transformer

Measured, not assumed. A 64×64 latent at patch size 2 is 1,024 tokens; attention
at that length is 1,024² scores per head per block, and on CPU PyTorch has no
fused attention kernel — it falls back to matmul chains, while convolutions go
through oneDNN. At 1.4 M parameters and 1,542 images, a DiT would train slower
and converge no better. The reasoning, and the conditions under which it should
be re-tested, are in `side/intelligent-loop-notes.md`.

Attention is not absent: it runs at the coarsest level (16×16 = 256 tokens),
which is where global structure matters and where the cost is affordable.

## Video

The video path slerps between two latents produced by real denoising runs and
decodes each frame. Both endpoints sit on the data manifold, so the path between
them stays close to it.

**This is motion in latent space, not a temporal model.** There is no learned
dynamics, no consistency of identity across frames beyond what the VAE latent
happens to preserve, and no understanding of what should happen next. A real
video generator needs video training data and GPU compute; this is the honest
version of what CPU can do.

## Train it

The workflow runs on a public-repo GitHub Actions runner (2 vCPU, 7 GB).

```bash
gh workflow run train-ldm-512 -f epochs=20 -f minutes=240
```

First run caches latents (~1 h) and trains (~4 h), then uploads `latents.npy`
and `ldm.pt` as artifacts. Later runs pass `cache_run_id` to reuse the cache
instead of re-encoding.

Locally:

```bash
pip install -r requirements.txt
python src/train.py --stage cache --data data/images --out runs/ldm
python src/train.py --stage train --out runs/ldm --epochs 30
python src/sample.py --checkpoint runs/ldm/ldm.pt --count 4 --video
```

## Honest expectations

- **20 epochs on 1,542 images will not produce photorealistic 512×512 faces.**
  Expect coherent colour, layout and stage lighting at 512×512 resolution, with
  soft or malformed detail. Diffusion at this data scale is data-starved, not
  architecture-starved.
- **The VAE is pretrained.** It carries whatever biases the Stable Diffusion
  VAE has, and it was trained on web imagery, not on Indian dance photography.
- **No text conditioning.** This is unconditional generation. Prompt-to-image
  needs a text encoder and paired captions, which is a larger project.
- **The test of success is the sample grid**, not a loss curve. Diffusion loss
  is a poor proxy for perceptual quality.

## Licence

Code: Apache-2.0. Training images: `tanmaykm/indian_dance_forms` (Apache-2.0),
downloaded at runtime. VAE weights: `stabilityai/sd-vae-ft-mse` (MIT), loaded at
runtime, not redistributed.
