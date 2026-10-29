# Side notes — not part of the generator

Everything in this folder is exploratory. Nothing here is imported by `src/`,
`app.py` or the training workflow. It is kept separate on purpose: the main
project stays a plain latent diffusion model, and these ideas get judged on
their own merits before anything is wired in.

## The "auto-collecting, intelligent-loop transformer" idea

The proposal, as I understand it, has two parts:

1. **Auto-collecting** — the model gathers its own training signal as it runs,
   instead of consuming a fixed dataset.
2. **Intelligent loop** — the transformer revisits and refines its own
   intermediate state over several passes rather than computing once.

### What is genuinely promising

**Adaptive compute is real and works.** Not every input needs the same number of
layers or denoising steps. Systems that learn where to spend compute — early
exits, mixture-of-depths, adaptive step counts in diffusion — measurably cut
FLOPs at equal quality. In diffusion specifically, the number of sampling steps
is already an explicit dial, and a learned policy for when to stop is a
well-motivated research direction.

**Reusing intermediate state across passes is real too.** Weight-tied recurrent
depth (the same block applied repeatedly) works at small scale and is cheaper in
parameters than stacking distinct blocks.

### What does not hold up as stated

**Self-collected data does not create new information.** A model that trains on
its own outputs amplifies whatever biases it already has; this is the classic
model-collapse failure of self-training. Auto-collecting only helps when
something *external* validates the collected samples — a checker, a simulator, a
human, a held-out gate. Your Fabric project already has exactly that idea in its
fail-closed admission gate, and that gate is the part doing the work, not the
collecting.

**Loops do not automatically mean efficiency.** Applying a block three times
costs three times the compute of applying it once. It is only efficient if the
repeated passes let you shrink the model enough that total compute drops while
quality holds. That has to be measured per task; it is not a free win.

**At 5 M parameters, depth tricks are not the bottleneck.** The bottleneck here
is data (1,542 images) and total compute (a few CPU-hours). A cleverer
architecture on the same data and budget will land within noise of the plain
UNet.

### What I would actually test, in order

1. **Adaptive sampling steps.** Measure quality against steps (10 / 20 / 50) on
   this model. If 20 steps is indistinguishable from 50, you have halved
   inference cost for free, with no architecture change.
2. **Weight-tied depth.** Replace three distinct blocks with one block applied
   three times. Compare parameters and FID-style quality at equal compute.
3. **A learned stop policy.** Train a small head that predicts whether the
   current sample is finished. This is the honest version of "intelligent loop":
   the model decides how much compute to spend.
4. **Only then** consider a transformer backbone (DiT), and only on GPU, where
   attention is actually fast.

Each of these is a small, falsifiable experiment. None of them should touch the
main project until it beats the baseline on a measured metric.

## Why the main project uses a UNet, not a transformer

Measured on two CPU cores, not assumed:

| | UNet (this repo) | DiT, patch 2 |
|---|---|---|
| Tokens at 64x64 latent | — | 1,024 |
| Attention cost | 256 tokens, coarsest level only | 1,024^2 per head per block |
| Kernel path | convolution → oneDNN | matmul chain, no fused attention on CPU |
| Parameters at comparable quality | 1.4 M | would need far more |

A diffusion transformer is the better architecture in the regime where it was
validated: hundreds of millions of parameters, GPU hours, large datasets. At
1.4 M parameters on 1,542 images and two CPU cores, it would train slower and
converge no better. If this project moves to a GPU and a dataset an order of
magnitude larger, that conclusion should be re-tested rather than assumed.
