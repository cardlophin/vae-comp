<div align="center">

# vae-comp

**Dense VAE vs. convolutional VAE on MNIST, written in Rust with [Burn](https://burn.dev)**

![Rust 2024](https://img.shields.io/badge/Rust-2024-b7410e?style=flat-square&logo=rust&logoColor=white)
![Burn 0.22](https://img.shields.io/badge/Burn-0.22-2a78d6?style=flat-square)
![Backend Metal](https://img.shields.io/badge/backend-Metal-52514e?style=flat-square&logo=apple&logoColor=white)
![Dataset MNIST](https://img.shields.io/badge/dataset-MNIST-eb6834?style=flat-square)

</div>

This repository trains two variational autoencoders under the **same data split, loss,
optimiser, schedule and seed**, so that the only thing that changes is the architecture:

- **Dense VAE**: a multilayer perceptron that treats each digit as a flat 784-vector.
- **CNN-VAE**: a convolutional encoder and a transposed-convolution decoder that keep the 28×28 spatial structure.

Both use a 16-dimensional Gaussian latent space and a Bernoulli decoder.

## Results at a glance

Test-set metrics for the checkpoint with the lowest validation loss, averaged over
5 Monte Carlo samples of `z`. Losses are per image, in nats (lower is better).

| Model | Params | Best epoch | **−ELBO** | Reconstruction (BCE) | KL | Train time / epoch¹ |
| :-- | --: | --: | --: | --: | --: | --: |
| Dense VAE | 829,232 | 95 | 100.75 | 76.83 | **23.92** | **0.52 s** |
| CNN-VAE | **220,385** | 98 | **96.66** | **72.65** | 24.00 | 3.04 s |

¹ Mean over epochs 2–100 on Apple Metal (epoch 1 includes kernel compilation).

> [!NOTE]
> The CNN-VAE reaches a **4.1 nat lower negative ELBO with 3.8× fewer parameters**.
> Almost all of the gain comes from reconstruction (−4.2 nats of BCE): both models spend
> roughly the same ~24 nats of KL on the latent code, but the convolutional decoder
> uses it more efficiently. The price is about 6× more compute per epoch.

## Method

### Variational autoencoder

A VAE learns a latent-variable model $p_\theta(x, z) = p_\theta(x \mid z)\,p(z)$ with prior
$p(z) = \mathcal{N}(0, I)$, and an amortised encoder
$q_\phi(z \mid x) = \mathcal{N}\big(\mu_\phi(x), \operatorname{diag}\sigma^2_\phi(x)\big)$.
Both networks are trained jointly by maximising the evidence lower bound (ELBO).
In this repository we minimise its negative:

$$
\mathcal{L}(x) = \underbrace{-\,\mathbb{E}_{q_\phi(z\mid x)}\big[\log p_\theta(x \mid z)\big]}_{\text{reconstruction (BCE)}}
\;+\; \beta\,\underbrace{D_{\mathrm{KL}}\big(q_\phi(z \mid x)\,\|\,p(z)\big)}_{\text{regulariser}}
$$

- **Reconstruction.** Pixels are scaled to $[0,1]$ and modelled as Bernoulli. The decoder
  outputs logits, and the BCE is computed with `log_sigmoid` for numerical stability,
  summed over the 784 pixels and averaged over the batch.
- **KL term.** Closed form for diagonal Gaussians:
  $\tfrac12 \sum_j \big(\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1\big)$.
- **Reparameterisation trick.** $z = \mu + \sigma \odot \varepsilon,\ \varepsilon \sim \mathcal{N}(0, I)$,
  so gradients flow through the sampling step. The encoder predicts $\log\sigma^2$ to keep $\sigma > 0$.

### Architectures

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/arch-vae-dark.svg">
  <img alt="Dense VAE: Linear 784→512 ReLU, heads μ and log σ² of size 16, decoder Linear 16→512 ReLU, Linear 512→784, sigmoid." src="docs/figures/arch-vae-light.svg" width="100%">
</picture>

The **dense VAE** (`src/models/vae.rs`) has one 512-unit hidden layer on each side.
Its weights use Xavier-uniform initialisation with a √2 gain for ReLU layers and
a gain of 1 for the output layer.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/arch-cnn-dark.svg">
  <img alt="CNN-VAE: two stride-2 4×4 convolutions (32, 64 channels) to 64×7×7, linear heads μ and log σ², decoder Linear 16→3136 reshaped to 64×7×7 and two stride-2 transposed convolutions back to 1×28×28." src="docs/figures/arch-cnn-light.svg" width="100%">
</picture>

The **CNN-VAE** (`src/models/cnnvae.rs`) downsamples 28→14→7 with two stride-2 convolutions,
then mirrors the path with transposed convolutions. Most of its parameters are in the
linear layers around the 3,136-dimensional bottleneck, not in the convolutions.

### Training protocol

Both runs share `TrainingConfig` in `src/main.rs`:

| | |
| :-- | :-- |
| Data split | 54,000 train / 6,000 validation (shuffled from the MNIST train set, seed 42) / 10,000 test |
| Optimiser | Adam, batch size 256, initial LR 1e-3 |
| LR schedule | Reduce on plateau: ×0.5 after 5 epochs without a 0.05 improvement, floor 1e-5 |
| β warm-up | β grows linearly from 0.2 to 1 over the first 5 epochs to avoid early posterior collapse |
| Early stopping | Patience 15 epochs (neither run triggered it within 100 epochs) |
| Model selection | Lowest validation −ELBO at β = 1, considered only after warm-up |
| Evaluation | Validation: 3 MC samples · Test: 5 MC samples, run once on the selected checkpoint |

## Training curves

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/loss-total-dark.svg">
  <img alt="Total loss per epoch for both models. CNN-VAE stays about 4 nats below the dense VAE through training; final validation 97.25 vs 101.46." src="docs/figures/loss-total-light.svg" width="100%">
</picture>

The CNN-VAE pulls ahead within the first ~10 epochs and the gap stays roughly constant.
The small steps around epochs 65, 77 and 85–97 come from learning-rate reductions.
The train/validation gap is small for both models (≈1.3–1.8 nats), so neither overfits noticeably.
Both curves are still trending down at epoch 100.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/loss-components-dark.svg">
  <img alt="Left: reconstruction BCE, CNN-VAE lower (73.4 vs 77.5). Right: KL divergence, both models converge to about 24 nats." src="docs/figures/loss-components-light.svg" width="100%">
</picture>

Splitting the loss shows where the difference lives. The **KL term converges to ≈24 nats
for both models**, so they encode a similar amount of information in `z`. The
**reconstruction term is consistently ~4 nats lower** for the CNN-VAE, whose locality
and weight sharing suit image data.

## Samples

64 digits decoded from $z \sim \mathcal{N}(0, I)$ with each model's best checkpoint:

| Dense VAE | CNN-VAE |
| :--: | :--: |
| <img src="artifacts/vae/generated-grid.png" width="360" alt="64 samples from the dense VAE"> | <img src="artifacts/cnn/generated-grid.png" width="360" alt="64 samples from the CNN-VAE"> |

The CNN samples are generally crisper, with more continuous strokes. The dense VAE produces
more blurry or ambiguous digits, the typical artefact of a decoder with no spatial prior.

## Reproduce

Requires a recent Rust toolchain and a Mac with Metal (the backend is set in `src/main.rs`).
MNIST is downloaded automatically by Burn on the first run.

```bash
cargo run --release -- vae   # → artifacts/vae/
cargo run --release -- cnn   # → artifacts/cnn/
```

Each run writes to `artifacts/<model>/`:

| File | Content |
| :-- | :-- |
| `history.csv` | Per-epoch LR, β, train/val total/BCE/KL, timings |
| `test.csv` | Best epoch and test −ELBO/BCE/KL |
| `best.bpk`, `last.bpk` | Model weights (git-ignored) |
| `generated-grid.png`, `generated-single.png` | Samples from the prior |

## Charts

All figures in this README come from a small dependency-free template in
[`docs/charts/`](docs/charts):

| File | Role |
| :-- | :-- |
| `charts.js` | Pure SVG renderers (loss curves, architecture diagrams) shared by browser and Node |
| `index.html` · `style.css` · `app.js` | Interactive viewer: metric tabs, train/val toggles, hover tooltips, light/dark theme, SVG download |
| `export.js` | Writes the light and dark SVGs in `docs/figures/` used above |

```bash
# Regenerate README figures after a new training run
node docs/charts/export.js

# Open the interactive viewer
python3 -m http.server     # from the repo root
open http://localhost:8000/docs/charts/
```

## Project structure

```
src/
├── main.rs          # CLI entry point, hyperparameters, model selection
├── training.rs      # ELBO loss, training loop, LR schedule, early stopping, evaluation
├── io.rs            # Saves sample grids as PNG
├── io/mnist.rs      # MNIST batcher and train/val/test loaders
├── models.rs        # VariationalModel trait shared by both models
└── models/
    ├── vae.rs       # Dense VAE
    ├── cnnvae.rs    # Convolutional VAE
    └── utils.rs     # Xavier-initialised linear layer helper
```
