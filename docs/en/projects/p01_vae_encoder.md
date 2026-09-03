---
title: P01 Train a VAE Encoder
---

# P01: Train a VAE Encoder

Train a compact convolutional Variational Autoencoder (VAE) on synthetic 64x64 RGB images. The encoder learns a 32-dimensional latent space that P02 reuses as its observation encoder. The goal here is not photorealistic generation. It is to learn a stable latent space that downstream notebooks can consume.

**Output**: this notebook trains from scratch (no prior checkpoint needed) and saves the trained weights to `vae_encoder.pt`, which P02 and P03 load as their observation encoder.

**Outline:**
1. Setup: synthetic data and DataLoader
2. Model: encoder, decoder, reparameterization
3. ELBO loss: reconstruction plus KL
4. Train: 30 epochs and loss curves
5. Inspect: reconstructions, traversals, random samples

> Notebook source: [p01_vae_encoder.ipynb](https://github.com/datawhalechina/learn-world-model/blob/main/docs/en/projects/p01_vae_encoder.ipynb)

```bash
%%bash
# Install dependencies for a fresh environment.
if command -v rocm-smi >/dev/null || [ -d /opt/rocm ]; then
  pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
  pip install matplotlib numpy
else
  pip install torch torchvision matplotlib numpy
fi
```
## 1. Setup

Generate synthetic colored shapes so the notebook runs offline with no external downloads.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
try:
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'inline')
except Exception:
    pass
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

try:
    import torch_xla.core.xla_model as xm
    _XLA_AVAILABLE = True
except Exception:
    xm = None
    _XLA_AVAILABLE = False


def _resolve_device():
    if _XLA_AVAILABLE:
        return xm.xla_device()
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


DEVICE = _resolve_device()
USE_TPU = DEVICE.type == 'xla'
USE_CUDA = DEVICE.type == 'cuda'
IS_ROCM = USE_CUDA and torch.version.hip is not None
LOAD_DEVICE = torch.device('cpu') if USE_TPU else DEVICE


def optimizer_step(optimizer, scaler=None):
    if USE_TPU:
        xm.optimizer_step(optimizer)
    elif scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

# Use faster kernels when CUDA is available.
if USE_CUDA:
    torch.backends.cudnn.benchmark = True
    if not IS_ROCM:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

print(f'Using device   : {DEVICE}')
if USE_TPU:
    print('TPU backend    : torch_xla')
elif USE_CUDA:
    print(f'GPU            : {torch.cuda.get_device_name(0)}')
    if IS_ROCM:
        print(f'ROCm/HIP       : {torch.version.hip}')
        print(f'GPU arch       : {torch.cuda.get_device_properties(0).gcnArchName}')
    else:
        print(f'CUDA capability: {torch.cuda.get_device_capability(0)}')
print(f'PyTorch version: {torch.__version__}')
```
With the runtime ready, generate a synthetic-shape dataset that the VAE can learn from without external downloads.

```python
def make_shape_image(img_size=64):
    """Generate a single 64x64 RGB image containing a random colored shape."""
    rng = np.random.default_rng()
    img = np.zeros((img_size, img_size, 3), dtype=np.float32)

    shape_type = rng.integers(0, 3)
    color = rng.uniform(0.3, 1.0, size=3).astype(np.float32)

    cx = rng.integers(16, img_size - 16)
    cy = rng.integers(16, img_size - 16)
    r  = rng.integers(8, 20)

    if shape_type == 0:
        x0, x1 = max(0, cx - r), min(img_size, cx + r)
        y0, y1 = max(0, cy - r), min(img_size, cy + r)
        img[y0:y1, x0:x1] = color
    elif shape_type == 1:
        ys, xs = np.mgrid[0:img_size, 0:img_size]
        mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= r ** 2
        img[mask] = color
    else:
        for row in range(img_size):
            half_w = int(r * (1 - abs(row - cy) / max(r, 1)))
            if half_w > 0:
                c0 = max(0, cx - half_w)
                c1 = min(img_size, cx + half_w)
                img[row, c0:c1] = color

    return torch.from_numpy(img.transpose(2, 0, 1))


class ShapeDataset(Dataset):
    def __init__(self, n_samples=1000, img_size=64, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        # Store the dataset as one tensor for cheap indexing.
        self.images = torch.stack([make_shape_image(img_size) for _ in range(n_samples)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


dataset = ShapeDataset(n_samples=1000)

dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=2 if USE_CUDA else 0,
    pin_memory=USE_CUDA,
)

print(f'Dataset size : {len(dataset)}')
print(f'Image shape  : {dataset[0].shape}  (C, H, W)')
print(f'Batches/epoch: {len(dataloader)}')

# Quick sanity check.
fig, axes = plt.subplots(1, 8, figsize=(16, 2))
for i, ax in enumerate(axes):
    ax.imshow(dataset[i].permute(1, 2, 0).numpy())
    ax.axis('off')
fig.suptitle('Sample images from synthetic dataset', y=1.02)
plt.tight_layout()
plt.show()
```
## 2. Model: Encoder

The `Encoder` is the CNN from [L02 Observation Encoding](../lectures/lecture-02-encode-and-dynamics/01-encoding): four stride-2 convolutions halve the spatial resolution each time (64 -> 32 -> 16 -> 8 -> 4), so the final feature map is `256 x 4 x 4`. Flattening that map and passing it through two linear heads gives `mu` and `log_var`, the mean and log-variance of the approximate posterior $q(z \mid x)$ introduced on that page. We predict `log_var` rather than `var` or `sigma` directly so the network's raw output can be any real number; exponentiating it later guarantees the variance stays positive without needing a constrained activation function.

```python
LATENT_DIM = 32
IMG_SIZE   = 64
IMG_CH     = 3


class Encoder(nn.Module):
    """Encode 3x64x64 images into latent mean and log-variance."""

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(IMG_CH, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.flat_dim = 256 * 4 * 4
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x):
        h = self.conv(x).flatten(start_dim=1)
        return self.fc_mu(h), self.fc_log_var(h)
```
The `Decoder` mirrors the encoder with transposed convolutions (see the "Transposed Convolution" callout on the encoding page) that each double spatial resolution, undoing the encoder's compression: `4 -> 8 -> 16 -> 32 -> 64`. The final `Sigmoid` keeps output pixels in `[0, 1]`, matching how the synthetic images were generated.

```python
class Decoder(nn.Module):
    """Decode latent vectors back to 3x64x64 images."""

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.flat_dim = 256 * 4 * 4
        self.fc = nn.Linear(latent_dim, self.flat_dim)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, IMG_CH, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 256, 4, 4)
        return self.deconv(h)
```
`VAE` combines the encoder and decoder. `reparameterize` is the reparameterization trick from the lecture, `z = mu + sigma * epsilon`, implemented with `std = exp(0.5 * log_var)` because `log_var = log(sigma^2)`, so `0.5 * log_var = log(sigma)` and exponentiating recovers `sigma`. `torch.randn_like(std)` draws `epsilon ~ N(0, I)` independently of the network parameters, which is exactly what keeps this operation differentiable: gradients flow through `mu` and `std`, never through the random draw itself.

```python
class VAE(nn.Module):
    """Variational Autoencoder combining encoder and decoder."""

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, log_var):
        """Sample z = mu + sigma * epsilon."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var

    def encode(self, x):
        """Return the latent mean without sampling."""
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z):
        return self.decoder(z)


model = VAE(latent_dim=LATENT_DIM).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f'Total parameters: {total_params:,}')

# Verify shapes with a dummy forward pass.
dummy = torch.zeros(4, IMG_CH, IMG_SIZE, IMG_SIZE).to(DEVICE)
recon_dummy, mu_dummy, lv_dummy = model(dummy)
print(f'Input  shape : {dummy.shape}')
print(f'mu shape     : {mu_dummy.shape}')
print(f'log_var shape: {lv_dummy.shape}')
print(f'Recon  shape : {recon_dummy.shape}')
```
**Checking the lecture's worked example**: the lecture hand-computes $z = 0.76$ from $\mu = 0.50$, $\sigma = 0.20$, $\varepsilon = 1.30$. Running the same numbers through `reparameterize`'s formula confirms the code matches the math exactly.

```python
demo_mu = torch.tensor([[0.50]])
demo_std = torch.tensor([[0.20]])
demo_eps = torch.tensor([[1.30]])
demo_z = demo_mu + demo_std * demo_eps
print(f'z = mu + std * eps = {demo_mu.item()} + {demo_std.item()} * {demo_eps.item()} = {demo_z.item():.2f}')
assert abs(demo_z.item() - 0.76) < 1e-6, "does not match the lecture's hand-worked result"
```
## 3. ELBO Loss

`elbo_loss` implements the two terms from the [ELBO derivation on the encoding page](../lectures/lecture-02-encode-and-dynamics/01-encoding#elbo-loss-balancing-two-objectives):

$$
\mathcal{L} = \underbrace{\mathbb{E}_{q(z|x)}[-\log p(x|z)]}_{\text{reconstruction}} + \underbrace{D_{KL}(q(z|x) \| p(z))}_{\text{KL divergence}}
$$

**Reconstruction term.** `F.mse_loss(recon_x, x, reduction='mean')` stands in for $-\log p(x \mid z)$. This substitution is standard but worth naming: if the decoder's output distribution is assumed to be a Gaussian with fixed variance centered at the reconstructed pixels, $-\log p(x \mid z)$ reduces to squared error between `recon_x` and `x` up to an additive constant, which is exactly MSE. Using MSE is therefore not an approximation of convenience but the exact negative log-likelihood under that Gaussian-output assumption.

**KL term, and where the closed form comes from.** For two Gaussians, $q(z \mid x) = \mathcal{N}(\mu, \sigma^2)$ and the standard normal prior $p(z) = \mathcal{N}(0, 1)$, the KL divergence has an exact closed form (no Monte Carlo sampling needed), derived by expanding the definition $D_{KL}(q \| p) = \int q(z) \log \frac{q(z)}{p(z)} \, dz$ for Gaussian densities:

$$
D_{KL}\big(\mathcal{N}(\mu, \sigma^2) \,\|\, \mathcal{N}(0, 1)\big) = -\frac{1}{2}\left(1 + \log \sigma^2 - \mu^2 - \sigma^2\right)
$$

Summing this over all 32 latent dimensions (assumed independent, so the joint KL is the sum of per-dimension KLs) gives exactly the code's `torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)` term, negated and halved. `log_var.exp()` recovers $\sigma^2$ from $\log \sigma^2$, the same relationship used in `reparameterize` above.

**Why the KL term is rescaled.** The code additionally divides `kl_loss` by `IMG_CH * IMG_SIZE * IMG_SIZE` (12,288, the same pixel count motivating compression in [Why Compress?](../lectures/lecture-02-encode-and-dynamics/01-encoding#why-compress)). Reconstruction MSE is already an average over 12,288 pixels, but KL is a sum over only 32 latent dimensions, so without rescaling the KL term would be several orders of magnitude smaller than reconstruction and have negligible effect on training. Dividing by the same pixel count puts both terms on a comparable scale before `kl_weight` is applied.

The sanity check below runs one forward pass before any training so you can confirm the loss function executes and produces the expected shapes; the printed KL loss should be close to 0 here, since an untrained encoder's `mu` and `log_var` are close to their initialization near `(0, 0)`, which is close to matching the standard normal prior by coincidence rather than by learning.

```python
def elbo_loss(recon_x, x, mu, log_var, kl_weight=1.0):
    """
    ELBO loss = reconstruction loss + KL divergence.

    Returns:
        total_loss : scalar tensor (for backward)
        recon_loss : scalar tensor (for logging)
        kl_loss    : scalar tensor (for logging)
    """
    # Mean squared reconstruction error.
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')

    # Closed-form KL for a diagonal Gaussian posterior.
    kl_loss = -0.5 * torch.mean(
        torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    )
    # Scale KL so it is comparable to the reconstruction term.
    kl_loss = kl_loss / (IMG_CH * IMG_SIZE * IMG_SIZE)

    total_loss = recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss


# Quick sanity check.
with torch.no_grad():
    sample_batch = dataset[:4].to(DEVICE)
    r, mu_, lv_ = model(sample_batch)
    total, recon, kl = elbo_loss(r, sample_batch, mu_, lv_)

print(f'Initial total loss  : {total.item():.4f}')
print(f'Initial recon loss  : {recon.item():.4f}')
print(f'Initial KL loss     : {kl.item():.6f}')
```
## 4. Training Loop

Three hyperparameters matter here beyond the standard `EPOCHS`/`LR` pair:

- **`KL_WEIGHT = 3.0`** is the $\beta$ coefficient from the $\beta$-VAE formulation: multiplying the KL term before adding it to the reconstruction loss. A weight above 1 pushes the latent distribution closer to $\mathcal{N}(0, I)$ than the plain ELBO would, at some cost to reconstruction sharpness. This matters directly for Section 5 below: a latent space that is not close to standard normal produces incoherent images when you sample `z ~ N(0, I)` and decode, because the decoder never saw points from that region during training. The comment in the code above explains why 3.0 was chosen for this dataset specifically: because the KL term is already divided by the pixel count, a weight of 3 still leaves reconstruction quality visually unaffected while measurably tightening the latent distribution.
- **Mixed precision (`torch.amp.autocast`, `GradScaler`)** runs the forward pass in float16 on CUDA to speed up training, while `GradScaler` rescales gradients before the backward pass to prevent float16 underflow. This is a standard training-speed optimization, unrelated to the VAE's statistical properties; on CPU or when `USE_CUDA` is `False`, `autocast` and the scaler are no-ops and training proceeds in float32.
- **Tracking `recon_loss` and `kl_loss` separately** (not just their sum) is what makes the two-axis plot in the next cell diagnostic rather than decorative: watching KL alone is how you catch **posterior collapse**, the failure mode where the KL term drops to near zero early in training because the encoder has learned to ignore the input and output `mu ≈ 0, log_var ≈ 0` for every image, matching the prior exactly but destroying the reconstruction path in the process. If you see `Recon` plateau at a high value while `KL` is near zero from the first few epochs, that is what has happened; `KL_WEIGHT` set too high, or too low a reconstruction signal early in training, are the usual causes.

With the training budget and loss weighting fixed, wire the dataset into an epoch iterator and a DataLoader for minibatch updates, then run the loop.

```python
EPOCHS     = 30
LR         = 1e-3
# KL is averaged over pixels in elbo_loss, so a weight above 1 keeps the
# latent space close to N(0, I) (better prior samples and traversals)
# without measurably hurting reconstruction on this simple data.
KL_WEIGHT  = 3.0

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Mixed precision speeds up CUDA training and falls back cleanly on CPU.
scaler = torch.amp.GradScaler('cuda', enabled=USE_CUDA)

history_recon = []
history_kl    = []

model.train()
for epoch in range(1, EPOCHS + 1):
    epoch_recon = 0.0
    epoch_kl    = 0.0
    n_batches   = 0

    for batch in dataloader:
        batch = batch.to(DEVICE, non_blocking=USE_CUDA)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=USE_CUDA):
            recon_batch, mu, log_var = model(batch)
            loss, recon_loss, kl_loss = elbo_loss(
                recon_batch, batch, mu, log_var, kl_weight=KL_WEIGHT
            )

        scaler.scale(loss).backward()
        optimizer_step(optimizer, scaler)

        epoch_recon += recon_loss.item()
        epoch_kl    += kl_loss.item()
        n_batches   += 1

    avg_recon = epoch_recon / n_batches
    avg_kl    = epoch_kl    / n_batches
    history_recon.append(avg_recon)
    history_kl.append(avg_kl)

    if epoch % 5 == 0 or epoch == 1:
        print(f'Epoch {epoch:3d}/{EPOCHS} | '
              f'Recon: {avg_recon:.5f} | KL: {avg_kl:.6f}')

print('\nTraining complete.')
```
After training, plot both loss components against epoch. The expected shape: `Reconstruction Loss` decreases steadily as the encoder-decoder pair learns to compress and restore the synthetic shapes, while `KL Divergence` should decrease more gently and then stabilize at a small positive value, not collapse to zero. A KL curve that hits zero within the first few epochs is the posterior collapse warning sign described above, not a sign that training finished early.

```python
epochs_range = range(1, EPOCHS + 1)

fig, ax1 = plt.subplots(figsize=(10, 4))

color_recon = '#2196F3'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Reconstruction Loss (MSE)', color=color_recon)
ax1.plot(epochs_range, history_recon, color=color_recon, linewidth=2, label='Reconstruction')
ax1.tick_params(axis='y', labelcolor=color_recon)

ax2 = ax1.twinx()
color_kl = '#F44336'
ax2.set_ylabel('KL Divergence', color=color_kl)
ax2.plot(epochs_range, history_kl, color=color_kl, linewidth=2, linestyle='--', label='KL Divergence')
ax2.tick_params(axis='y', labelcolor=color_kl)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.title('VAE Training: Reconstruction Loss and KL Divergence')
fig.tight_layout()
plt.show()

print(f'Final reconstruction loss : {history_recon[-1]:.5f}')
print(f'Final KL divergence       : {history_kl[-1]:.6f}')
```
Numbers on a loss curve confirm the model is optimizing something, but only a visual check confirms it optimized the right thing. Compare original held-out images against their reconstructions: a healthy encoder-decoder pair should preserve shape, color, and rough position, even though the latent bottleneck (32 dimensions, versus 12,288 input pixel values) rules out pixel-perfect recovery.

```python
# Visual comparison: original vs reconstruction for a held-out batch
model.eval()
with torch.no_grad():
    sample = dataset[200:208].to(DEVICE)
    recon, _, _ = model(sample)

n_show = 8
fig, axes = plt.subplots(2, n_show, figsize=(16, 4))
for i in range(n_show):
    axes[0, i].imshow(sample[i].cpu().permute(1, 2, 0).numpy())
    axes[0, i].axis('off')
    axes[1, i].imshow(recon[i].cpu().permute(1, 2, 0).numpy())
    axes[1, i].axis('off')

axes[0, 0].set_title('Original', loc='left', fontsize=12)
axes[1, 0].set_title('Reconstruction', loc='left', fontsize=12)
plt.suptitle('Original vs Reconstructed (after 30 epochs)', y=1.02)
plt.tight_layout()
plt.show()
```
## 5. Latent Space Visualization

**Latent traversal.** Fixing all dimensions of `z` at 0 except one, then sweeping that one dimension from -2 to 2 and decoding at each value, tests the **continuity** property described in the [VAE Intuition section](../lectures/lecture-02-encode-and-dynamics/01-encoding#vae-intuition-learning-to-compress-and-reconstruct): nearby points in latent space should decode to visually similar images, with no discontinuous jumps as `z` moves smoothly through a dimension. This is the KL term's job: without it pulling the posterior toward a single shared, densely-populated region ($\mathcal{N}(0, I)$), the encoder would be free to scatter different images into disconnected, arbitrarily-shaped regions of latent space, and interpolating between them could pass through decoded garbage.

Look for whether each swept dimension changes one recognizable attribute (position, size, or color) while leaving the others roughly fixed. Perfect **disentanglement**, where every dimension controls exactly one independent factor, is not guaranteed by the plain ELBO objective (it is the subject of follow-up work such as $\beta$-VAE and FactorVAE), so partial or blended effects across dimensions 0-2 are expected on this simple dataset, not a bug.

```python
model.eval()

# Dimensions to sweep and sweep values
dims_to_vary = [0, 1, 2]
sweep_values = np.linspace(-2, 2, 5)

n_rows = len(dims_to_vary)
n_cols = len(sweep_values)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2 + 0.5))

with torch.no_grad():
    for row_idx, dim in enumerate(dims_to_vary):
        for col_idx, val in enumerate(sweep_values):
            z = torch.zeros(1, LATENT_DIM, device=DEVICE)
            z[0, dim] = float(val)
            decoded = model.decode(z)  # shape: (1, 3, 64, 64)
            img = decoded[0].cpu().permute(1, 2, 0).numpy()  # (64, 64, 3)

            ax = axes[row_idx, col_idx]
            ax.imshow(img)
            ax.axis('off')
            if col_idx == 0:
                ax.set_ylabel(f'dim {dim}', fontsize=10, rotation=0, labelpad=30, va='center')

# Column headers
for col_idx, val in enumerate(sweep_values):
    axes[0, col_idx].set_title(f'z={val:.1f}', fontsize=9)

plt.suptitle('Latent Space Traversal: varying one dimension at a time', fontsize=12, y=1.02)
plt.tight_layout()
plt.show()
```
**Prior sampling.** This cell draws `z ~ N(0, I)` directly, with no encoder involved at all, and decodes it. This is a stricter test than the traversal above: it checks whether the *entire* trained latent distribution matches the standard normal prior, not just the three dimensions swept earlier. If reconstructions in Section 4 looked good but these random samples look incoherent, the encoder has learned a valid compression for real images without actually matching $p(z) = \mathcal{N}(0, I)$ everywhere, meaning the KL term was too weak relative to reconstruction. Raising `KL_WEIGHT` and retraining is the direct fix, at some cost to reconstruction sharpness; this reconstruction-versus-prior-matching tension is the same two-objective balance introduced in Section 3.

```python
# Sample from the learned prior.
model.eval()
torch.manual_seed(42)

n_samples = 16
with torch.no_grad():
    z_random = torch.randn(n_samples, LATENT_DIM, device=DEVICE)
    samples  = model.decode(z_random)

fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(samples[i].cpu().permute(1, 2, 0).numpy())
    ax.axis('off')

plt.suptitle('Random samples from learned latent space (z ~ N(0, I))', fontsize=12)
plt.tight_layout()
plt.show()
```
## Save Checkpoint

Save the checkpoint as `vae_encoder.pt` so P02 and P03 can reuse the encoder weights. A good run should end with low reconstruction loss, visually coherent reconstructions, and latent traversals that change the decoded shape in a predictable way. P02 loads this checkpoint via `model.encoder`, using it as a frozen (non-trainable) feature extractor: it maps each raw observation `o_t` to the latent `z_t` that the dynamics models (GRU, MDN-RNN, RSSM) then learn to predict forward in time, exactly the encoder-to-dynamics handoff described in the [Dreamer pipeline summary](../lectures/lecture-02-encode-and-dynamics/03-dynamics-dreamer-series#the-encoder-s-role-as-a-bridge-in-dreamer).

```python
import os

checkpoint_path = 'vae_encoder.pt'
torch.save({
    'model_state_dict': model.state_dict(),
    'encoder':          model.encoder.state_dict(),
    'decoder':          model.decoder.state_dict(),
    'latent_dim':       LATENT_DIM,
    'img_size':         IMG_SIZE,
    'img_channels':     IMG_CH,
    'final_recon_loss': history_recon[-1],
    'final_kl_loss':    history_kl[-1],
    'epochs_trained':   EPOCHS,
    'checkpoint_format': 'vae-v2',
}, checkpoint_path)

size_kb = os.path.getsize(checkpoint_path) / 1024
print(f'Checkpoint saved to: {checkpoint_path}  ({size_kb:.1f} KB)')
print(f'Latent dim          : {LATENT_DIM}')
print(f'Final recon loss    : {history_recon[-1]:.5f}')
print(f'Final KL divergence : {history_kl[-1]:.6f}')
print()
print('To load in a downstream project:')
print("  ckpt = torch.load('vae_encoder.pt', map_location='cpu')")
print("  model = VAE(latent_dim=ckpt['latent_dim'])")
print("  model.load_state_dict(ckpt['model_state_dict'])")
```
