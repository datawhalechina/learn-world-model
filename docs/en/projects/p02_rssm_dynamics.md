---
title: P02 Build an RSSM Dynamics Model
---

# P02: Build an RSSM Dynamics Model

Train and compare GRU, MDN-RNN, and RSSM dynamics models on synthetic pixel trajectories. The point of this notebook is comparison, not leaderboard chasing: GRU is the simplest baseline, MDN-RNN adds predictive uncertainty, and RSSM introduces a latent stochastic state for world-model style rollouts.

**Prerequisite**: P01 (`vae_encoder.pt`) if present; otherwise the notebook falls back to a randomly initialized encoder so it still runs, but the rollout comparison is only meaningful with the pretrained checkpoint. This notebook trains the dynamics models and saves the RSSM to `rssm.pt` for P03 and P04.

> Notebook source: [p02_rssm_dynamics.ipynb](https://github.com/datawhalechina/learn-world-model/blob/main/docs/en/projects/p02_rssm_dynamics.ipynb)

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

Build the frozen encoder, synthetic trajectories, and latent dataset.

```python
import os
import math
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'inline')
except Exception:
    pass
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

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
LOAD_DEVICE = torch.device('cpu') if USE_TPU else DEVICE


def optimizer_step(optimizer, scaler=None):
    if USE_TPU:
        xm.optimizer_step(optimizer)
    elif scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer_step(optimizer)

LATENT_DIM = 32
HIDDEN_DIM = 128
ACTION_DIM = 1
N_TRAJ     = 200
T_STEPS    = 20
IMG_SIZE   = 64

print(f'Device: {DEVICE}')
if USE_TPU:
    print('TPU backend    : torch_xla')
print(f'LATENT_DIM={LATENT_DIM}, HIDDEN_DIM={HIDDEN_DIM}')
```
With setup done, reuse the P01 encoder and decoder so the dynamics model works in the same latent space as the rest of the world-model stack.

```python
# P01-compatible VAE encoder and decoder

class VAEEncoder(nn.Module):
    """Encode 64x64 RGB frames into latent mean and logvar."""
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv = nn.Sequential(
            nn.Conv2d(3,   32,  4, stride=2, padding=1),  # 64->32
            nn.ReLU(),
            nn.Conv2d(32,  64,  4, stride=2, padding=1),  # 32->16
            nn.ReLU(),
            nn.Conv2d(64,  128, 4, stride=2, padding=1),  # 16->8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 8->4
            nn.ReLU(),
        )
        self.fc_mu  = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_var = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.conv(x).reshape(x.size(0), -1)
        return self.fc_mu(h), self.fc_var(h)

    def encode(self, x):
        """Reparameterized sample from q(z|x)."""
        mu, logvar = self.forward(x)
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)


class VAEDecoder(nn.Module):
    """Mirror decoder: latent_dim -> 64x64 RGB in [0,1]."""
    def __init__(self, latent_dim=32):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32,  3,   4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z).reshape(-1, 256, 4, 4)
        return self.deconv(h)


encoder = VAEEncoder(LATENT_DIM).to(DEVICE)
decoder = VAEDecoder(LATENT_DIM).to(DEVICE)

def _load_vae_checkpoint(path):
    ckpt = torch.load(path, map_location=DEVICE)
    if 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
        enc_state = {
            k.removeprefix('encoder.').replace('fc_log_var', 'fc_var'): v
            for k, v in state.items()
            if k.startswith('encoder.')
        }
        dec_state = {
            k.removeprefix('decoder.'): v
            for k, v in state.items()
            if k.startswith('decoder.')
        }
        encoder.load_state_dict(enc_state)
        decoder.load_state_dict(dec_state)
        return True
    if 'encoder' in ckpt and 'decoder' in ckpt:
        encoder.load_state_dict(ckpt['encoder'])
        decoder.load_state_dict(ckpt['decoder'])
        return True
    raise KeyError(f'Unrecognized checkpoint format: {list(ckpt.keys())[:10]}')

ckpt_path = Path('vae_encoder.pt')
try:
    _load_vae_checkpoint(ckpt_path)
    print(f'Loaded VAE weights from {ckpt_path}')
except Exception as e:
    print(f'Could not load VAE checkpoint from {ckpt_path} ({e}); using random init.')

VAE_CHECKPOINT_PATH = ckpt_path

encoder.eval()
decoder.eval()
for p in list(encoder.parameters()) + list(decoder.parameters()):
    p.requires_grad_(False)

print(f'Encoder params: {sum(p.numel() for p in encoder.parameters()):,}')
print(f'Decoder params: {sum(p.numel() for p in decoder.parameters()):,}')
```
Next, generate synthetic trajectories; these rollouts become the supervision signal for the sequence models.

```python
# Generate synthetic trajectory data.

def make_trajectory(T=20, img_size=64, seed=None):
    rng = np.random.RandomState(seed)
    color = rng.rand(3).astype(np.float32)
    w  = rng.randint(10, 20)
    h  = rng.randint(10, 20)
    x  = float(rng.randint(0, img_size - w))
    y  = float(rng.randint(0, img_size - h))
    vx = float(rng.randint(-3, 4))
    vy = float(rng.randint(-3, 4))
    frames  = []
    actions = []
    for _ in range(T):
        img = np.zeros((img_size, img_size, 3), dtype=np.float32)
        x1, y1 = int(np.clip(x, 0, img_size - w)), int(np.clip(y, 0, img_size - h))
        img[y1:y1 + h, x1:x1 + w] = color
        frames.append(img)
        actions.append(int(rng.randint(0, 2)))
        x = float(np.clip(x + vx + rng.uniform(-1, 1), 0, img_size - w))
        y = float(np.clip(y + vy + rng.uniform(-1, 1), 0, img_size - h))
    obs = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)  # [T,3,H,W]
    act = torch.tensor(actions, dtype=torch.float32)               # [T]
    return {'obs': obs, 'actions': act}


print('Generating 200 synthetic trajectories...')
trajectories = [make_trajectory(T=T_STEPS, img_size=IMG_SIZE, seed=i) for i in range(N_TRAJ)]
print(f"obs shape per trajectory:     {trajectories[0]['obs'].shape}")
print(f"actions shape per trajectory: {trajectories[0]['actions'].shape}")
```
Once the trajectories exist, encode each observation into z so the rest of the notebook can learn dynamics over latent sequences.

```python
# Encode observations into latent sequences z [N, T, 32]

print('Encoding observations...')
latent_list = []
with torch.no_grad():
    for traj in trajectories:
        obs = traj['obs'].to(DEVICE)      # [T,3,H,W]
        z   = encoder.encode(obs)         # [T, latent_dim]
        latent_list.append(z.cpu())

Z_all = torch.stack(latent_list, dim=0)                                     # [N,T,32]
A_all = torch.stack([t['actions'] for t in trajectories], dim=0)            # [N,T]

# Train/test split
N_TRAIN = 180
Z_train, A_train = Z_all[:N_TRAIN].to(DEVICE), A_all[:N_TRAIN].to(DEVICE)
Z_test,  A_test  = Z_all[N_TRAIN:].to(DEVICE), A_all[N_TRAIN:].to(DEVICE)

print(f'Z_all shape: {Z_all.shape}  (N, T, latent_dim)')
print(f'Train: {N_TRAIN} trajectories | Test: {N_TRAJ - N_TRAIN} trajectories')
```
## 2. Dynamics Models

Start with the simple GRU baseline, then add mixture density outputs and RSSM latent state.

```python
class GRUDynamics(nn.Module):
    """GRU with hidden_dim=128, input=(latent_dim+action_dim), output=latent_dim.
    Takes (z_t, a_t) -> h_{t+1} -> predicts z_{t+1}.
    """
    def __init__(self, latent_dim=32, action_dim=1, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru    = nn.GRUCell(latent_dim + action_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_seq, a_seq):
        """z_seq [B,T,D], a_seq [B,T] -> pred_z [B,T-1,D] for steps 1..T."""
        B, T, _ = z_seq.shape
        h = torch.zeros(B, self.hidden_dim, device=z_seq.device)
        preds = []
        for t in range(T - 1):
            inp = torch.cat([z_seq[:, t], a_seq[:, t].unsqueeze(-1)], dim=-1)
            h   = self.gru(inp, h)
            preds.append(self.output(h))
        return torch.stack(preds, dim=1)  # [B, T-1, D]

    def rollout(self, z0, a_seq):
        """Open-loop rollout from z0 [1,D] for len(a_seq) steps.
        Returns [1, steps+1, D] (includes z0).
        """
        z = z0
        h = torch.zeros(1, self.hidden_dim, device=z0.device)
        zs = [z]
        for a in a_seq:
            inp = torch.cat([z, a.view(1, 1)], dim=-1)
            h   = self.gru(inp, h)
            z   = self.output(h)
            zs.append(z)
        return torch.stack(zs, dim=1)  # [1, steps+1, D]


gru_model = GRUDynamics(LATENT_DIM, ACTION_DIM, HIDDEN_DIM).to(DEVICE)
print(f'GRUDynamics params: {sum(p.numel() for p in gru_model.parameters()):,}')
```
With a plain GRU baseline in hand, add an MDN-RNN to model the multimodal uncertainty that a single prediction cannot capture.

```python
class MDNRNN(nn.Module):
    """GRU + MDN head predicting a mixture of 3 Gaussians over z_{t+1}.
    MDN loss: negative log-likelihood of the mixture.
    """
    def __init__(self, latent_dim=32, action_dim=1, hidden_dim=128, n_mix=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_mix  = n_mix
        self.gru    = nn.GRUCell(latent_dim + action_dim, hidden_dim)
        # logits (K), mu (K*D), log_sigma (K*D)
        self.mdn_head = nn.Linear(hidden_dim, n_mix + 2 * n_mix * latent_dim)

    def _split(self, out):
        K, D = self.n_mix, self.latent_dim
        logits = out[..., :K]
        mu     = out[..., K:K + K * D].reshape(*out.shape[:-1], K, D)
        log_s  = out[..., K + K * D:].reshape(*out.shape[:-1], K, D)
        return logits, mu, log_s

    def forward(self, z_seq, a_seq):
        """Returns (logits, mu, log_sigma) each [B, T-1, ...] for MDN loss."""
        B, T, _ = z_seq.shape
        h = torch.zeros(B, self.hidden_dim, device=z_seq.device)
        all_logits, all_mu, all_ls = [], [], []
        for t in range(T - 1):
            inp = torch.cat([z_seq[:, t], a_seq[:, t].unsqueeze(-1)], dim=-1)
            h   = self.gru(inp, h)
            lg, mu, ls = self._split(self.mdn_head(h))
            all_logits.append(lg)
            all_mu.append(mu)
            all_ls.append(ls)
        return (
            torch.stack(all_logits, dim=1),
            torch.stack(all_mu,     dim=1),
            torch.stack(all_ls,     dim=1),
        )

    def mdn_loss(self, logits, mu, log_sigma, target):
        """Negative log-likelihood of mixture.
        logits [B,T,K], mu [B,T,K,D], log_sigma [B,T,K,D], target [B,T,D].
        """
        B, T, K, D = mu.shape
        tgt = target.unsqueeze(2).expand_as(mu)  # [B,T,K,D]
        sigma = log_sigma.exp().clamp(min=1e-4)
        log_p = -0.5 * (((tgt - mu) / sigma) ** 2 + 2 * log_sigma
                        + math.log(2 * math.pi))
        log_p  = log_p.sum(-1)                          # [B,T,K]
        log_pi = F.log_softmax(logits, dim=-1)          # [B,T,K]
        return -torch.logsumexp(log_pi + log_p, dim=-1).mean()

    def rollout(self, z0, a_seq):
        """Open-loop rollout using the most likely mixture component.
        Returns [1, steps+1, D].
        """
        z  = z0
        h  = torch.zeros(1, self.hidden_dim, device=z0.device)
        zs = [z]
        for a in a_seq:
            inp = torch.cat([z, a.view(1, 1)], dim=-1)
            h   = self.gru(inp, h)
            lg, mu, _ = self._split(self.mdn_head(h))
            k = lg[0].argmax().item()
            z = mu[0, k].unsqueeze(0)   # [1, D]
            zs.append(z)
        return torch.stack(zs, dim=1)


mdn_model = MDNRNN(LATENT_DIM, ACTION_DIM, HIDDEN_DIM, n_mix=3).to(DEVICE)
print(f'MDNRNN params: {sum(p.numel() for p in mdn_model.parameters()):,}')
```
After the two baselines are defined, introduce RSSM as the structured latent-state model we want to compare against them.

```python
class RSSM(nn.Module):
    """Recurrent State Space Model.
    Deterministic path: h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})
    Stochastic prior:   z_t ~ N(mu_prior(h_t), sigma_prior(h_t))
    Stochastic posterior: z_t ~ N(mu_post(h_t, o_t), sigma_post(h_t, o_t))
    Training: ELBO = reconstruction + KL(posterior || prior)
    hidden_dim=128, latent_dim=32
    """
    def __init__(self, latent_dim=32, action_dim=1, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Deterministic recurrence
        self.gru = nn.GRUCell(latent_dim + action_dim, hidden_dim)

        # Prior: h_t -> (mu, logvar)
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )

        # Posterior: (h_t, o_t) -> (mu, logvar),  o_t = encoded observation
        self.post_net = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )

        # Reconstruction head: z -> predicted z (latent reconstruction target)
        self.recon = nn.Linear(latent_dim, latent_dim)

    def _rsample(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def forward(self, z_seq, a_seq):
        """Compute ELBO over a full trajectory.
        z_seq [B,T,D], a_seq [B,T].
        Returns scalar ELBO loss.
        """
        B, T, D = z_seq.shape
        h = torch.zeros(B, self.hidden_dim, device=z_seq.device)
        z = torch.zeros(B, D, device=z_seq.device)
        recon_loss = z_seq.new_zeros(())
        kl_loss    = z_seq.new_zeros(())
        for t in range(T):
            inp = torch.cat([z, a_seq[:, t].unsqueeze(-1)], dim=-1)
            h   = self.gru(inp, h)

            # Prior
            pr   = self.prior_net(h)
            mu_pr, lv_pr = pr.chunk(2, dim=-1)

            # Posterior conditioned on observed latent o_t = z_seq[:, t]
            po   = self.post_net(torch.cat([h, z_seq[:, t]], dim=-1))
            mu_po, lv_po = po.chunk(2, dim=-1)

            z = self._rsample(mu_po, lv_po)

            # Reconstruction: predict the observation latent
            recon_loss = recon_loss + F.mse_loss(self.recon(z), z_seq[:, t])

            # KL(posterior || prior)
            kl = 0.5 * (
                lv_pr - lv_po
                + (lv_po.exp() + (mu_po - mu_pr) ** 2) / lv_pr.exp().clamp(min=1e-4)
                - 1
            )
            kl_loss = kl_loss + kl.mean()

        return (recon_loss + kl_loss) / T

    def rollout(self, z0, a_seq):
        """Open-loop rollout using the prior only (no observations).
        Returns [1, steps+1, D].
        """
        z  = z0
        h  = torch.zeros(1, self.hidden_dim, device=z0.device)
        zs = [z]
        for a in a_seq:
            inp = torch.cat([z, a.view(1, 1)], dim=-1)
            h   = self.gru(inp, h)
            pr  = self.prior_net(h)
            mu, _ = pr.chunk(2, dim=-1)
            z  = mu  # use prior mean for deterministic rollout
            zs.append(z)
        return torch.stack(zs, dim=1)


rssm_model = RSSM(LATENT_DIM, ACTION_DIM, HIDDEN_DIM).to(DEVICE)
print(f'RSSM params: {sum(p.numel() for p in rssm_model.parameters()):,}')
```
## 3. Training

All three models are trained for 20 epochs on the 180 training trajectories using Adam (lr=1e-3).
Loss functions:
- GRU: MSE between predicted z and actual z
- MDN-RNN: negative log-likelihood of the Gaussian mixture
- RSSM: ELBO = MSE reconstruction of z + KL divergence

```python
EPOCHS = 20
BATCH  = 32
LR     = 1e-3

opt_gru  = torch.optim.Adam(gru_model.parameters(),  lr=LR)
opt_mdn  = torch.optim.Adam(mdn_model.parameters(),  lr=LR)
opt_rssm = torch.optim.Adam(rssm_model.parameters(), lr=LR)


def run_epoch(model, optimizer, Z, A, loss_fn):
    model.train()
    N   = Z.shape[0]
    idx = torch.randperm(N)
    total, nb = 0.0, 0
    for s in range(0, N, BATCH):
        bi  = idx[s:s + BATCH]
        zb, ab = Z[bi], A[bi]
        optimizer.zero_grad()
        loss = loss_fn(model, zb, ab)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item(); nb += 1
    return total / nb


def gru_loss(m, zb, ab):
    return F.mse_loss(m(zb, ab), zb[:, 1:])


def mdn_loss_fn(m, zb, ab):
    logits, mu, ls = m(zb, ab)
    return m.mdn_loss(logits, mu, ls, zb[:, 1:])


def rssm_loss(m, zb, ab):
    return m(zb, ab)


losses_gru, losses_mdn, losses_rssm = [], [], []

print(f'Training 3 models for {EPOCHS} epochs...')
for epoch in range(1, EPOCHS + 1):
    lg = run_epoch(gru_model,  opt_gru,  Z_train, A_train, gru_loss)
    lm = run_epoch(mdn_model,  opt_mdn,  Z_train, A_train, mdn_loss_fn)
    lr = run_epoch(rssm_model, opt_rssm, Z_train, A_train, rssm_loss)
    losses_gru.append(lg)
    losses_mdn.append(lm)
    losses_rssm.append(lr)
    if epoch % 5 == 0 or epoch == 1:
        print(f'Epoch {epoch:3d} | GRU: {lg:.4f} | MDN-RNN: {lm:.4f} | RSSM: {lr:.4f}')

print('Training complete.')
```
Now that the model family is defined, set the epoch schedule and train the three dynamics variants side by side.

```python
# Plot normalized training loss curves.

def norm01(curve):
    a = np.array(curve, dtype=np.float64)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo + 1e-9)

fig, ax = plt.subplots(figsize=(8, 4))
xs = np.arange(1, EPOCHS + 1)
ax.plot(xs, norm01(losses_gru),  label='GRU (MSE)',     color='tab:blue')
ax.plot(xs, norm01(losses_mdn),  label='MDN-RNN (NLL)', color='tab:orange')
ax.plot(xs, norm01(losses_rssm), label='RSSM (ELBO)',   color='tab:green')
ax.set_xlabel('Epoch')
ax.set_ylabel('Normalized Loss [0, 1]')
ax.set_title('Training Loss Curves (normalized for comparability)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```
## 4. Rollout Comparison

Starting from the first frame of a test trajectory, each model rolls forward 10 steps without seeing future observations.
The predicted latents are decoded back to pixel space and displayed in a grid (3 rows x 10 columns).

```python
ROLLOUT_STEPS = 10

# Use the first test trajectory.
z_traj = Z_test[0]           # [T, D]
a_traj = A_test[0]           # [T]
z0     = z_traj[0].unsqueeze(0)        # [1, D]
a_seq  = a_traj[:ROLLOUT_STEPS]        # [10]

gru_model.eval()
mdn_model.eval()
rssm_model.eval()

with torch.no_grad():
    zs_gru  = gru_model.rollout(z0, a_seq).squeeze(0)    # [11, D]
    zs_mdn  = mdn_model.rollout(z0, a_seq).squeeze(0)
    zs_rssm = rssm_model.rollout(z0, a_seq).squeeze(0)

    def decode_seq(zs):
        """zs [S, D] -> numpy [S, H, W, 3] in [0,1]."""
        imgs = decoder(zs.to(DEVICE))  # [S, 3, H, W]
        return imgs.cpu().permute(0, 2, 3, 1).numpy()

    imgs_gru  = decode_seq(zs_gru)
    imgs_mdn  = decode_seq(zs_mdn)
    imgs_rssm = decode_seq(zs_rssm)
    imgs_gt   = decode_seq(z_traj[:ROLLOUT_STEPS + 1])   # ground truth

print(f'Decoded rollout shape (GRU): {imgs_gru.shape}  (steps+1, H, W, 3)')
```
With training complete, move to rollout evaluation and measure how errors accumulate as we predict further into the future.

```python
# Image grid: GT, GRU, MDN-RNN, RSSM.
N_COLS      = ROLLOUT_STEPS + 1
row_labels  = ['Ground Truth', 'GRU', 'MDN-RNN', 'RSSM']
row_imgs    = [imgs_gt, imgs_gru, imgs_mdn, imgs_rssm]

fig, axes = plt.subplots(
    4,
    N_COLS,
    figsize=(N_COLS * 1.7, 4.4),
    constrained_layout=True,
)
fig.patch.set_facecolor('white')
for r, (label, imgs) in enumerate(zip(row_labels, row_imgs)):
    for c in range(N_COLS):
        ax = axes[r, c]
        ax.imshow(np.clip(imgs[c], 0, 1), interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if c == 0:
            ax.set_ylabel(
                label,
                fontsize=9,
                rotation=0,
                labelpad=36,
                va='center',
                ha='right',
            )
        if r == 0:
            ax.set_title(f'Step {c}', fontsize=9, pad=8)
from IPython.display import display
fig.suptitle('10-step Imagined Rollouts vs Ground Truth', fontsize=12, y=1.05)
display(fig)
plt.close(fig)
```
The image grid gives the visual story; the next block turns that into a per-step pixel MSE curve for a quantitative view.

```python
# Per-step pixel MSE vs ground truth.

def pixel_mse_per_step(pred, gt):
    return [float(((p - g) ** 2).mean()) for p, g in zip(pred, gt)]

mse_gru  = pixel_mse_per_step(imgs_gru,  imgs_gt)
mse_mdn  = pixel_mse_per_step(imgs_mdn,  imgs_gt)
mse_rssm = pixel_mse_per_step(imgs_rssm, imgs_gt)
steps_x  = list(range(N_COLS))

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(steps_x, mse_gru,  marker='o', label='GRU',     color='tab:blue')
ax.plot(steps_x, mse_mdn,  marker='s', label='MDN-RNN', color='tab:orange')
ax.plot(steps_x, mse_rssm, marker='^', label='RSSM',    color='tab:green')
ax.set_xlabel('Rollout Step')
ax.set_ylabel('Pixel MSE')
ax.set_title('Per-Step Pixel MSE vs Ground Truth')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```
## 5. 1-step vs 5-step Prediction Error

On the held-out test trajectories, we compute average latent-space MSE at horizons 1 to 5 for all three models.

```python
MAX_H = 5


def horizon_mse(model_rollout_fn, Z, A, max_h=5):
    """Average latent MSE at horizons 1..max_h on test set.
    model_rollout_fn(z0, a_seq) -> [1, steps+1, D]
    """
    N, T, D = Z.shape
    errs   = np.zeros(max_h)
    counts = np.zeros(max_h)
    with torch.no_grad():
        for i in range(N):
            for t0 in range(T - max_h):
                z0   = Z[i, t0].unsqueeze(0)          # [1, D]
                a_s  = A[i, t0:t0 + max_h]            # [max_h]
                zs   = model_rollout_fn(z0, a_s).squeeze(0)  # [max_h+1, D]
                for h in range(1, max_h + 1):
                    errs[h - 1]   += F.mse_loss(zs[h], Z[i, t0 + h]).item()
                    counts[h - 1] += 1
    return errs / counts


print('Computing horizon errors on test set...')
gru_model.eval(); mdn_model.eval(); rssm_model.eval()

err_gru  = horizon_mse(gru_model.rollout,  Z_test, A_test, MAX_H)
err_mdn  = horizon_mse(mdn_model.rollout,  Z_test, A_test, MAX_H)
err_rssm = horizon_mse(rssm_model.rollout, Z_test, A_test, MAX_H)

horizons = list(range(1, MAX_H + 1))
print('\nLatent MSE by horizon:')
print(f'{"Horizon":>8}  {"GRU":>10}  {"MDN-RNN":>10}  {"RSSM":>10}')
for h, (eg, em, er) in enumerate(zip(err_gru, err_mdn, err_rssm), start=1):
    print(f'{h:>8}  {eg:>10.4f}  {em:>10.4f}  {er:>10.4f}')
```
With the horizon range fixed, plot step-wise prediction error so short-horizon and long-horizon behavior can be compared directly.

```python
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(horizons, err_gru,  marker='o', label='GRU',     color='tab:blue')
ax.plot(horizons, err_mdn,  marker='s', label='MDN-RNN', color='tab:orange')
ax.plot(horizons, err_rssm, marker='^', label='RSSM',    color='tab:green')
ax.set_xlabel('Prediction Horizon (steps)')
ax.set_ylabel('Average Latent MSE')
ax.set_title('1-step to 5-step Prediction Error (held-out test set)')
ax.set_xticks(horizons)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```
## Save Checkpoint

```python
checkpoint = {
    'rssm_state_dict': rssm_model.state_dict(),
    'hidden_dim':      HIDDEN_DIM,
    'latent_dim':      LATENT_DIM,
    'action_dim':      ACTION_DIM,
    'epochs_trained':  EPOCHS,
    'final_loss':      losses_rssm[-1],
}
torch.save(checkpoint, 'rssm.pt')
print('RSSM checkpoint saved to rssm.pt')
print(f'  hidden_dim={HIDDEN_DIM}, latent_dim={LATENT_DIM}')
print(f'  final ELBO loss: {losses_rssm[-1]:.4f}')
```
