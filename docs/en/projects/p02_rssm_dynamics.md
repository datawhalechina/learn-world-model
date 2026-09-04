---
title: P02 Build an RSSM Dynamics Model
---

# P02: Build an RSSM Dynamics Model

Train and compare GRU, MDN-RNN, and RSSM dynamics models on synthetic pixel trajectories. The point of this notebook is comparison, not leaderboard chasing: GRU is the simplest baseline, MDN-RNN adds predictive uncertainty, and RSSM introduces a latent stochastic state for world-model style rollouts.

**Prerequisite**: P01 (`vae_encoder.pt`) if present. Otherwise the notebook falls back to a randomly initialized encoder so it still runs, but the rollout comparison is only meaningful with the pretrained checkpoint. This notebook trains the dynamics models and saves the RSSM to `rssm.pt` for P03 and P04.

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
With setup done, reuse the P01 encoder and decoder so the dynamics models operate in the same latent space as the rest of the world-model stack. `_load_vae_checkpoint` handles a small naming mismatch (`fc_log_var` vs `fc_var`) between this notebook's `VAEEncoder` class and P01's `Encoder`, but the underlying weights are identical if `vae_encoder.pt` is present. Both networks are set to `.eval()` and every parameter gets `requires_grad_(False)`: the encoder and decoder are frozen for the rest of this notebook, used only to convert between pixels and latents, never updated by the dynamics training below.

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
Next, generate synthetic trajectories. These rollouts become the supervision signal for the sequence models. Each trajectory is a colored rectangle drifting across the frame with a fixed velocity `(vx, vy)` plus small per-step noise, and a binary `action` at every step (currently decorative: it is recorded and fed to every model below, but the box's motion does not depend on it). This keeps the focus of the comparison on how GRU, MDN-RNN, and RSSM differ in modeling *uncertainty* over an otherwise simple, near-deterministic dynamic, rather than on whether they can discover a complex action-conditioned rule.

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
Once the trajectories exist, encode each observation into z so the rest of the notebook can learn dynamics over latent sequences, not raw pixels. `encoder.encode(obs)` calls the *sampling* path (`mu + std * epsilon`) from P01's `reparameterize`, not just the mean, so each `z_t` here already carries the small amount of stochasticity the VAE's posterior assigns to that frame. This matters for interpreting the RSSM section below: `z_seq` supplies the *posterior target* the RSSM's own posterior network learns to reproduce.

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

Start with the simple GRU baseline, then add mixture density outputs and RSSM latent state. All three classes below implement the models from [Latent Dynamics](../lectures/lecture-02-encode-and-dynamics/02-dynamics): the same input `(z_t, a_t)`, the same goal (predict `z_{t+1}`), but three different ways of handling uncertainty. Reading the three `forward` methods side by side is the fastest way to see the actual code-level difference the lecture describes only in equations.

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
With a plain GRU baseline in hand, add an MDN-RNN to model the multimodal uncertainty a single prediction cannot capture. `GRUDynamics.forward` is the direct code form of the lecture's $\mathbf{z}_{t+1} = \text{GRU}(\mathbf{z}_t, \mathbf{a}_t)$: `torch.cat([z_seq[:, t], a_seq[:, t].unsqueeze(-1)], dim=-1)` builds the concatenated `(z_t, a_t)` input, `self.gru(inp, h)` is the gated update itself (PyTorch's built-in `GRUCell`, which implements the reset/update gate mechanism from the lecture's callout internally), and `self.output(h)` is a linear readout turning the hidden state into a predicted `z_{t+1}`. There is no sampling step anywhere in this class: given the same `(z_t, a_t, h_{t-1})`, the GRU always produces the same `z_{t+1}`, which is exactly the "deterministic output" limitation the lecture names.

`MDNRNN` replaces that single linear readout with `mdn_head`, a linear layer producing three groups of outputs per mixture component `k`: `logits` (before softmax, these become the mixture weights $\pi_k$), `mu` (component means $\mu_k$), and `log_s` (log-variance per component, mirroring `log_var` from P01). `_split` slices these three groups out of the head's flat output tensor. `mdn_loss` is the negative log-likelihood of the Gaussian mixture from the lecture's $p(\mathbf{z}_{t+1} \mid \mathbf{z}_t, \mathbf{a}_t) = \sum_k \pi_k \mathcal{N}(\mathbf{z}_{t+1}; \mu_k, \sigma_k^2)$: `log_p` computes each component's log-density (the Gaussian log-likelihood formula, summed over the 32 latent dimensions since they are treated as independent), `log_pi` normalizes the raw logits into $\log \pi_k$ via log-softmax, and `torch.logsumexp(log_pi + log_p, dim=-1)` combines them into $\log \sum_k \pi_k \mathcal{N}(\ldots)$, computed in log-space for numerical stability rather than exponentiating and summing directly (which risks overflow for large log-densities). Negating and averaging gives the loss to minimize.

At rollout time, `MDNRNN.rollout` does not sample from the mixture; it deterministically picks the highest-weight component (`lg[0].argmax()`) and uses its mean. This means MDN-RNN's *training* objective captures multimodality, but the *rollout* shown later in this notebook does not exercise that multimodality since it always follows the single most likely branch.

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
After the two baselines are defined, introduce RSSM as the structured latent-state model to compare against them. `RSSM.forward` implements the three core equations from [the RSSM section of the Latent Dynamics lecture](../lectures/lecture-02-encode-and-dynamics/02-dynamics#rssm-separating-deterministic-and-stochastic-components) inside a single `for t in range(T)` loop:

- `h = self.gru(inp, h)` is the deterministic update $\mathbf{h}_t = f_\phi(\mathbf{h}_{t-1}, \mathbf{z}_{t-1}, \mathbf{a}_{t-1})$, where `z` on the right-hand side of `inp` is last step's *sampled* latent, carried over from the previous loop iteration (initialized to zeros at `t=0`).
- `pr = self.prior_net(h)`, split into `mu_pr, lv_pr`, is the prior $\mathbf{z}_t \sim p_\phi(\mathbf{z}_t \mid \mathbf{h}_t)$: it depends on `h` alone, with no access to the current observation, matching the lecture's description of the prior as the branch used for pure imagination.
- `po = self.post_net(torch.cat([h, z_seq[:, t]], dim=-1))`, split into `mu_po, lv_po`, is the posterior $\mathbf{z}_t \sim q_\phi(\mathbf{z}_t \mid \mathbf{h}_t, \mathbf{o}_t)$: note it additionally consumes `z_seq[:, t]`, this trajectory's actual encoded observation at this step, which is the "correction from the real observation" the lecture describes. `self._rsample(mu_po, lv_po)` then draws the value of `z` that both continues the recurrence (as next step's `inp`) and is compared against the prior in the KL term below.

The KL term in `forward`, `kl = 0.5 * (lv_pr - lv_po + (lv_po.exp() + (mu_po - mu_pr)**2) / lv_pr.exp().clamp(min=1e-4) - 1)`, is the closed-form KL divergence between two general (not necessarily standard-normal) diagonal Gaussians, $D_{KL}(q_\phi(\mathbf{z}_t \mid \mathbf{h}_t, \mathbf{o}_t) \,\|\, p_\phi(\mathbf{z}_t \mid \mathbf{h}_t))$: this is a more general form of the same Gaussian KL formula used in P01, generalized from "posterior vs. standard normal prior" to "posterior vs. a *learned* prior network," which is precisely what separates RSSM's KL term from a plain VAE's. Minimizing this KL pulls the prior (which has no access to `o_t`) toward matching the posterior (which does), so that at rollout time, when only the prior is available, it has learned to approximate what the posterior would have said.

`RSSM.rollout` uses only `prior_net`, never `post_net`, and takes the prior's mean (`mu`) rather than sampling, exactly matching the lecture's description of imagination as prior-only rollout with no real observations involved.

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
**Checking the lecture's worked example**: the lecture hand-computes one simplified (gate-free, linear) RSSM transition step, getting $\mathbf{h}_t = [0.60, 0.20]$, $\mu_{\text{pr}} = 0.40$, $z_t = 0.50$. The cell below runs the same numbers and weights through the same computation. Note this deliberately does not use the `RSSM` class above (which internally uses a real `GRUCell` plus a multi-layer prior network, so its numbers will not match this simplified version); this cell only checks that the formula's derivation itself is correct.

```python
# Simplified single-step transition matching the lecture's worked example (not the real RSSM class).
h_prev = torch.tensor([0.0, 0.0])
z_prev = torch.tensor(0.50)
a_prev = torch.tensor(1.0)

w_z = torch.tensor([0.4, 0.2])
w_a = torch.tensor([0.3, 0.1])
b_h = torch.tensor([0.1, 0.0])
h_t = w_z * z_prev + w_a * a_prev + b_h
print(f'h_t = {h_t.tolist()}')
assert torch.allclose(h_t, torch.tensor([0.60, 0.20]), atol=1e-6)

w_mu = torch.tensor([0.5, 0.5])
sigma_pr = 0.20
mu_pr = (w_mu * h_t).sum()
print(f'mu_pr = {mu_pr.item():.2f}')
assert abs(mu_pr.item() - 0.40) < 1e-6

eps = torch.tensor(0.50)
z_t = mu_pr + sigma_pr * eps
print(f'z_t = {z_t.item():.2f}')
assert abs(z_t.item() - 0.50) < 1e-6
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
`run_epoch` is the shared training loop used for all three models: standard minibatch SGD with `clip_grad_norm_(model.parameters(), 1.0)` guarding against exploding gradients, which recurrent models are especially prone to over even the 20-step sequences used here. The three `*_loss` wrapper functions (`gru_loss`, `mdn_loss_fn`, `rssm_loss`) each adapt one model's own loss computation to the same `loss_fn(model, zb, ab)` signature `run_epoch` expects, so the same training loop drives all three without duplicating logic.

Note that `RSSM.forward` (used by `rssm_loss`) reconstructs the *latent* `z_seq[:, t]` via `self.recon(z)`, not raw pixels. This is a simplification specific to this notebook: the lecture's RSSM reconstructs observations directly, $o_t \sim p(o_t \mid h_t, z_t)$, but since `z_seq` here is already the frozen P01 encoder's output, reconstructing it is a proxy for reconstructing pixels, one step removed. The three losses are not on a comparable numeric scale (MSE, mixture NLL, and ELBO have different units and typical magnitudes), which is exactly why the plot in the next cell normalizes each curve before comparing shapes rather than raw values.

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
With training complete, move to rollout evaluation and measure how errors accumulate as we predict further into the future. `ROLLOUT_STEPS = 10` open-loop steps means each model is given only the first frame `z0` and the true action sequence, then must predict every subsequent latent from its own previous predictions, with no further correction from real observations. This is precisely the **teacher forcing gap** described in [L03's STORM discussion](../lectures/lecture-03-architecture-patterns/01-architectures-rnn-transformer-diffusion#storm-s-key-improvement-single-token-stochastic-latent-variable) and diagnosed formally in [L04](../lectures/lecture-04-evaluation-by-model/00-diagnostic-framework): training used real `z_seq` at every step, but this evaluation forces each model to compound its own errors, which is the harder and more realistic test of whether the learned dynamics are actually useful for imagination-based planning.

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
The image grid gives the visual story. The next block turns that into a per-step pixel MSE curve for a quantitative view. Expect GRU to drift first and hardest (no mechanism for expressing or correcting uncertainty), MDN-RNN to do somewhat better by modeling multiple possible outcomes during training even though its rollout picks only one, and RSSM to hold up best on this near-deterministic synthetic data because its explicit prior network was trained specifically to match what the posterior would have inferred from a real observation.

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

On the held-out test trajectories, we compute average latent-space MSE at horizons 1 to 5 for all three models. `horizon_mse` slides a window across every test trajectory (`for t0 in range(T - max_h)`), rolls each model forward `max_h` steps from that window's start, and accumulates the squared error against the true latent at each horizon separately. This produces the **error-versus-horizon curve** that L04's diagnostic framework calls out as standard evidence for the long-horizon rollout layer: a curve that stays low and flat means a model's errors are not compounding, while a curve that rises steeply with horizon means small per-step mistakes are snowballing, the same **horizon drift** failure mode documented for every architecture in [L04](../lectures/lecture-04-evaluation-by-model/05-diffusion-drift#horizon-drift-the-universal-failure-mode-across-all-world-models).

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
With the horizon range fixed, plot step-wise prediction error so short-horizon and long-horizon behavior can be compared directly. Reading this plot alongside the qualitative rollout grid above is deliberate: a model can have a low horizon-1 error (accurate immediate predictions) while still drifting badly by horizon 5, and only the multi-horizon curve, not a single aggregate number, reveals that.

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

`rssm_model.state_dict()` is saved, not the GRU or MDN-RNN checkpoints, because RSSM is the dynamics core P03 builds on: the prior/posterior split that lets rollout run from `prior_net` alone (no real observations needed) is exactly what makes "training entirely in imagination" possible, the property the [Dreamer pipeline](../lectures/lecture-02-encode-and-dynamics/03-dynamics-dreamer-series#the-encoder-s-role-as-a-bridge-in-dreamer) depends on. P03 loads this checkpoint, wraps `rssm_model` with an actor and critic, and trains the policy entirely on `RSSM.rollout`-style imagined trajectories rather than on real environment interaction.

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
