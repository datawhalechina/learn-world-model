---
title: P04 동역학 백본 교체
---

# P04: 동역학 백본 교체

P02의 RSSM을 인과적 Transformer로 교체하고, 같은 합성 데이터에서 두 백본을 비교합니다. 이 튜토리얼은 엔지니어링 트레이드오프에 초점을 맞춥니다. RSSM의 더 강한 귀납적 편향과, 어텐션의 더 쉬운 병렬화 및 더 유연한 장기 컨텍스트 처리 사이의 교환입니다. 파이프라인은 CatVAE 토큰화, 인과적 Transformer 학습, RSSM과의 롤아웃 비교로 이어지며, 이는 통제된 비교로 읽어야지 Transformer가 더 낫다는 일반적인 주장으로 읽으면 안 됩니다.

**선행 조건**: P02(`rssm.pt`)가 있으면 사용합니다. 없으면 롤아웃 비교가 무작위로 초기화된 RSSM으로 대체되어 노트북은 여전히 실행되지만, RSSM 대 Transformer 수치는 사전학습된 체크포인트가 있어야만 의미가 있습니다. 이 노트북은 CatVAE와 Transformer를 처음부터 학습시켜 `transformer_wm.pt`로 저장해 P05에 전달합니다.

> Notebook 원본: [p04_transformer_backbone.ipynb](https://github.com/datawhalechina/learn-world-model/blob/main/docs/ko/projects/p04_transformer_backbone.ipynb)

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
의존성을 설치했으니, 핵심 라이브러리를 임포트하고 VAE와 Transformer 두 절이 공유할 런타임을 설정합니다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
try:
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'inline')
except Exception:
    pass
import matplotlib.pyplot as plt
import time
import math
from pathlib import Path

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
LOAD_DEVICE = torch.device('cpu') if USE_TPU else DEVICE


def optimizer_step(optimizer, scaler=None):
    if USE_TPU:
        xm.optimizer_step(optimizer)
    elif scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

PATH = Path('.')
print('Device:', DEVICE)
if USE_TPU:
    print('TPU backend    : torch_xla')
print('PyTorch version:', torch.__version__)
```
## 1. 범주형 VAE

각 프레임을 하나의 32차원 이산 코드로 토큰화합니다. 이는 [IRIS와 STORM](../lectures/lecture-03-architecture-patterns/01-architectures-rnn-transformer-diffusion#architecture-2-transformer-based-2022-2023) 뒤에 있는 토크나이저를 단순화한 버전입니다. STORM은 각각 32차원인 32개 범주(프레임당 가능한 결합 상태 32 x 32 = 1024개)로 이루어진 범주형 VAE를 쓰지만, 이 노트북의 `CatVAE`는 CPU에서도 학습이 빠르도록 단일 32방향 범주형 코드(`NUM_CATEGORIES = 32`개 선택지, 프레임당 하나)를 씁니다. 연속적인 인코더 출력을 미분 가능한 완화를 통해 이산 토큰으로 매핑하는 근본 메커니즘은, 강의가 IRIS의 VQ-VAE와 STORM의 범주형 VAE 모두에 대해 지목한 것과 같습니다.

```python
# Synthetic shape image dataset.
def make_shape_images(n=1000, size=64, seed=0):
    rng = np.random.RandomState(seed)
    imgs = np.zeros((n, 3, size, size), dtype=np.float32)
    for i in range(n):
        # Background
        bg = rng.uniform(0.05, 0.2, (3, 1, 1)).astype(np.float32)
        imgs[i] = bg
        # Random shape: circle or rectangle
        color = rng.uniform(0.4, 1.0, 3).astype(np.float32)
        cx = rng.randint(10, size - 10)
        cy = rng.randint(10, size - 10)
        r = rng.randint(5, 14)
        shape_type = rng.randint(0, 2)
        for c in range(3):
            if shape_type == 0:  # circle
                for y in range(size):
                    for x in range(size):
                        if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                            imgs[i, c, y, x] = color[c]
            else:  # rectangle
                x0, x1 = max(0, cx - r), min(size, cx + r)
                y0, y1 = max(0, cy - r), min(size, cy + r)
                imgs[i, c, y0:y1, x0:x1] = color[c]
    return torch.from_numpy(imgs)

print('Generating 1000 synthetic shape images...')
images = make_shape_images(n=1000, size=64, seed=42)
print('Image tensor shape:', images.shape, '  dtype:', images.dtype)

# Quick sanity check
fig, axes = plt.subplots(1, 5, figsize=(12, 2.5))
for i, ax in enumerate(axes):
    ax.imshow(images[i].permute(1, 2, 0).numpy())
    ax.axis('off')
    ax.set_title(f'img {i}')
plt.suptitle('Sample synthetic images', y=1.02)
plt.tight_layout()
plt.show()
```
합성 도형 데이터셋이 준비되었으니, 범주형 VAE를 처음부터 끝까지 학습 가능하게 만드는 스트레이트-스루 Gumbel-softmax 경로를 정의합니다. 이는 [백본 선택 페이지의 VQ 설명 상자](../lectures/lecture-03-architecture-patterns/01-architectures-rnn-transformer-diffusion#iris-turning-images-into-sentences)에서 설명한 **스트레이트-스루 추정기**(straight-through estimator)를 코드 수준에서 구현한 것입니다. `straight_through_gumbel`은 32개 이산 범주 중 하나를 골라야 하지만, (순전파에 필요한) `argmax`는 어디서든 그래디언트가 0이라서 인코더를 학습할 수 없게 만듭니다.

`F.gumbel_softmax(logits, tau=tau, hard=False)`는 `y_soft`를 만듭니다. 이는 원-핫 벡터의 *연속적인* 완화판으로, 32개 범주 전체에 대한 완전한 확률분포이며 가장 가능성 높은 범주 쪽으로 치우쳐 있고 `tau -> 0`일수록 진짜 원-핫 벡터에 가까워집니다. `y_hard`는 `argmax`로 얻은 진짜 이산 원-핫 벡터입니다. 반환 줄 `(y_hard - y_soft).detach() + y_soft`가 바로 그 트릭입니다. 순전파에서는 정확히 `y_hard`로 계산되므로(수치적으로 `(y_hard - y_soft).detach() + y_soft = y_hard`이기 때문입니다), 모델은 진짜 이산 토큰을 봅니다. 역전파에서는 `.detach()`가 `(y_hard - y_soft)`를 그래디언트 0인 상수로 만들어, 그래디언트가 `+ y_soft` 항만을 통해 흐르므로 인코더는 마치 자신이 *연속적인* 완화판을 출력한 것처럼 그래디언트를 받습니다. 이는 강의의 설명 그대로입니다. "순전파는 이산 샘플을 쓰고, 역전파는 이 연산을 항등함수로 취급해 그래디언트가 그대로 흐르게 한다."

`CatVAEEncoder`와 `CatVAEDecoder`는 그 외에는 P01의 VAE와 같은 합성곱 구조이지만, 인코더의 출력에서 핵심적인 차이가 있습니다. `CatVAEEncoder.forward`는 (연속 가우시안을 위한 `mu, log_var` 쌍이 아니라) 32개 범주에 대한 원 로짓을 반환하고, `CatVAE.encode`는 그 로짓에 스트레이트-스루 Gumbel-softmax를 적용해 학습 가능한 이산 토큰을 얻습니다.

```python
# Straight-through Gumbel-softmax.
def straight_through_gumbel(logits, tau=1.0):
    """Returns a straight-through estimator for discrete sampling."""
    y_soft = F.gumbel_softmax(logits, tau=tau, hard=False)
    y_hard = F.one_hot(y_soft.argmax(-1), num_classes=logits.shape[-1]).float()
    # Straight-through: forward uses y_hard, backward flows through y_soft
    return (y_hard - y_soft).detach() + y_soft


# --- Categorical VAE ---
NUM_CATEGORIES = 32   # discrete vocabulary size
Z_DIM = 32            # embedding dimension per token

class CatVAEEncoder(nn.Module):
    """CNN that maps a 3x64x64 frame to logits over NUM_CATEGORIES."""
    def __init__(self, num_categories=NUM_CATEGORIES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),   # 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # 8x8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),# 4x4
            nn.ReLU(),
            nn.Flatten(),                 # 256*4*4 = 4096
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_categories),
        )

    def forward(self, x):
        return self.net(x)  # (B, num_categories)


class CatVAEDecoder(nn.Module):
    """MLP + ConvTranspose that maps a 32-dim one-hot embedding back to 3x64x64."""
    def __init__(self, num_categories=NUM_CATEGORIES):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_categories, 256),
            nn.ReLU(),
            nn.Linear(256, 256 * 4 * 4),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),     # 64x64
            nn.Sigmoid(),
        )

    def forward(self, z_onehot):
        h = self.fc(z_onehot)
        h = h.view(-1, 256, 4, 4)
        return self.deconv(h)  # (B, 3, 64, 64)


class CatVAE(nn.Module):
    def __init__(self, num_categories=NUM_CATEGORIES, tau=1.0):
        super().__init__()
        self.encoder = CatVAEEncoder(num_categories)
        self.decoder = CatVAEDecoder(num_categories)
        self.tau = tau

    def encode(self, x):
        """Returns straight-through one-hot and argmax indices."""
        logits = self.encoder(x)           # (B, K)
        z = straight_through_gumbel(logits, tau=self.tau)  # (B, K)
        idx = logits.argmax(-1)            # (B,)
        return z, idx, logits

    def forward(self, x):
        z, idx, logits = self.encode(x)
        recon = self.decoder(z)
        return recon, z, idx, logits

catvae = CatVAE(num_categories=NUM_CATEGORIES, tau=1.0).to(DEVICE)
total_params = sum(p.numel() for p in catvae.parameters())
print(f'CatVAE parameters: {total_params:,}')
```
완화 기법을 마련했으니, 합성 이미지에서 CatVAE를 학습시켜 이산 잠재 코드가 압축된 표현을 학습하도록 합니다. 손실은 (P01과 같은 MSE 재구성 항인) `recon_loss`와, P01에는 없던 `entropy_reg` 항을 결합합니다. `probs = F.softmax(logits, dim=-1).mean(0)`는 배치 전체에 걸친 평균 범주 사용 분포이고, `entropy_reg = (probs * (probs + 1e-8).log()).sum()`은 그 분포의 음의 엔트로피입니다. `0.01 * entropy_reg`를 최소화하는 것은 곧 엔트로피를 *최대화*하는 것이므로, 모델이 소수의 범주로 붕괴하지 않고 32개 범주를 대략 고르게 사용하도록 유도합니다. 이는 P01의 연속 VAE에서 쓴 KL 항의 이산 잠재 버전입니다. 두 항 모두 잠재 공간이 퇴화하지 않고 잘 정돈되도록(여기서는 범주 사용을 고르게 퍼뜨리도록) 존재하지만, 이산 잠재에 맞는 다른 메커니즘으로 강제된다는 점이 다릅니다.

```python
# Train CatVAE.
from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(images)
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0 if USE_TPU else 2, pin_memory=USE_CUDA)

opt_vae = torch.optim.Adam(catvae.parameters(), lr=3e-4)

VAE_EPOCHS = 30
vae_losses = []

print('Training CatVAE...')
for epoch in range(VAE_EPOCHS):
    epoch_loss = 0.0
    for (batch,) in loader:
        batch = batch.to(DEVICE)
        recon, z, idx, logits = catvae(batch)
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, batch)
        # Entropy regularization: encourage uniform category use
        probs = F.softmax(logits, dim=-1).mean(0)  # (K,)
        entropy_reg = (probs * (probs + 1e-8).log()).sum()  # negative entropy
        loss = recon_loss + 0.01 * entropy_reg
        opt_vae.zero_grad()
        loss.backward()
        optimizer_step(opt_vae)
        epoch_loss += recon_loss.item()
    vae_losses.append(epoch_loss / len(loader))
    if (epoch + 1) % 10 == 0:
        print(f'  Epoch {epoch+1:3d}/{VAE_EPOCHS}  recon_loss={vae_losses[-1]:.4f}')

print('CatVAE training complete.')

plt.figure(figsize=(7, 3))
plt.plot(vae_losses, linewidth=2, color='steelblue')
plt.xlabel('Epoch')
plt.ylabel('Reconstruction Loss (MSE)')
plt.title('Categorical VAE: Reconstruction Loss')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```
학습이 시작되었으니, 범주형 병목이 여전히 이미지 구조를 보존하는지 확인하는 간단한 재구성 점검을 추가합니다. 이는 P01이 연속 잠재 공간에 대해 실행한 것과 같은 점검을, 이제 프레임당 하나의 이산 32방향 코드에 적용한 것입니다. P01의 32개 연속 차원보다 훨씬 좁은 병목이므로, 재구성이 대략적인 형태와 색상은 보존하되 미세한 위치 정밀도는 잃을 것으로 예상하세요.

```python
# Visual check of CatVAE reconstructions.
catvae.eval()
with torch.no_grad():
    sample = images[:8].to(DEVICE)
    recon, _, _, _ = catvae(sample)

fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i in range(8):
    axes[0, i].imshow(sample[i].cpu().permute(1, 2, 0).numpy())
    axes[0, i].axis('off')
    axes[0, i].set_title('Original' if i == 0 else '')
    axes[1, i].imshow(recon[i].cpu().permute(1, 2, 0).numpy())
    axes[1, i].axis('off')
    axes[1, i].set_title('Recon' if i == 0 else '')
plt.suptitle('CatVAE: Originals (top) vs Reconstructions (bottom)')
plt.tight_layout()
plt.show()
catvae.train()
```
## 2. 인과적 Transformer

인과적 어텐션으로 미래 토큰과 보상을 예측합니다.

월드모델 Transformer는 (z, a) 토큰이 번갈아 나열된 시퀀스에서 동작합니다. 각 스텝 t에서:
- z_t는 스텝 t 관측의 CatVAE 인코딩입니다(32차원 원-핫).
- a_t는 동작 임베딩입니다(이산 동작 2개를 32차원으로 투영).

인과적 마스킹은 위치 t가 t까지의 위치만(t 포함) 주목할 수 있도록 보장하므로, 모델은 학습 중 미래 관측을 엿볼 수 없습니다.

위치 t의 출력은 다음 잠재 토큰 z_{t+1}(교차 엔트로피), 보상 r_t(MSE), 종료 플래그 d_t(BCE)를 예측합니다.

이 아키텍처는 STORM보다 [IRIS의 설계](../lectures/lecture-03-architecture-patterns/01-architectures-rnn-transformer-diffusion#iris-turning-images-into-sentences)에 더 가깝습니다. IRIS도 프레임 토큰과 동작 토큰을 하나의 시퀀스로 번갈아 배치하는데(`torch.stack([z_emb, a_emb], dim=2).view(B, 2*T, D)`가 정확히 그 `[z0, a0, z1, a1, ...]` 배치를 만듭니다), STORM은 Transformer에 넣기 전에 `z_t`와 `a_t`를 스텝당 하나의 토큰으로 융합해 시퀀스 길이를 절반으로 줄입니다. IRIS의 세 결합 예측 목표, 즉 전이, 보상, 종료는 아래의 `token_head`, `reward_head`, `done_head`와 정확히 대응됩니다. `_causal_mask` 메서드는 [셀프 어텐션 설명 상자](../lectures/lecture-03-architecture-patterns/01-architectures-rnn-transformer-diffusion#core-mechanism-1)에 나온 상삼각 마스크를 그대로 만듭니다. `torch.triu(..., diagonal=1)`은 대각선보다 엄격히 위에 있는 모든 위치(모든 미래 위치)를 `True`, 즉 "무시"로 표시하므로, 위치 `t`는 `0..t` 위치만 주목할 수 있고 그 이후는 볼 수 없습니다. 이것이 강의가 지목한 자기회귀 제약 그대로입니다.

```python
# Causal Transformer world model.
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
SEQ_LEN = 20   # trajectory length
N_ACTIONS = 2


class CausalTransformerWM(nn.Module):
    def __init__(self, num_categories=NUM_CATEGORIES, d_model=D_MODEL,
                 n_heads=N_HEADS, n_layers=N_LAYERS, n_actions=N_ACTIONS,
                 max_len=SEQ_LEN):
        super().__init__()
        self.d_model = d_model
        self.num_categories = num_categories

        # Project z (one-hot) and action to d_model
        self.z_proj = nn.Linear(num_categories, d_model)
        self.a_embed = nn.Embedding(n_actions, d_model)

        # Positional encoding
        pos = torch.arange(max_len * 2).unsqueeze(1)  # *2 for (z,a) interleaved
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len * 2, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

        # Transformer layers using nn.MultiheadAttention
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model * 4,
                batch_first=True, norm_first=True
            )
            for _ in range(n_layers)
        ])

        # Output heads
        self.token_head = nn.Linear(d_model, num_categories)   # next-token prediction
        self.reward_head = nn.Linear(d_model, 1)               # reward regression
        self.done_head = nn.Linear(d_model, 1)                 # done classification

    def _causal_mask(self, T):
        """Upper-triangular mask: True means 'ignore this position'."""
        return torch.triu(torch.ones(T, T, device=self.pe.device), diagonal=1).bool()

    def forward(self, z_seq, a_seq):
        """
        z_seq: (B, T, num_categories)  one-hot latents
        a_seq: (B, T)                  integer actions
        Returns:
          token_logits: (B, T, num_categories)
          reward_pred:  (B, T, 1)
          done_pred:    (B, T, 1)
        """
        B, T, _ = z_seq.shape

        z_emb = self.z_proj(z_seq)          # (B, T, D)
        a_emb = self.a_embed(a_seq)         # (B, T, D)

        # Interleave z and a tokens: [z0, a0, z1, a1, ...] -> length 2T
        tokens = torch.stack([z_emb, a_emb], dim=2).view(B, 2 * T, self.d_model)
        tokens = tokens + self.pe[:2 * T].unsqueeze(0)

        mask = self._causal_mask(2 * T)
        h = tokens
        for layer in self.layers:
            h = layer(h, src_mask=mask, is_causal=False)

        # Extract z positions (even indices) for prediction heads
        z_h = h[:, 0::2, :]   # (B, T, D)  -- output at each z position

        token_logits = self.token_head(z_h)   # (B, T, K)
        reward_pred  = self.reward_head(z_h)  # (B, T, 1)
        done_pred    = self.done_head(z_h)    # (B, T, 1)
        return token_logits, reward_pred, done_pred


transformer_wm = CausalTransformerWM().to(DEVICE)
total_params_t = sum(p.numel() for p in transformer_wm.parameters())
print(f'Causal Transformer parameters: {total_params_t:,}')
```
## 3. 학습

P02와 같은 합성 궤적 데이터를 생성합니다. 각각 20스텝인 200개의 궤적으로, 에이전트가 화면을 가로질러 도형을 미는 두 동작짜리 환경을 사용합니다. Transformer를 학습시키기 전에 모든 관측은 CatVAE로 인코딩됩니다.

```python
# Synthetic trajectory data.
def make_obs(cx, cy, size=64):
    img = np.zeros((3, size, size), dtype=np.float32)
    r = 8
    color = np.array([0.9, 0.3, 0.3], dtype=np.float32)
    for y in range(size):
        for x in range(size):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                img[:, y, x] = color
    return img


def generate_trajectories(n_traj=200, horizon=20, size=64, seed=0):
    rng = np.random.RandomState(seed)
    obs_list, act_list, rew_list, done_list = [], [], [], []
    for _ in range(n_traj):
        cx = rng.randint(20, size - 20)
        cy = rng.randint(20, size - 20)
        traj_obs, traj_act, traj_rew, traj_done = [], [], [], []
        for t in range(horizon):
            traj_obs.append(make_obs(cx, cy, size))
            action = rng.randint(0, 2)
            traj_act.append(action)
            # Action 0: move right, Action 1: move left
            cx = np.clip(cx + (4 if action == 0 else -4), 10, size - 10)
            rew = 1.0 if cx > size // 2 else 0.0
            traj_rew.append(rew)
            traj_done.append(0.0)
        obs_list.append(traj_obs)
        act_list.append(traj_act)
        rew_list.append(traj_rew)
        done_list.append(traj_done)
    obs_arr  = torch.tensor(np.array(obs_list),  dtype=torch.float32)   # (N, T, 3, 64, 64)
    act_arr  = torch.tensor(np.array(act_list),  dtype=torch.long)       # (N, T)
    rew_arr  = torch.tensor(np.array(rew_list),  dtype=torch.float32)   # (N, T)
    done_arr = torch.tensor(np.array(done_list), dtype=torch.float32)   # (N, T)
    return obs_arr, act_arr, rew_arr, done_arr


print('Generating 200 synthetic trajectories (20 steps each)...')
obs_arr, act_arr, rew_arr, done_arr = generate_trajectories(n_traj=200, horizon=SEQ_LEN)
print(f'obs: {obs_arr.shape}, act: {act_arr.shape}, rew: {rew_arr.shape}')
```
이미지 쪽이 끝났으니, 궤적 쪽으로 넘어가 Transformer 월드모델이 학습할 시간적 데이터를 마련합니다. `generate_trajectories`는 P02/P03보다 더 단순한 새 합성 환경을 만듭니다. 공 하나가 동작에 따라 고정된 스텝만큼 왼쪽이나 오른쪽으로 움직이고, 공이 중앙보다 오른쪽에 있을 때마다 보상 `1.0`을 받습니다. 이는 P03의 환경보다 의도적으로 더 단순한데, 여기서의 목표는 정책 학습을 보여주는 것이 아니라 RSSM과 Transformer의 예측 품질을 통제된 조건에서 비교하는 것이기 때문입니다. 이 노트북에서는 액터나 크리틱을 전혀 학습시키지 않습니다.

```python
# Encode all observations with CatVAE.
catvae.eval()
N, T, C, H, W = obs_arr.shape
z_encoded = torch.zeros(N, T, NUM_CATEGORIES)  # one-hot tokens

with torch.no_grad():
    flat_obs = obs_arr.view(N * T, C, H, W).to(DEVICE)
    logits_all = catvae.encoder(flat_obs)                              # (N*T, K)
    idx_all = logits_all.argmax(-1)                                    # (N*T,)
    z_onehot_all = F.one_hot(idx_all, num_classes=NUM_CATEGORIES).float()  # (N*T, K)
    z_encoded = z_onehot_all.view(N, T, NUM_CATEGORIES).cpu()

print('Encoded latents shape:', z_encoded.shape)
unique_tokens = idx_all.unique().numel()
print(f'Unique token categories used: {unique_tokens} / {NUM_CATEGORIES}')
catvae.train()
```
관측이 토큰화되었으니, Transformer는 잠재 시퀀스에 대해 시간적 동역학을 직접 모델링할 수 있습니다. 200개 궤적 전체의 모든 관측은 고정된 `catvae.encoder`를 통해 하나의 배치 순전파로 처리되고(`obs_arr.view(N * T, C, H, W)`), CatVAE 학습 때 쓴 스트레이트-스루 완화가 아니라 `argmax`를 통해 단단한 원-핫 벡터로 변환됩니다. 이 단계에서는 인코더가 고정되어 있고 Transformer의 학습 목표만을 만들어내므로, 토큰화를 거슬러 그래디언트가 흐를 필요가 없고, (부드러운 완화가 아니라) 실제 이산 선택을 쓰면 Transformer가 학습할 명확하고 모호하지 않은 토큰 레이블을 얻을 수 있습니다.

```python
# Train the Causal Transformer.
from torch.utils.data import TensorDataset, DataLoader

traj_dataset = TensorDataset(z_encoded, act_arr, rew_arr, done_arr)
traj_loader  = DataLoader(traj_dataset, batch_size=32, shuffle=True)

opt_t = torch.optim.Adam(transformer_wm.parameters(), lr=1e-3)

TRANS_EPOCHS = 20
token_losses, reward_losses = [], []
epoch_times = []   # wall-clock seconds per epoch

print('Training Causal Transformer...')
for epoch in range(TRANS_EPOCHS):
    t0 = time.time()
    ep_tok, ep_rew = 0.0, 0.0
    for z_b, a_b, r_b, d_b in traj_loader:
        z_b = z_b.to(DEVICE)
        a_b = a_b.to(DEVICE)
        r_b = r_b.to(DEVICE)
        d_b = d_b.to(DEVICE)

        # Predict: at position t, predict token t+1, reward t, done t
        token_logits, reward_pred, done_pred = transformer_wm(z_b, a_b)  # (B, T, K/1/1)

        # Next-token labels: shift by 1, ignore last position
        target_idx = z_b[:, 1:, :].argmax(-1)           # (B, T-1)
        pred_logits = token_logits[:, :-1, :]            # (B, T-1, K)
        tok_loss = F.cross_entropy(
            pred_logits.reshape(-1, NUM_CATEGORIES),
            target_idx.reshape(-1)
        )

        # Reward prediction at all positions
        rew_loss = F.mse_loss(reward_pred.squeeze(-1), r_b)

        # Done prediction
        done_loss = F.binary_cross_entropy_with_logits(done_pred.squeeze(-1), d_b)

        loss = tok_loss + 0.5 * rew_loss + 0.1 * done_loss
        opt_t.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(transformer_wm.parameters(), 1.0)
        opt_t.step()

        ep_tok += tok_loss.item()
        ep_rew += rew_loss.item()

    elapsed = time.time() - t0
    epoch_times.append(elapsed)
    token_losses.append(ep_tok / len(traj_loader))
    reward_losses.append(ep_rew / len(traj_loader))
    if (epoch + 1) % 5 == 0:
        print(f'  Epoch {epoch+1:2d}/{TRANS_EPOCHS}  '
              f'tok_loss={token_losses[-1]:.4f}  '
              f'rew_loss={reward_losses[-1]:.4f}  '
              f'time={elapsed:.2f}s')

print('Transformer training complete.')
```
최적화가 진행 중이니, 토큰 손실과 보상 손실을 함께 추적해 두 예측 헤드가 모두 시야에 들어오도록 합니다. 레이블 구성은 자세히 읽어볼 가치가 있습니다. `target_idx = z_b[:, 1:, :].argmax(-1)`은 인덱스 1부터 모든 위치의 *실제* 토큰을 가져오고, `pred_logits = token_logits[:, :-1, :]`는 마지막에서 두 번째까지 모든 위치에서의 모델 예측을 가져옵니다. 인과적 마스크 때문에 `token_logits[:, t, :]`는 위치 `t` *다음에* 오는 것에 대한 모델의 예측이므로, `pred_logits[t]`를 `target_idx[t]`(즉 `z_b[t+1]`)와 짝짓는 것이 바로 강의가 설명하는 "지금까지 본 모든 것으로부터 다음 토큰을 예측하기" 목표이며, IRIS와 STORM에서와 정확히 같은 일반적인 다음 토큰 교차 엔트로피로 구현됩니다. 전체 손실은 세 항에 가중치를 둡니다(`tok_loss + 0.5 * rew_loss + 0.1 * done_loss`). 이 가중치는 가장 어렵고 가장 많은 샘플을 필요로 하는 목표인 토큰 예측이 학습 초반 그래디언트를 지배하도록 선택된 것입니다.

```python
# --- Plot token and reward losses ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(token_losses,  label='Token prediction loss (CE)',  color='steelblue',  linewidth=2)
ax.plot(reward_losses, label='Reward prediction loss (MSE)', color='darkorange', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Causal Transformer: Training Losses')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```
## 4. 롤아웃 품질 비교

두 백본에서 롤아웃을 디코딩해 여러 지평선에 걸쳐 이미지 품질을 비교합니다.

RSSM(P02)과 인과적 Transformer의 상상된 롤아웃을 비교합니다. P02의 `rssm.pt`가 있으면 불러옵니다. 없으면 P02와 같은 무작위 시드로 RSSM을 초기화합니다. 두 모델 모두 같은 시작 상태에서 10스텝 롤아웃을 생성하고, CatVAE 디코더를 사용해 예측된 잠재값을 다시 픽셀로 디코딩합니다. PSNR은 각 지평선 스텝에서 픽셀 수준의 품질을 측정합니다.

짚어둘 만한 비대칭이 하나 있습니다. (바로 위에서 정의한, P02와 호환되는 사본인) `RSSM.prior_step`은 사전분포의 *평균*만 반환하며(`mu, _ = pr.chunk(2, dim=-1); return mu, h`), P02 자체의 `RSSM.rollout`과 정확히 같은 샘플링 없는 결정론적 롤아웃입니다. 아래의 Transformer 롤아웃도 결정론적입니다(`tok_logits[:, -1, :].argmax(-1)`, 항상 가장 확률이 높은 토큰). 그래서 두 모델은 짝을 맞춘 결정론적 롤아웃 조건에서 비교되며, 이렇게 하면 아키텍처의 효과(순환 상태 대 어텐션)를 확률적 샘플링의 효과와 분리할 수 있습니다. 그렇지 않으면 이런 지평선 품질 비교에서 교란 요인이 되었을 것입니다.

```python
# PSNR utility.
def psnr(pred, target):
    mse = F.mse_loss(pred, target)
    return 10 * torch.log10(1.0 / (mse + 1e-8))


# P02-compatible RSSM so the saved checkpoint can load cleanly here.
class RSSM(nn.Module):
    def __init__(self, latent_dim=32, action_dim=1, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        self.gru = nn.GRUCell(latent_dim + action_dim, hidden_dim)
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )
        self.post_net = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )
        self.recon = nn.Linear(latent_dim, latent_dim)

    def _rsample(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def prior_step(self, z, h, a):
        if a.dim() == 1:
            a = a.unsqueeze(-1)
        h = self.gru(torch.cat([z, a.float()], dim=-1), h)
        pr = self.prior_net(h)
        mu, _ = pr.chunk(2, dim=-1)
        return mu, h

    def posterior_step(self, h, a, z_obs):
        if a.dim() == 1:
            a = a.unsqueeze(-1)
        po = self.post_net(torch.cat([h, z_obs], dim=-1))
        mu, logvar = po.chunk(2, dim=-1)
        z = self._rsample(mu, logvar)
        h = self.gru(torch.cat([z, a.float()], dim=-1), h)
        return z, h


# --- Load or initialize RSSM ---
rssm_path = next((p for p in [PATH / 'rssm.pt', PATH / 'notebooks' / 'rssm.pt'] if p.exists()), None)
if rssm_path is not None:
    try:
        state = torch.load(rssm_path, map_location=DEVICE)
        if isinstance(state, dict) and 'rssm_state_dict' in state:
            rssm = RSSM(
                latent_dim=int(state.get('latent_dim', 32)),
                action_dim=int(state.get('action_dim', 1)),
                hidden_dim=int(state.get('hidden_dim', 128)),
            ).to(DEVICE)
            rssm.load_state_dict(state['rssm_state_dict'])
            print(
                f"Loaded RSSM weights from {rssm_path} "
                f"(hidden_dim={rssm.hidden_dim}, latent_dim={rssm.latent_dim}, action_dim={rssm.action_dim})"
            )
        elif isinstance(state, dict) and 'rssm' in state:
            rssm = RSSM().to(DEVICE)
            rssm.load_state_dict(state['rssm'])
            print(f'Loaded RSSM weights from {rssm_path} (legacy rssm key)')
        else:
            rssm = RSSM().to(DEVICE)
            rssm.load_state_dict(state)
            print(f'Loaded RSSM weights from {rssm_path} (raw state_dict)')
    except Exception as e:
        rssm = RSSM().to(DEVICE)
        print(f'Could not load rssm.pt ({e}). Using randomly initialized RSSM.')
else:
    rssm = RSSM().to(DEVICE)
    print('rssm.pt not found. Using randomly initialized RSSM (P02 baseline).')

rssm.eval()
transformer_wm.eval()
catvae.eval()
```
PSNR 유틸리티를 정의했으니, 여러 롤아웃 지평선에 걸쳐 평가해 예측 품질이 시간이 지나며 어떻게 저하되는지 살펴봅니다. `psnr`은 [L04의 STORM 지표 페이지](../lectures/lecture-04-evaluation-by-model/04-storm-diffusion-drift#long-horizon-psnr)의 공식을 구현합니다: $\text{PSNR} = 10 \log_{10}(\text{MAX}^2 / \text{MSE})$이며, 여기서는 픽셀이 `[0, 1]`로 정규화되어 있으므로 `MAX = 1.0`입니다. 두 롤아웃 모두 공유된 시작 프레임 `z0`에서 전적으로 개방루프로 진행됩니다. RSSM 루프는 `rssm.prior_step`을 반복 호출하고(스텝 0 이후로는 관측이 없습니다), Transformer 루프는 자신이 예측한 토큰을 `z_seq`에 이어붙여 점점 길어지는 시퀀스를 다시 입력합니다(`z_seq = torch.cat([z_seq, next_z.unsqueeze(1)], dim=1)`). 그래서 둘 다 L04에서 STORM에 대해 논의한 것과 같은 teacher forcing 격차를 겪습니다. 각 모델이 예측한 것이 자신의 다음 입력이 되므로, 초반의 작은 오차가 10스텝 지평선에 걸쳐 누적될 수 있습니다.

```python
# Compute PSNR vs horizon.
ROLLOUT_LEN = 10
N_EVAL = 5  # trajectories to average over

# Use the first N_EVAL trajectories as eval set
eval_obs  = obs_arr[:N_EVAL].to(DEVICE)   # (N_EVAL, T, 3, 64, 64)
eval_act  = act_arr[:N_EVAL].to(DEVICE)   # (N_EVAL, T)
eval_z    = z_encoded[:N_EVAL].to(DEVICE) # (N_EVAL, T, K)

horizons = [1, 3, 5, 10]
psnr_rssm_all  = {h: [] for h in horizons}
psnr_trans_all = {h: [] for h in horizons}

with torch.no_grad():
    for traj_i in range(N_EVAL):
        # Starting state: encode step 0
        z0 = eval_z[traj_i, 0:1]         # (1, K)  one-hot
        acts = eval_act[traj_i]           # (T,)

        # ---- RSSM rollout ----
        z = z0.clone()
        h = torch.zeros(1, rssm.hidden_dim, device=DEVICE)
        rssm_preds = []  # decoded frames at each step
        for t in range(ROLLOUT_LEN):
            a_t = acts[t:t+1].float().unsqueeze(-1)  # (1, 1)
            z, h = rssm.prior_step(z, h, a_t)
            frame = catvae.decoder(z)      # (1, 3, 64, 64)
            rssm_preds.append(frame)

        # ---- Transformer rollout ----
        # Seed with observed z0, then autoregressively predict
        z_seq = z0.unsqueeze(0)            # (1, 1, K)
        trans_preds = []
        for t in range(ROLLOUT_LEN):
            current_len = z_seq.shape[1]
            a_prefix = acts[:current_len].unsqueeze(0)  # (1, current_len)
            tok_logits, _, _ = transformer_wm(z_seq, a_prefix)  # (1, L, K)
            next_logits = tok_logits[:, -1, :]           # (1, K)
            next_z = F.one_hot(next_logits.argmax(-1), num_classes=NUM_CATEGORIES).float()  # (1, K)
            frame = catvae.decoder(next_z)               # (1, 3, 64, 64)
            trans_preds.append(frame)
            z_seq = torch.cat([z_seq, next_z.unsqueeze(1)], dim=1)  # (1, L+1, K)

        # Ground truth frames
        for h in horizons:
            if h <= ROLLOUT_LEN and h < eval_obs.shape[1]:
                gt = eval_obs[traj_i, h:h+1]             # (1, 3, 64, 64)
                p_rssm  = psnr(rssm_preds[h-1],  gt).item()
                p_trans = psnr(trans_preds[h-1], gt).item()
                psnr_rssm_all[h].append(p_rssm)
                psnr_trans_all[h].append(p_trans)

psnr_rssm_mean  = [np.mean(psnr_rssm_all[h])  for h in horizons]
psnr_trans_mean = [np.mean(psnr_trans_all[h]) for h in horizons]

print('PSNR (dB) by horizon step:')
print(f'{"Horizon":>10}  {"RSSM":>10}  {"Transformer":>12}')
for h, r, t in zip(horizons, psnr_rssm_mean, psnr_trans_mean):
    print(f'{h:>10}  {r:>10.2f}  {t:>12.2f}')
```
점수를 모았으니, 지평선 곡선을 그려 전체적인 추세를 더 쉽게 읽을 수 있게 합니다. 이 그래프를 [L04의 지평선 드리프트 논의](../lectures/lecture-04-evaluation-by-model/05-diffusion-drift#horizon-drift-the-universal-failure-mode-across-all-world-models)와 함께 읽어보세요. 두 곡선 모두 지평선이 늘어날수록 하락할 것으로 예상됩니다(드리프트는 모든 아키텍처에 걸친 보편적인 현상입니다). 하지만 실제로 비교할 만한 지점은 RSSM의 고정 크기 순환 상태와 Transformer의 전체 컨텍스트 어텐션 사이에서 하락하는 *속도*입니다.

```python
# --- Plot PSNR vs horizon ---
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(horizons, psnr_rssm_mean,  'o-', color='royalblue',  linewidth=2, label='RSSM (P02)', markersize=7)
ax.plot(horizons, psnr_trans_mean, 's-', color='tomato',     linewidth=2, label='Transformer (P04)', markersize=7)
ax.set_xlabel('Rollout Horizon (steps)')
ax.set_ylabel('PSNR (dB)')
ax.set_title('Rollout Quality: PSNR vs Horizon')
ax.set_xticks(horizons)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```
지평선 곡선은 전체적인 패턴을 보여줍니다. 이미지 격자는 스텝별 차이를 구체적으로 보여줍니다. 스텝 1, 5, 10에서 실제로 디코딩된 프레임을 실제값과 나란히 보면 각 백본이 어떤 *종류*의 오차를 누적하는지(흐림, 색상 드리프트, 위치 드리프트) 드러납니다. 이는 위의 단일 PSNR 수치로는 평균화되어 사라지는 정보입니다.

```python
# Image grid: GT / RSSM / Transformer.
display_steps = [1, 5, 10]
traj_i = 0  # use first eval trajectory

with torch.no_grad():
    z0 = eval_z[traj_i, 0:1].to(DEVICE)
    acts = eval_act[traj_i].to(DEVICE)

    # RSSM rollout
    z = z0.clone()
    h = torch.zeros(1, rssm.hidden_dim, device=DEVICE)
    rssm_frames = []
    for t in range(10):
        a_t = acts[t:t+1].float().unsqueeze(-1)
        z, h = rssm.prior_step(z, h, a_t)
        rssm_frames.append(catvae.decoder(z).cpu().squeeze(0))

    # Transformer rollout
    z_seq = z0.unsqueeze(0)
    trans_frames = []
    for t in range(10):
        current_len = z_seq.shape[1]
        a_prefix = acts[:current_len].unsqueeze(0)
        tok_logits, _, _ = transformer_wm(z_seq, a_prefix)
        next_z = F.one_hot(tok_logits[:, -1, :].argmax(-1), num_classes=NUM_CATEGORIES).float()
        trans_frames.append(catvae.decoder(next_z).cpu().squeeze(0))
        z_seq = torch.cat([z_seq, next_z.unsqueeze(1)], dim=1)

fig, axes = plt.subplots(3, len(display_steps), figsize=(10, 7))
row_labels = ['Ground Truth', 'RSSM (P02)', 'Transformer (P04)']
for col, step in enumerate(display_steps):
    gt_frame = eval_obs[traj_i, step].cpu().permute(1, 2, 0).numpy()
    rssm_frame = rssm_frames[step - 1].permute(1, 2, 0).numpy()
    trans_frame = trans_frames[step - 1].permute(1, 2, 0).numpy()
    for row, (frame, label) in enumerate(zip(
            [gt_frame, rssm_frame, trans_frame], row_labels)):
        ax = axes[row, col]
        ax.imshow(np.clip(frame, 0, 1))
        ax.axis('off')
        if col == 0:
            ax.set_ylabel(label, fontsize=10)
        if row == 0:
            ax.set_title(f'Step {step}', fontsize=10)

plt.suptitle('Imagined Rollouts: Ground Truth vs RSSM vs Transformer', y=1.01)
plt.tight_layout()
plt.show()
```
## 5. 학습 효율

실측 학습 시간은 동역학 백본을 고를 때 실용적으로 고려할 사항입니다. 여기서는 (위에서 측정한) Transformer의 에폭당 시간을 간단한 RSSM 재실행과 비교한 뒤, 두 모델 모두에 대해 검증 손실을 누적 실측 시간에 대해 그래프로 그립니다.

`RSSMForTiming`은 오직 이 시간 비교만을 위해 정의된 새롭고 최소한의 RSSM 클래스입니다(위 롤아웃 비교에서 쓴 `RSSM` 클래스와 같은 객체가 아닙니다). 다음 절의 두 손실 곡선이 비교 가능한 스케일에 있도록 Transformer와 같은 토큰 예측 교차 엔트로피 목적함수로 학습됩니다. 이는 강의가 말한 트레이드오프를 직접 보여줍니다. RSSM은 궤적의 20개 스텝 각각을 파이썬 `for t in range(T)` 루프 안에서 `self.gru(inp, h)`를 통해 순차적으로 처리하며(스텝당 GRU 셀 호출 한 번), Transformer의 어텐션 층들은 번갈아 배치된 전체 시퀀스를 하나의 순전파로 처리합니다. 여기서 출력되는 실측 시간 수치는 아래 아키텍처 비교 표의 "학습 병렬성: 높음 대 낮음" 행을 이 노트북이 직접 실증적으로 보여주는 버전입니다.

```python
# Brief RSSM timing run.
class RSSMForTiming(nn.Module):
    """Compact RSSM for training-time comparison."""
    def __init__(self, z_dim=NUM_CATEGORIES, a_dim=N_ACTIONS, h_dim=128):
        super().__init__()
        self.h_dim = h_dim
        self.a_embed = nn.Embedding(a_dim, 32)
        self.gru = nn.GRUCell(z_dim + 32, h_dim)
        self.prior = nn.Sequential(nn.Linear(h_dim, 128), nn.ELU(), nn.Linear(128, z_dim))
        self.post  = nn.Sequential(nn.Linear(h_dim + z_dim, 128), nn.ELU(), nn.Linear(128, z_dim))

    def forward(self, z_seq, a_seq):
        B, T, K = z_seq.shape
        h = torch.zeros(B, self.h_dim, device=z_seq.device)
        prior_logits, post_logits = [], []
        for t in range(T):
            a_emb = self.a_embed(a_seq[:, t])
            inp = torch.cat([z_seq[:, t], a_emb], -1)
            h = self.gru(inp, h)
            prior_logits.append(self.prior(h))
            post_logits.append(self.post(torch.cat([h, z_seq[:, t]], -1)))
        prior_logits = torch.stack(prior_logits, 1)  # (B, T, K)
        post_logits  = torch.stack(post_logits, 1)
        return prior_logits, post_logits

torch.manual_seed(42)
rssm_timing = RSSMForTiming().to(DEVICE)
opt_rssm = torch.optim.Adam(rssm_timing.parameters(), lr=1e-3)

RSSM_EPOCHS = 20
rssm_losses, rssm_times = [], []

print('Brief RSSM training run for timing comparison...')
for epoch in range(RSSM_EPOCHS):
    t0 = time.time()
    ep_loss = 0.0
    for z_b, a_b, r_b, d_b in traj_loader:
        z_b = z_b.to(DEVICE)
        a_b = a_b.to(DEVICE)
        prior_logits, post_logits = rssm_timing(z_b, a_b)
        # KL + reconstruction loss
        target_idx = z_b[:, 1:, :].argmax(-1)          # (B, T-1)
        loss = F.cross_entropy(
            prior_logits[:, :-1, :].reshape(-1, NUM_CATEGORIES),
            target_idx.reshape(-1)
        )
        opt_rssm.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(rssm_timing.parameters(), 1.0)
        opt_rssm.step()
        ep_loss += loss.item()
    rssm_losses.append(ep_loss / len(traj_loader))
    rssm_times.append(time.time() - t0)
    if (epoch + 1) % 5 == 0:
        print(f'  Epoch {epoch+1:2d}/{RSSM_EPOCHS}  loss={rssm_losses[-1]:.4f}  time={rssm_times[-1]:.2f}s')

print(f'RSSM avg time/epoch: {np.mean(rssm_times):.3f}s')
print(f'Transformer avg time/epoch: {np.mean(epoch_times):.3f}s')
```
시간 측정 기준선을 확보했으니, 검증 손실을 실측 시간에 대해 비교해 효율성 트레이드오프를 명확히 드러냅니다. 왼쪽 패널은 각 모델의 손실 곡선을 에폭 수가 아니라 자신의 누적 학습 시간에 대해 그리는데, 이것이 더 공정한 비교입니다. 두 모델은 같은 수의 에폭을 완료하는 데 서로 다른 실측 시간이 걸릴 수 있으므로, 손실 대 에폭 곡선을 겹쳐 그리면 이 절이 드러내려는 연산 효율성 차이가 바로 가려져 버립니다.

```python
# Plot validation loss vs wall-clock time.
trans_cumtime = np.cumsum(epoch_times)
rssm_cumtime  = np.cumsum(rssm_times)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Left: loss vs wall-clock time
axes[0].plot(trans_cumtime, token_losses, 'o-', color='tomato',    linewidth=2, markersize=4, label='Transformer (token CE)')
axes[0].plot(rssm_cumtime,  rssm_losses,  's-', color='royalblue', linewidth=2, markersize=4, label='RSSM (token CE)')
axes[0].set_xlabel('Wall-clock time (s)')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss vs Wall-clock Time')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Right: per-epoch time bar chart
x = np.arange(1, TRANS_EPOCHS + 1)
axes[1].bar(x - 0.2, epoch_times,  width=0.4, color='tomato',    label='Transformer', alpha=0.8)
axes[1].bar(x + 0.2, rssm_times,   width=0.4, color='royalblue', label='RSSM',        alpha=0.8)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Time (s)')
axes[1].set_title('Per-Epoch Training Time')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```
## 아키텍처 비교

RSSM과 Transformer는 상태 표현과 연산에서 서로 다른 트레이드오프를 취합니다.

| 속성 | RSSM (P02) | 인과적 Transformer (P04) |
|---|---|---|
| **잠재 표현** | 연속 가우시안(결정론적 + 확률적) | 이산 범주형(Gumbel-softmax) |
| **시퀀스 처리** | 순차적 GRU, 한 번에 한 스텝 | 전체 컨텍스트 창에 대한 병렬 어텐션 |
| **그래디언트 흐름** | GRU 은닉 상태를 통해 | 어텐션 가중치를 통해 |
| **인과 구조** | GRU 재귀에 암묵적으로 내재 | 명시적인 상삼각 마스크 |
| **장거리 기억** | 거리에 따라 흐려짐(그래디언트 소실) | 모든 위치에 대한 어텐션으로 보존 |
| **학습 병렬성** | 낮음: 순차적으로 펼쳐야 함 | 높음: 모든 위치를 한 번의 순전파로 계산 |
| **파라미터 수** | 더 적음(GRU 셀) | 더 많음(어텐션 층) |
| **일반적인 에폭당 시간** | 짧은 시퀀스에서 더 빠름 | 더 느리지만 긴 시퀀스에서 더 잘 확장됨 |

핵심 트레이드오프는 귀납적 편향 대 유연성입니다. RSSM의 GRU는 최근성과 매끄러운 잠재 보간에 대한 내재된 개념을 가지고 있어 단기 지평선 예측 과제에서 흔히 도움이 됩니다. Transformer는 스텝당 더 높은 연산 비용을 치르지만 컨텍스트 창 안의 어떤 위치에도 똑같이 잘 주목할 수 있는데, 이는 장거리 의존성이 있는 과제에서 중요합니다(예를 들어 여러 스텝 전에 열린 문은 지금 에이전트가 할 수 있는 일에 여전히 영향을 미칩니다).

실제로 STORM 스타일 모델들은 언어모델과 비슷하게 시퀀스 길이에 따라 더 예측 가능하게 확장되고 토큰 수준 추론에 잘 맞기 때문에 정확히 이 이유로 Transformer를 동역학 백본으로 사용합니다.

```python
# Save model weights.
save_path = PATH / 'transformer_wm.pt'
save_path.parent.mkdir(parents=True, exist_ok=True)

torch.save({
    'catvae': catvae.state_dict(),
    'transformer_wm': transformer_wm.state_dict(),
    'token_losses': token_losses,
    'reward_losses': reward_losses,
    'psnr_rssm': psnr_rssm_mean,
    'psnr_transformer': psnr_trans_mean,
    'horizons': horizons,
}, save_path)

print(f'Saved to {save_path}')
print(f'  CatVAE final recon loss:            {vae_losses[-1]:.4f}')
print(f'  Transformer final token loss:        {token_losses[-1]:.4f}')
print(f'  Transformer final reward loss:       {reward_losses[-1]:.4f}')
print(f'  RSSM PSNR at horizon 10:             {psnr_rssm_mean[-1]:.2f} dB')
print(f'  Transformer PSNR at horizon 10:      {psnr_trans_mean[-1]:.2f} dB')
print(f'  Total Transformer training time:     {trans_cumtime[-1]:.1f}s')
print(f'  Total RSSM training time:            {rssm_cumtime[-1]:.1f}s')
```
