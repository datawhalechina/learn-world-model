---
title: P01 VAE 인코더 학습
---

# P01: VAE 인코더 학습

합성 64x64 RGB 이미지에 대해 컴팩트한 합성곱 변이형 오토인코더(Variational Autoencoder, VAE)를 학습시킵니다. 인코더는 32차원 잠재 공간을 학습하며, P02는 이를 관측 인코더로 재사용합니다. 여기서 목표는 사실적인 이미지 생성이 아니라, 이후 노트북들이 활용할 수 있는 안정적인 잠재 공간을 학습하는 것입니다.

**출력**: 이 노트북은 사전 체크포인트 없이 처음부터 학습하며, 학습된 가중치를 `vae_encoder.pt`로 저장합니다. P02와 P03은 이 파일을 관측 인코더로 불러옵니다.

**개요:**
1. 준비: 합성 데이터와 DataLoader
2. 모델: 인코더, 디코더, 재매개변수화
3. ELBO 손실: 재구성과 KL
4. 학습: 30 에폭과 손실 곡선
5. 점검: 재구성, 순회, 무작위 샘플

> Notebook 원본: [p01_vae_encoder.ipynb](https://github.com/datawhalechina/learn-world-model/blob/main/docs/ko/projects/p01_vae_encoder.ipynb)

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
## 1. 준비

노트북이 외부 다운로드 없이 오프라인에서 실행되도록 합성 색상 도형을 생성합니다.

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
런타임이 준비되었으니, 외부 다운로드 없이 VAE가 학습할 수 있는 합성 도형 데이터셋을 생성합니다.

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
## 2. 모델

`Encoder`는 [L02 관측 인코딩](../lectures/lecture-02-encode-and-dynamics/01-encoding)에서 다룬 CNN입니다. 스트라이드 2인 합성곱 네 개가 매번 공간 해상도를 절반으로 줄여 64에서 32, 16, 8을 거쳐 4까지 줄어들므로, 최종 특징 맵은 `256 x 4 x 4`가 됩니다. 이 맵을 평탄화한 뒤 두 개의 선형 헤드에 통과시키면 근사 사후분포 $q(z \mid x)$의 평균과 로그분산인 `mu`와 `log_var`가 나옵니다. `var`나 `sigma`를 직접 예측하지 않고 `log_var`를 예측하는 이유는, 네트워크의 원출력이 임의의 실수가 될 수 있게 하기 위해서입니다. 나중에 지수를 취하면 제약된 활성화 함수 없이도 분산이 항상 양수가 되도록 보장할 수 있습니다.

`Decoder`는 인코더를 거울처럼 뒤집은 구조로, 전치 합성곱(인코딩 페이지의 "전치 합성곱(Transposed Convolution)" 설명 참고)이 매번 공간 해상도를 두 배로 늘려 인코더의 압축을 되돌리는데, 4에서 8, 16, 32를 거쳐 64까지 늘어납니다. 마지막의 `Sigmoid`는 출력 픽셀 값을 `[0, 1]` 범위로 유지하는데, 이는 합성 이미지가 생성된 방식과 일치합니다.

`reparameterize`는 강의에서 다룬 재매개변수화 트릭 `z = mu + sigma * epsilon`을 구현한 것입니다. `log_var = log(sigma^2)`이므로 `0.5 * log_var = log(sigma)`가 되고, 여기에 지수를 취하면 `sigma`를 복원할 수 있습니다. 그래서 코드는 `std = exp(0.5 * log_var)`로 구현되어 있습니다. `torch.randn_like(std)`는 네트워크 파라미터와 무관하게 `epsilon ~ N(0, I)`를 뽑는데, 이것이 바로 이 연산을 미분 가능하게 만드는 핵심입니다. 그래디언트는 `mu`와 `std`를 통해서만 흐르고, 무작위 추출 자체를 통해서는 흐르지 않습니다.

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
## 3. ELBO 손실

`elbo_loss`는 [인코딩 페이지의 ELBO 유도](../lectures/lecture-02-encode-and-dynamics/01-encoding#elbo-loss-balancing-two-objectives)에서 다룬 두 항을 구현합니다.

$$
\mathcal{L} = \underbrace{\mathbb{E}_{q(z|x)}[-\log p(x|z)]}_{\text{재구성}} + \underbrace{D_{KL}(q(z|x) \| p(z))}_{\text{KL 발산}}
$$

**재구성 항.** `F.mse_loss(recon_x, x, reduction='mean')`가 $-\log p(x \mid z)$를 대신합니다. 이 대체는 표준적이지만 짚고 넘어갈 가치가 있습니다. 디코더의 출력 분포가 재구성된 픽셀을 중심으로 한 고정 분산 가우시안이라고 가정하면, $-\log p(x \mid z)$는 상수항을 제외하면 `recon_x`와 `x` 사이의 제곱 오차로 환원되며, 이것이 바로 MSE입니다. 따라서 MSE를 쓰는 것은 편의를 위한 근사가 아니라, 이런 가우시안 출력을 가정했을 때 나오는 정확한 음의 로그가능도 그 자체입니다.

**KL 항, 그리고 닫힌 형태가 나오는 이유.** 두 가우시안, 즉 $q(z \mid x) = \mathcal{N}(\mu, \sigma^2)$와 표준정규 사전분포 $p(z) = \mathcal{N}(0, 1)$ 사이에서는 KL 발산이 정확한 닫힌 형태로 구해집니다(몬테카를로 샘플링이 필요 없습니다). 가우시안 밀도에 대해 정의식 $D_{KL}(q \| p) = \int q(z) \log \frac{q(z)}{p(z)} \, dz$를 전개하면 다음이 유도됩니다.

$$
D_{KL}\big(\mathcal{N}(\mu, \sigma^2) \,\|\, \mathcal{N}(0, 1)\big) = -\frac{1}{2}\left(1 + \log \sigma^2 - \mu^2 - \sigma^2\right)
$$

이를 32개 잠재 차원 전체에 대해 합하면(각 차원이 독립이라고 가정하므로, 전체 KL은 차원별 KL의 합입니다) 코드의 `torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)` 항에 부호를 바꾸고 절반을 취한 것과 정확히 같아집니다. `log_var.exp()`는 $\log \sigma^2$에서 $\sigma^2$를 복원하는데, 이는 위 `reparameterize`에서 쓴 것과 같은 관계입니다.

**KL 항을 다시 스케일링하는 이유.** 코드는 추가로 `kl_loss`를 `IMG_CH * IMG_SIZE * IMG_SIZE`(12,288, [왜 압축하는가](../lectures/lecture-02-encode-and-dynamics/01-encoding#why-compress)에서도 압축의 이유로 든 것과 같은 픽셀 수)로 나눕니다. 재구성 MSE는 이미 12,288개 픽셀에 대한 평균이지만, KL은 32개 잠재 차원에 대한 합일 뿐이므로, 재스케일링하지 않으면 KL 항이 재구성 항보다 몇 자릿수나 작아져서 학습에 거의 영향을 미치지 못합니다. 같은 픽셀 수로 나누면 `kl_weight`를 곱하기 전에 두 항을 비슷한 스케일로 맞출 수 있습니다.

아래 점검은 학습을 시작하기 전에 순전파를 한 번 실행해, 손실 함수가 정상적으로 실행되고 예상한 형태를 만드는지 확인합니다. 여기서 출력되는 KL 손실은 0에 가까워야 하는데, 학습되지 않은 인코더의 `mu`와 `log_var`가 초기화 값인 `(0, 0)` 근처에 있기 때문입니다. 이는 학습을 통해서가 아니라 우연히 표준정규 사전분포와 비슷해진 것일 뿐입니다.

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
## 4. 학습 루프

표준적인 `EPOCHS`/`LR` 쌍 외에 중요한 하이퍼파라미터가 세 가지 있습니다.

- **`KL_WEIGHT = 3.0`**은 $\beta$-VAE 공식화에서 나온 $\beta$ 계수입니다. 재구성 손실에 더하기 전에 KL 항에 곱해집니다. 1보다 큰 가중치는 일반 ELBO보다 잠재 분포를 $\mathcal{N}(0, I)$에 더 가깝게 밀어붙이는데, 그 대가로 재구성의 선명도가 어느 정도 떨어집니다. 이는 아래 5절과 직접 관련이 있습니다. 잠재 공간이 표준정규분포에 가깝지 않으면 `z ~ N(0, I)`를 샘플링해 디코딩했을 때 이미지가 뒤죽박죽으로 나오는데, 디코더가 학습 중 그 영역의 점들을 한 번도 보지 못했기 때문입니다. 위 코드의 주석은 이 데이터셋에서 왜 3.0을 선택했는지 설명합니다. KL 항이 이미 픽셀 수로 나뉘어 있으므로, 가중치 3은 재구성 품질에 시각적으로 영향을 주지 않으면서도 잠재 분포를 눈에 띄게 조여줍니다.
- **혼합 정밀도**(`torch.amp.autocast`, `GradScaler`)는 CUDA에서 순전파를 float16으로 실행해 학습을 빠르게 하고, `GradScaler`는 역전파 전에 그래디언트를 재조정해 float16 언더플로를 방지합니다. 이는 VAE의 통계적 속성과는 무관한, 표준적인 학습 속도 최적화입니다. CPU에서나 `USE_CUDA`가 `False`일 때는 `autocast`와 스케일러가 아무 일도 하지 않고 학습이 float32로 진행됩니다.
- **`recon_loss`와 `kl_loss`를 따로 추적하는 것**(단순히 합만 보지 않는 것)이 다음 셀의 2축 그래프를 장식이 아니라 진단 도구로 만들어줍니다. KL만 따로 관찰하면 **사후붕괴**(posterior collapse)를 잡아낼 수 있습니다. 이는 학습 초반에 KL 항이 거의 0으로 떨어지는 실패 모드로, 인코더가 입력을 무시하도록 학습되어 모든 이미지에 대해 `mu ≈ 0, log_var ≈ 0`을 출력하게 됩니다. 이는 사전분포와는 정확히 일치하지만, 그 과정에서 재구성 경로를 망가뜨립니다. 처음 몇 에폭부터 `KL`이 0에 가까운 채로 `Recon`이 높은 값에서 정체된다면 바로 이 현상이 일어난 것입니다. `KL_WEIGHT`를 너무 높게 설정했거나, 학습 초반 재구성 신호가 너무 약한 것이 흔한 원인입니다.

학습 예산과 손실 가중치를 고정했으니, 데이터셋을 에폭 반복자와 미니배치 업데이트용 DataLoader에 연결하고 루프를 실행합니다.

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
학습이 끝나면 두 손실 성분을 에폭에 대해 그래프로 그립니다. 예상되는 형태는 다음과 같습니다. `Reconstruction Loss`는 인코더-디코더 쌍이 합성 도형을 압축하고 복원하는 법을 배우면서 꾸준히 감소하고, `KL Divergence`는 더 완만하게 감소하다가 작은 양수 값에서 안정되어야 하며, 0으로 붕괴해서는 안 됩니다. 처음 몇 에폭 안에 KL 곡선이 0에 도달한다면 위에서 설명한 사후붕괴의 경고 신호이지, 학습이 일찍 끝났다는 뜻이 아닙니다.

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
손실 곡선의 숫자는 모델이 무언가를 최적화하고 있다는 것만 확인해줄 뿐, 그것이 올바른 것을 최적화했는지는 시각적 확인으로만 알 수 있습니다. 원본 홀드아웃 이미지와 그 재구성을 비교해보세요. 온전한 인코더-디코더 쌍이라면 잠재 병목(32차원, 입력 픽셀 값 12,288개 대비)이 픽셀 단위의 완벽한 복원은 불가능하게 하더라도, 형태와 색상, 대략적인 위치는 보존해야 합니다.

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
## 5. 잠재 공간 시각화

**잠재 순회(latent traversal).** `z`의 모든 차원을 0으로 고정한 채 한 차원만 -2에서 2까지 훑으면서 각 값마다 디코딩하면, [VAE 직관 절](../lectures/lecture-02-encode-and-dynamics/01-encoding#vae-intuition-learning-to-compress-and-reconstruct)에서 설명한 **연속성(continuity)** 속성을 검증할 수 있습니다. 잠재 공간에서 가까운 점들은 시각적으로 비슷한 이미지로 디코딩되어야 하며, `z`가 한 차원을 따라 매끄럽게 움직일 때 불연속적인 도약이 없어야 합니다. 이는 KL 항의 역할입니다. KL 항이 사후분포를 하나의 공유되고 밀집된 영역($\mathcal{N}(0, I)$)으로 끌어당기지 않는다면, 인코더는 서로 다른 이미지들을 잠재 공간의 연결되지 않은 임의 형태의 영역에 흩어놓을 수 있고, 그 사이를 보간하면 디코딩된 결과가 뒤죽박죽이 될 수 있습니다.

각 차원을 훑을 때 하나의 인식 가능한 속성(위치, 크기, 색상)만 바뀌고 나머지는 대체로 고정되어 있는지 살펴보세요. 모든 차원이 정확히 하나의 독립적 요인만 제어하는 완벽한 **분리**(disentanglement)는 단순한 ELBO 목적함수만으로는 보장되지 않습니다(이는 $\beta$-VAE, FactorVAE 같은 후속 연구의 주제입니다). 그래서 이 단순한 데이터셋에서도 0~2번 차원에 걸쳐 부분적이거나 뒤섞인 효과가 나타나는 것은 버그가 아니라 예상된 결과입니다.

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
**사전분포 샘플링.** 이 셀은 인코더를 전혀 거치지 않고 `z ~ N(0, I)`를 직접 뽑아 디코딩합니다. 이는 위 순회보다 더 엄격한 검증입니다. 앞서 훑은 세 차원뿐 아니라, 학습된 잠재 분포 *전체*가 표준정규 사전분포와 일치하는지를 확인하기 때문입니다. 4절의 재구성은 괜찮아 보이는데 이 무작위 샘플들이 뒤죽박죽이라면, 인코더가 실제 이미지에 대한 유효한 압축은 학습했지만 모든 곳에서 $p(z) = \mathcal{N}(0, I)$와 실제로 일치시키지는 못했다는 뜻이며, 이는 KL 항이 재구성에 비해 너무 약했음을 의미합니다. `KL_WEIGHT`를 높여 재학습하는 것이 직접적인 해법이며, 그 대가로 재구성 선명도가 어느 정도 떨어집니다. 재구성과 사전분포 일치 사이의 이 긴장 관계는 3절에서 소개한 것과 같은 두 목적함수 간의 균형입니다.

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
## 체크포인트 저장

P02와 P03이 인코더 가중치를 재사용할 수 있도록 체크포인트를 `vae_encoder.pt`로 저장합니다. 잘된 학습이라면 낮은 재구성 손실, 시각적으로 일관된 재구성, 그리고 예측 가능한 방식으로 디코딩된 형태를 바꾸는 잠재 순회로 끝나야 합니다. P02는 이 체크포인트를 `model.encoder`를 통해 불러와 고정된(학습되지 않는) 특징 추출기로 사용합니다. 이 인코더는 각 원본 관측 `o_t`를 잠재 `z_t`로 매핑하고, 동역학 모델(GRU, MDN-RNN, RSSM)은 이 잠재값의 미래를 예측하도록 학습합니다. 이는 [Dreamer 파이프라인 요약](../lectures/lecture-02-encode-and-dynamics/03-dynamics-dreamer-series#the-encoders-role-as-a-bridge-in-dreamer)에서 설명한 인코더-동역학 인계와 정확히 같습니다.

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
