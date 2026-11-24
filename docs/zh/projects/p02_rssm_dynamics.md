---
title: P02 构建 RSSM 动力学模型
---

# P02: 构建 RSSM 动力学模型

在合成像素轨迹上训练并对比 GRU、MDN-RNN 和 RSSM 三种动力学模型。本 notebook 的重点在于对比，而非追求排行榜成绩：GRU 是最简单的基线，MDN-RNN 引入预测不确定性，RSSM 则通过潜在随机状态实现世界模型风格的 rollout。

**前置条件**：P01 生成的 `vae_encoder.pt` 权重文件（如有）。否则 notebook 会使用随机初始化的编码器继续运行，但此时 rollout 对比结果仅供参考，意义有限。本 notebook 将训练动力学模型，并将 RSSM 保存为 `rssm.pt`，供 P03 和 P04 使用。

> Notebook 源文件: [p02_rssm_dynamics.ipynb](https://github.com/datawhalechina/learn-world-model/blob/main/docs/zh/projects/p02_rssm_dynamics.ipynb)

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
## 1. 环境准备

构建冻结编码器、合成轨迹数据集及潜在数据集。

```python
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
import matplotlib as mpl
from matplotlib import font_manager

# 让 Colab 和新环境优先使用支持中文的字体，避免标题和坐标轴显示成方框。
def _configure_cjk_font():
    preferred = [
        "Noto Sans CJK SC",
        "Noto Sans SC",
        "Source Han Sans SC",
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "WenQuanYi Micro Hei",
    ]
    for family in preferred:
        try:
            font_manager.findfont(family, fallback_to_default=False)
            mpl.rcParams["font.family"] = "sans-serif"
            mpl.rcParams["font.sans-serif"] = [family] + [f for f in mpl.rcParams.get("font.sans-serif", []) if f != family]
            mpl.rcParams["axes.unicode_minus"] = False
            return family
        except Exception:
            pass

    font_path = Path.home() / ".cache" / "notebook-fonts" / "NotoSansCJKsc-Regular.otf"
    if not font_path.exists():
        try:
            import urllib.request
            font_path.parent.mkdir(parents=True, exist_ok=True)
            url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf"
            urllib.request.urlretrieve(url, font_path)
        except Exception:
            font_path = None

    if font_path and font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        family = font_manager.FontProperties(fname=str(font_path)).get_name()
        mpl.rcParams["font.family"] = "sans-serif"
        mpl.rcParams["font.sans-serif"] = [family] + [f for f in preferred if f != family]
        mpl.rcParams["axes.unicode_minus"] = False
        return family

    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False
    return None

_CJK_FONT = _configure_cjk_font()

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
        optimizer.step()

LATENT_DIM = 32
HIDDEN_DIM = 128
ACTION_DIM = 1
N_TRAJ     = 200
T_STEPS    = 20
IMG_SIZE   = 64

print(f'设备: {DEVICE}')
if USE_TPU:
    print('TPU 后端    : torch_xla')
print(f'LATENT_DIM={LATENT_DIM}, HIDDEN_DIM={HIDDEN_DIM}')
```
环境准备完成后，复用 P01 的编码器和解码器，让动力学模型和整个 world model 栈处在同一个潜在空间里。`_load_vae_checkpoint` 处理了一个小的命名差异：本 notebook 的 `VAEEncoder` 类用 `fc_log_var`，P01 的 `Encoder` 用 `fc_var`，但如果 `vae_encoder.pt` 存在，底层权重是完全相同的。两个网络都被设为 `.eval()`，每个参数都设置了 `requires_grad_(False)`：编码器和解码器在本 notebook 剩余部分保持冻结，只用来在像素和潜在向量之间转换，不会被下面的动力学训练更新。

```python
# 与 P01 兼容的 VAE 编码器和解码器

class VAEEncoder(nn.Module):
    """将 64x64 RGB 帧编码为潜在均值和对数方差。"""
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
        """从 q(z|x) 进行重参数化采样。"""
        mu, logvar = self.forward(x)
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)


class VAEDecoder(nn.Module):
    """镜像解码器：latent_dim -> 64x64 RGB，值域 [0,1]。"""
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
    raise KeyError(f'未识别的权重文件格式: {list(ckpt.keys())[:10]}')

ckpt_path = Path('../../public/vae_encoder.pt')
try:
    _load_vae_checkpoint(ckpt_path)
    print(f'已从 {ckpt_path} 加载 VAE 权重')
except Exception as e:
    print(f'无法从 {ckpt_path} 加载 VAE 权重文件（{e}），使用随机初始化。')

VAE_CHECKPOINT_PATH = ckpt_path

encoder.eval()
decoder.eval()
for p in list(encoder.parameters()) + list(decoder.parameters()):
    p.requires_grad_(False)

print(f'编码器参数量: {sum(p.numel() for p in encoder.parameters()):,}')
print(f'解码器参数量: {sum(p.numel() for p in decoder.parameters()):,}')
```
接下来生成合成轨迹，这些 rollouts 将成为序列模型的监督信号。每条轨迹是一个彩色矩形以固定速度 `(vx, vy)` 在画面中漂移，每步再叠加一点噪声，同时每一步都带一个二元 `action`（目前只是装饰性的：它被记录下来并喂给下面每个模型，但矩形的运动并不依赖它）。这样设计是为了让对比聚焦在 GRU、MDN-RNN、RSSM 三者对一个本身简单、近乎确定性的动力学如何建模*不确定性*上，而不是聚焦在它们能否学出一条复杂的、以动作为条件的规律。矩形的颜色范围（0.3 到 1）和尺寸（16 到 32 像素）与 P01 训练 VAE 时用的图形保持一致。如果方块更暗或更小，冻结的编码器会把大部分帧编码成接近空白的潜在向量，下面每个模型都会在 VAE 自己都还原不出来的帧上被评判。

```python
# 生成合成轨迹数据。

def make_trajectory(T=20, img_size=64, seed=None):
    rng = np.random.RandomState(seed)
    color = rng.uniform(0.3, 1.0, size=3).astype(np.float32)
    w  = rng.randint(16, 33)
    h  = rng.randint(16, 33)
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


print('正在生成 200 条合成轨迹...')
trajectories = [make_trajectory(T=T_STEPS, img_size=IMG_SIZE, seed=i) for i in range(N_TRAJ)]
print(f"每条轨迹的观测形状:     {trajectories[0]['obs'].shape}")
print(f"每条轨迹的动作形状: {trajectories[0]['actions'].shape}")
```
轨迹数据构建好后，把每个观测编码成 z，这样 notebook 剩余部分就能在潜在序列上学习动力学，而不是在原始像素上。`encoder(obs)` 返回后验的均值和对数方差，这里只保留均值 `mu`，而不使用 P01 `reparameterize` 里的*采样*路径 `encoder.encode(obs)`（`mu + std * epsilon`）。在这些帧上有不少潜在维度并不活跃，它们的后验接近 N(0, 1)；如果采样，每一步都会叠加一份任何动力学模型都预测不了的独立噪声，下面所有的误差数字都会被这份噪声主导。用均值可以得到干净、确定的目标，这也是把冻结的 VAE 当作观测编码器时的常见做法。`z_seq` 仍然是 RSSM 自己的后验网络要吸收的观测，下面 RSSM 一节会讲到。

```python
# 将观测编码为潜在序列 z [N, T, 32]

print('正在编码观测...')
latent_list = []
with torch.no_grad():
    for traj in trajectories:
        obs = traj['obs'].to(DEVICE)      # [T,3,H,W]
        mu, _ = encoder(obs)              # [T, latent_dim]，后验均值
        z   = mu                          # 用均值而不是 encode()：采样噪声会淹没动力学信号
        latent_list.append(z.cpu())

Z_all = torch.stack(latent_list, dim=0)                                     # [N,T,32]
A_all = torch.stack([t['actions'] for t in trajectories], dim=0)            # [N,T]

# 训练/测试划分
N_TRAIN = 180
Z_train, A_train = Z_all[:N_TRAIN].to(DEVICE), A_all[:N_TRAIN].to(DEVICE)
Z_test,  A_test  = Z_all[N_TRAIN:].to(DEVICE), A_all[N_TRAIN:].to(DEVICE)

print(f'Z_all 形状: {Z_all.shape}  (N, T, latent_dim)')
print(f'训练集: {N_TRAIN} 条轨迹 | 测试集: {N_TRAJ - N_TRAIN} 条轨迹')
```
## 2. 动力学模型

从最简单的 GRU 基线出发，依次加入混合密度输出和 RSSM 潜在状态。下面三个类都实现了[潜在动力学](../lectures/lecture-02-encode-and-dynamics/02-dynamics)里讲的模型：相同的输入 `(z_t, a_t)`，相同的目标（预测 `z_{t+1}`），但处理不确定性的方式各不相同。把三个 `forward` 方法放在一起对照阅读，是看清讲义里只用公式描述的差异在代码层面究竟是什么样子的最快方式。

```python
class GRUDynamics(nn.Module):
    """GRU 动力学模型，hidden_dim=128，输入为 (latent_dim+action_dim)，输出为 latent_dim。
    接受 (z_t, a_t) 作为输入，经 GRU 更新后预测 z_{t+1}。
    """
    def __init__(self, latent_dim=32, action_dim=1, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru    = nn.GRUCell(latent_dim + action_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_seq, a_seq):
        """z_seq [B,T,D], a_seq [B,T] -> pred_z [B,T-1,D]，对应步骤 1..T。"""
        B, T, _ = z_seq.shape
        h = torch.zeros(B, self.hidden_dim, device=z_seq.device)
        preds = []
        for t in range(T - 1):
            inp = torch.cat([z_seq[:, t], a_seq[:, t].unsqueeze(-1)], dim=-1)
            h   = self.gru(inp, h)
            preds.append(self.output(h))
        return torch.stack(preds, dim=1)  # [B, T-1, D]

    def rollout(self, z_ctx, a_seq):
        """先用观测到的潜在向量预热隐藏状态，再开环 rollout。
        z_ctx [1, C, D]：观测 z_0..z_{C-1}（传 [1, D] 即 C=1）。
        a_seq：从 a_0 开始的动作；最后 len(a_seq) - C + 1 个驱动开环步。
        返回 [1, steps+1, D]：最后一个观测，接着是预测。
        """
        z_ctx = z_ctx.reshape(1, -1, z_ctx.shape[-1])
        C = z_ctx.shape[1]
        h = torch.zeros(1, self.hidden_dim, device=z_ctx.device)
        for t in range(C - 1):  # 在上下文上 teacher forcing
            h = self.gru(torch.cat([z_ctx[:, t], a_seq[t].view(1, 1)], dim=-1), h)
        z  = z_ctx[:, -1]
        zs = [z]
        for a in a_seq[C - 1:]:
            inp = torch.cat([z, a.view(1, 1)], dim=-1)
            h   = self.gru(inp, h)
            z   = self.output(h)
            zs.append(z)
        return torch.stack(zs, dim=1)  # [1, steps+1, D]


gru_model = GRUDynamics(LATENT_DIM, ACTION_DIM, HIDDEN_DIM).to(DEVICE)
print(f'GRUDynamics 参数量: {sum(p.numel() for p in gru_model.parameters()):,}')
```
有了普通 GRU 基线以后，再补上 MDN-RNN，用多峰分布去描述单点预测无法捕捉的不确定性。`GRUDynamics.forward` 就是讲义里 $\mathbf{z}_{t+1} = \text{GRU}(\mathbf{z}_t, \mathbf{a}_t)$ 的直接代码形式：`torch.cat([z_seq[:, t], a_seq[:, t].unsqueeze(-1)], dim=-1)` 把 `(z_t, a_t)` 拼接成输入，`self.gru(inp, h)` 就是门控更新本身（PyTorch 内置的 `GRUCell`，内部实现了讲义深挖框里说的重置门/更新门机制），`self.output(h)` 是一个线性读出层，把隐状态变成预测的 `z_{t+1}`。这个类里没有任何采样步骤：给定同样的 `(z_t, a_t, h_{t-1})`，GRU 永远输出同样的 `z_{t+1}`，这正是讲义里说的「确定性输出」这一局限。

`MDNRNN` 把这个单一的线性读出层换成了 `mdn_head`，一个线性层，对每个混合分量 $k$ 输出三组结果：`logits`（softmax 之前的值，会变成混合权重 $\pi_k$）、`mu`（分量均值 $\mu_k$）、`log_s`（每个分量的对数方差，与 P01 的 `log_var` 是同一套写法）。`_split` 把这三组结果从 head 的扁平输出张量里切分出来。`mdn_loss` 就是讲义里 $p(\mathbf{z}_{t+1} \mid \mathbf{z}_t, \mathbf{a}_t) = \sum_k \pi_k \mathcal{N}(\mathbf{z}_{t+1}; \mu_k, \sigma_k^2)$ 这个高斯混合分布的负对数似然：`log_p` 计算每个分量的对数密度（高斯对数似然公式，对 32 个潜在维度求和，因为它们被当作相互独立），`log_pi` 用 log-softmax 把原始 logits 归一化成 $\log \pi_k$，`torch.logsumexp(log_pi + log_p, dim=-1)` 把它们组合成 $\log \sum_k \pi_k \mathcal{N}(\ldots)$，之所以在对数空间里计算是为了数值稳定，而不是直接取指数再求和（那样对较大的对数密度容易溢出）。取负再求平均，就是要最小化的损失。

在 rollout 阶段，`MDNRNN.rollout` 并不从混合分布里采样，而是确定性地选出权重最高的分量（`lg[0].argmax()`），使用它的均值。这意味着 MDN-RNN 的*训练*目标捕捉到了多模态，但本 notebook 后面展示的 *rollout* 并没有真正展现这种多模态，因为它总是沿着单一的最可能分支走。

```python
class MDNRNN(nn.Module):
    """GRU + MDN 输出头，对 z_{t+1} 预测由 3 个高斯分量组成的混合分布。
    MDN 损失：混合分布的负对数似然。
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
        """返回 (logits, mu, log_sigma)，各自形状为 [B, T-1, ...]，用于计算 MDN 损失。"""
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
        """混合分布的负对数似然损失。
        logits [B,T,K], mu [B,T,K,D], log_sigma [B,T,K,D], target [B,T,D]。
        """
        B, T, K, D = mu.shape
        tgt = target.unsqueeze(2).expand_as(mu)  # [B,T,K,D]
        sigma = log_sigma.exp().clamp(min=1e-4)
        log_p = -0.5 * (((tgt - mu) / sigma) ** 2 + 2 * log_sigma
                        + math.log(2 * math.pi))
        log_p  = log_p.sum(-1)                          # [B,T,K]
        log_pi = F.log_softmax(logits, dim=-1)          # [B,T,K]
        return -torch.logsumexp(log_pi + log_p, dim=-1).mean()

    def rollout(self, z_ctx, a_seq):
        """先用观测到的潜在向量预热隐藏状态，再用概率最高的混合分量开环 rollout。
        z_ctx [1, C, D]：观测 z_0..z_{C-1}（传 [1, D] 即 C=1）。
        a_seq：从 a_0 开始的动作；最后 len(a_seq) - C + 1 个驱动开环步。
        返回 [1, steps+1, D]：最后一个观测，接着是预测。
        """
        z_ctx = z_ctx.reshape(1, -1, z_ctx.shape[-1])
        C = z_ctx.shape[1]
        h = torch.zeros(1, self.hidden_dim, device=z_ctx.device)
        for t in range(C - 1):  # 在上下文上 teacher forcing
            h = self.gru(torch.cat([z_ctx[:, t], a_seq[t].view(1, 1)], dim=-1), h)
        z  = z_ctx[:, -1]
        zs = [z]
        for a in a_seq[C - 1:]:
            inp = torch.cat([z, a.view(1, 1)], dim=-1)
            h   = self.gru(inp, h)
            lg, mu, _ = self._split(self.mdn_head(h))
            k = lg[0].argmax().item()
            z = mu[0, k].unsqueeze(0)   # [1, D]
            zs.append(z)
        return torch.stack(zs, dim=1)


mdn_model = MDNRNN(LATENT_DIM, ACTION_DIM, HIDDEN_DIM, n_mix=3).to(DEVICE)
print(f'MDNRNN 参数量: {sum(p.numel() for p in mdn_model.parameters()):,}')
```
两个基线都定义完后，接着实现 RSSM，作为要与它们对比的结构化 latent-state 模型。`RSSM.forward` 在一个 `for t in range(T)` 循环里实现了[潜在动力学讲义 RSSM 一节](../lectures/lecture-02-encode-and-dynamics/02-dynamics#rssm-分离确定性与随机性)里的三个核心方程：

- `h = self.gru(inp, h)` 是确定性更新 $\mathbf{h}_t = f_\phi(\mathbf{h}_{t-1}, \mathbf{z}_{t-1}, \mathbf{a}_{t-1})$，其中 `inp` 里的 `z` 是上一步*采样得到*的潜在状态，从上一轮循环延续下来（在 `t=0` 时初始化为零）。
- `pr = self.prior_net(h)`，切分成 `mu_pr, lv_pr`，就是先验 $\mathbf{z}_t \sim p_\phi(\mathbf{z}_t \mid \mathbf{h}_t)$：它只依赖 `h`，不能访问当前观测，与讲义里「先验是用于纯想象的分支」这一描述一致。
- `po = self.post_net(torch.cat([h, z_seq[:, t]], dim=-1))`，切分成 `mu_po, lv_po`，就是后验 $\mathbf{z}_t \sim q_\phi(\mathbf{z}_t \mid \mathbf{h}_t, \mathbf{o}_t)$：注意它额外接收了 `z_seq[:, t]`，也就是这条轨迹在这一步的真实编码观测，这正是讲义里说的「用真实观测修正」。`self._rsample(mu_po, lv_po)` 采样出的 `z` 值，既延续了递归（作为下一步的 `inp`），也会在下面的 KL 项里与先验做对比。

`forward` 里的 KL 项，`kl = 0.5 * (lv_pr - lv_po + (lv_po.exp() + (mu_po - mu_pr)**2) / lv_pr.exp().clamp(min=1e-4) - 1)`，是两个一般（不一定是标准正态）对角高斯分布之间的解析 KL 散度 $D_{KL}(q_\phi(\mathbf{z}_t \mid \mathbf{h}_t, \mathbf{o}_t) \,\|\, p_\phi(\mathbf{z}_t \mid \mathbf{h}_t))$：这是 P01 里那个高斯 KL 公式的更一般形式，从「后验 vs. 标准正态先验」推广到「后验 vs. 一个*可学习的*先验网络」，这正是 RSSM 的 KL 项与朴素 VAE 的区别所在。最小化这个 KL，会把先验（它接触不到 $o_t$）拉向匹配后验（它能接触到），这样在 rollout 阶段只有先验可用时，它已经学会了逼近后验原本会给出的答案。

`forward` 在把这个 KL 加到重建损失之前，先乘上 `kl_weight=0.1`。在这种近乎确定性的数据上，如果权重是 1，降低损失最省事的办法是让后验几乎忽略 `z_seq[:, t]`、直接照抄先验，也就是后验塌缩（posterior collapse）；这时模型预测出来的差不多只是一个平均潜在向量。权重小一些，观测信息就能留在 `z` 里。Dreamer 出于同样的原因使用了 KL balancing 和 free bits 这类相关技巧。`kl_weight` 只是普通属性，不是参数，所以保存的 state dict 仍与 P03、P04 兼容。

`RSSM.rollout` 先在观测到的上下文帧 `z_ctx` 上跑一遍与 `forward` 相同的过滤循环：从零状态、零动作出发，每个观测步都更新 `h` 并取 `post_net` 的后验均值，因此起始状态已经包含了位置和速度信息。上下文结束后只使用 `prior_net`，不再使用 `post_net`，并且取先验的均值（`mu`）而不是采样，这与讲义里「想象就是只用先验、不涉及真实观测的 rollout」这一描述完全一致。有一个细节对下面的比较很关键：RSSM 内部的 `z` 是模型自己的随机状态，并不在 VAE 的潜在空间里，训练时唯一和观测潜在向量对齐的是 `self.recon(z)`。所以 `rollout` 每一步返回的都是 `self.recon(z)`，这样它的输出才和 GRU、MDN-RNN 的预测处在同一空间，可以被解码，也可以和真实的 `z_seq` 比较。如果把观测到的潜在向量当成 RSSM 状态直接喂进 GRU，或者直接返回状态 `z`，都等于把两套不同的坐标系当成一套，RSSM 的结果就会显得远比实际更差。

```python
class RSSM(nn.Module):
    """循环状态空间模型（RSSM）。
    确定性路径：h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})
    随机先验：   z_t ~ N(mu_prior(h_t), sigma_prior(h_t))
    随机后验：   z_t ~ N(mu_post(h_t, o_t), sigma_post(h_t, o_t))
    训练目标：   ELBO = 重建损失 + kl_weight * KL(后验 || 先验)
    hidden_dim=128, latent_dim=32
    """
    def __init__(self, latent_dim=32, action_dim=1, hidden_dim=128, kl_weight=0.1):
        super().__init__()
        self.kl_weight  = kl_weight  # 小于 1，防止后验塌缩到先验
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # 确定性循环
        self.gru = nn.GRUCell(latent_dim + action_dim, hidden_dim)

        # 先验网络：h_t -> (mu, logvar)
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )

        # 后验网络：(h_t, o_t) -> (mu, logvar)，o_t 为编码后的观测
        self.post_net = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )

        # 重建头：z -> 预测 z（潜在重建目标）
        self.recon = nn.Linear(latent_dim, latent_dim)

    def _rsample(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def forward(self, z_seq, a_seq):
        """计算整条轨迹的 ELBO 损失。
        z_seq [B,T,D], a_seq [B,T]。
        返回标量 ELBO 损失。
        """
        B, T, D = z_seq.shape
        h = torch.zeros(B, self.hidden_dim, device=z_seq.device)
        z = torch.zeros(B, D, device=z_seq.device)
        recon_loss = z_seq.new_zeros(())
        kl_loss    = z_seq.new_zeros(())
        for t in range(T):
            # a_{t-1}：t=0 时用零动作，与类注释及 rollout 保持一致
            a_prev = a_seq[:, t - 1] if t > 0 else torch.zeros_like(a_seq[:, 0])
            inp = torch.cat([z, a_prev.unsqueeze(-1)], dim=-1)
            h   = self.gru(inp, h)

            # 先验
            pr   = self.prior_net(h)
            mu_pr, lv_pr = pr.chunk(2, dim=-1)

            # 以观测潜在向量 o_t = z_seq[:, t] 为条件的后验
            po   = self.post_net(torch.cat([h, z_seq[:, t]], dim=-1))
            mu_po, lv_po = po.chunk(2, dim=-1)

            z = self._rsample(mu_po, lv_po)

            # 重建损失：预测观测潜在向量
            recon_loss = recon_loss + F.mse_loss(self.recon(z), z_seq[:, t])

            # KL(后验 || 先验)
            kl = 0.5 * (
                lv_pr - lv_po
                + (lv_po.exp() + (mu_po - mu_pr) ** 2) / lv_pr.exp().clamp(min=1e-4)
                - 1
            )
            kl_loss = kl_loss + kl.mean()

        return (recon_loss + self.kl_weight * kl_loss) / T

    def rollout(self, z_ctx, a_seq):
        """先用后验过滤观测到的上下文，之后每一步只用先验（不观测）。
        z_ctx [1, C, D]、a_seq 与 GRUDynamics.rollout 相同。
        返回预测的观测潜在向量 self.recon(z)，形状 [1, steps+1, D]，
        与 z_seq 同一空间（z 是 RSSM 自己的状态，不是 VAE 潜在向量）。
        """
        z_ctx = z_ctx.reshape(1, -1, z_ctx.shape[-1])
        C = z_ctx.shape[1]
        h = torch.zeros(1, self.hidden_dim, device=z_ctx.device)
        z = torch.zeros_like(z_ctx[:, 0])
        a = torch.zeros(1, 1, device=z_ctx.device)  # 与 forward 相同，从零状态、零动作出发
        for t in range(C):  # 用后验均值过滤上下文
            h = self.gru(torch.cat([z, a], dim=-1), h)
            z, _ = self.post_net(torch.cat([h, z_ctx[:, t]], dim=-1)).chunk(2, dim=-1)
            a = a_seq[t].view(1, 1)
        zs = [self.recon(z)]
        for a in a_seq[C - 1:]:
            inp = torch.cat([z, a.view(1, 1)], dim=-1)
            h   = self.gru(inp, h)
            pr  = self.prior_net(h)
            mu, _ = pr.chunk(2, dim=-1)
            z  = mu  # 确定性 rollout 使用先验均值
            zs.append(self.recon(z))
        return torch.stack(zs, dim=1)


rssm_model = RSSM(LATENT_DIM, ACTION_DIM, HIDDEN_DIM).to(DEVICE)
print(f'RSSM 参数量: {sum(p.numel() for p in rssm_model.parameters()):,}')
```
**核对讲义里的手算例子**：讲义用一个去掉门控的简化线性版本手算了一步 RSSM 转移，得到 $\mathbf{h}_t = [0.60, 0.20]$、$\mu_{\text{pr}} = 0.40$、$z_t = 0.50$。下面用同样的数字和权重跑一遍；注意这里刻意没有用上面的 `RSSM` 类（它内部是真正的 `GRUCell` 加多层先验网络，数值和这个简化版对不上），这个 cell 只是为了验证公式本身的推导没有算错。

```python
# 简化版单步转移，对应讲义里的手算例子（不是真正的 RSSM 类）。
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
## 3. 训练

三个模型均在 180 条训练轨迹上使用 Adam（lr=1e-3）训练 50 轮。
各模型的损失函数如下：
- GRU：预测 z 与真实 z 之间的 MSE
- MDN-RNN：高斯混合分布的负对数似然
- RSSM：ELBO = z 的 MSE 重建损失 + 0.1 × KL 散度

```python
EPOCHS = 50
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

print(f'训练 3 个模型，共 {EPOCHS} 轮...')
for epoch in range(1, EPOCHS + 1):
    lg = run_epoch(gru_model,  opt_gru,  Z_train, A_train, gru_loss)
    lm = run_epoch(mdn_model,  opt_mdn,  Z_train, A_train, mdn_loss_fn)
    lr = run_epoch(rssm_model, opt_rssm, Z_train, A_train, rssm_loss)
    losses_gru.append(lg)
    losses_mdn.append(lm)
    losses_rssm.append(lr)
    if epoch % 5 == 0 or epoch == 1:
        print(f'第 {epoch:3d} 轮 | GRU: {lg:.4f} | MDN-RNN: {lm:.4f} | RSSM: {lr:.4f}')

print('训练完成。')
```
`run_epoch` 是三个模型共用的训练循环：标准的 minibatch SGD，加上 `clip_grad_norm_(model.parameters(), 1.0)` 防止梯度爆炸，循环神经网络即使在这里只有 20 步的序列长度下也格外容易出现这个问题。三个 `*_loss` 包装函数（`gru_loss`、`mdn_loss_fn`、`rssm_loss`）各自把对应模型自己的损失计算适配成 `run_epoch` 期望的统一签名 `loss_fn(model, zb, ab)`，这样同一个训练循环就能驱动全部三个模型，不需要重复逻辑。

注意 `RSSM.forward`（被 `rssm_loss` 调用）重建的是*潜在*的 `z_seq[:, t]`，通过 `self.recon(z)`，而不是原始像素。这是本 notebook 特有的简化：讲义里的 RSSM 直接重建观测，$o_t \sim p(o_t \mid h_t, z_t)$，但因为这里的 `z_seq` 本身就是冻结的 P01 编码器的输出，重建它相当于隔了一层去重建像素的代理目标。三个损失并不在同一个数值量级上（MSE、混合分布负对数似然、ELBO 的单位和典型量级都不同），这正是下一个 cell 的图表在比较形状之前先对每条曲线做归一化的原因，而不是直接比较原始数值。

模型家族定义完成后，设置训练轮数，把三个动力学版本放在一起训练。

```python
# 绘制归一化训练损失曲线。

def norm01(curve):
    a = np.array(curve, dtype=np.float64)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo + 1e-9)

fig, ax = plt.subplots(figsize=(8, 4))
xs = np.arange(1, EPOCHS + 1)
ax.plot(xs, norm01(losses_gru),  label='GRU (MSE)',     color='tab:blue')
ax.plot(xs, norm01(losses_mdn),  label='MDN-RNN (NLL)', color='tab:orange')
ax.plot(xs, norm01(losses_rssm), label='RSSM (ELBO)',   color='tab:green')
ax.set_xlabel('轮次')
ax.set_ylabel('归一化损失 [0, 1]')
ax.set_title('训练损失曲线（归一化以便对比）')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```
## 4. Rollout 对比

每个模型先看测试轨迹的前 `CONTEXT = 5` 帧，然后在不观测未来帧的情况下向前推演 10 步。预测得到的潜在向量被解码回像素空间，以网格形式展示（4 行 x 11 列，第 0 步是最后一个观测帧）。

```python
ROLLOUT_STEPS = 10

CONTEXT       = 5   # 每个模型先看 5 帧真实观测，再开始想象

# 选运动幅度最大的测试轨迹，让 rollout 的差异看得出来。
traj_idx = int((Z_test[:, 1:] - Z_test[:, :-1]).pow(2).sum(dim=(1, 2)).argmax())
z_traj = Z_test[traj_idx]    # [T, D]
a_traj = A_test[traj_idx]    # [T]
z_ctx  = z_traj[:CONTEXT].unsqueeze(0)              # [1, C, D]
a_seq  = a_traj[:CONTEXT - 1 + ROLLOUT_STEPS]       # [C-1+10]

gru_model.eval()
mdn_model.eval()
rssm_model.eval()

with torch.no_grad():
    zs_gru  = gru_model.rollout(z_ctx, a_seq).squeeze(0)    # [11, D]
    zs_mdn  = mdn_model.rollout(z_ctx, a_seq).squeeze(0)
    zs_rssm = rssm_model.rollout(z_ctx, a_seq).squeeze(0)

    def decode_seq(zs):
        """zs [S, D] -> numpy [S, H, W, 3]，值域 [0,1]。"""
        imgs = decoder(zs.to(DEVICE))  # [S, 3, H, W]
        return imgs.cpu().permute(0, 2, 3, 1).numpy()

    imgs_gru  = decode_seq(zs_gru)
    imgs_mdn  = decode_seq(zs_mdn)
    imgs_rssm = decode_seq(zs_rssm)
    imgs_gt   = decode_seq(z_traj[CONTEXT - 1:CONTEXT + ROLLOUT_STEPS])   # 真实帧

print(f'解码后的 rollout 形状（GRU）: {imgs_gru.shape}  (steps+1, H, W, 3)')
```
训练完成后，进入 rollout 评估，衡量预测越往未来走误差是如何累积的。`ROLLOUT_STEPS = 10` 步开环预测意味着每个模型拿到前 `CONTEXT` 帧真实观测和真实的动作序列，之后每一步都必须基于自己前面的预测继续往下预测，不再有真实观测来修正。上下文很重要：单独一帧只能看出方块在哪，看不出它移动多快；如果只从一帧和空的隐藏状态开始，模型只能去猜速度，比较结果反映的就是这个猜测，而不是学到的动力学。演示选用潜在空间里运动幅度最大的测试轨迹，因为贴在墙边不动的方块在所有模型下看起来都一样。这正是[L03 STORM 一节](../lectures/lecture-03-architecture-patterns/01-architectures-rnn-transformer-diffusion#storm-的核心改进-单-token-随机潜变量)里说的**教师强制（teacher forcing）差距**，[L04](../lectures/lecture-04-evaluation-by-model/00-diagnostic-framework)对此有正式的诊断方法：训练时每一步用的都是真实的 `z_seq`，但这里的评估强迫每个模型累积自己的误差，这是检验学到的动力学是否真的能用于基于想象的规划的更难、也更贴近现实的测试。

```python
# 图像网格：真实帧、GRU、MDN-RNN、RSSM。
N_COLS      = ROLLOUT_STEPS + 1
row_labels  = ['真实帧', 'GRU', 'MDN-RNN', 'RSSM']
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
            ax.set_title(f'步骤 {c}', fontsize=9, pad=8)
from IPython.display import display
fig.suptitle('10 步想象 Rollout 与真实帧对比', fontsize=12, y=1.05)
display(fig)
plt.close(fig)
```
图像网格给出直观的视觉印象，下一个代码块把它转换成逐步的像素 MSE 曲线，提供定量视角。不要指望 RSSM 在这里处处胜出。这份数据近乎确定：方块以恒定速度运动，只有少量噪声，动作也不影响运动，所以随机潜变量几乎没有随机性可以建模，确定性的 GRU 本来就很合适。RSSM 的优势在于它的先验被训练成贴近后验会做出的推断，因此在长 rollout 中误差往往增长得更慢。在一次参考运行中，RSSM 那一行 10 步都保持了方块的形状和移动方向，GRU 那一行大约 5 步后开始漂移、变模糊，而只跟随概率最高混合分量的 MDN-RNN 则让方块逐渐缩小、融进背景。这只是一条轨迹，下一节会在整个测试集上检验这个规律。

```python
# 各步像素 MSE（相对于真实帧）。

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
ax.set_xlabel('Rollout 步骤')
ax.set_ylabel('像素 MSE')
ax.set_title('各步像素 MSE（相对于真实帧）')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```
## 5. 各步长的预测误差（1 到 10 步）

在保留的测试轨迹上，为三个模型分别计算预测步长 1 到 10 的平均潜在空间 MSE。`horizon_mse` 在每条测试轨迹上滑动窗口（`for t0 in range(context - 1, T - max_h)`），先给模型截至 `t0` 的 `context` 帧真实观测，再从那里向前滚动 `max_h` 步，分别在每个步长上累积与真实潜在状态之间的平方误差。两条平凡基线给这些数字提供参照：`copy_rollout` 重复最后一个观测到的潜在向量，`mean_rollout` 始终预测训练集均值。如果学到的模型连「复制上一帧」都赢不了，说明它没有学到任何运动。这样得到的**误差-步长曲线**，正是 L04 诊断框架里长时程 rollout 这一层的标准证据：曲线保持低且平坦，说明模型的误差没有累积；曲线随步长陡峭上升，说明每步的小误差在滚雪球，这与[L04](../lectures/lecture-04-evaluation-by-model/05-diffusion-drift#时程漂移-所有世界模型的共同失效模式)里记录的、所有架构共有的**时程漂移**失效模式是同一回事。

```python
MAX_H = 10


def horizon_mse(model_rollout_fn, Z, A, max_h=5, context=CONTEXT):
    """计算测试集上步长 1..max_h 的平均潜在 MSE。
    每个窗口先给模型 context 帧真实观测，再从最后一帧开环预测。
    model_rollout_fn(z_ctx, a_seq) -> [1, steps+1, D]
    """
    N, T, D = Z.shape
    errs   = np.zeros(max_h)
    counts = np.zeros(max_h)
    with torch.no_grad():
        for i in range(N):
            for t0 in range(context - 1, T - max_h):
                z_ctx = Z[i, t0 - context + 1:t0 + 1].unsqueeze(0)   # [1, C, D]
                a_s   = A[i, t0 - context + 1:t0 + max_h]            # [C-1+max_h]
                zs    = model_rollout_fn(z_ctx, a_s).squeeze(0)      # [max_h+1, D]
                for h in range(1, max_h + 1):
                    errs[h - 1]   += F.mse_loss(zs[h], Z[i, t0 + h]).item()
                    counts[h - 1] += 1
    return errs / counts


# 平凡基线：复制上一步潜在向量，或始终预测训练集均值。
Z_MEAN = Z_train.reshape(-1, LATENT_DIM).mean(0, keepdim=True)  # [1, D]

def copy_rollout(z_ctx, a_seq):
    steps = len(a_seq) - z_ctx.shape[1] + 1
    return z_ctx[:, -1:].expand(1, steps + 1, -1)

def mean_rollout(z_ctx, a_seq):
    steps = len(a_seq) - z_ctx.shape[1] + 1
    return torch.cat([z_ctx[:, -1], Z_MEAN.expand(steps, -1)], dim=0).unsqueeze(0)


print('正在计算测试集上的步长误差...')
gru_model.eval(); mdn_model.eval(); rssm_model.eval()

err_gru  = horizon_mse(gru_model.rollout,  Z_test, A_test, MAX_H)
err_mdn  = horizon_mse(mdn_model.rollout,  Z_test, A_test, MAX_H)
err_rssm = horizon_mse(rssm_model.rollout, Z_test, A_test, MAX_H)
err_copy = horizon_mse(copy_rollout,      Z_test, A_test, MAX_H)
err_mean = horizon_mse(mean_rollout,      Z_test, A_test, MAX_H)

horizons = list(range(1, MAX_H + 1))
print('\n各步长潜在 MSE:')
print(f'{"步长":>8}  {"GRU":>10}  {"MDN-RNN":>10}  {"RSSM":>10}  {"复制上一步":>10}  {"训练集均值":>10}')
for h, (eg, em, er, ec, eb) in enumerate(zip(err_gru, err_mdn, err_rssm, err_copy, err_mean), start=1):
    print(f'{h:>8}  {eg:>10.4f}  {em:>10.4f}  {er:>10.4f}  {ec:>10.4f}  {eb:>10.4f}')
```
预测步数范围已经固定，下面把逐步误差画出来，方便直接比较短步长和长步长的表现。训练集均值基线只在上面打印，没有画进图里，因为它的误差大好几倍，会把其他曲线压平。刻意把这张图和上面定性的 rollout 网格放在一起看：一个模型可以在步长 1 上误差最低，后面却被反超，只有多步长曲线才能揭示这一点，单一的聚合数字做不到。在一次参考运行中，GRU 在所有步长上都赢过「复制上一帧」基线，RSSM 从步长 2 起赢过它，MDN-RNN 要到步长 4 才赢过它。GRU 在步长 4 以内误差最低，步长 5 两者持平，步长 6 到 10 则是 RSSM 更低，因为它的误差增长得更慢。MDN-RNN 在三者中一直最高，因为它的 rollout 每一步都只押注一个混合分量。

```python
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(horizons, err_gru,  marker='o', label='GRU',     color='tab:blue')
ax.plot(horizons, err_mdn,  marker='s', label='MDN-RNN', color='tab:orange')
ax.plot(horizons, err_rssm, marker='^', label='RSSM',    color='tab:green')
ax.plot(horizons, err_copy, linestyle='--', label='复制上一步', color='gray')
ax.set_xlabel('预测步长（步）')
ax.set_ylabel('平均潜在 MSE')
ax.set_title('1 到 10 步预测误差（保留测试集）')
ax.set_xticks(horizons)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```
## 保存权重文件

保存的是 `rssm_model.state_dict()`，而不是 GRU 或 MDN-RNN 的权重，因为 RSSM 是 P03 要在其上构建的动力学核心：先验/后验的分离让 rollout 可以只靠 `prior_net`（不需要真实观测）向前滚动，这正是「完全在想象中训练」得以成立的关键，也是 [Dreamer 流程](../lectures/lecture-02-encode-and-dynamics/03-dynamics-dreamer-series#dreamer-中编码器的桥梁作用) 依赖的性质。P03 会加载这份权重，给 `rssm_model` 包上一个 actor 和一个 critic，完全在 `RSSM.rollout` 风格的想象轨迹上训练策略，而不依赖与真实环境的交互。

```python
checkpoint = {
    'rssm_state_dict': rssm_model.state_dict(),
    'hidden_dim':      HIDDEN_DIM,
    'latent_dim':      LATENT_DIM,
    'action_dim':      ACTION_DIM,
    'epochs_trained':  EPOCHS,
    'final_loss':      losses_rssm[-1],
}
Path('../../public').mkdir(parents=True, exist_ok=True)
torch.save(checkpoint, '../../public/rssm.pt')
print('RSSM 权重文件已保存至 ../../public/rssm.pt')
print(f'  hidden_dim={HIDDEN_DIM}, latent_dim={LATENT_DIM}')
print(f'  最终 ELBO 损失: {losses_rssm[-1]:.4f}')
```
