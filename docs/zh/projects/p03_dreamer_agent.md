---
title: P03：训练 Dreamer 智能体
---

# P03：训练 Dreamer 智能体

训练一个包含世界模型与潜在 Actor-Critic 策略的紧凑型 Dreamer 智能体。本项目为教程规模的演示：目标是展示 Dreamer 训练循环、权重文件的衔接方式以及指标诊断流程，而非求解高难度控制基准。本项目不依赖外部 gym 库，由 `SyntheticEnv` 生成 64×64 RGB 帧并附带简单奖励信号。

**前置条件**：若存在 P01 的 `vae_encoder.pt` 和 P02 的 `rssm.pt`，将自动加载。否则缺失部分退化为随机初始化，笔记本仍可运行，但只有在使用预训练权重文件的情况下，训练出的智能体才具有实际意义。本笔记本将完整智能体保存为 `dreamer.pt`，供 P05 使用。

此处出现嘈杂的奖励曲线是可以接受的。教程目标是构建一个可运行的世界模型加策略流水线，而非追求基准得分。

> Notebook 源文件: [p03_dreamer_agent.ipynb](https://github.com/datawhalechina/learn-world-model/blob/main/docs/zh/projects/p03_dreamer_agent.ipynb)

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
## 1. 初始化设置

定义共享环境、模型维度与训练计划。本 notebook 把前面讲义里的四块内容组装成[完整的 Dreamer 流程](../lectures/lecture-02-encode-and-dynamics/03-dynamics-dreamer-series#dreamer-中编码器的桥梁作用)：编码、用 RSSM 向前预测、用 actor 和 critic 评估想象轨迹、再在真实环境里执行。这里先提前说明一处刻意的简化，免得后面的代码显得奇怪：本 notebook 里的 actor 是通过模仿一个手写的专家策略训练的（第 4 节的 `supervised_policy_update`），而不是像[讲义里的潜在 Actor-Critic](../lectures/lecture-03-architecture-patterns/05-planning-cem-ac#机制二-潜在空间中的-actor-critic-dreamer-的做法)描述的那样通过想象奖励反向传播训练。这里只有 critic 是在想象回报上训练的。这样能让教程规模的智能体在几分钟内稳定训练完成；第 3 节仍然实现了真正 Dreamer actor 更新会用到的完整可微想象机制。

```python
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
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
from collections import deque

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

print(f'使用设备: {DEVICE}')
if USE_CUDA:
    print(f'CUDA 可用: {torch.cuda.is_available()}')
```
### 1.1 超参数

保持参数规模较小，使完整训练循环能快速运行。其中三个直接对应讲义里的概念：`IMAGINE_H = 10` 是[CEM-MPC 与潜在 Actor-Critic](../lectures/lecture-03-architecture-patterns/05-planning-cem-ac)里的想象步数 $H$，也就是在世界模型里向前滚动多少步才计算回报。`LAMBDA_RETURN = 0.95` 是讲义 λ-return 里的 $\lambda$：越接近 1 越信任真实（想象）rollout，越接近 0 越信任 critic 自己的价值估计。`GAMMA = 0.99` 是同一页 CEM 伪代码里的折扣因子 $\gamma$，控制未来想象奖励被打折的速度。

```python
# 模型维度
IMG_SIZE    = 64
LATENT_DIM  = 32      # VAE / 随机潜在变量 z
HIDDEN_DIM  = 128     # RSSM 确定性隐状态 h
ACTION_DIM  = 2       # 二值动作空间
AC_HIDDEN   = 128     # Actor / Critic 隐层大小

# 训练计划
EPISODE_LEN   = 20    # 每个合成回合的步数
N_ITERATIONS  = 30    # 外层训练迭代次数
BATCH_SIZE    = 4     # 每次世界模型更新使用的轨迹数
IMAGINE_H     = 10    # 想象时域
LAMBDA_RETURN = 0.95  # TD(lambda)
GAMMA         = 0.99  # 折扣因子

# 学习率
LR_WM   = 3e-4
LR_AC   = 3e-4

# 权重文件路径（来自前序项目）
ENCODER_PATH = 'vae_encoder.pt'
RSSM_PATH    = 'rssm.pt'
SAVE_PATH    = 'dreamer.pt'
```
### 1.2 VAE 编码器

使用与 P01 相同的编码器结构，多了一处改动：这里的 `VAEDecoder.forward` 同时接收 `z` 和 `h`（`torch.cat([z, h], dim=-1)`），而不是像 P01 和 P02 那样只用 `z`。这对应[RSSM 一节](../lectures/lecture-02-encode-and-dynamics/02-dynamics#rssm-分离确定性与随机性)里的观测模型 $o_t \sim p(o_t \mid h_t, z_t)$：重建同时以确定性记忆 `h_t` 和随机感知 `z_t` 为条件，而不是像 P01/P02 更简单的解码器那样只用 `z_t`。

```python
class VAEEncoder(nn.Module):
    """将 64x64 RGB 帧编码为潜在均值和对数方差。"""
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 8x8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),# 4x4
            nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        """x: (B, 3, 64, 64) -> mu, logvar 各为 (B, latent_dim)"""
        h = self.conv(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def encode(self, x):
        mu, logvar = self.forward(x)
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std, mu, logvar


class VAEDecoder(nn.Module):
    """解码器：潜在维度 -> 64x64 RGB 重建。"""
    def __init__(self, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim + hidden_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z, h):
        """z: (B, latent_dim), h: (B, hidden_dim) -> (B, 3, 64, 64)"""
        x = self.fc(torch.cat([z, h], dim=-1))
        x = x.view(x.size(0), 256, 4, 4)
        return self.deconv(x)
```
### 1.3 RSSM

复用 P02 中的潜在动态模型接口，但重新拆分成三个独立方法（`prior`、`posterior`、`step`），而不是 P02 那种单一的 `forward` 循环，这样本 notebook 就能独立调用每一块：`step` 推进 `h`，`prior`/`posterior` 采样 `z`。正是这个拆分让下面的 `imagined_rollout` 得以实现，因为想象只需要 `prior`（不需要观测），而训练需要 `posterior`（需要观测），这正是[讲义里先验-后验深挖框](../lectures/lecture-02-encode-and-dynamics/02-dynamics#rssm-分离确定性与随机性)描述的那个拆分。

有一个实现细节值得指出：`prior` 和 `posterior` 用的是 `std = F.softplus(logvar) + 0.1`，而不是 P02 的 `std = (0.5 * logvar).exp()`。两者都是从无约束的网络输出保证标准差为正的有效方式；`softplus(x) + 0.1` 额外强制了 0.1 的最小标准差，防止 KL 项被一个几乎确定性的后验压到恰好为零，这是对[L04 Dreamer 诊断](../lectures/lecture-04-evaluation-by-model/01-model-metrics-dreamer#想象轨迹熵-imagined-trajectory-entropy)里讨论的后验坍缩失效模式的直接防护。

```python
class RSSM(nn.Module):
    """兼容 P02 的 RSSM，提供适合 P03 的动作接口。"""
    def __init__(self, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # P02 权重文件期望标量动作特征。
        self.gru = nn.GRUCell(latent_dim + 1, hidden_dim)

        # 先验：p(z_t | h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )

        # 后验：q(z_t | h_t, e_t)
        self.post_net = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )
        self.recon = nn.Linear(latent_dim, latent_dim)

    def _action_feature(self, action):
        """将标量或 one-hot 动作转换为 P02 所需的标量动作输入。"""
        if action.dim() == 0:
            action = action.view(1, 1)
        elif action.dim() == 1:
            action = action.unsqueeze(-1)
        if action.shape[-1] > 1:
            # 二值动作：保留动作 1 的概率质量，使想象推演中的软动作
            # 在 Actor 训练时保持可微。
            action = action[..., 1:2]
        return action.float()

    def initial_state(self, batch_size):
        h = torch.zeros(batch_size, self.hidden_dim, device=DEVICE)
        z = torch.zeros(batch_size, self.latent_dim, device=DEVICE)
        return h, z

    def prior(self, h):
        out = self.prior_net(h)
        mu, logvar = out.chunk(2, dim=-1)
        std = F.softplus(logvar) + 0.1
        z = mu + std * torch.randn_like(std)
        return z, mu, std

    def posterior(self, h, enc_z):
        out = self.post_net(torch.cat([h, enc_z], dim=-1))
        mu, logvar = out.chunk(2, dim=-1)
        std = F.softplus(logvar) + 0.1
        z = mu + std * torch.randn_like(std)
        return z, mu, std

    def step(self, h, z, action_onehot):
        """推进确定性状态，返回新的 h。"""
        action_feat = self._action_feature(action_onehot)
        inp = torch.cat([z, action_feat], dim=-1)
        h_new = self.gru(inp, h)
        return h_new
```
### 1.4 Actor 与 Critic

两者均完全在潜在空间中训练，处理的是 `(h, z)` 对而不是像素，这正是让纯想象训练成为可能的「不需要真实观测」这一性质。`Actor.forward` 还接受一个可选的 `bar_pos` 特征；这是为接下来定义的合成平衡杆环境准备的任务专属捷径，不属于通用的 Dreamer 架构，省略时默认为零（下面 `imagined_rollout` 里就是这种情况，因为没有真实观测可以用来计算它）。`RewardModel` 和 Actor、Critic 一起定义：Dreamer 需要在想象 rollout *内部*预测奖励，因为想象出来的状态没有真实环境可以查询奖励信号，所以一个学出来的奖励模型和动力学模型本身一样是必不可少的组件。

```python
class Actor(nn.Module):
    """将潜在状态 (h, z) 和轻量级观测特征映射为动作 logits。"""
    def __init__(self, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM, ac_hidden=AC_HIDDEN, obs_feat_dim=1):
        super().__init__()
        inp = hidden_dim + latent_dim + obs_feat_dim
        self.net = nn.Sequential(
            nn.Linear(inp, ac_hidden),
            nn.ELU(),
            nn.Linear(ac_hidden, ac_hidden),
            nn.ELU(),
            nn.Linear(ac_hidden, action_dim),
        )

    def forward(self, h, z, bar_pos=None):
        if bar_pos is None:
            bar_pos = torch.zeros(h.shape[0], 1, device=h.device)
        logits = self.net(torch.cat([h, z, bar_pos], dim=-1))
        return logits

    def sample(self, h, z, bar_pos=None):
        logits = self.forward(h, z, bar_pos=bar_pos)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist


class Critic(nn.Module):
    """将潜在状态 (h, z) 映射为标量价值估计。"""
    def __init__(self, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, ac_hidden=AC_HIDDEN):
        super().__init__()
        inp = hidden_dim + latent_dim
        self.net = nn.Sequential(
            nn.Linear(inp, ac_hidden),
            nn.ELU(),
            nn.Linear(ac_hidden, ac_hidden),
            nn.ELU(),
            nn.Linear(ac_hidden, 1),
        )

    def forward(self, h, z):
        return self.net(torch.cat([h, z], dim=-1)).squeeze(-1)


class RewardModel(nn.Module):
    """从潜在状态和动作预测即时奖励。"""
    def __init__(self, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM, ac_hidden=AC_HIDDEN):
        super().__init__()
        inp = hidden_dim + latent_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(inp, ac_hidden),
            nn.ELU(),
            nn.Linear(ac_hidden, ac_hidden),
            nn.ELU(),
            nn.Linear(ac_hidden, 1),
        )

    def forward(self, h, z, a):
        return self.net(torch.cat([h, z, a], dim=-1)).squeeze(-1)
```
### 1.5 合成环境

`SyntheticEnv` 是专门为本 notebook 设计的一个极简控制任务，讲义里没有介绍过：一根红色竖杆的水平位置 `pos` 根据二元动作向左或向右移动固定步长，只要动作让杆更靠近中心（`abs(pos) < prev_abs`）奖励就是 `+1`，否则是 `-1`。这使得「让杆保持在中心附近」成为最优策略，足够简单，在 CPU 上训练几分钟就能演示出一个能跑通的 Dreamer 循环，同时又要求智能体从像素里读出位置（通过编码器）并根据一个多步信号采取行动（通过 RSSM 和 actor），而不是用一行启发式规则就能解决。

```python
class SyntheticEnv:
    """带图像观测的简单合成控制环境。"""
    def __init__(self, episode_len=EPISODE_LEN, img_size=IMG_SIZE, seed=None):
        self.episode_len = episode_len
        self.img_size    = img_size
        self.rng         = np.random.default_rng(seed)
        self.pos         = 0.0
        self.step_count  = 0

    def _render(self):
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
        bar_x = int((self.pos + 1.0) / 2.0 * (self.img_size - 1))
        bar_x = np.clip(bar_x, 0, self.img_size - 1)
        img[:, max(0, bar_x - 2): bar_x + 3, 0] = 1.0  # 红色通道
        # 加入轻微背景噪声，使编码器面临非平凡任务
        img += self.rng.uniform(0, 0.05, img.shape).astype(np.float32)
        return np.clip(img, 0, 1)

    def reset(self):
        self.pos        = float(self.rng.uniform(-0.8, 0.8))
        self.step_count = 0
        return self._render()

    def step(self, action):
        """action: int（0 或 1）。返回 (obs, reward, done)。"""
        prev_abs = abs(self.pos)
        delta    = 0.1 if action == 1 else -0.1
        self.pos = float(np.clip(self.pos + delta, -1.0, 1.0))
        reward   = 1.0 if abs(self.pos) < prev_abs else -1.0
        self.step_count += 1
        done = self.step_count >= self.episode_len
        return self._render(), reward, done


# 快速完整性检查
env = SyntheticEnv(seed=0)
obs = env.reset()
print(f'观测形状: {obs.shape}, 数据类型: {obs.dtype}, 范围: [{obs.min():.2f}, {obs.max():.2f}]')
obs2, r, done = env.step(1)
print(f'执行动作后: 奖励={r}, 回合结束={done}')
```
### 1.6 加载或初始化模型

如果存在的话，加载 P01 的 `vae_encoder.pt` 和 P02 的 `rssm.pt`。注意 `_load_encoder_decoder_from_vae_checkpoint` 里的注释：从 P01 的权重文件里只复用了*编码器*的权重；P03 的解码器是刻意重新初始化、从头训练的，因为 P03 解码器的结构和 P01 不一样（如上面 1.2 节所说，它多接收了 `h`），所以即使复用 P01 解码器的权重，形状也对不上。如果任一权重文件缺失，对应的模块会退化为随机初始化，notebook 仍然能运行，但训练出的智能体只是一个较弱的演示，因为它没有建立在 P01/P02 学到的表示之上。

```python
def obs_to_tensor(obs):
    """将 HWC numpy float32 转换为 (1, 3, H, W) 张量。"""
    t = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0)
    return t.to(DEVICE)


encoder = VAEEncoder(latent_dim=LATENT_DIM).to(DEVICE)
decoder = VAEDecoder(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
rssm    = RSSM(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM).to(DEVICE)
actor   = Actor(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM, ac_hidden=AC_HIDDEN).to(DEVICE)
critic  = Critic(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, ac_hidden=AC_HIDDEN).to(DEVICE)
reward_model = RewardModel(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM, ac_hidden=AC_HIDDEN).to(DEVICE)

# 尝试从前序项目加载权重。
def _load_encoder_decoder_from_vae_checkpoint(path):
    ckpt = torch.load(path, map_location=DEVICE)
    state = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt

    if isinstance(ckpt, dict) and 'encoder' in ckpt:
        enc_state = {k.replace('fc_log_var', 'fc_logvar'): v for k, v in ckpt['encoder'].items()}
        encoder.load_state_dict(enc_state, strict=True)
        return '仅编码器（解码器在 P03 中有意重新初始化）'

    enc_state = {}
    dec_state = {}
    for key, value in state.items():
        if key.startswith('encoder.'):
            enc_key = key[len('encoder.'):].replace('fc_log_var', 'fc_logvar')
            enc_state[enc_key] = value
        elif key.startswith('decoder.'):
            dec_state[key[len('decoder.'):]] = value

    if not enc_state:
        raise KeyError(f'无法识别的 VAE 权重文件格式: {list(ckpt.keys())[:10] if isinstance(ckpt, dict) else type(ckpt)}')

    encoder.load_state_dict(enc_state, strict=True)
    return '仅编码器来自 model_state_dict（解码器在 P03 中有意重新初始化）'

vae_ckpt_candidates = [Path(ENCODER_PATH), Path('notebooks') / ENCODER_PATH]
vae_ckpt_path = next((p for p in vae_ckpt_candidates if p.exists()), None)
if vae_ckpt_path is not None:
    try:
        vae_ckpt_format = _load_encoder_decoder_from_vae_checkpoint(vae_ckpt_path)
        print(f'已从 {vae_ckpt_path} 加载编码器/解码器权重（{vae_ckpt_format}）')
    except Exception as e:
        print(f'无法从 {vae_ckpt_path} 加载编码器/解码器（{e}），使用随机初始化')
else:
    print('未找到 vae_encoder.pt，使用随机初始化编码器')

rssm_path = next((p for p in [Path(RSSM_PATH), Path('notebooks') / RSSM_PATH] if p.exists()), None)
if rssm_path is not None:
    try:
        state = torch.load(rssm_path, map_location=DEVICE)
        if isinstance(state, dict) and 'rssm_state_dict' in state:
            sd = state['rssm_state_dict']
            rssm.load_state_dict(sd, strict=True)
            print(
                f"已从 {rssm_path} 加载 RSSM 权重 "
                f"(hidden_dim={state.get('hidden_dim', HIDDEN_DIM)}, latent_dim={state.get('latent_dim', LATENT_DIM)}, action_dim={state.get('action_dim', 1)})"
            )
        elif isinstance(state, dict) and 'rssm' in state:
            rssm.load_state_dict(state['rssm'], strict=False)
            print(f'已从 {rssm_path} 加载 RSSM 权重（旧版 rssm 键）')
        else:
            rssm.load_state_dict(state, strict=False)
            print(f'已从 {rssm_path} 加载 RSSM 权重（原始 state_dict）')
    except Exception as e:
        print(f'无法加载 RSSM（{e}），使用随机初始化')
else:
    print('未找到 rssm.pt，使用随机初始化 RSSM')

print('\n各模型参数量:')
for name, m in [('encoder', encoder), ('decoder', decoder), ('rssm', rssm), ('actor', actor), ('critic', critic), ('reward_model', reward_model)]:
    n = sum(p.numel() for p in m.parameters())
    print(f'  {name}: {n:,}')
```
### 1.7 回放缓冲区与优化器

`replay_buffer` 是一个固定大小的 FIFO 队列（`deque(maxlen=200)`），存放采集到的轨迹，是世界模型更新和后面监督式策略更新共用的训练数据来源。注意 `opt_wm` 把 `encoder + decoder + rssm + reward_model` 联合优化成一组参数，而 `opt_actor` 和 `opt_critic` 是各自独立的优化器：这个分离很重要，因为第 1 节里提到过，actor 和 critic 是用不同机制训练的（模仿 vs. 想象回报），不应该共享梯度更新。

```python
# 回放缓冲区存储轨迹字典。
replay_buffer = deque(maxlen=200)

# 世界模型优化器涵盖编码器、解码器和 RSSM
wm_params = list(encoder.parameters()) + list(decoder.parameters()) + list(rssm.parameters()) + list(reward_model.parameters())
opt_wm    = optim.Adam(wm_params, lr=LR_WM)

opt_actor  = optim.Adam(actor.parameters(),  lr=LR_AC)
opt_critic = optim.Adam(critic.parameters(), lr=LR_AC)

print('优化器已初始化。')
```
## 2. 世界模型更新

`world_model_update` 是本 notebook 版本的世界模型训练循环，源自 [P02](../projects/p02_rssm_dynamics)，扩展成同时预测奖励。对轨迹里的每一步：编码当前观测（`encoder.encode`），同时算出后验 `z_post`（通过 `rssm.posterior`，以真实观测为条件）和先验（通过 `rssm.prior`，只以历史为条件），从 `(z_post, h_next)` 重建*下一步*观测，再从 `(h, z_post, a_oh)` 预测即时奖励。

`kl` 的计算和 P02 的 RSSM 用的是同一个解析形式的高斯-高斯 KL，只是直接用 `std` 写出来（`(post_std / prior_std).pow(2) + ...`），而不是像 P02 那样用 `log_var`：两者在代数上是等价的，因为 `log_var = 2 * log(std)`。每一步结尾的 `h = h_next.detach()` 和 `z = z_post.detach()` 刻意切断了相邻时间步之间的反向传播图（截断式随时间反向传播），用一点梯度精度换取训练稳定性，并在这里用到的 20 步 episode 上降低显存占用。

```python
def action_to_onehot(action_int, action_dim=ACTION_DIM):
    """标量整数 -> (1, action_dim) one-hot 张量。"""
    oh = torch.zeros(1, action_dim, device=DEVICE)
    oh[0, action_int] = 1.0
    return oh


def world_model_update(batch):
    """执行一次世界模型 ELBO 更新。"""
    encoder.train()
    decoder.train()
    rssm.train()
    opt_wm.zero_grad()

    total_recon = 0.0
    total_kl    = 0.0
    total_reward = 0.0
    count       = 0

    for traj in batch:
        obs_list  = traj['obs']     # T+1 个 numpy 数组列表 (H,W,3)
        act_list  = traj['actions'] # T 个整数列表
        T         = len(act_list)

        h, z = rssm.initial_state(1)

        for t in range(T):
            obs_t    = obs_to_tensor(obs_list[t])           # (1,3,64,64)
            obs_next = obs_to_tensor(obs_list[t + 1])
            a_oh     = action_to_onehot(act_list[t])        # (1, action_dim)

            # 编码当前观测
            enc_z, enc_mu, enc_logvar = encoder.encode(obs_t)

            # 由编码器嵌入得到后验
            z_post, post_mu, post_std = rssm.posterior(h, enc_z)

            # 由确定性状态得到先验
            _, prior_mu, prior_std = rssm.prior(h)

            # 重建下一帧观测
            h_next = rssm.step(h, z_post.detach(), a_oh)
            recon  = decoder(z_post, h_next)

            # 从潜在状态预测实际转移奖励
            reward_target = torch.tensor([traj['rewards'][t]], device=DEVICE, dtype=torch.float32)
            reward_pred   = reward_model(h, z_post, a_oh)

            # 重建损失（每像素 MSE，在空间维度上取均值）
            recon_loss = F.mse_loss(recon, obs_next, reduction='mean')

            # KL 散度：后验 || 先验（封闭形式，均为高斯分布）
            kl = 0.5 * (
                (post_std / prior_std).pow(2)
                + ((post_mu - prior_mu) / prior_std).pow(2)
                - 1
                + 2 * prior_std.log()
                - 2 * post_std.log()
            ).sum(dim=-1).mean()

            reward_loss = F.mse_loss(reward_pred, reward_target)

            total_recon = total_recon + recon_loss
            total_kl    = total_kl    + kl
            total_reward = total_reward + reward_loss
            count       += 1

            # 推进状态（detach 以避免跨步的时间反向传播）
            h = h_next.detach()
            z = z_post.detach()

    loss = (total_recon + total_kl + total_reward) / max(count, 1)
    loss.backward()
    nn.utils.clip_grad_norm_(wm_params, 100.0)
    opt_wm.step()

    return (total_recon / count).item(), (total_kl / count).item(), (total_reward / count).item()


print('world_model_update 已定义。')
```
## 3. 行为学习（想象推演）

`imagined_rollout` 实现的是[讲义潜在 Actor-Critic 训练流程](../lectures/lecture-03-architecture-patterns/05-planning-cem-ac#机制二-潜在空间中的-actor-critic-dreamer-的做法)里的想象步骤：从一个真实的 `(h, z)` 对出发，反复从 actor 采样动作，推进 RSSM 的先验（永远不用后验，因为一旦开始想象就没有真实观测了），每一步都预测奖励，全程不接触真实环境。`differentiable=True` 这个分支专门是为了支持本 notebook 默认没有使用的*讲义精确版* actor 训练：它把 `dist.sample()`（不可微，因为从类别分布采样会阻断梯度）换成 `F.gumbel_softmax(..., hard=True)`，这是从离散分布近似出可微样本的标准技巧，正是这个技巧能让 actor 的梯度像讲义描述的那样反向流过整条想象轨迹。`behavior_update` 在不可微模式下（`differentiable=False`）调用 `imagined_rollout`，只用结果通过 `lambda_returns`（讲义引入的同一个 λ-return 公式）训练 critic；这个函数完全不涉及 actor。

`lambda_returns` 直接实现了讲义里的 λ-return：`td = rewards[t] + gamma * values[t+1]` 是单步 TD 目标，反向递推 `G_next = (1 - lam) * td + lam * gamma * G_next` 构造出讲义描述的同一个「对所有 k 步回报加权平均」，只是通过在步长上反向高效计算，而不是显式地做加权求和。

```python
def lambda_returns(rewards, values, gamma=GAMMA, lam=LAMBDA_RETURN):
    """计算想象推演的 lambda 回报目标。"""
    H = rewards.shape[0]
    G = torch.zeros(H, device=DEVICE)
    G_next = values[H]
    for t in reversed(range(H)):
        td = rewards[t] + gamma * values[t + 1]
        G_next = (1 - lam) * td + lam * gamma * G_next
        G[t] = G_next
    return G


def set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad_(flag)


def imagined_rollout(start_h, start_z, horizon=IMAGINE_H, differentiable=False, tau=0.8):
    """在想象空间中展开潜在轨迹。"""
    h, z = start_h, start_z
    h_seq, z_seq, r_seq, ent_seq = [], [], [], []

    for _ in range(horizon):
        logits = actor(h, z)
        dist = torch.distributions.Categorical(logits=logits)
        ent_seq.append(dist.entropy().mean())

        if differentiable:
            a_oh = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
            r_hat = reward_model(h, z, a_oh)
            h_next = rssm.step(h, z, a_oh)
            prior_out = rssm.prior_net(h_next)
            prior_mu, _ = prior_out.chunk(2, dim=-1)
            z_next = prior_mu
        else:
            a = dist.sample()
            a_oh = F.one_hot(a, num_classes=ACTION_DIM).float()
            with torch.no_grad():
                r_hat = reward_model(h, z, a_oh)
                h_next = rssm.step(h, z, a_oh)
                prior_out = rssm.prior_net(h_next)
                prior_mu, _ = prior_out.chunk(2, dim=-1)
                z_next = prior_mu

        h_seq.append(h)
        z_seq.append(z)
        r_seq.append(r_hat)
        h, z = h_next, z_next

    h_all = torch.stack(h_seq, dim=0)
    z_all = torch.stack(z_seq, dim=0)
    r_all = torch.stack(r_seq, dim=0).mean(dim=-1)
    return h_all, z_all, r_all, ent_seq


def behavior_update(start_h, start_z, horizon=IMAGINE_H):
    """在想象回报上训练 Critic。"""
    critic.train()
    reward_model.eval()
    rssm.eval()

    with torch.no_grad():
        h_all, z_all, r_all, ent_seq = imagined_rollout(start_h.detach(), start_z.detach(), horizon=horizon, differentiable=False)

    v_all = torch.zeros(horizon + 1, device=DEVICE)
    for t in range(horizon):
        v_all[t] = critic(h_all[t], z_all[t]).mean().detach()
    v_all[horizon] = critic(h_all[-1], z_all[-1]).mean().detach()
    G = lambda_returns(r_all, v_all)

    opt_critic.zero_grad()
    v_pred = torch.stack([critic(h_all[t], z_all[t]).mean() for t in range(horizon)])
    critic_loss = F.mse_loss(v_pred, G.detach())
    critic_loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), 100.0)
    opt_critic.step()

    return torch.stack(ent_seq).mean().item()


print('behavior_update 已定义。')
```
## 4. 训练循环

`N_ITERATIONS` 次外层迭代，每次都跑四个子步骤：用当前 actor 采集一条真实 episode（`collect_episode`）、在一批回放轨迹上更新世界模型（`world_model_update`）、用这条 episode 状态出发的想象 rollout 更新 critic（`behavior_update`）、再用模仿的方式更新 actor（`supervised_policy_update`）。最后这一步正是第 1 节里提到过的那处简化：`expert_action_from_obs` 是直接写死在这个 cell 里的手写启发式规则（「往中心移动」），actor 用普通的 `F.cross_entropy` 对着专家的动作标签训练，是标准的监督式行为克隆，不是策略梯度或者穿过想象的反向传播。这是教程规模演示的一个务实取舍：端到端可微的 actor 训练（通过 `imagined_rollout(differentiable=True)`）已经实现并可用（见第 3 节），但对超参数更敏感，在 `N_ITERATIONS = 30` 次迭代内更难稳定收敛，所以默认训练循环改用更稳定的模仿信号。如果想看讲义精确版的损失实际运行，可以把 `supervised_policy_update` 这一调用换成用 `imagined_rollout(h, z, differentiable=True)` 返回的 `r_seq` 算出的 actor 损失，像讲义描述的那样通过 critic 最大化它。

```python
def collect_episode(env_seed=None, deterministic=False, epsilon=0.05):
    """使用当前 Actor 收集一个回合的数据。"""
    env = SyntheticEnv(seed=env_seed)
    obs = env.reset()
    traj = {'obs': [obs], 'actions': [], 'rewards': []}

    h, z = rssm.initial_state(1)
    actor.eval()
    encoder.eval()
    rssm.eval()

    total_reward = 0.0
    done = False

    with torch.no_grad():
        while not done:
            obs_t = obs_to_tensor(obs)
            enc_z, _, _ = encoder.encode(obs_t)
            z_post, _, _ = rssm.posterior(h, enc_z)
            bar_pos = obs_to_bar_pos(obs)
            logits = actor(h, z_post, bar_pos=bar_pos)
            dist   = torch.distributions.Categorical(logits=logits)
            if deterministic:
                a_int = int(torch.argmax(logits, dim=-1).item())
            else:
                if random.random() < epsilon:
                    a_int = random.randint(0, ACTION_DIM - 1)
                else:
                    a_int = int(dist.sample().item())

            obs_next, reward, done = env.step(a_int)

            a_oh = action_to_onehot(a_int)
            h = rssm.step(h, z_post, a_oh)
            z = z_post

            traj['obs'].append(obs_next)
            traj['actions'].append(a_int)
            traj['rewards'].append(reward)
            obs = obs_next
            total_reward += reward

    return traj, total_reward


def obs_to_bar_pos(obs):
    """从红色通道估计滑块位置，返回归一化标量。"""
    red_profile = obs[:, :, 0].mean(axis=0)
    bar_x = int(np.argmax(red_profile))
    denom = max(obs.shape[1] - 1, 1)
    bar_pos = (2.0 * bar_x / denom) - 1.0
    return torch.tensor([[bar_pos]], device=DEVICE, dtype=torch.float32)


def get_rssm_states_from_traj(traj):
    """在轨迹上重新运行 RSSM，获取后验 (h, z) 对，用于想象推演。"""
    encoder.eval()
    rssm.eval()
    h, z = rssm.initial_state(1)
    h_list, z_list = [], []
    with torch.no_grad():
        for t, a_int in enumerate(traj['actions']):
            obs_t = obs_to_tensor(traj['obs'][t])
            enc_z, _, _ = encoder.encode(obs_t)
            z_post, _, _ = rssm.posterior(h, enc_z)
            h_list.append(h)
            z_list.append(z_post)
            a_oh = action_to_onehot(a_int)
            h = rssm.step(h, z_post, a_oh)
    return torch.cat(h_list, dim=0), torch.cat(z_list, dim=0)  # (T, dim)


def expert_action_from_obs(obs):
    """为合成滑块任务返回向中心移动的专家动作。"""
    red_profile = obs[:, :, 0].mean(axis=0)
    bar_x = int(np.argmax(red_profile))
    center = obs.shape[1] // 2
    return 1 if bar_x < center else 0


def supervised_policy_update(batch):
    """训练 Actor 模仿回放轨迹上向中心移动的专家策略。"""
    actor.train()
    losses = []
    for traj in batch:
        h_states, z_states = get_rssm_states_from_traj(traj)
        targets = torch.tensor([expert_action_from_obs(obs) for obs in traj['obs'][:-1]], device=DEVICE, dtype=torch.long)
        bar_pos = torch.cat([obs_to_bar_pos(obs) for obs in traj['obs'][:-1]], dim=0)
        logits = actor(h_states, z_states, bar_pos=bar_pos)
        losses.append(F.cross_entropy(logits, targets))

    loss = torch.stack(losses).mean()
    opt_actor.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), 100.0)
    opt_actor.step()
    return loss.item()


print('数据收集工具已定义，开始训练...')
```
轨迹采样已经准备好，先把指标历史记录搭起来，方便训练循环持续写入结果。

```python
# 指标历史记录。
ep_rewards      = []
recon_losses    = []
kl_losses       = []
reward_losses   = []
policy_losses   = []
actor_entropies = []

for iteration in range(N_ITERATIONS):
    # --- 收集一个回合 ---
    traj, ep_reward = collect_episode(env_seed=iteration, deterministic=False, epsilon=0.10)
    replay_buffer.append(traj)
    ep_rewards.append(ep_reward)

    # --- 世界模型更新 ---
    buf_list = list(replay_buffer)
    n_sample = min(BATCH_SIZE, len(buf_list))
    batch    = random.sample(buf_list, n_sample)
    recon_l, kl_l, reward_l = world_model_update(batch)
    recon_losses.append(recon_l)
    kl_losses.append(kl_l)
    reward_losses.append(reward_l)

    # --- Critic 更新（想象推演）---
    h_states, z_states = get_rssm_states_from_traj(traj)
    entropy = behavior_update(h_states, z_states, horizon=IMAGINE_H)
    actor_entropies.append(entropy)

    # --- 从回放中使用简单专家策略更新 Actor ---
    policy_l = supervised_policy_update(batch)
    policy_losses.append(policy_l)

    if (iteration + 1) % 10 == 0:
        print(
            f'迭代 {iteration+1:3d} | '
            f'回合奖励={ep_reward:+.1f} | '
            f'重建={recon_l:.4f} | '
            f'KL={kl_l:.4f} | '
            f'奖励={reward_l:.4f} | '
            f'策略={policy_l:.4f} | '
            f'Actor熵={entropy:.4f}'
        )

print('\n训练完成。')
```
指标记录好之后，把它们画成学习曲线，方便观察智能体是否在持续变好。如果模仿训练出的 actor 成功学到了「往中心移动」这个行为，`ep_reward` 应该随迭代次数上升；`recon_losses` 和 `kl_losses` 是 P01 和 P02 里介绍过的同两个 ELBO 分量，现在用来诊断世界模型本身（而不是策略）是否在退化。尤其要留意 `kl_losses` 是否出现第 1.3 节提到过的接近零坍缩模式：如果 KL 在最初几次迭代内就跌到接近零，而 `recon_losses` 仍然很高，说明发生了后验坍缩，而不是训练成功。

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Dreamer 训练指标', fontsize=14)

axes[0, 0].plot(ep_rewards, color='steelblue')
axes[0, 0].set_title('回合奖励')
axes[0, 0].set_xlabel('迭代次数')
axes[0, 0].set_ylabel('总奖励')
axes[0, 0].axhline(0, color='gray', linestyle='--', alpha=0.5)

axes[0, 1].plot(recon_losses, color='tomato')
axes[0, 1].set_title('重建损失（MSE）')
axes[0, 1].set_xlabel('迭代次数')
axes[0, 1].set_ylabel('损失')

axes[1, 0].plot(kl_losses, color='darkorange')
axes[1, 0].set_title('KL 散度损失')
axes[1, 0].set_xlabel('迭代次数')
axes[1, 0].set_ylabel('KL')

axes[1, 1].plot(actor_entropies, color='mediumpurple')
axes[1, 1].set_title('Actor 熵（想象推演）')
axes[1, 1].set_xlabel('迭代次数')
axes[1, 1].set_ylabel('熵（奈特）')

plt.tight_layout()
plt.show()
```
## 5. 自评估指标

本节实现的正是[L04 Dreamer 案例](../lectures/lecture-04-evaluation-by-model/01-model-metrics-dreamer#奖励相关性-reward-correlation)里的诊断方法：对同样的起始状态，比较世界模型*想象*出的奖励和真实环境*实际返回*的奖励。`imagined_rollout_rewards` 是 `imagined_rollout`（第 3 节）的只读变体，额外记录了每一步想象中 actor 的熵，因为讲义里健康 Dreamer 的诊断规则要求同时检查奖励相关性和熵，而不能只看奖励相关性。

```python
def imagined_rollout_rewards(start_h, start_z, horizon=10, deterministic=True):
    """使用 Actor 和奖励模型在想象空间中前向推演。"""
    actor.eval()
    reward_model.eval()
    rssm.eval()
    h, z = start_h.clone(), start_z.clone()
    im_rewards = []
    im_entropies = []

    with torch.no_grad():
        for _ in range(horizon):
            logits = actor(h, z)
            dist   = torch.distributions.Categorical(logits=logits)
            im_entropies.append(dist.entropy().mean().item())
            if deterministic:
                a = torch.argmax(logits, dim=-1)
            else:
                a = dist.sample()
            a_oh   = F.one_hot(a, num_classes=ACTION_DIM).float()
            r_hat  = reward_model(h, z, a_oh).mean().item()
            im_rewards.append(r_hat)
            h = rssm.step(h, z, a_oh)
            prior_out = rssm.prior_net(h)
            z, _ = prior_out.chunk(2, dim=-1)

    return im_rewards, im_entropies


N_EVAL     = 10
EVAL_H     = EPISODE_LEN

real_reward_sums  = []
imag_reward_sums  = []
imag_entropies_ev = []

encoder.eval()
rssm.eval()
actor.eval()
reward_model.eval()

for ep_i in range(N_EVAL):
    # 收集真实回合
    traj, ep_r = collect_episode(env_seed=1000 + ep_i, deterministic=True)
    real_reward_sums.append(sum(traj['rewards']))

    # 以第一步的 RSSM 状态作为想象推演的起点
    h0, z0 = get_rssm_states_from_traj(traj)
    seed_h  = h0[0:1]   # 单步
    seed_z  = z0[0:1]

    # 想象奖励与熵
    im_r, ents = imagined_rollout_rewards(seed_h, seed_z, horizon=EVAL_H, deterministic=True)
    imag_reward_sums.append(sum(im_r))
    imag_entropies_ev.append(float(np.mean(ents)))

print(f'在 {N_EVAL} 个回合上的评估完成。')
print(f'  真实奖励均值（完整回合）:          {np.mean(real_reward_sums):.3f}')
print(f'  预测奖励均值（想象推演）:           {np.mean(imag_reward_sums):.3f}')
print(f'  想象轨迹熵均值:                    {np.mean(imag_entropies_ev):.4f}')
```
想象推演函数定义好之后，比较想象奖励和真实回报，检查规划是否真的对齐了环境信号。用 `deterministic=True`（始终选择 actor 里 logit 最高的动作，不加探索噪声）采集 `N_EVAL = 10` 条全新 episode，然后对每条 episode，用第一次真实观测之后 RSSM 的后验状态作为种子，展开一条同样长度的想象 rollout。对同样的起始状态比较 `real_reward_sums` 和 `imag_reward_sums`，正是讲义推荐的「从同一个初始状态出发」的比较方式：如果两者走势一致，说明世界模型的奖励预测是规划可以信赖的输入；如果两者出现分歧，想象奖励就不能用来指导动作选择，这正是[Actor-Critic 规划讲义](../lectures/lecture-03-architecture-patterns/05-planning-cem-ac#机制二-潜在空间中的-actor-critic-dreamer-的做法)里说的模型漏洞风险。

```python
# 计算想象奖励与真实奖励之和的 Pearson 相关系数
r_real = np.array(real_reward_sums)
r_imag = np.array(imag_reward_sums)

if r_real.std() > 1e-8 and r_imag.std() > 1e-8:
    rho = np.corrcoef(r_real, r_imag)[0, 1]
else:
    rho = 0.0

print(f'奖励相关性 rho（预测值与真实值，{EVAL_H} 步）: {rho:.4f}')
print(f'想象轨迹熵均值:                              {np.mean(imag_entropies_ev):.4f}')
```
相关性摘要算好之后，把各类诊断图并排放出来，做一次快速的 sanity check。上一个 cell 里用 `np.corrcoef` 算出的 `rho`，正是 [L04](../lectures/lecture-04-evaluation-by-model/01-model-metrics-dreamer#奖励相关性-reward-correlation)里的皮尔逊相关系数 $\rho = \text{Pearson}(r_{\text{imagined}}, r_{\text{real}})$，讲义建议健康的世界模型应该达到 `ρ ≥ 0.8`；在这个只训练了 30 次迭代的小型教程规模智能体上，数值会比这个生产级目标更嘈杂。散点图上的对角虚线代表完全一致（`想象 = 真实`）：远离这条线的点，尤其是系统性地分布在线上方，正是讲义警告过的世界模型「撒谎」的视觉信号。

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Dreamer 自评估', fontsize=13)

# 训练过程中的回合奖励
axes[0].plot(ep_rewards, color='steelblue', label='训练')
axes[0].set_title('回合奖励（训练阶段）')
axes[0].set_xlabel('迭代次数')
axes[0].set_ylabel('总奖励')
axes[0].legend()

# 奖励相关性散点图
axes[1].scatter(r_real, r_imag, color='tomato', alpha=0.7)
axes[1].set_title(f'奖励相关性（rho={rho:.3f}）')
axes[1].set_xlabel(f'真实奖励（完整回合）')
axes[1].set_ylabel('预测奖励（想象推演）')
lims = [min(r_real.min(), r_imag.min()) - 0.5, max(r_real.max(), r_imag.max()) + 0.5]
axes[1].plot(lims, lims, 'k--', alpha=0.3, label='理想线')
axes[1].legend()

# 各评估回合的想象轨迹熵
axes[2].bar(range(N_EVAL), imag_entropies_ev, color='mediumpurple', alpha=0.8)
axes[2].set_title('想象轨迹熵')
axes[2].set_xlabel('评估回合')
axes[2].set_ylabel('平均熵（奈特）')
axes[2].axhline(np.mean(imag_entropies_ev), color='black', linestyle='--', label=f'均值={np.mean(imag_entropies_ev):.3f}')
axes[2].legend()

plt.tight_layout()
plt.show()
```
## 6. 保存权重文件

保存每一个训练过的组件（`encoder`、`decoder`、`rssm`、`actor`、`critic`、`reward_model`）以及完整的指标历史，而不只是像 P02 那样只存 RSSM，因为 P05 的评估仪表盘需要完整的智能体，而不只是它的动力学核心，才能按需复现第 5 节的奖励相关性和熵诊断。

```python
checkpoint = {
    'encoder': encoder.state_dict(),
    'decoder': decoder.state_dict(),
    'rssm':    rssm.state_dict(),
    'actor':   actor.state_dict(),
    'critic':  critic.state_dict(),
    'reward_model': reward_model.state_dict(),
    'hyperparams': {
        'latent_dim':  LATENT_DIM,
        'hidden_dim':  HIDDEN_DIM,
        'action_dim':  ACTION_DIM,
        'ac_hidden':   AC_HIDDEN,
    },
    'metrics': {
        'ep_rewards':      ep_rewards,
        'recon_losses':    recon_losses,
        'kl_losses':       kl_losses,
        'reward_losses':   reward_losses,
        'policy_losses':   policy_losses,
        'actor_entropies': actor_entropies,
        'reward_corr_rho': float(rho),
        'mean_imag_entropy': float(np.mean(imag_entropies_ev)),
        'mean_eval_reward': float(np.mean(real_reward_sums)),
        'mean_pred_reward': float(np.mean(imag_reward_sums)),
    },
}
torch.save(checkpoint, SAVE_PATH)
print(f'权重文件已保存至 {SAVE_PATH}')

# 训练摘要
print('\n--- 训练摘要 ---')
print(f'  末尾 10 轮探索回合奖励均值:      {np.mean(ep_rewards[-10:]):.2f}')
print(f'  最终重建损失:                    {recon_losses[-1]:.4f}')
print(f'  最终 KL 损失:                    {kl_losses[-1]:.4f}')
print(f'  最终奖励损失:                    {reward_losses[-1]:.4f}')
print(f'  最终策略损失:                    {policy_losses[-1]:.4f}')
print(f'  最终 Actor 熵:                   {actor_entropies[-1]:.4f}')
print(f'  奖励相关性 rho:                  {rho:.4f}')
print(f'  想象轨迹熵均值:                  {np.mean(imag_entropies_ev):.4f}')
print(f'  评估真实奖励均值:                {np.mean(real_reward_sums):.2f}')
```
