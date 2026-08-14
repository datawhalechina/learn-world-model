---
title: P05 世界模型评估仪表盘
---

# P05: 世界模型评估仪表盘

加载 P03 的 Dreamer 权重文件和 P04 的 Transformer 权重文件，在留出 episode 上进行评估，并将各项指标并排比较。本仪表盘有意保持保守风格：优先采用显式的权重文件加载和诚实的回退机制，避免任何隐含假设。

**前提条件**：P03 生成的 `dreamer.pt` 和 P04 生成的 `transformer_wm.pt`（如存在）。否则，每个缺失的权重文件将回退到随机初始化的模型，以便本 notebook 仍可作为冒烟测试运行。只有在加载了预训练权重文件时，所报告的指标才有实际意义，因此正式评估路径即为加载权重文件的路径。

**指标**：Dreamer 的奖励相关性、PSNR、潜在漂移。Transformer 的 token 预测损失、PSNR、潜在漂移。

> Notebook 源文件: [p05_evaluation_dashboard.ipynb](https://github.com/datawhalechina/learn-world-model/blob/main/docs/zh/projects/p05_evaluation_dashboard.ipynb)

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
环境准备完成后，先导入整块 dashboard 会用到的轨迹生成和打分工具。

```python
import math
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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

# 共享超参数，，必须与 P03 和 P04 保持一致
HIDDEN_DIM   = 128   # GRU / Transformer d_model
LATENT_DIM   = 32    # 随机状态维度（RSSM）/ 嵌入维度
N_CATEGORIES = 32    # CatVAE 离散词表大小
N_ACTIONS    = 2
SEQ_LEN      = 20    # 轨迹时域长度
N_EVAL_TRAJ  = 20    # 留出 episode 数量
ROLLOUT_LEN  = 10    # 用于时域指标的步数

NOTEBOOKS_DIR = Path('.')
DREAMER_CKPT  = NOTEBOOKS_DIR / 'dreamer.pt'
TRANS_CKPT    = NOTEBOOKS_DIR / 'transformer_wm.pt'

print('设备:', DEVICE)
if USE_TPU:
    print('TPU 后端: torch_xla')
print('PyTorch 版本:', torch.__version__)
print('Dreamer 权重文件存在:', DREAMER_CKPT.exists())
print('Transformer 权重文件存在:', TRANS_CKPT.exists())
```
## 1. 合成环境与轨迹生成

从 P03 使用的同一环境中生成 20 条留出 episode。

```python
class SyntheticEnv:
    """64x64 画布上移动的红色圆球。两个动作：向右（0）或向左（1）。"""
    SIZE = 64
    RADIUS = 8

    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)
        self.cx = self.cy = self.SIZE // 2

    def reset(self):
        self.cx = self.rng.randint(20, self.SIZE - 20)
        self.cy = self.rng.randint(20, self.SIZE - 20)
        return self._obs()

    def step(self, action):
        self.cx = int(np.clip(self.cx + (4 if action == 0 else -4), 10, self.SIZE - 10))
        reward = 1.0 if self.cx > self.SIZE // 2 else 0.0
        return self._obs(), reward, False

    def _obs(self):
        img = np.zeros((3, self.SIZE, self.SIZE), dtype=np.float32)
        color = np.array([0.9, 0.3, 0.3], dtype=np.float32)
        cx, cy, r = self.cx, self.cy, self.RADIUS
        for y in range(self.SIZE):
            for x in range(self.SIZE):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                    img[:, y, x] = color
        return img


def generate_eval_trajectories(n_traj=N_EVAL_TRAJ, horizon=SEQ_LEN, base_seed=999):
    """生成训练阶段未见过的留出轨迹。"""
    obs_list, act_list, rew_list = [], [], []
    for i in range(n_traj):
        env = SyntheticEnv(seed=base_seed + i)
        obs = env.reset()
        traj_obs, traj_act, traj_rew = [obs], [], []
        rng = np.random.RandomState(base_seed + i + 10000)
        for _ in range(horizon):
            action = rng.randint(0, N_ACTIONS)
            next_obs, rew, _ = env.step(action)
            traj_act.append(action)
            traj_rew.append(rew)
            traj_obs.append(next_obs)
        # obs_seq 共有 horizon+1 帧。保留前 horizon 帧作为输入
        obs_list.append(traj_obs[:horizon])
        act_list.append(traj_act)
        rew_list.append(traj_rew)
    obs_t = torch.tensor(np.array(obs_list), dtype=torch.float32)  # (N, T, 3, 64, 64)
    act_t = torch.tensor(np.array(act_list), dtype=torch.long)      # (N, T)
    rew_t = torch.tensor(np.array(rew_list), dtype=torch.float32)  # (N, T)
    return obs_t, act_t, rew_t


print(f'正在生成 {N_EVAL_TRAJ} 条留出评估轨迹（每条 {SEQ_LEN} 步）...')
eval_obs, eval_act, eval_rew = generate_eval_trajectories()
print('eval_obs :', eval_obs.shape)
print('eval_act :', eval_act.shape)
print('eval_rew :', eval_rew.shape)
print('奖励均值（应接近 0.5）:', eval_rew.mean().item())
```
## 2. 模型架构定义

内联定义 Dreamer 和 Transformer 组件，使本仪表盘完全自包含。

所有类均内联定义，确保本 notebook 自包含。架构维度与 P03 和 P04 完全一致：

- `HIDDEN_DIM = 128`，`LATENT_DIM = 32`，`N_CATEGORIES = 32`
- **Dreamer 侧：** CNN VAE 编码器/解码器、RSSM（GRU + 先验/后验网络）、Actor、Critic
- **Transformer 侧：** CatVAE（与 P04 相同）、CausalTransformerWM（与 P04 相同）

```python
# Dreamer 组件。

class Encoder(nn.Module):
    """CNN 编码器：将 3x64x64 帧映射到 LATENT_DIM 维的均值和对数方差。"""
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),    # 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),   # 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 8x8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), # 4x4
            nn.ReLU(),
            nn.Flatten(),                  # 256*4*4 = 4096
        )
        self.fc_mu     = nn.Linear(4096, latent_dim)
        self.fc_logvar = nn.Linear(4096, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def encode(self, x):
        mu, logvar = self.forward(x)
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)


class Decoder(nn.Module):
    """转置 CNN 解码器：(HIDDEN_DIM + LATENT_DIM) -> 3x64x64。"""
    def __init__(self, in_dim=HIDDEN_DIM + LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(in_dim, 256 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),    # 64x64
            nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.fc(x).view(-1, 256, 4, 4)
        return self.net(h)


class RSSM(nn.Module):
    """循环状态空间模型，包含确定性状态（h）和随机状态（s）。"""
    def __init__(self, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, n_actions=N_ACTIONS):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.gru = nn.GRUCell(latent_dim + 1, hidden_dim)
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        self.post_net = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        self.recon = nn.Linear(latent_dim, latent_dim)

    def _action_feature(self, action):
        if action.dim() == 0:
            action = action.view(1, 1)
        elif action.dim() == 1:
            action = action.unsqueeze(-1)
        if action.shape[-1] > 1:
            action = action[..., 1:2]
        return action.float()

    def initial_state(self, batch_size):
        h = torch.zeros(batch_size, self.hidden_dim, device=DEVICE)
        s = torch.zeros(batch_size, self.latent_dim, device=DEVICE)
        return h, s

    def prior(self, h):
        mu, lv = self.prior_net(h).chunk(2, dim=-1)
        std = F.softplus(lv) + 0.1
        s = mu + std * torch.randn_like(std)
        return s, mu, std

    def posterior(self, h, z_obs):
        mu, lv = self.post_net(torch.cat([h, z_obs], dim=-1)).chunk(2, dim=-1)
        std = F.softplus(lv) + 0.1
        s = mu + std * torch.randn_like(std)
        return s, mu, std

    def prior_step(self, h, a):
        s, _, _ = self.prior(h)
        a_feat = self._action_feature(a)
        h_next = self.gru(torch.cat([s, a_feat], dim=-1), h)
        return s, h_next

    def posterior_step(self, h, a, z_obs):
        s, _, _ = self.posterior(h, z_obs)
        a_feat = self._action_feature(a)
        h_next = self.gru(torch.cat([s, a_feat], dim=-1), h)
        return s, h_next


class RewardModel(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, n_actions=N_ACTIONS, ac_hidden=128):
        super().__init__()
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim + n_actions, ac_hidden),
            nn.ELU(),
            nn.Linear(ac_hidden, ac_hidden),
            nn.ELU(),
            nn.Linear(ac_hidden, 1)
        )

    def _one_hot(self, action):
        if action.dim() == 0:
            action = action.view(1)
        if action.dim() == 1 and action.dtype != torch.float32:
            action = F.one_hot(action.long(), num_classes=self.n_actions).float()
        elif action.dim() == 1:
            action = action.unsqueeze(-1)
        elif action.dim() == 2 and action.shape[-1] == 1:
            action = F.one_hot(action.squeeze(-1).long(), num_classes=self.n_actions).float()
        return action.float()

    def forward(self, h, s, a):
        a_oh = self._one_hot(a)
        return self.net(torch.cat([h, s, a_oh], dim=-1)).squeeze(-1)


class Actor(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, action_dim=N_ACTIONS, ac_hidden=128, obs_feat_dim=1):
        super().__init__()
        inp = hidden_dim + latent_dim + obs_feat_dim
        self.net = nn.Sequential(
            nn.Linear(inp, ac_hidden), nn.ELU(),
            nn.Linear(ac_hidden, ac_hidden), nn.ELU(),
            nn.Linear(ac_hidden, action_dim)
        )

    def forward(self, h, s, bar_pos=0.0):
        if not torch.is_tensor(bar_pos):
            bar_pos = torch.full((h.shape[0], 1), float(bar_pos), device=h.device)
        elif bar_pos.dim() == 0:
            bar_pos = bar_pos.view(1, 1).expand(h.shape[0], 1)
        elif bar_pos.dim() == 1:
            bar_pos = bar_pos.unsqueeze(-1)
        return self.net(torch.cat([h, s, bar_pos], -1))


class Critic(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, ac_hidden=128):
        super().__init__()
        inp = hidden_dim + latent_dim
        self.net = nn.Sequential(
            nn.Linear(inp, ac_hidden), nn.ELU(),
            nn.Linear(ac_hidden, ac_hidden), nn.ELU(),
            nn.Linear(ac_hidden, 1)
        )

    def forward(self, h, s):
        return self.net(torch.cat([h, s], -1)).squeeze(-1)


print('Dreamer 架构类定义完毕。')
print(f'  RSSM hidden_dim={HIDDEN_DIM}, latent_dim={LATENT_DIM}')
```
Dreamer 侧的组件已经定义好，接着补上 Transformer 世界模型部分，这样后面就能直接并排比较。

```python
# 基于 Transformer 的世界模型组件。

def straight_through_gumbel(logits, tau=1.0):
    y_soft = F.gumbel_softmax(logits, tau=tau, hard=False)
    y_hard = F.one_hot(y_soft.argmax(-1), num_classes=logits.shape[-1]).float()
    return (y_hard - y_soft).detach() + y_soft


class CatVAEEncoder(nn.Module):
    def __init__(self, num_categories=N_CATEGORIES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_categories),
        )

    def forward(self, x):
        return self.net(x)


class CatVAEDecoder(nn.Module):
    def __init__(self, num_categories=N_CATEGORIES):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_categories, 256),
            nn.ReLU(),
            nn.Linear(256, 256 * 4 * 4),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z_onehot):
        h = self.fc(z_onehot).view(-1, 256, 4, 4)
        return self.deconv(h)


class CatVAE(nn.Module):
    def __init__(self, num_categories=N_CATEGORIES, tau=1.0):
        super().__init__()
        self.encoder = CatVAEEncoder(num_categories)
        self.decoder = CatVAEDecoder(num_categories)
        self.tau = tau

    def encode(self, x):
        logits = self.encoder(x)
        z = straight_through_gumbel(logits, tau=self.tau)
        idx = logits.argmax(-1)
        return z, idx, logits

    def forward(self, x):
        z, idx, logits = self.encode(x)
        return self.decoder(z), z, idx, logits


class CausalTransformerWM(nn.Module):
    def __init__(self, num_categories=N_CATEGORIES, d_model=HIDDEN_DIM,
                 n_heads=4, n_layers=2, n_actions=N_ACTIONS, max_len=SEQ_LEN):
        super().__init__()
        self.d_model = d_model
        self.num_categories = num_categories
        self.z_proj   = nn.Linear(num_categories, d_model)
        self.a_embed  = nn.Embedding(n_actions, d_model)
        pos = torch.arange(max_len * 2).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe  = torch.zeros(max_len * 2, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model * 4,
                batch_first=True, norm_first=True
            ) for _ in range(n_layers)
        ])
        self.token_head  = nn.Linear(d_model, num_categories)
        self.reward_head = nn.Linear(d_model, 1)
        self.done_head   = nn.Linear(d_model, 1)

    def _causal_mask(self, T):
        return torch.triu(torch.ones(T, T, device=self.pe.device), diagonal=1).bool()

    def forward(self, z_seq, a_seq):
        B, T, _ = z_seq.shape
        z_emb = self.z_proj(z_seq)
        a_emb = self.a_embed(a_seq)
        tokens = torch.stack([z_emb, a_emb], dim=2).view(B, 2 * T, self.d_model)
        tokens = tokens + self.pe[:2 * T].unsqueeze(0)
        mask = self._causal_mask(2 * T)
        h = tokens
        for layer in self.layers:
            h = layer(h, src_mask=mask, is_causal=False)
        z_h = h[:, 0::2, :]
        return self.token_head(z_h), self.reward_head(z_h), self.done_head(z_h)


print('Transformer 架构类定义完毕。')
print(f'  CausalTransformerWM d_model={HIDDEN_DIM}, n_categories={N_CATEGORIES}')
```
## 3. 加载或初始化两个模型

有权重文件时加载，否则回退到随机初始化权重。

```python
# 构建模型实例。
encoder      = Encoder(LATENT_DIM).to(DEVICE)
decoder      = Decoder(HIDDEN_DIM + LATENT_DIM).to(DEVICE)
rssm         = RSSM(LATENT_DIM, HIDDEN_DIM, N_ACTIONS).to(DEVICE)
reward_model = RewardModel(LATENT_DIM, HIDDEN_DIM, N_ACTIONS).to(DEVICE)
actor        = Actor(LATENT_DIM, HIDDEN_DIM, N_ACTIONS).to(DEVICE)
critic       = Critic(LATENT_DIM, HIDDEN_DIM).to(DEVICE)

catvae       = CatVAE(N_CATEGORIES).to(DEVICE)
transformer  = CausalTransformerWM().to(DEVICE)

dreamer_loaded = False
trans_loaded   = False

# 加载 Dreamer 权重文件（P03）。
if DREAMER_CKPT.exists():
    try:
        ckpt = torch.load(DREAMER_CKPT, map_location=DEVICE, weights_only=False)
        if isinstance(ckpt, dict):
            if 'encoder' in ckpt:
                enc_state = {k.replace('conv.', 'net.'): v for k, v in ckpt['encoder'].items()}
                encoder.load_state_dict(enc_state, strict=True)
            if 'decoder' in ckpt:
                dec_state = {k.replace('deconv.', 'net.'): v for k, v in ckpt['decoder'].items()}
                decoder.load_state_dict(dec_state, strict=True)
            if 'rssm' in ckpt:
                rssm.load_state_dict(ckpt['rssm'], strict=True)
            if 'reward_model' in ckpt:
                reward_model.load_state_dict(ckpt['reward_model'], strict=True)
            if 'actor' in ckpt:
                actor.load_state_dict(ckpt['actor'], strict=True)
            if 'critic' in ckpt:
                critic.load_state_dict(ckpt['critic'], strict=True)
        dreamer_loaded = True
        print(f'已从 {DREAMER_CKPT} 加载 Dreamer 权重文件')
    except Exception as e:
        print(f'无法加载 Dreamer 权重文件（{e}），使用随机初始化。')
else:
    print('未找到 dreamer.pt，使用随机初始化的 Dreamer。')

# 加载 Transformer 权重文件（P04）。
if TRANS_CKPT.exists():
    try:
        ckpt = torch.load(TRANS_CKPT, map_location=DEVICE, weights_only=False)
        if isinstance(ckpt, dict):
            if 'catvae' in ckpt:
                catvae.load_state_dict(ckpt['catvae'], strict=True)
            if 'transformer_wm' in ckpt:
                transformer.load_state_dict(ckpt['transformer_wm'], strict=True)
        trans_loaded = True
        print(f'已从 {TRANS_CKPT} 加载 Transformer 权重文件')
    except Exception as e:
        print(f'无法加载 Transformer 权重文件（{e}），使用随机初始化。')
else:
    print('未找到 transformer_wm.pt，使用随机初始化的 Transformer。')

# --- 冻结所有参数 ---
for m in [encoder, decoder, rssm, reward_model, actor, critic, catvae, transformer]:
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)

print()
print(f'Dreamer 已从权重文件加载: {dreamer_loaded}')
print(f'Transformer 已从权重文件加载: {trans_loaded}')
print('所有模型已冻结并处于评估模式。')
```
## 4. 各模型指标计算

计算每个模型在留出轨迹上的指标。

下面为每个指标定义辅助函数，然后遍历 20 条留出轨迹进行计算。

**PSNR** 衡量像素级重建保真度，越高越好。
**潜在漂移**（想象潜在向量与真实观测潜在向量之间的 L2 距离）量化想象轨迹偏离真实值的速度。
**奖励相关性**（皮尔逊相关系数 rho）检验 RSSM 在想象推演中能否预测哪些步骤会获得奖励。
**token 预测损失**（交叉熵）是 Transformer 在测试时的训练信号。

```python
def psnr_fn(pred, target):
    """峰值信噪比（dB）。两个张量均在 [0,1] 范围内。"""
    mse = F.mse_loss(pred.clamp(0, 1), target.clamp(0, 1))
    return 10.0 * torch.log10(1.0 / (mse + 1e-8)).item()


def pearson_rho(x, y):
    """计算一维张量的皮尔逊相关系数。"""
    x = x - x.mean()
    y = y - y.mean()
    denom = (x.norm() * y.norm()).clamp(min=1e-8)
    return (x @ y / denom).item()


PSNR_STEPS   = [1, 3, 5, 10]
DRIFT_STEPS  = list(range(1, ROLLOUT_LEN + 1))

print('辅助函数已定义。')
print('PSNR 评估步骤:', PSNR_STEPS)
print('潜在漂移步骤 :', DRIFT_STEPS)
```
共享指标函数已经准备好，先把 Dreamer 的 rollout 统计量算出来。

```python
# Dreamer 指标。

dreamer_psnr        = {s: [] for s in PSNR_STEPS}
dreamer_drift       = {s: [] for s in DRIFT_STEPS}
dreamer_rew_corr    = []

with torch.no_grad():
    for traj_i in range(N_EVAL_TRAJ):
        obs_seq = eval_obs[traj_i].to(DEVICE)   # (T, 3, 64, 64)
        act_seq = eval_act[traj_i].to(DEVICE)   # (T,)
        rew_seq = eval_rew[traj_i].to(DEVICE)   # (T,)
        T = obs_seq.shape[0]

        # 对所有观测帧编码，获取真实潜在向量
        mu_all, _ = encoder(obs_seq)             # (T, LATENT_DIM)

        # 从第一帧（后验）初始化 RSSM 状态
        h = torch.zeros(1, rssm.hidden_dim, device=DEVICE)
        z0, _ = encoder.forward(obs_seq[0:1])    # 用均值作为点估计
        z0 = z0  # (1, LATENT_DIM)

        # 从第 0 步开始进行 10 步想象推演
        imagined_rewards = []
        imagined_latents = []   # 每步的随机状态 s
        imagined_frames  = []

        h_cur = h.clone()
        s_cur = z0.clone()      # 起始随机状态 = 编码后的 obs[0]

        for t in range(ROLLOUT_LEN):
            a_t = act_seq[t:t+1]
            s_next, h_next = rssm.prior_step(h_cur, a_t)
            r_pred = reward_model(h_next, s_next, a_t)   # 标量
            imagined_rewards.append(r_pred.squeeze())
            imagined_latents.append(s_next.squeeze(0))     # (LATENT_DIM,)
            frame = decoder(torch.cat([h_next, s_next], dim=-1))  # (1, 3, 64, 64)
            imagined_frames.append(frame.squeeze(0))
            h_cur, s_cur = h_next, s_next

        imagined_rewards = torch.stack(imagined_rewards)    # (ROLLOUT_LEN,)
        imagined_latents = torch.stack(imagined_latents)    # (ROLLOUT_LEN, LATENT_DIM)

        # 奖励相关性：想象奖励与真实奖励在推演窗口内的相关性
        actual_rew = rew_seq[:ROLLOUT_LEN]
        dreamer_rew_corr.append(pearson_rho(imagined_rewards.cpu(), actual_rew.cpu()))

        # 在指定步骤计算 PSNR
        for step in PSNR_STEPS:
            if step <= T and step <= ROLLOUT_LEN:
                gt = obs_seq[step - 1]           # 该步骤的真实帧
                pred_frame = imagined_frames[step - 1]
                dreamer_psnr[step].append(psnr_fn(pred_frame.unsqueeze(0), gt.unsqueeze(0)))

        # 潜在漂移：想象 s_t 与编码 obs_t 之间的 L2 距离
        for step in DRIFT_STEPS:
            if step <= T and step <= ROLLOUT_LEN:
                gt_latent = mu_all[step - 1]     # (LATENT_DIM,)
                img_latent = imagined_latents[step - 1]
                drift = (img_latent - gt_latent).norm().item()
                dreamer_drift[step].append(drift)

dreamer_psnr_mean  = {s: float(np.mean(dreamer_psnr[s]))  for s in PSNR_STEPS}
dreamer_drift_mean = {s: float(np.mean(dreamer_drift[s])) for s in DRIFT_STEPS}
dreamer_rho        = float(np.mean(dreamer_rew_corr))

print('Dreamer 指标计算完毕。')
print(f'  奖励相关性 rho : {dreamer_rho:.4f}')
print(f'  PSNR@1  : {dreamer_psnr_mean[1]:.2f} dB')
print(f'  PSNR@5  : {dreamer_psnr_mean[5]:.2f} dB')
print(f'  PSNR@10 : {dreamer_psnr_mean[10]:.2f} dB')
print(f'  潜在漂移@10 : {dreamer_drift_mean[10]:.4f}')
```
Dreamer 的指标收集完毕后，再对 Transformer 基线做同样的评估，把结果并排放在一起。

```python
# Transformer 指标。

trans_psnr      = {s: [] for s in PSNR_STEPS}
trans_drift     = {s: [] for s in DRIFT_STEPS}
trans_tok_loss  = []

with torch.no_grad():
    for traj_i in range(N_EVAL_TRAJ):
        obs_seq = eval_obs[traj_i].to(DEVICE)   # (T, 3, 64, 64)
        act_seq = eval_act[traj_i].to(DEVICE)   # (T,)
        T = obs_seq.shape[0]

        # 对所有帧编码为离散 token（真实潜在向量）
        logits_all = catvae.encoder(obs_seq)                         # (T, K)
        idx_all    = logits_all.argmax(-1)                           # (T,)
        z_oh_all   = F.one_hot(idx_all, num_classes=N_CATEGORIES).float()  # (T, K)

        # 对完整轨迹计算 token 预测损失（teacher forcing）
        z_seq_in  = z_oh_all.unsqueeze(0)        # (1, T, K)
        a_seq_in  = act_seq.unsqueeze(0)         # (1, T)
        tok_logits, _, _ = transformer(z_seq_in, a_seq_in)  # (1, T, K)
        # 用位置 t 的输出预测 t+1 的 token
        target_idx = idx_all[1:]                 # (T-1,)
        pred_logits = tok_logits[0, :-1, :]     # (T-1, K)
        loss_val = F.cross_entropy(pred_logits, target_idx).item()
        trans_tok_loss.append(loss_val)

        # 从第 0 步开始自回归推演（不使用 teacher forcing）
        z_context = z_oh_all[0:1].unsqueeze(0)  # (1, 1, K)  -- 以 obs[0] 作为种子
        imagined_z  = []   # 每步的 one-hot 潜在向量
        imagined_f  = []   # 解码帧

        for t in range(ROLLOUT_LEN):
            cur_len = z_context.shape[1]
            a_prefix = act_seq[:cur_len].unsqueeze(0)         # (1, cur_len)
            tok_out, _, _ = transformer(z_context, a_prefix)  # (1, cur_len, K)
            next_logits = tok_out[0, -1, :]                   # (K,)
            next_idx    = next_logits.argmax().unsqueeze(0)   # (1,)
            next_z      = F.one_hot(next_idx, num_classes=N_CATEGORIES).float()  # (1, K)
            frame       = catvae.decoder(next_z)              # (1, 3, 64, 64)
            imagined_z.append(next_z.squeeze(0))              # (K,)
            imagined_f.append(frame.squeeze(0))               # (3, 64, 64)
            z_context = torch.cat([z_context, next_z.unsqueeze(0)], dim=1)  # 扩展上下文

        imagined_z = torch.stack(imagined_z)   # (ROLLOUT_LEN, K)

        # PSNR
        for step in PSNR_STEPS:
            if step <= T and step <= ROLLOUT_LEN:
                gt = obs_seq[step - 1]
                pred_frame = imagined_f[step - 1]
                trans_psnr[step].append(psnr_fn(pred_frame.unsqueeze(0), gt.unsqueeze(0)))

        # 潜在漂移：想象 one-hot 与真实 one-hot 之间的 L2 距离
        for step in DRIFT_STEPS:
            if step <= T and step <= ROLLOUT_LEN:
                gt_z    = z_oh_all[step - 1]       # (K,)
                img_z   = imagined_z[step - 1]     # (K,)
                drift   = (img_z - gt_z).norm().item()
                trans_drift[step].append(drift)

trans_psnr_mean  = {s: float(np.mean(trans_psnr[s]))  for s in PSNR_STEPS}
trans_drift_mean = {s: float(np.mean(trans_drift[s])) for s in DRIFT_STEPS}
trans_tok_mean   = float(np.mean(trans_tok_loss))

print('Transformer 指标计算完毕。')
print(f'  Token 预测损失 : {trans_tok_mean:.4f}')
print(f'  PSNR@1  : {trans_psnr_mean[1]:.2f} dB')
print(f'  PSNR@5  : {trans_psnr_mean[5]:.2f} dB')
print(f'  PSNR@10 : {trans_psnr_mean[10]:.2f} dB')
print(f'  潜在漂移@10 : {trans_drift_mean[10]:.4f}')
```
## 5. 指标汇总表

将关键指标汇集到一张表中。

下表汇总了两个模型的所有计算指标。标注 `N/A` 的条目表示该指标在概念上对该架构无意义：奖励相关性要求 RSSM 中存在专用奖励头，token 预测损失则要求离散类别潜在表示（Transformer 架构所具备）。

```python
# 打印指标汇总表。
header = f"{'模型':<15} | {'PSNR@1':>8} | {'PSNR@5':>8} | {'PSNR@10':>9} | {'LatentDrift@10':>14} | {'RewardCorr':>11} | {'TokenLoss':>10}"
sep    = '-' * len(header)
row_d  = (
    f"{'Dreamer':<15} | "
    f"{dreamer_psnr_mean[1]:>8.2f} | "
    f"{dreamer_psnr_mean[5]:>8.2f} | "
    f"{dreamer_psnr_mean[10]:>9.2f} | "
    f"{dreamer_drift_mean[10]:>14.4f} | "
    f"{dreamer_rho:>11.4f} | "
    f"{'N/A':>10}"
)
row_t  = (
    f"{'Transformer':<15} | "
    f"{trans_psnr_mean[1]:>8.2f} | "
    f"{trans_psnr_mean[5]:>8.2f} | "
    f"{trans_psnr_mean[10]:>9.2f} | "
    f"{trans_drift_mean[10]:>14.4f} | "
    f"{'N/A':>11} | "
    f"{trans_tok_mean:>10.4f}"
)
print(sep)
print(header)
print(sep)
print(row_d)
print(row_t)
print(sep)
```
## 6. 指标并排可视化

四个子图直观展示 L04 指标：

1. PSNR 随时域步骤的变化（两个模型绘制在同一坐标轴）
2. 潜在漂移（L2 范数）随步骤的变化（两个模型）
3. Dreamer 的奖励相关性 rho（柱状图）
4. Transformer 的 token 预测损失（柱状图）

```python
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle('P05: 世界模型评估仪表盘', fontsize=14, fontweight='bold')

# 子图 1：PSNR 随时域变化。
ax = axes[0, 0]
d_psnr_vals = [dreamer_psnr_mean[s] for s in PSNR_STEPS]
t_psnr_vals = [trans_psnr_mean[s]   for s in PSNR_STEPS]
ax.plot(PSNR_STEPS, d_psnr_vals, 'o-', color='royalblue',  linewidth=2, markersize=7, label='Dreamer (RSSM)')
ax.plot(PSNR_STEPS, t_psnr_vals, 's-', color='tomato',     linewidth=2, markersize=7, label='Transformer')
ax.set_xlabel('时域步骤')
ax.set_ylabel('PSNR (dB)')
ax.set_title('长时域 PSNR')
ax.set_xticks(PSNR_STEPS)
ax.legend()
ax.grid(True, alpha=0.3)

# 子图 2：潜在漂移随步骤变化。
ax = axes[0, 1]
d_drift_vals = [dreamer_drift_mean[s] for s in DRIFT_STEPS]
t_drift_vals = [trans_drift_mean[s]   for s in DRIFT_STEPS]
ax.plot(DRIFT_STEPS, d_drift_vals, 'o-', color='royalblue', linewidth=2, markersize=5, label='Dreamer (RSSM)')
ax.plot(DRIFT_STEPS, t_drift_vals, 's-', color='tomato',    linewidth=2, markersize=5, label='Transformer')
ax.set_xlabel('步骤')
ax.set_ylabel('L2 距离（想象潜在向量 vs 真实潜在向量）')
ax.set_title('潜在漂移随步骤变化')
ax.legend()
ax.grid(True, alpha=0.3)

# 子图 3：奖励相关性（Dreamer）。
ax = axes[1, 0]
per_traj_rho = dreamer_rew_corr
ax.bar(['Dreamer (RSSM)'], [dreamer_rho], color='royalblue', alpha=0.85, width=0.4)
ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
ax.set_ylim(-1.1, 1.1)
ax.set_ylabel('皮尔逊相关系数 rho')
ax.set_title('奖励相关性（10 步推演）')
# 叠加各轨迹散点
ax.scatter(
    np.zeros(len(per_traj_rho)),
    per_traj_rho, color='steelblue', alpha=0.5, zorder=3, s=30
)
ax.text(0, dreamer_rho + 0.05, f'均值={dreamer_rho:.3f}', ha='center', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# 子图 4：Token 预测损失（Transformer）。
ax = axes[1, 1]
ax.bar(['Transformer'], [trans_tok_mean], color='tomato', alpha=0.85, width=0.4)
ax.set_ylabel('交叉熵损失')
ax.set_title('Token 预测损失（teacher forcing）')
ax.scatter(
    np.zeros(len(trans_tok_loss)),
    trans_tok_loss, color='firebrick', alpha=0.5, zorder=3, s=30
)
ax.text(0, trans_tok_mean + 0.02, f'均值={trans_tok_mean:.3f}', ha='center', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```
## 7. 解码帧序列：并排可视化

3 行图像网格将真实观测与两个模型在推演步骤 1、5、10 以及最终步骤（第 20 步，即轨迹最后一帧）的想象帧并排对比。这使得 PSNR 随时域的衰减一目了然。

```python
DISPLAY_STEPS = [1, 5, 10, SEQ_LEN - 1]  # 1-indexed，最后一个除外
TRAJ_IDX = 0  # 使用第一条评估轨迹进行可视化

obs_seq_vis = eval_obs[TRAJ_IDX].to(DEVICE)  # (T, 3, 64, 64)
act_seq_vis = eval_act[TRAJ_IDX].to(DEVICE)  # (T,)
T_vis = obs_seq_vis.shape[0]

with torch.no_grad():
    # --- Dreamer 想象帧 ---
    h_vis = torch.zeros(1, rssm.hidden_dim, device=DEVICE)
    mu0, _ = encoder.forward(obs_seq_vis[0:1])
    s_vis  = mu0
    dreamer_vis_frames = []
    for t in range(max(DISPLAY_STEPS)):
        a_t = act_seq_vis[t:t+1]
        s_vis, h_vis = rssm.prior_step(h_vis, a_t)
        frame = decoder(torch.cat([h_vis, s_vis], dim=-1))
        dreamer_vis_frames.append(frame.squeeze(0).cpu())

    # --- Transformer 想象帧 ---
    logits0  = catvae.encoder(obs_seq_vis[0:1])
    idx0     = logits0.argmax(-1)
    z0_oh    = F.one_hot(idx0, num_classes=N_CATEGORIES).float()  # (1, K)
    z_ctx    = z0_oh.unsqueeze(0)   # (1, 1, K)
    trans_vis_frames = []
    for t in range(max(DISPLAY_STEPS)):
        a_prefix = act_seq_vis[:z_ctx.shape[1]].unsqueeze(0)
        tok_out, _, _ = transformer(z_ctx, a_prefix)
        next_z = F.one_hot(tok_out[0, -1, :].argmax().unsqueeze(0), num_classes=N_CATEGORIES).float()
        frame  = catvae.decoder(next_z)
        trans_vis_frames.append(frame.squeeze(0).cpu())
        z_ctx = torch.cat([z_ctx, next_z.unsqueeze(0)], dim=1)

# --- 构建 3 行图像网格 ---
n_cols = len(DISPLAY_STEPS)
fig, axes = plt.subplots(3, n_cols, figsize=(3.5 * n_cols, 10))
row_labels = ['真实值', 'Dreamer (RSSM)', 'Transformer']

for col, step in enumerate(DISPLAY_STEPS):
    gt_idx = min(step - 1, T_vis - 1) if step >= 1 else step
    gt_frame = obs_seq_vis[gt_idx].cpu().permute(1, 2, 0).numpy()

    dream_frame = dreamer_vis_frames[step - 1].permute(1, 2, 0).numpy()
    trans_frame = trans_vis_frames[step - 1].permute(1, 2, 0).numpy()

    for row, (frame, label) in enumerate(zip(
            [gt_frame, dream_frame, trans_frame], row_labels)):
        ax = axes[row, col]
        ax.imshow(np.clip(frame, 0, 1))
        ax.axis('off')
        if col == 0:
            ax.set_ylabel(label, fontsize=11)
        if row == 0:
            ax.set_title(f'第 {step} 步', fontsize=11)

plt.suptitle('想象推演：真实值 / Dreamer / Transformer', fontsize=13, y=1.01)
plt.tight_layout()
plt.show()
```
## 8. 诊断总结

总结主要失效模式与权衡取舍。

### PSNR 随时域衰减

两个模型的 PSNR 都随时域增长而下降，原因在于想象推演时均无法访问未来观测。RSSM 的连续高斯随机状态在帧间平滑插值，在短时域上往往产生较模糊但指标更接近真实值的重建。Transformer 的离散 token 瓶颈在第 1 步（种子 token 可靠时）更清晰，但误差累积迅速：每个预测错误的 token 都会成为下一步预测的上下文。若权重文件缺失，回退路径使用随机权重，适合冒烟测试，但不适合对比正式指标。

### 潜在漂移与误差累积

潜在漂移（想象潜在向量与真实潜在向量之间的 L2 距离）对两种架构都随步骤单调增长。在 RSSM 中，GRU 的递归引入平滑偏置：隐状态逐渐漂移，而非突变。在 Transformer 中，漂移通常更大，因为单个分类错误的 token 会改变整个上下文窗口，进而偏置后续所有预测。这正是 DreamerV3 通过在训练时混合先验与后验来解决的误差累积问题。

### 奖励相关性作为 RSSM 专属指标

奖励相关性仅对 Dreamer 模型定义，因为 RSSM 拥有与 actor-critic 损失端到端训练的专用奖励头。Transformer 也将奖励预测作为辅助任务，但未经策略梯度信号训练，其奖励预测不用于规划，因此不纳入本指标。

### Teacher forcing 偏差

token 预测损失在 teacher forcing 条件下测量：模型在每一步都以真实 token 作为上下文。推理时模型必须以自身的历史预测为条件，引入分布偏移。teacher forcing 损失与开环 PSNR 之间的差距揭示了这一偏移的程度。差距越大，说明模型对真实上下文过拟合，自由推演时性能下降越剧烈。

```python
# 最终内联汇总表。
print('=' * 90)
print('P05 最终指标汇总')
print('=' * 90)
header2 = (
    f"{'模型':<14} | "
    f"{'PSNR@1':>8} | "
    f"{'PSNR@5':>8} | "
    f"{'PSNR@10':>9} | "
    f"{'LatentDrift@10':>14} | "
    f"{'RewardCorr':>11} | "
    f"{'TokenLoss':>10}"
)
sep2 = '-' * len(header2)
print(header2)
print(sep2)
print(
    f"{'Dreamer':<14} | "
    f"{dreamer_psnr_mean[1]:>8.2f} | "
    f"{dreamer_psnr_mean[5]:>8.2f} | "
    f"{dreamer_psnr_mean[10]:>9.2f} | "
    f"{dreamer_drift_mean[10]:>14.4f} | "
    f"{dreamer_rho:>11.4f} | "
    f"{'N/A':>10}"
)
print(
    f"{'Transformer':<14} | "
    f"{trans_psnr_mean[1]:>8.2f} | "
    f"{trans_psnr_mean[5]:>8.2f} | "
    f"{trans_psnr_mean[10]:>9.2f} | "
    f"{trans_drift_mean[10]:>14.4f} | "
    f"{'N/A':>11} | "
    f"{trans_tok_mean:>10.4f}"
)
print(sep2)
print()
print('说明：')
print('  PSNR 单位为 dB（越高越好）。潜在漂移为 L2 范数（越低越好）')
print('  RewardCorr：10 步想象推演的皮尔逊相关系数 rho（仅 Dreamer）')
print('  TokenLoss：teacher forcing 条件下的交叉熵损失（仅 Transformer）')
print(f'  已从权重文件加载的模型：Dreamer={dreamer_loaded}, Transformer={trans_loaded}')
```
