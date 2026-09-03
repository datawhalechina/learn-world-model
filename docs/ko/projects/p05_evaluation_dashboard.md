---
title: P05 월드모델 평가 대시보드
---

# P05: 월드모델 평가 대시보드

P03의 Dreamer와 P04의 Transformer 체크포인트를 불러와 홀드아웃 에피소드에서 평가하고, P04 지표를 나란히 비교합니다. 이 대시보드는 의도적으로 보수적입니다. 숨겨진 가정보다 명시적인 체크포인트 로딩과 정직한 대체 동작을 우선합니다.

**선행 조건**: P03(`dreamer.pt`)과 P04(`transformer_wm.pt`)가 있으면 사용합니다. 없으면 누락된 체크포인트마다 무작위로 초기화된 모델로 대체되어 노트북은 여전히 스모크 테스트로 실행됩니다. 보고되는 지표는 사전학습된 체크포인트가 있어야만 의미가 있으므로, 배포 준비가 된 경로는 체크포인트를 불러온 경로입니다.

**지표**: Dreamer의 보상 상관관계, PSNR, 잠재 드리프트. Transformer의 토큰 손실, PSNR, 잠재 드리프트.

> Notebook 원본: [p05_evaluation_dashboard.ipynb](https://github.com/datawhalechina/learn-world-model/blob/main/docs/ko/projects/p05_evaluation_dashboard.ipynb)

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
환경이 준비되었으니, 대시보드 전체에서 쓸 궤적 생성과 채점 유틸리티를 임포트합니다. 이 노트북은 [L04의 모델 독립적 진단 프레임워크](../lectures/lecture-04-evaluation-by-model/00-diagnostic-framework)를 그대로 구현합니다. 단일 집계 점수를 보고하는 대신, P03의 Dreamer와 P04의 Transformer 모두에 대해 같은 홀드아웃 궤적에서 같은 소수의 지표를 계산하므로, 각 모델을 자신만의 기준으로 채점하는 것이 아니라 수치를 직접 비교할 수 있습니다.

```python
import math
from pathlib import Path

try:
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'inline')
except Exception:
    pass
import matplotlib.pyplot as plt
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

# Shared hyperparameters -- must match P03 and P04
HIDDEN_DIM   = 128   # GRU / Transformer d_model
LATENT_DIM   = 32    # stochastic state dim (RSSM) / embedding dim
N_CATEGORIES = 32    # CatVAE discrete vocabulary
N_ACTIONS    = 2
SEQ_LEN      = 20    # trajectory horizon
N_EVAL_TRAJ  = 20    # held-out episodes
ROLLOUT_LEN  = 10    # steps used for horizon metrics

NOTEBOOKS_DIR = Path('.')
DREAMER_CKPT  = NOTEBOOKS_DIR / 'dreamer.pt'
TRANS_CKPT    = NOTEBOOKS_DIR / 'transformer_wm.pt'

print('Device:', DEVICE)
if USE_TPU:
    print('TPU backend    : torch_xla')
print('PyTorch version:', torch.__version__)
print('Dreamer checkpoint exists:', DREAMER_CKPT.exists())
print('Transformer checkpoint exists:', TRANS_CKPT.exists())
```
## 1. 합성 환경과 궤적 생성

P03에서 쓴 것과 같은 환경에서 20개의 홀드아웃 에피소드를 생성합니다. 여기서 "홀드아웃"이 중요한 이유는 이 에피소드들이 (두 학습 실행 중 어디에서도 재사용되지 않고) 새로 생성되기 때문입니다. 그래서 아래의 지표는 각 모델이 자신의 학습 궤적을 암기했는지 확인하는 것이 아니라 진정한 일반화 테스트가 됩니다. 이는 진단 프레임워크 강의가 "세 가지 평가 조건" 절에서 지적한 것과 같은 분포 내 대 홀드아웃의 구분입니다.

```python
class SyntheticEnv:
    """Moving red circle on a 64x64 canvas. Two actions: right (0) or left (1)."""
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
    """Generate held-out trajectories not seen during training."""
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
        # obs_seq has horizon+1 frames; we keep the first horizon as input
        obs_list.append(traj_obs[:horizon])
        act_list.append(traj_act)
        rew_list.append(traj_rew)
    obs_t = torch.tensor(np.array(obs_list), dtype=torch.float32)  # (N, T, 3, 64, 64)
    act_t = torch.tensor(np.array(act_list), dtype=torch.long)      # (N, T)
    rew_t = torch.tensor(np.array(rew_list), dtype=torch.float32)  # (N, T)
    return obs_t, act_t, rew_t


print(f'Generating {N_EVAL_TRAJ} held-out evaluation trajectories ({SEQ_LEN} steps each)...')
eval_obs, eval_act, eval_rew = generate_eval_trajectories()
print('eval_obs :', eval_obs.shape)
print('eval_act :', eval_act.shape)
print('eval_rew :', eval_rew.shape)
print('Reward mean (should be ~0.5):', eval_rew.mean().item())
```
## 2. 모델 아키텍처 정의

대시보드가 독립적으로 동작하도록 Dreamer와 Transformer 구성 요소를 인라인으로 정의합니다. 아키텍처 차원은 P03, P04와 정확히 일치합니다.

- `HIDDEN_DIM = 128`, `LATENT_DIM = 32`, `N_CATEGORIES = 32`
- **Dreamer 쪽:** CNN VAE 인코더/디코더, RSSM(GRU + 사전/사후 네트워크), 액터, 크리틱
- **Transformer 쪽:** CatVAE(P04와 동일), CausalTransformerWM(P04와 동일)

P03/P04에서 임포트하지 않고 다시 구현하는 것은 의도적입니다. 이 노트북은 `state_dict`를 통해 학습된 *가중치*(`dreamer.pt`, `transformer_wm.pt`)만 불러오므로, 여기서의 클래스 정의는 그 체크포인트들의 형태와 정확히 일치해야 하지만, P05의 어떤 부분도 P03이나 P04의 노트북 코드가 여전히 남아 있는지, 혹은 그 코드가 그 뒤에 바뀌었는지에 의존하지 않습니다.

```python
# Dreamer components.

class Encoder(nn.Module):
    """CNN encoder: 3x64x64 frame -> LATENT_DIM-dim mean and log-var."""
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
    """Transposed-CNN decoder: (HIDDEN_DIM + LATENT_DIM) -> 3x64x64."""
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
    """Recurrent State-Space Model with deterministic (h) and stochastic (s) states."""
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


print('Dreamer architecture classes defined.')
print(f'  RSSM hidden_dim={HIDDEN_DIM}, latent_dim={LATENT_DIM}')
```
Dreamer 스택을 정의했으니, Transformer 월드모델 구성 요소도 추가해 대시보드가 이 둘을 직접 비교할 수 있게 합니다. 이는 P04의 `CatVAE`, `straight_through_gumbel`, `CausalTransformerWM`을 그대로 복사한 것으로, P04 끝에서 저장된 체크포인트가 형태 불일치 없이 그대로 불러와지도록 동일하게 유지됩니다.

```python
# Transformer-based world model components.

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


print('Transformer architecture classes defined.')
print(f'  CausalTransformerWM d_model={HIDDEN_DIM}, n_categories={N_CATEGORIES}')
```
## 3. 두 모델 불러오기 또는 초기화

가능하면 체크포인트를 불러오고, 없으면 무작위 가중치로 대체합니다. 노트북 첫 셀에서 밝혔듯, 이 대체 동작은 순전히 사전 체크포인트가 없어도 대시보드가 스모크 테스트로 실행되도록 하기 위해 존재합니다. 무작위로 초기화된 모델에서 계산된 모든 지표는 비교로서는 무의미하며, 코드가 오류 없이 실행된다는 확인으로만 받아들여야지 모델 품질에 대한 근거로 여겨서는 안 됩니다.

```python
# Build model instances.
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

# Load Dreamer checkpoint (P03).
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
        print(f'Loaded Dreamer checkpoint from {DREAMER_CKPT}')
    except Exception as e:
        print(f'Could not load Dreamer checkpoint ({e}). Using random initialization.')
else:
    print('dreamer.pt not found. Using randomly initialized Dreamer.')

# Load Transformer checkpoint (P04).
if TRANS_CKPT.exists():
    try:
        ckpt = torch.load(TRANS_CKPT, map_location=DEVICE, weights_only=False)
        if isinstance(ckpt, dict):
            if 'catvae' in ckpt:
                catvae.load_state_dict(ckpt['catvae'], strict=True)
            if 'transformer_wm' in ckpt:
                transformer.load_state_dict(ckpt['transformer_wm'], strict=True)
        trans_loaded = True
        print(f'Loaded Transformer checkpoint from {TRANS_CKPT}')
    except Exception as e:
        print(f'Could not load Transformer checkpoint ({e}). Using random initialization.')
else:
    print('transformer_wm.pt not found. Using randomly initialized Transformer.')

# --- Freeze all parameters ---
for m in [encoder, decoder, rssm, reward_model, actor, critic, catvae, transformer]:
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)

print()
print(f'Dreamer loaded from checkpoint : {dreamer_loaded}')
print(f'Transformer loaded from checkpoint: {trans_loaded}')
print('All models frozen in eval mode.')
```
## 4. 모델별 지표 계산

각 모델에 대해 홀드아웃 지표를 계산합니다.

각 지표에 대한 헬퍼 함수를 정의한 뒤, 20개의 홀드아웃 궤적에 대해 반복합니다.

**PSNR**은 픽셀 수준의 재구성 품질을 측정합니다. 높을수록 좋습니다.
**잠재 드리프트**(상상된 잠재 벡터와 관측된 잠재 벡터 사이의 L2 거리)는 상상된 궤적이 실제로부터 얼마나 빨리 벗어나는지를 정량화합니다.
**보상 상관관계**(피어슨 rho)는 상상 속에서도 RSSM이 어떤 스텝이 보상을 낳는지 예측할 수 있는지 확인합니다.
**토큰 예측 손실**(교차 엔트로피)은 테스트 시점의 Transformer 학습 신호입니다.

각 지표는 [L04의 여섯 가지 질문 프레임워크](../lectures/lecture-04-evaluation-by-model/00-diagnostic-framework#six-diagnostic-questions)의 특정 진단 계층에 대응됩니다. PSNR과 잠재 드리프트는 모두 "장기 롤아웃" 계층을 검사합니다(예측이 스스로의 다음 입력이 되어도 오차가 통제된 상태로 유지되는가). 보상 상관관계는 특히 Dreamer에 대한 "과제 수행 신호" 계층을 검사하는데, 이는 학습 궤적이 아니라 새로운 홀드아웃 데이터에서 계산된 [L04 자체의 보상 상관관계 진단](../lectures/lecture-04-evaluation-by-model/01-model-metrics-dreamer#reward-correlation)과 일치합니다. 토큰 예측 손실은 "1스텝 동역학" 계층에 대한 Transformer 전용 대리 지표입니다. Transformer에는 Dreamer의 Actor-Critic처럼 검사할 별도의 보상 조건부 롤아웃 메커니즘이 없기 때문입니다.

```python
def psnr_fn(pred, target):
    """Peak Signal-to-Noise Ratio in dB. Both tensors in [0,1]."""
    mse = F.mse_loss(pred.clamp(0, 1), target.clamp(0, 1))
    return 10.0 * torch.log10(1.0 / (mse + 1e-8)).item()


def pearson_rho(x, y):
    """Pearson correlation between 1-D tensors."""
    x = x - x.mean()
    y = y - y.mean()
    denom = (x.norm() * y.norm()).clamp(min=1e-8)
    return (x @ y / denom).item()


PSNR_STEPS   = [1, 3, 5, 10]
DRIFT_STEPS  = list(range(1, ROLLOUT_LEN + 1))

print('Utility functions defined.')
print('PSNR evaluation steps :', PSNR_STEPS)
print('Latent drift steps     :', DRIFT_STEPS)
```
공유 지표 헬퍼가 준비되었으니, 먼저 Dreamer 롤아웃 통계를 계산합니다. 코드에서 눈여겨볼 세부사항이 두 가지 있습니다. `s_cur = z0.clone()`은 롤아웃을 *인코딩된 실제* 첫 관측(`z0 = encoder.forward(obs_seq[0:1])`, 인코더의 평균값으로 샘플이 아니라 점 추정값)으로 시작하고, 이후의 모든 스텝은 더 이상의 관측 없이 `rssm.prior_step`만 호출하는데, 이는 강의가 상상을 사전분포 전용 롤아웃으로 설명한 것과 일치합니다. `imagined_rewards`는 각 상상 스텝에서 보상 모델의 예측을 누적합니다. 이 시퀀스를 `pearson_rho`를 통해 `rew_seq[:ROLLOUT_LEN]`(같은 궤적에 대한 실제 환경의 보상)과 비교하는 것이 바로 L04의 [보상 상관관계 진단](../lectures/lecture-04-evaluation-by-model/01-model-metrics-dreamer#reward-correlation) 그 자체이며, 이제는 Dreamer 에이전트가 P03 학습 중 한 번도 보지 못한 20개의 새 에피소드에서 측정됩니다.

```python
# Dreamer metrics.

dreamer_psnr        = {s: [] for s in PSNR_STEPS}
dreamer_drift       = {s: [] for s in DRIFT_STEPS}
dreamer_rew_corr    = []

with torch.no_grad():
    for traj_i in range(N_EVAL_TRAJ):
        obs_seq = eval_obs[traj_i].to(DEVICE)   # (T, 3, 64, 64)
        act_seq = eval_act[traj_i].to(DEVICE)   # (T,)
        rew_seq = eval_rew[traj_i].to(DEVICE)   # (T,)
        T = obs_seq.shape[0]

        # Encode all observations to get ground-truth latent vectors
        mu_all, _ = encoder(obs_seq)             # (T, LATENT_DIM)

        # Initialize RSSM state from first observation (posterior)
        h = torch.zeros(1, rssm.hidden_dim, device=DEVICE)
        z0, _ = encoder.forward(obs_seq[0:1])    # mu as point estimate
        z0 = z0  # (1, LATENT_DIM)

        # 10-step imagination rollout starting from step 0
        imagined_rewards = []
        imagined_latents = []   # s at each step
        imagined_frames  = []

        h_cur = h.clone()
        s_cur = z0.clone()      # start stochastic state = encoded obs[0]

        for t in range(ROLLOUT_LEN):
            a_t = act_seq[t:t+1]
            s_next, h_next = rssm.prior_step(h_cur, a_t)
            r_pred = reward_model(h_next, s_next, a_t)   # scalar
            imagined_rewards.append(r_pred.squeeze())
            imagined_latents.append(s_next.squeeze(0))     # (LATENT_DIM,)
            frame = decoder(torch.cat([h_next, s_next], dim=-1))  # (1, 3, 64, 64)
            imagined_frames.append(frame.squeeze(0))
            h_cur, s_cur = h_next, s_next

        imagined_rewards = torch.stack(imagined_rewards)    # (ROLLOUT_LEN,)
        imagined_latents = torch.stack(imagined_latents)    # (ROLLOUT_LEN, LATENT_DIM)

        # Reward correlation: imagined vs actual rewards over rollout
        actual_rew = rew_seq[:ROLLOUT_LEN]
        dreamer_rew_corr.append(pearson_rho(imagined_rewards.cpu(), actual_rew.cpu()))

        # PSNR at specified steps
        for step in PSNR_STEPS:
            if step <= T and step <= ROLLOUT_LEN:
                gt = obs_seq[step - 1]           # ground-truth frame at this step
                pred_frame = imagined_frames[step - 1]
                dreamer_psnr[step].append(psnr_fn(pred_frame.unsqueeze(0), gt.unsqueeze(0)))

        # Latent drift: L2 between imagined s_t and encoded obs_t
        for step in DRIFT_STEPS:
            if step <= T and step <= ROLLOUT_LEN:
                gt_latent = mu_all[step - 1]     # (LATENT_DIM,)
                img_latent = imagined_latents[step - 1]
                drift = (img_latent - gt_latent).norm().item()
                dreamer_drift[step].append(drift)

dreamer_psnr_mean  = {s: float(np.mean(dreamer_psnr[s]))  for s in PSNR_STEPS}
dreamer_drift_mean = {s: float(np.mean(dreamer_drift[s])) for s in DRIFT_STEPS}
dreamer_rho        = float(np.mean(dreamer_rew_corr))

print('Dreamer metrics computed.')
print(f'  Reward correlation rho : {dreamer_rho:.4f}')
print(f'  PSNR@1  : {dreamer_psnr_mean[1]:.2f} dB')
print(f'  PSNR@5  : {dreamer_psnr_mean[5]:.2f} dB')
print(f'  PSNR@10 : {dreamer_psnr_mean[10]:.2f} dB')
print(f'  Latent drift@10 : {dreamer_drift_mean[10]:.4f}')
```
Dreamer 수치를 기록했으니, Transformer 기준선에 대해서도 같은 평가를 실행해 결과를 나란히 정렬합니다. 이 셀은 `trans_tok_loss`를 **teacher forcing** 아래에서 계산한다는 점에 유의하세요(`transformer(z_seq_in, a_seq_in)`은 모든 위치에서 실제 토큰 시퀀스를 컨텍스트로 받으며, 이는 P04에서 Transformer가 학습된 방식과 일치합니다). 하지만 `trans_psnr`과 `trans_drift`는 **완전한 자기회귀** 롤아웃에서 계산됩니다(`z_context = torch.cat([z_context, next_z.unsqueeze(0)], dim=1)`는 스텝 0 이후로는 실제값이 아니라 모델 자신이 예측한 토큰으로 매 스텝마다 컨텍스트를 확장합니다). 둘 다 보고하는 것은 의도적입니다. teacher forcing 손실과 개방루프 PSNR/드리프트 수치 사이의 격차가 바로 [L04의 STORM 페이지](../lectures/lecture-04-evaluation-by-model/04-storm-diffusion-drift#long-horizon-psnr)가 자기회귀 월드모델의 주된 실패 모드로 꼽는 **teacher forcing 격차**이며, 아래 8절이 이 격차를 명시적으로 다시 다룹니다.

```python
# Transformer metrics.

trans_psnr      = {s: [] for s in PSNR_STEPS}
trans_drift     = {s: [] for s in DRIFT_STEPS}
trans_tok_loss  = []

with torch.no_grad():
    for traj_i in range(N_EVAL_TRAJ):
        obs_seq = eval_obs[traj_i].to(DEVICE)   # (T, 3, 64, 64)
        act_seq = eval_act[traj_i].to(DEVICE)   # (T,)
        T = obs_seq.shape[0]

        # Encode all frames to discrete tokens (ground-truth latents)
        logits_all = catvae.encoder(obs_seq)                         # (T, K)
        idx_all    = logits_all.argmax(-1)                           # (T,)
        z_oh_all   = F.one_hot(idx_all, num_classes=N_CATEGORIES).float()  # (T, K)

        # Token prediction loss over the full trajectory (teacher-forced)
        z_seq_in  = z_oh_all.unsqueeze(0)        # (1, T, K)
        a_seq_in  = act_seq.unsqueeze(0)         # (1, T)
        tok_logits, _, _ = transformer(z_seq_in, a_seq_in)  # (1, T, K)
        # Predict token t+1 from position t
        target_idx = idx_all[1:]                 # (T-1,)
        pred_logits = tok_logits[0, :-1, :]     # (T-1, K)
        loss_val = F.cross_entropy(pred_logits, target_idx).item()
        trans_tok_loss.append(loss_val)

        # Autoregressive rollout from step 0 (no teacher forcing)
        z_context = z_oh_all[0:1].unsqueeze(0)  # (1, 1, K)  -- seed with obs[0]
        imagined_z  = []   # one-hot latent at each imagined step
        imagined_f  = []   # decoded frames

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
            z_context = torch.cat([z_context, next_z.unsqueeze(0)], dim=1)  # extend

        imagined_z = torch.stack(imagined_z)   # (ROLLOUT_LEN, K)

        # PSNR
        for step in PSNR_STEPS:
            if step <= T and step <= ROLLOUT_LEN:
                gt = obs_seq[step - 1]
                pred_frame = imagined_f[step - 1]
                trans_psnr[step].append(psnr_fn(pred_frame.unsqueeze(0), gt.unsqueeze(0)))

        # Latent drift: L2 between imagined one-hot and real one-hot
        for step in DRIFT_STEPS:
            if step <= T and step <= ROLLOUT_LEN:
                gt_z    = z_oh_all[step - 1]       # (K,)
                img_z   = imagined_z[step - 1]     # (K,)
                drift   = (img_z - gt_z).norm().item()
                trans_drift[step].append(drift)

trans_psnr_mean  = {s: float(np.mean(trans_psnr[s]))  for s in PSNR_STEPS}
trans_drift_mean = {s: float(np.mean(trans_drift[s])) for s in DRIFT_STEPS}
trans_tok_mean   = float(np.mean(trans_tok_loss))

print('Transformer metrics computed.')
print(f'  Token prediction loss : {trans_tok_mean:.4f}')
print(f'  PSNR@1  : {trans_psnr_mean[1]:.2f} dB')
print(f'  PSNR@5  : {trans_psnr_mean[5]:.2f} dB')
print(f'  PSNR@10 : {trans_psnr_mean[10]:.2f} dB')
print(f'  Latent drift@10 : {trans_drift_mean[10]:.4f}')
```
## 5. 요약 지표 표

두 모델의 핵심 지표를 하나의 표에 모읍니다.

아래 표는 두 모델에 대해 계산된 모든 지표를 모읍니다. `N/A`로 표시된 항목은
해당 아키텍처에서 개념적으로 정의되지 않는 지표를 나타냅니다. 보상 상관관계는
RSSM 안의 명시적인 보상 헤드가 필요하고, 토큰 예측 손실은 Transformer에서처럼
이산 범주형 잠재값이 필요합니다.

이 `N/A` 처리 자체가 [L04의 도입 원칙](../lectures/lecture-04-evaluation-by-model/00-diagnostic-framework#a-model-independent-diagnostic-framework)을 보여주는 작은 예시입니다. "단일 지표로 여섯 계층을 모두 다룰 수는 없다." 모든 아키텍처를 같은 지표 집합에 억지로 끼워 맞추면(예를 들어 토큰이 없는 Dreamer의 연속 잠재값에 대해 토큰 예측 손실을 계산하면) 비교 가능해 보이지만 실제로는 아무것도 측정하지 않는 수치가 나옵니다.

```python
# Print summary metrics table.
header = f"{'Model':<15} | {'PSNR@1':>8} | {'PSNR@5':>8} | {'PSNR@10':>9} | {'LatentDrift@10':>14} | {'RewardCorr':>11} | {'TokenLoss':>10}"
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
## 6. 나란히 배치한 지표 그래프

네 패널이 L04 지표를 시각적으로 보여줍니다.

1. 지평선 스텝에 대한 PSNR(두 모델을 같은 축에)
2. 스텝에 대한 잠재 드리프트(L2 노름, 두 모델)
3. Dreamer의 보상 상관관계 rho(막대 그래프)
4. Transformer의 토큰 예측 손실(막대 그래프)

패널 1과 2는 두 모델을 공유 축에 배치해 저하되는 *속도*를 비교할 수 있게 합니다. 이는 P02와 P04가 각자의 아키텍처 비교에서 만들었던 것과 같은 지평선 대 오차 곡선 진단입니다. 패널 3과 4는 5절에서 언급했듯 아키텍처 간 대응되는 것이 없어 그래프로 견줄 수 없으므로, 아키텍처별 지표로 따로 표시됩니다.

```python
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle('P05: World Model Evaluation Dashboard', fontsize=14, fontweight='bold')

# Plot 1: PSNR vs horizon.
ax = axes[0, 0]
d_psnr_vals = [dreamer_psnr_mean[s] for s in PSNR_STEPS]
t_psnr_vals = [trans_psnr_mean[s]   for s in PSNR_STEPS]
ax.plot(PSNR_STEPS, d_psnr_vals, 'o-', color='royalblue',  linewidth=2, markersize=7, label='Dreamer (RSSM)')
ax.plot(PSNR_STEPS, t_psnr_vals, 's-', color='tomato',     linewidth=2, markersize=7, label='Transformer')
ax.set_xlabel('Horizon step')
ax.set_ylabel('PSNR (dB)')
ax.set_title('Long-horizon PSNR')
ax.set_xticks(PSNR_STEPS)
ax.legend()
ax.grid(True, alpha=0.3)

# --- Plot 2: Latent drift vs step ---
ax = axes[0, 1]
d_drift_vals = [dreamer_drift_mean[s] for s in DRIFT_STEPS]
t_drift_vals = [trans_drift_mean[s]   for s in DRIFT_STEPS]
ax.plot(DRIFT_STEPS, d_drift_vals, 'o-', color='royalblue', linewidth=2, markersize=5, label='Dreamer (RSSM)')
ax.plot(DRIFT_STEPS, t_drift_vals, 's-', color='tomato',    linewidth=2, markersize=5, label='Transformer')
ax.set_xlabel('Step')
ax.set_ylabel('L2 distance (imagined vs real latent)')
ax.set_title('Latent Drift vs Step')
ax.legend()
ax.grid(True, alpha=0.3)

# --- Plot 3: Reward correlation (Dreamer) ---
ax = axes[1, 0]
per_traj_rho = dreamer_rew_corr
ax.bar(['Dreamer (RSSM)'], [dreamer_rho], color='royalblue', alpha=0.85, width=0.4)
ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
ax.set_ylim(-1.1, 1.1)
ax.set_ylabel('Pearson rho')
ax.set_title('Reward Correlation (10-step rollout)')
# Overlay individual trajectory values as scatter
ax.scatter(
    np.zeros(len(per_traj_rho)),
    per_traj_rho, color='steelblue', alpha=0.5, zorder=3, s=30
)
ax.text(0, dreamer_rho + 0.05, f'mean={dreamer_rho:.3f}', ha='center', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# --- Plot 4: Token prediction loss (Transformer) ---
ax = axes[1, 1]
ax.bar(['Transformer'], [trans_tok_mean], color='tomato', alpha=0.85, width=0.4)
ax.set_ylabel('Cross-entropy loss')
ax.set_title('Token Prediction Loss (teacher-forced)')
ax.scatter(
    np.zeros(len(trans_tok_loss)),
    trans_tok_loss, color='firebrick', alpha=0.5, zorder=3, s=30
)
ax.text(0, trans_tok_mean + 0.02, f'mean={trans_tok_mean:.3f}', ha='center', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```
## 7. 디코딩된 프레임 시퀀스: 나란히 비교

3행짜리 이미지 격자가 롤아웃 스텝 1, 5, 10, 마지막 스텝(스텝 20, 즉 궤적의
마지막 프레임)에서 실제 관측과 각 모델의 상상된 프레임을 비교합니다. 이렇게 하면
PSNR 저하가 한눈에 보입니다.

이는 P02의 롤아웃 격자와 P04의 이미지 비교가 모두 의존했던 것과 같은 "수치만으로는 실패 모드가 가려지고, 이미지가 어떤 종류의 오차인지 드러낸다"는 원칙입니다. PSNR 곡선은 예측 품질이 *얼마나* 떨어졌는지 말해주지만, 각 스텝의 디코딩된 프레임은 *무엇이* 떨어졌는지(흐림, 색상 드리프트, 위치 드리프트) 보여주며, 이는 단일 스칼라 값으로는 구분할 수 없습니다.

```python
DISPLAY_STEPS = [1, 5, 10, SEQ_LEN - 1]  # 1-indexed except the last
TRAJ_IDX = 0  # use the first evaluation trajectory for visualization

obs_seq_vis = eval_obs[TRAJ_IDX].to(DEVICE)  # (T, 3, 64, 64)
act_seq_vis = eval_act[TRAJ_IDX].to(DEVICE)  # (T,)
T_vis = obs_seq_vis.shape[0]

with torch.no_grad():
    # --- Dreamer imagined frames ---
    h_vis = torch.zeros(1, rssm.hidden_dim, device=DEVICE)
    mu0, _ = encoder.forward(obs_seq_vis[0:1])
    s_vis  = mu0
    dreamer_vis_frames = []
    for t in range(max(DISPLAY_STEPS)):
        a_t = act_seq_vis[t:t+1]
        s_vis, h_vis = rssm.prior_step(h_vis, a_t)
        frame = decoder(torch.cat([h_vis, s_vis], dim=-1))
        dreamer_vis_frames.append(frame.squeeze(0).cpu())

    # --- Transformer imagined frames ---
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

# --- Build the 3-row grid ---
n_cols = len(DISPLAY_STEPS)
fig, axes = plt.subplots(3, n_cols, figsize=(3.5 * n_cols, 10))
row_labels = ['Ground Truth', 'Dreamer (RSSM)', 'Transformer']

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
            ax.set_title(f'Step {step}', fontsize=11)

plt.suptitle('Imagined Rollouts: Ground Truth / Dreamer / Transformer', fontsize=13, y=1.01)
plt.tight_layout()
plt.show()
```
## 8. 진단 요약

주요 실패 모드와 트레이드오프를 요약합니다.

### PSNR 저하

두 모델 모두 상상 중에는 미래 관측에 접근할 수 없으므로 지평선이 늘어날수록 PSNR이
떨어집니다. RSSM의 연속 가우시안 확률 상태는 프레임 사이를 매끄럽게 보간하는데, 그 때문에
단기 지평선에서는 더 흐릿하지만 수치상으로는 더 가까운 재구성을 만드는
경향이 있습니다. Transformer의 이산 토큰 병목은 시드 토큰을 신뢰할 수 있는
스텝 1에서는 더 선명하지만, 오차가 빠르게 누적됩니다. 잘못 예측된 토큰 하나하나가
다음 토큰의 컨텍스트가 되기 때문입니다. 체크포인트가 없으면 대체 경로가 무작위
가중치를 사용하는데, 스모크 테스트에는 유용하지만 배포용 지표를 비교하는 데는
쓸 수 없습니다.

### 잠재 드리프트와 오차의 누적

잠재 드리프트(상상된 잠재 벡터와 실제 잠재 벡터 사이의 L2 거리)는 두 아키텍처
모두에서 스텝이 늘어날수록 단조롭게 증가합니다. RSSM에서는 GRU의 재귀가 매끄러움 편향을 만들어, 은닉 상태가 불연속적으로 튀지 않고 점진적으로 드리프트합니다.
Transformer에서는 잘못 분류된 토큰 하나가 전체 컨텍스트 창을 바꾸고, 그것이 이후의
모든 예측에 편향을 주기 때문에 드리프트가 더 큰 경우가 많습니다. 이는 DreamerV3가
학습 중 사전분포와 사후분포를 섞어 대응하는 바로 그 오차 누적 문제입니다.

### RSSM 전용 지표로서의 보상 상관관계

보상 상관관계는 Dreamer 모델에서만 정의됩니다. RSSM에는 Actor-Critic 손실로
처음부터 끝까지 함께 학습되는 전용 보상 헤드가 있기 때문입니다. Transformer도
부차적인 과제로 보상을 예측하지만 정책 그래디언트 신호로 학습되지는 않으므로,
그 보상 예측은 계획에 쓰이지 않으며 이 지표에서 제외됩니다.

### teacher forcing 격차

토큰 예측 손실은 teacher forcing 아래에서 측정되는데, 이는 모델이 모든 스텝에서 실제 토큰을 컨텍스트로 받는다는 뜻입니다. 추론 시점에는 모델이 자신의 이전 예측에 조건화해야 하므로
분포 이동이 발생합니다. teacher forcing 손실과 개방루프 PSNR 사이의 격차가 이 이동의
크기를 드러냅니다. 격차가 클수록 모델이 실제 컨텍스트에 과적합되어 자유 롤아웃에서
더 급격히 저하된다는 뜻입니다.

```python
# Final inline summary table.
print('=' * 90)
print('P05 FINAL METRICS SUMMARY')
print('=' * 90)
header2 = (
    f"{'Model':<14} | "
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
print('Notes:')
print('  PSNR in dB (higher is better); Latent Drift in L2 norm (lower is better)')
print('  RewardCorr: Pearson rho over 10-step imagined rollout (Dreamer only)')
print('  TokenLoss: cross-entropy under teacher forcing (Transformer only)')
print(f'  Models loaded from checkpoint: Dreamer={dreamer_loaded}, Transformer={trans_loaded}')
```
