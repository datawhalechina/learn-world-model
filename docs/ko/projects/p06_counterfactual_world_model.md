---
title: P06 반사실적 동작 조건화 월드모델
---

# P06: 반사실적 동작 조건화 월드모델

P01부터 P05까지는 하나의 질문을 던집니다. 모델이 실제를 얼마나 정확하게 재구성하는가? 이 프로젝트는 다른 질문을 던집니다. 모델의 예측이 정말로 동작에 의존하는가, 아니면 어떤 과거 다음에 어떤 미래가 오는지를 그저 암기한 것뿐인가? 동작을 무시하는 모델은 월드모델의 옷을 입은 비디오 예측기일 뿐입니다. 이 둘을 가르는 검증은 반사실적입니다. 과거를 고정한 채 동작만 바꾸고, 상상된 미래가 실제로 달라지는지 확인하는 것입니다.

이 프로젝트는 주디아 펄(Judea Pearl)의 인과의 사다리를 중심으로 구성됩니다. 맨 아래 단은 연관 `P(Y | X)`로, 일반적인 시퀀스 모델이 학습하는 것입니다. 가운데 단은 개입 `P(Y | do(a))`로, 동작을 관측하는 대신 *설정*했을 때 뒤따르는 미래입니다. 맨 위 단은 반사실입니다. 실제로 일어난 궤적이 주어졌을 때, 에이전트가 다르게 행동했다면 나머지는 모두 고정한 채 무슨 일이 *일어났을지*를 묻습니다. P02와 P03에서 나온 세 가지 RSSM 수식, 즉 상태 전이, 관측, 보상이 구조 모델의 역할을 하므로, 이 사다리는 우리가 이미 구축한 월드모델에 그대로 적용됩니다.

미리 정직하게 밝혀둘 것이 하나 있습니다. 펄의 do-연산은 동작이 숨은 교란 요인과 얽혀 있을 때 관측 데이터로부터 `P(Y | do(a))`를 복원할 수 있는지를 판단하기 위해 존재합니다. 그 문제는 여기서는 일어나지 않습니다. 동작은 우리가 선택하는 외생 입력이므로, 우리가 선택한 동작과 잠재 상태 사이에는 후문 경로(back-door path)가 없습니다. 그래서 개입 `do(a_t = a')`은 그래프 수술이 아니라 롤아웃에서 동작을 고정하는 방식으로, 실용적으로 구현됩니다. do 표기법을 쓰는 이유는 그것이 올바른 개념을 정확히 지칭하기 때문이며, 그 적용 범위를 정직하게 유지합니다.

**선행 조건**: P03(`dreamer.pt`)과 P04(`transformer_wm.pt`)가 있으면 사용합니다. 없으면 누락된 체크포인트마다 무작위로 초기화된 모델로 대체되어 노트북은 여전히 스모크 테스트로 실행됩니다. 반사실적 비교는 실제 체크포인트를 불러왔을 때만 의미가 있습니다. 이 노트북은 작은 동작 정규화 월드모델 하나를 학습시켜 `causal_wm.pt`로 저장합니다.

> Notebook 원본: [p06_counterfactual_world_model.ipynb](https://github.com/datawhalechina/learn-world-model/blob/main/docs/ko/projects/p06_counterfactual_world_model.ipynb)

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

앞선 프로젝트들과 정확히 같은 방식으로 런타임을 임포트하고, 시드를 고정하고, 디바이스를 결정합니다.

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


def optimizer_step(optimizer, scaler=None):
    if USE_TPU:
        xm.optimizer_step(optimizer)
    elif scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

# Shared hyperparameters: must match P03 and P04
HIDDEN_DIM   = 128
LATENT_DIM   = 32
N_CATEGORIES = 32
N_ACTIONS    = 2
SEQ_LEN      = 20
ROLLOUT_LEN  = 10

PATH         = Path('.')
DREAMER_CKPT = PATH / 'dreamer.pt'
TRANS_CKPT   = PATH / 'transformer_wm.pt'
CAUSAL_CKPT  = PATH / 'causal_wm.pt'

print('Device:', DEVICE)
print('PyTorch version:', torch.__version__)
print('Dreamer checkpoint exists    :', DREAMER_CKPT.exists())
print('Transformer checkpoint exists:', TRANS_CKPT.exists())
```
## 2. 환경과 홀드아웃 궤적

P05의 환경을 재사용해, 반사실적 분석이 앞선 평가와 같은 데이터 분포에서 이루어지도록 합니다. 동작은 이진값입니다. 오른쪽으로 밀거나 왼쪽으로 밉니다.

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


def generate_eval_trajectories(n_traj=20, horizon=SEQ_LEN, base_seed=999):
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
        obs_list.append(traj_obs[:horizon])
        act_list.append(traj_act)
        rew_list.append(traj_rew)
    obs_t = torch.tensor(np.array(obs_list), dtype=torch.float32)
    act_t = torch.tensor(np.array(act_list), dtype=torch.long)
    rew_t = torch.tensor(np.array(rew_list), dtype=torch.float32)
    return obs_t, act_t, rew_t


eval_obs, eval_act, eval_rew = generate_eval_trajectories()
print('eval_obs:', eval_obs.shape, '  eval_act:', eval_act.shape)
```
## 3. P03과 P04 모델 불러오기

P03, P04와 같은 차원으로 Dreamer와 Transformer 구성 요소를 인라인으로 정의한 뒤 두 체크포인트를 모두 불러옵니다. 체크포인트가 없으면 무작위 가중치로 대체되어 노트북은 여전히 처음부터 끝까지 실행됩니다.

```python
class Encoder(nn.Module):
    """CNN encoder: 3x64x64 frame -> LATENT_DIM mean and log-var."""
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu     = nn.Linear(4096, latent_dim)
        self.fc_logvar = nn.Linear(4096, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """Transposed-CNN decoder: (HIDDEN_DIM + LATENT_DIM) -> 3x64x64."""
    def __init__(self, in_dim=HIDDEN_DIM + LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(in_dim, 256 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
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

    def prior(self, h):
        mu, lv = self.prior_net(h).chunk(2, dim=-1)
        std = F.softplus(lv) + 0.1
        return mu + std * torch.randn_like(std), mu, std

    def posterior(self, h, z_obs):
        mu, lv = self.post_net(torch.cat([h, z_obs], dim=-1)).chunk(2, dim=-1)
        std = F.softplus(lv) + 0.1
        return mu + std * torch.randn_like(std), mu, std

    def prior_step(self, h, a):
        s, _, _ = self.prior(h)
        h_next = self.gru(torch.cat([s, self._action_feature(a)], dim=-1), h)
        return s, h_next

    def posterior_step(self, h, a, z_obs):
        s, _, _ = self.posterior(h, z_obs)
        h_next = self.gru(torch.cat([s, self._action_feature(a)], dim=-1), h)
        return s, h_next


def straight_through_gumbel(logits, tau=1.0):
    y_soft = F.gumbel_softmax(logits, tau=tau, hard=False)
    y_hard = F.one_hot(y_soft.argmax(-1), num_classes=logits.shape[-1]).float()
    return (y_hard - y_soft).detach() + y_soft


class CatVAEEncoder(nn.Module):
    def __init__(self, num_categories=N_CATEGORIES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256), nn.ReLU(),
            nn.Linear(256, num_categories),
        )

    def forward(self, x):
        return self.net(x)


class CatVAEDecoder(nn.Module):
    def __init__(self, num_categories=N_CATEGORIES):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_categories, 256), nn.ReLU(),
            nn.Linear(256, 256 * 4 * 4), nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
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


class CausalTransformerWM(nn.Module):
    def __init__(self, num_categories=N_CATEGORIES, d_model=HIDDEN_DIM,
                 n_heads=4, n_layers=2, n_actions=N_ACTIONS, max_len=SEQ_LEN):
        super().__init__()
        self.d_model = d_model
        self.num_categories = num_categories
        self.z_proj  = nn.Linear(num_categories, d_model)
        self.a_embed = nn.Embedding(n_actions, d_model)
        pos = torch.arange(max_len * 2).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe  = torch.zeros(max_len * 2, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
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


encoder     = Encoder(LATENT_DIM).to(DEVICE)
decoder     = Decoder(HIDDEN_DIM + LATENT_DIM).to(DEVICE)
rssm        = RSSM(LATENT_DIM, HIDDEN_DIM, N_ACTIONS).to(DEVICE)
catvae      = CatVAE(N_CATEGORIES).to(DEVICE)
transformer = CausalTransformerWM().to(DEVICE)

dreamer_loaded = False
trans_loaded   = False

if DREAMER_CKPT.exists():
    try:
        ckpt = torch.load(DREAMER_CKPT, map_location=DEVICE, weights_only=False)
        if isinstance(ckpt, dict):
            if 'encoder' in ckpt:
                encoder.load_state_dict({k.replace('conv.', 'net.'): v for k, v in ckpt['encoder'].items()}, strict=True)
            if 'decoder' in ckpt:
                decoder.load_state_dict({k.replace('deconv.', 'net.'): v for k, v in ckpt['decoder'].items()}, strict=True)
            if 'rssm' in ckpt:
                rssm.load_state_dict(ckpt['rssm'], strict=True)
        dreamer_loaded = True
        print(f'Loaded Dreamer checkpoint from {DREAMER_CKPT}')
    except Exception as e:
        print(f'Could not load Dreamer checkpoint ({e}). Using random initialization.')
else:
    print('dreamer.pt not found. Using randomly initialized Dreamer.')

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

for m in [encoder, decoder, rssm, catvae, transformer]:
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)

print(f'Dreamer loaded: {dreamer_loaded}   Transformer loaded: {trans_loaded}')
```
## 4. 둘째 단: 개입적 롤아웃

첫 번째 반사실적 검증은 개입적 검증입니다. 궤적 시작 시점의 잠재 상태를 고정한 뒤, 두 가지 다른 고정 동작 시퀀스, `do(a = 항상 오른쪽)`과 `do(a = 항상 왼쪽)` 아래에서 모델을 두 번 앞으로 굴립니다. 모델이 동작에 조건화되어 있다면 두 상상된 미래는 서로 갈라집니다. 그렇지 않다면 동작과 무관하게 두 미래는 서로 겹칩니다.

`dreamer_rollout`은 P02~P05 전체에서 순수 상상에 쓴 것과 같은 `rssm.prior_step`으로 이를 구현하지만, 여기서 `action_seq`는 어떤 정책이 샘플링하거나 환경에서 읽어온 것이 아닙니다. `right = [0] * ROLLOUT_LEN`과 `left = [1] * ROLLOUT_LEN`은 실험자가 정해서 고정한 동작 시퀀스이며, 이는 노트북 첫 셀에서 밝혔듯 `do(a_t = a')`이 실용적으로 의미하는 것과 정확히 같습니다. 관측된 동작에 조건화하는 것이 아니라 동작 자체를 설정하는 것입니다. `transformer_rollout`은 같은 개입을 P04의 Transformer를 통해 실행하되, 독스트링에 명시된 한 가지 조정이 있습니다. Transformer의 롤아웃은 매 스텝마다 이산적인 `argmax` 토큰으로 딱 떨어지므로, 같은 최고 확률 토큰에 대해 단지 *다른 확률*만 만들어내는 두 동작 체제는 차이가 0이라고 잘못 보고될 것입니다. `token_probs`(argmax 이전의 소프트맥스 분포)를 보존해두면 다음 셀의 발산 측정이 그 약한 신호를 이산화로 잃지 않고 볼 수 있습니다.

```python
@torch.no_grad()
def dreamer_rollout(action_seq, seed_obs):
    """Roll the RSSM forward from seed_obs under a fixed action sequence."""
    h = torch.zeros(1, rssm.hidden_dim, device=DEVICE)
    mu0, _ = encoder(seed_obs.unsqueeze(0).to(DEVICE))
    s = mu0
    latents, frames = [], []
    for a in action_seq:
        a_t = torch.tensor([a], device=DEVICE)
        s, h = rssm.prior_step(h, a_t)
        latents.append(s.squeeze(0))
        frames.append(decoder(torch.cat([h, s], dim=-1)).squeeze(0))
    return torch.stack(latents), torch.stack(frames)


@torch.no_grad()
def transformer_rollout(action_seq, seed_obs):
    """Autoregressively roll the Transformer forward under a fixed action sequence.

    Returns decoded frames and the per-step categorical token distribution. The
    argmax frames can land on identical tokens when the action signal is weak, so
    the soft token distribution is kept to measure divergence without that snap.
    """
    logits0 = catvae.encoder(seed_obs.unsqueeze(0).to(DEVICE))
    z0 = F.one_hot(logits0.argmax(-1), num_classes=N_CATEGORIES).float()
    z_ctx = z0.unsqueeze(0)
    a_full = torch.tensor(action_seq, device=DEVICE)
    frames, token_probs = [], []
    for t in range(len(action_seq)):
        a_prefix = a_full[:z_ctx.shape[1]].unsqueeze(0)
        tok_out, _, _ = transformer(z_ctx, a_prefix)
        probs = F.softmax(tok_out[0, -1, :], dim=-1)
        token_probs.append(probs)
        next_z = F.one_hot(probs.argmax().unsqueeze(0), num_classes=N_CATEGORIES).float()
        frames.append(catvae.decoder(next_z).squeeze(0))
        z_ctx = torch.cat([z_ctx, next_z.unsqueeze(0)], dim=1)
    return torch.stack(frames), torch.stack(token_probs)


seed = eval_obs[0, 0]                       # fix the past: one starting observation
right = [0] * ROLLOUT_LEN                    # do(a = always right)
left  = [1] * ROLLOUT_LEN                    # do(a = always left)

d_lat_r, d_frm_r = dreamer_rollout(right, seed)
d_lat_l, d_frm_l = dreamer_rollout(left,  seed)
t_frm_r, t_prob_r = transformer_rollout(right, seed)
t_frm_l, t_prob_l = transformer_rollout(left,  seed)

print('Interventional rollouts computed for both models.')
```
두 개입적 롤아웃을 모두 얻었으니, 각 지평선 스텝에서 두 동작 체제가 얼마나 멀어지는지 측정합니다. Dreamer에 대해서는 디코딩된 프레임 사이의 픽셀 수준 RMS 차이를 사용합니다. Transformer의 경우 디코딩된 프레임은 범주형 토큰에 대한 `argmax`를 거치는데, 이는 미세한 동작 신호가 있어도 두 동작 체제를 같은 토큰으로 딱 떨어뜨려 정확히 0으로 보고합니다. 그 신호를 보려면 대신 두 예측 토큰 분포 사이의 대칭 KL 발산을 측정합니다. 인과적인 모델은 이 곡선들을 위로 끌어올립니다. 동작을 못 보는 모델은 곡선을 바닥 근처에 묶어둡니다.

구체적으로: `frame_divergence`는 두 디코딩된 프레임 시퀀스(Dreamer의 `d_frm_r` 대 `d_frm_l`) 사이의 평균제곱근 픽셀 차이를 계산해, "두 고정 동작 롤아웃이 눈에 띄게 다른 이미지를 만드는가"에 직접 답합니다. Transformer의 발산은 대신 `t_prob_r`과 `t_prob_l`, 즉 32개 가능한 토큰에 대한 스텝별 두 범주형 분포를 대칭 KL로 비교하는데, 이는 두 분포의 `argmax`가 우연히 일치하더라도 여전히 정보를 담고 있습니다. 어느 모델이든 지평선에 따라 상승하는 곡선은 월드모델의 *예측*이(단순히 학습 손실이 아니라) 동작에 의존한다는 직접적인 시각적 증거입니다. 이는 P01~P05가 한 번도 검증하지 않은 속성입니다. 그 평가들은 과거를 고정한 채 동작만 바꾸는 실험을 한 적이 없기 때문입니다.

```python
def frame_divergence(frames_a, frames_b):
    return [(fa - fb).pow(2).mean().sqrt().item() for fa, fb in zip(frames_a, frames_b)]


def token_kl_divergence(probs_a, probs_b, eps=1e-8):
    """Symmetric KL between two per-step categorical token distributions."""
    pa = probs_a.clamp_min(eps)
    pb = probs_b.clamp_min(eps)
    kl = (pa * (pa / pb).log()).sum(-1) + (pb * (pb / pa).log()).sum(-1)
    return kl.cpu().tolist()


d_div = frame_divergence(d_frm_r, d_frm_l)       # Dreamer: pixel RMS
t_div = token_kl_divergence(t_prob_r, t_prob_l)  # Transformer: token KL
steps = list(range(1, ROLLOUT_LEN + 1))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(steps, d_div, 'o-', color='royalblue', linewidth=2, label='Dreamer (RSSM)')
ax1.set_xlabel('Rollout step')
ax1.set_ylabel('RMS frame divergence')
ax1.set_title('Dreamer: do(right) vs do(left)')
ax1.set_xticks(steps)
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(steps, t_div, 's-', color='tomato', linewidth=2, label='Transformer')
ax2.set_xlabel('Rollout step')
ax2.set_ylabel('Symmetric KL of token distribution')
ax2.set_title('Transformer: do(right) vs do(left)')
ax2.set_xticks(steps)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```
## 5. 셋째 단: 귀추(abduction)를 통한 반사실적 롤아웃

개입적 검증은 빈 잠재 상태에서 시작합니다. 반사실적 질문은 더 강력합니다. *실제로* 일어난 궤적을 가져와, 어느 스텝에서 에이전트가 다른 동작을 선택했다면 나머지 세계는 그대로인 채 무슨 일이 일어났을지를 묻습니다. 펄의 처방에는 세 단계가 있습니다. 귀추, 그다음 동작, 그다음 예측입니다.

귀추(abduction, 관측된 결과로부터 그것을 만들어낸 원인을 역으로 추론하는 것)는 실제로 관측된 궤적을 만들어낸 잠재 상태를 추론합니다. RSSM 사후분포가 정확히 이 일을 하므로, 분기점까지 실제 프레임과 동작에 대해 앞으로 실행합니다. 동작은 그 분기점에서 다른 선택으로 대체합니다. 예측은 귀추된 상태로부터 새 동작 아래에서 사전분포를 앞으로 굴립니다. 사실 분기와 반사실 분기는 같은 과거를 공유하고 개입된 동작에서만 차이가 나는데, 바로 이것이 이 비교를 깔끔한 반사실로 만드는 요소입니다.

`abduct_state`는 코드로 구현된 귀추 단계입니다. `branch_t`까지 모든 타임스텝에서 (`prior_step`이 아니라) `rssm.posterior_step`을 호출하는데, 이는 각 스텝에서 *실제* 인코딩된 관측을 사용한다는 뜻이며, 정확히 [L02의 사전분포 대 사후분포 구분](../lectures/lecture-02-encode-and-dynamics/02-dynamics#rssm-separating-deterministic-and-stochastic-components)입니다. 사후분포는 "데이터를 본 뒤 갱신된 추정"이며, 이는 정확히 실제로 관측된 궤적을 만들어낸 잠재 상태를 귀추하는 데 필요한 것입니다. `counterfactual_branch`는 동작과 예측 단계를 합친 것입니다. 귀추된 `(h_b, s_b)`에서 시작해, `rssm.prior_step`만으로(더 이상의 실제 관측 없이) 앞으로 굴립니다. 이는 이 커리큘럼 전체에서 쓰인 것과 같은 상상 메커니즘이지만, 처음부터가 아니라 실제 이력에서 귀추된 상태로 시작한다는 점이 다릅니다.

`cf_actions = [1 - a for a in factual_actions]`가 실제 개입입니다. 분기점 이후 실제로 일어난 모든 이진 동작을 뒤집습니다(0은 1이 되고 1은 0이 됩니다). `factual_frames`와 `cf_frames` 모두 똑같이 귀추된 상태 `h_b, s_b`에서 시작하므로, 그 시점 이후 둘 사이의 어떤 차이든 그 이전에 무슨 일이 있었는지의 차이가 아니라 오직 뒤집힌 동작 하나에만 귀속될 수 있습니다. 이것이 서로 무관한 두 궤적을 비교하는 것과 구별되는, 반사실적 비교의 정의적 속성입니다.

```python
@torch.no_grad()
def abduct_state(obs_seq, act_seq, branch_t):
    """Run the RSSM posterior along the real trajectory up to branch_t, returning (h, s)."""
    h = torch.zeros(1, rssm.hidden_dim, device=DEVICE)
    mu0, _ = encoder(obs_seq[0:1].to(DEVICE))
    s = mu0
    for t in range(branch_t):
        mu_t, _ = encoder(obs_seq[t:t+1].to(DEVICE))
        a_t = act_seq[t:t+1].to(DEVICE)
        s, h = rssm.posterior_step(h, a_t, mu_t)
    return h, s


@torch.no_grad()
def counterfactual_branch(h, action_seq):
    """Predict the prior rollout from an abducted state under a chosen action sequence."""
    frames = []
    for a in action_seq:
        a_t = torch.tensor([a], device=DEVICE)
        s, h = rssm.prior_step(h, a_t)
        frames.append(decoder(torch.cat([h, s], dim=-1)).squeeze(0))
    return torch.stack(frames)


traj_i, BRANCH = 0, 4
obs_seq = eval_obs[traj_i]
act_seq = eval_act[traj_i]

h_b, s_b = abduct_state(obs_seq, act_seq, BRANCH)            # abduction
factual_actions = act_seq[BRANCH:ROLLOUT_LEN].tolist()       # what actually happened next
cf_actions      = [1 - a for a in factual_actions]           # the action, flipped

factual_frames = counterfactual_branch(h_b, factual_actions)
cf_frames      = counterfactual_branch(h_b, cf_actions)

print(f'Abducted state at step {BRANCH}; factual vs counterfactual branches rolled forward.')
```
두 분기를 굴렸으니, 나란히 표시합니다. 공유된 접두부는 구성상 동일하므로, 그 이후에 보이는 어떤 차이든 뒤집힌 동작의 인과적 효과입니다. 특히 갈라지는 지점을 눈여겨보세요. 두 분기는 분기 스텝까지(포함)는 픽셀 단위로 동일해야 하고, 그 직후부터 갈라지기 시작해야 합니다. 바로 그 지점이 `factual_actions`와 `cf_actions`가 처음으로 달라지는 곳이기 때문입니다.

```python
n_show = len(factual_actions)
fig, axes = plt.subplots(2, n_show, figsize=(2.2 * n_show, 4.6))
for col in range(n_show):
    axes[0, col].imshow(np.clip(factual_frames[col].cpu().permute(1, 2, 0).numpy(), 0, 1))
    axes[1, col].imshow(np.clip(cf_frames[col].cpu().permute(1, 2, 0).numpy(), 0, 1))
    for row in range(2):
        axes[row, col].axis('off')
    axes[0, col].set_title(f'+{col + 1}', fontsize=9)
axes[0, 0].set_ylabel('Factual', fontsize=10)
axes[1, 0].set_ylabel('Counterfactual', fontsize=10)
fig.suptitle(f'Counterfactual branch from abducted state (branch at step {BRANCH})', y=1.02)
plt.tight_layout()
plt.show()
```
## 6. 모델을 진짜로 동작-인과적으로 만들기

위의 롤아웃들은 인과적 정확도가 아니라 예측을 위해 학습된 모델을 검사한 것이므로, 재구성에서는 좋은 점수를 받으면서도 조용히 동작을 무시하는 모델일 수 있습니다. World-Action Model 계열 연구는 이를 역동역학 정규화 항으로 해결합니다. 순방향 예측 손실과 나란히, 작은 헤드 하나가 `s_t`와 `s_{t+1}` 사이의 잠재 전이로부터 동작 `a_t`를 복원해야 합니다. 전이로부터 동작을 다시 읽어낼 수 없다면, 동역학은 실제로 그 동작에 조건화되어 있지 않은 것입니다. 이 손실을 추가하면 잠재 전이가 동작의 효과를 담아내도록 강제됩니다.

합성 환경에서 컴팩트한 월드모델 두 개를 학습시킵니다. 하나는 역동역학 항을 넣고, 하나는 넣지 않은 채로, 각각이 얼마나 동작에 민감해지는지 비교합니다.

`CompactWM.inv`가 역동역학 헤드 그 자체입니다. `(s_t, s_next)`가 주어지면 그 특정 전이를 일으킨 동작 `a_t`가 무엇인지 예측해야 하는데, 이는 보통의 순방향 예측 방향을 뒤집은 것입니다. `train_compact_wm`에서 `fwd_loss = F.mse_loss(pred_next, s_next...)`는 두 모델이 공유하는 일반적인 순방향 동역학 손실이고, `use_inverse` 분기는 동작 정규화 모델에 대해서만 그 위에 `lam * F.cross_entropy(a_logits, a_t...)`를 더합니다. 이것이 작동하는 이유는 다음과 같습니다. 인코더가 동작의 효과를 뭉개도록(서로 다른 동작을 비슷한 `s_next`로 매핑하도록) 학습했다면, 역동역학 헤드는 `a_t`를 안정적으로 되짚어낼 수 없으므로 그 교차 엔트로피 손실은 높게 유지되고, 이는 다시 인코더와 순방향 동역학 네트워크를 동작의 효과가 잠재 공간에서 실제로 구분되는 표현 쪽으로 밀어붙이는 그래디언트를 만들어냅니다. 이는 [L03의 WAM 절](../lectures/lecture-03-architecture-patterns/04-architectures-loopwm-wam#architecture-nine-from-world-model-to-world-action-model-wam)에서 나온 "World-Action Model" 아이디어를 가능한 가장 작은 규모로 적용한 것입니다. 단일 보조 예측 과제를 강제해 동작의 인과적 흔적이 최적화 과정에서 사라지지 않도록 하는 것입니다.

```python
def make_training_set(n_traj=300, horizon=SEQ_LEN, base_seed=0):
    obs_list, act_list = [], []
    for i in range(n_traj):
        env = SyntheticEnv(seed=base_seed + i)
        obs = env.reset()
        traj_obs, traj_act = [obs], []
        rng = np.random.RandomState(base_seed + i + 50000)
        for _ in range(horizon):
            a = rng.randint(0, N_ACTIONS)
            nxt, _, _ = env.step(a)
            traj_act.append(a)
            traj_obs.append(nxt)
        obs_list.append(traj_obs[:horizon])
        act_list.append(traj_act)
    return (torch.tensor(np.array(obs_list), dtype=torch.float32),
            torch.tensor(np.array(act_list), dtype=torch.long))


class CompactWM(nn.Module):
    """Small action-conditioned latent dynamics model with an inverse-dynamics head."""
    def __init__(self, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, n_actions=N_ACTIONS):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),
            nn.Flatten(), nn.Linear(256 * 4 * 4, latent_dim),
        )
        self.a_embed = nn.Embedding(n_actions, latent_dim)
        self.fwd = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # Inverse-dynamics head: recover the action from (s_t, s_{t+1})
        self.inv = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def encode(self, x):
        return self.enc(x)

    def forward_dynamics(self, s, a):
        return self.fwd(torch.cat([s, self.a_embed(a)], dim=-1))

    def inverse_dynamics(self, s, s_next):
        return self.inv(torch.cat([s, s_next], dim=-1))


def train_compact_wm(use_inverse, epochs=8, lam=1.0, seed=0):
    torch.manual_seed(seed)
    model = CompactWM().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    obs, act = make_training_set()
    N, T = act.shape
    for _ in range(epochs):
        perm = torch.randperm(N)
        for i in range(0, N, 32):
            idx = perm[i:i + 32]
            o = obs[idx].to(DEVICE)                       # (B, T, 3, 64, 64)
            a = act[idx].to(DEVICE)                       # (B, T)
            B = o.shape[0]
            s = model.encode(o.reshape(B * T, 3, 64, 64)).reshape(B, T, -1)
            s_t, s_next = s[:, :-1], s[:, 1:]
            a_t = a[:, :-1]
            pred_next = model.forward_dynamics(s_t.reshape(-1, LATENT_DIM), a_t.reshape(-1))
            fwd_loss = F.mse_loss(pred_next, s_next.reshape(-1, LATENT_DIM).detach())
            loss = fwd_loss
            if use_inverse:
                a_logits = model.inverse_dynamics(s_t.reshape(-1, LATENT_DIM), s_next.reshape(-1, LATENT_DIM))
                loss = loss + lam * F.cross_entropy(a_logits, a_t.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


print('Training action-regularized world model...')
causal_wm   = train_compact_wm(use_inverse=True)
print('Training baseline world model (no inverse-dynamics term)...')
baseline_wm = train_compact_wm(use_inverse=False)
print('Both compact models trained.')
```
## 7. 동작 영향력 지표

동작 조건화를 수치로 나타내기 위해, 동작만 뒤집었을 때 예측된 다음 잠재값이 얼마나 바뀌는지를 홀드아웃 상태들에 대해 평균해 측정합니다. 동작을 존중하는 모델은 큰 영향력 점수를 만들어내고, 동작이 뭉개진 모델은 0에 가까운 점수를 만들어냅니다. 동작 정규화 모델, 기준선 모델, 그리고 불러온 Dreamer RSSM에 대해 이를 보고합니다.

`action_influence_compact`는 상태당 정확히 하나의 고정 개입 비교를 계산합니다. `s0 = model.forward_dynamics(s, a0)`와 `s1 = model.forward_dynamics(s, a1)`는 *같은* 인코딩된 상태 `s`로부터 `do(a=0)`과 `do(a=1)` 아래에서의 다음 잠재값을 예측하고, `(s0 - s1).pow(2).sum(-1).sqrt().mean()`은 그 두 예측 사이의 평균 유클리드 거리입니다. 이는 4절의 개입적 발산 곡선을 1스텝, 단일 수치 버전으로 만든 것입니다. 10스텝 롤아웃에 대한 곡선 대신, 256개의 홀드아웃 상태에 대해 평균한 단일 스칼라 값으로, 전체 곡선을 보기 전에 세 모델을 빠르게 헤드라인 비교하는 데 유용합니다.

```python
@torch.no_grad()
def action_influence_compact(model, obs, n=256):
    s = model.encode(obs[:n].to(DEVICE))
    a0 = torch.zeros(s.shape[0], dtype=torch.long, device=DEVICE)
    a1 = torch.ones_like(a0)
    s0 = model.forward_dynamics(s, a0)
    s1 = model.forward_dynamics(s, a1)
    return (s0 - s1).pow(2).sum(-1).sqrt().mean().item()


@torch.no_grad()
def action_influence_rssm(obs, n=256):
    h = torch.zeros(min(n, obs.shape[0]), rssm.hidden_dim, device=DEVICE)
    a0 = torch.zeros(h.shape[0], dtype=torch.long, device=DEVICE)
    a1 = torch.ones_like(a0)
    _, h0 = rssm.prior_step(h, a0)
    _, h1 = rssm.prior_step(h, a1)
    return (h0 - h1).pow(2).sum(-1).sqrt().mean().item()


flat_obs = eval_obs.reshape(-1, 3, 64, 64)
infl_causal   = action_influence_compact(causal_wm, flat_obs)
infl_baseline = action_influence_compact(baseline_wm, flat_obs)
infl_rssm     = action_influence_rssm(flat_obs)

print(f'Action influence (higher = more action-conditioned):')
print(f'  Action-regularized WM : {infl_causal:.4f}')
print(f'  Baseline WM           : {infl_baseline:.4f}')
print(f'  Dreamer RSSM (P03)    : {infl_rssm:.4f}')

fig, ax = plt.subplots(figsize=(6, 4))
names = ['Action-reg WM', 'Baseline WM', 'Dreamer RSSM']
vals  = [infl_causal, infl_baseline, infl_rssm]
ax.bar(names, vals, color=['seagreen', 'gray', 'royalblue'], alpha=0.85)
ax.set_ylabel('Action influence (L2 of latent delta)')
ax.set_title('Does the Model Respond to the Action?')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()
```
위의 스칼라는 1스텝 요약입니다. 효과가 누적되는 모습을 보려면, 동작 정규화 모델과 기준선 모델을 같은 인코딩된 상태에서 `do(오른쪽)` 대 `do(왼쪽)` 아래로 굴리고, 각 스텝에서 예측된 잠재값이 얼마나 벌어지는지 그래프로 그립니다. 정규화된 모델은 퍼져나가야 하고 기준선은 평평하게 유지되어야 하며, 이는 불러온 체크포인트로는 얻을 수 없었던 명확한 대비를 개입적 그래프에 부여합니다.

(아래 코드 셀에서 정의되며, 4절의 `dreamer_rollout`/`frame_divergence`와 같은 패턴을 따르는) `compact_intervention_divergence`는 각 고정 동작 아래에서 `forward_dynamics`를 반복 적용해 잠재 공간의 격차가 스텝마다 커지는 것을 추적하며, 위 7절의 단일 스칼라를, 4절이 사전학습된 Dreamer와 Transformer 체크포인트에 대해 만들어낸 것과 같은 종류의 지평선 곡선으로 바꾸는데, 이번에는 학습 중 역동역학 손실의 유무만 다른 통제된 모델 쌍을 대상으로 합니다.

```python
@torch.no_grad()
def compact_intervention_divergence(model, seed_obs, steps=ROLLOUT_LEN):
    """Roll a CompactWM forward under do(right) vs do(left) from one encoded state."""
    s = model.encode(seed_obs.unsqueeze(0).to(DEVICE))
    s_r = s_l = s
    div = []
    for _ in range(steps):
        a_r = torch.zeros(1, dtype=torch.long, device=DEVICE)
        a_l = torch.ones(1, dtype=torch.long, device=DEVICE)
        s_r = model.forward_dynamics(s_r, a_r)
        s_l = model.forward_dynamics(s_l, a_l)
        div.append((s_r - s_l).pow(2).sum(-1).sqrt().item())
    return div


causal_div   = compact_intervention_divergence(causal_wm, seed)
baseline_div = compact_intervention_divergence(baseline_wm, seed)
steps = list(range(1, ROLLOUT_LEN + 1))

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(steps, causal_div, 'o-', color='seagreen', linewidth=2, label='Action-regularized WM')
ax.plot(steps, baseline_div, 's-', color='gray', linewidth=2, label='Baseline WM')
ax.set_xlabel('Rollout step')
ax.set_ylabel('L2 latent divergence: do(right) vs do(left)')
ax.set_title('Interventional Divergence after Inverse-Dynamics Training')
ax.set_xticks(steps)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```
## 8. 체크포인트 저장과 요약

재사용을 위해 동작 정규화 모델을 저장하고 핵심 수치를 기록합니다. 역동역학 정규화 항으로 학습된 모델인 `causal_wm`이 저장되는 모델입니다. 이 노트북이 실제로 동작-인과적이라고 보여준 모델이므로 앞으로 가져갈 가치가 있는 산출물이고, `baseline_wm`은 이 노트북 안에서 정규화 항 없이는 어떤 일이 일어나는지 보여주는 비교 대상으로만 존재하는 것과 대조됩니다.

```python
torch.save({
    'causal_wm': causal_wm.state_dict(),
    'action_influence': {
        'causal': infl_causal,
        'baseline': infl_baseline,
        'dreamer_rssm': infl_rssm,
    },
}, CAUSAL_CKPT)
print(f'Saved action-regularized world model to {CAUSAL_CKPT}')

print('\n--- P06 Summary ---')
print(f'  Dreamer / Transformer checkpoints loaded : {dreamer_loaded} / {trans_loaded}')
print(f'  Dreamer interventional divergence (step 10, RMS px) : {d_div[-1]:.4f}')
print(f'  Transformer interventional divergence (step 10, token KL) : {t_div[-1]:.4f}')
print(f'  Action influence (regularized)           : {infl_causal:.4f}')
print(f'  Action influence (baseline)              : {infl_baseline:.4f}')
print('  A large gap between the two confirms the inverse-dynamics term induces action-conditioning.')
```
P06의 교훈은 정확도와 인과적 정확도가 서로 다른 축이라는 것입니다. 월드모델은 프레임을 잘 재구성하면서도(P05가 측정한 질문), 여전히 동작의 효과를 인코딩하는 데는 실패할 수 있습니다(여기서 측정한 질문). 반사실적 롤아웃은 그 격차를 직접 드러내며, 역동역학 정규화는 그 격차를 좁히는 하나의 구체적인 방법입니다. 이것이 바로 구조화된 압축이라는 인터뷰 프레이밍이 중요한 이유입니다. 쓸모 있는 월드모델은 관측을 단순히 학습 분포를 재현하는 통계치가 아니라, 개입에 반응하는 변수로 압축합니다.

