---
title: "Part A (Continued): TD-MPC and Planning Mechanism Comparison"
description: TD-MPC's temporal-difference hybrid planning scheme, its comparison with DreamerV3, and a comprehensive summary of the three planning mechanisms.
lecture: 3
---

# Part A (Continued): TD-MPC and Planning Mechanism Comparison

## Mechanism 3: TD-MPC, the Bridge Between the Two

TD-MPC (Temporal Difference Model Predictive Control) [Hansen et al., 2022] combines the lookahead planning capability of MPC with the temporal-difference learning efficiency of Actor-Critic.

**Core design**:

| Component | Role |
|-----------|------|
| Latent consistency loss | Trains the implicit dynamics model: $\hat{z}_{t+1} = f(z_t, a_t)$ should be consistent with the encoder output $\text{sg}(z_{t+1})$ |
| Temporal-difference target | Updates the Q function (action-value function, $Q(s,a)$ represents the expected cumulative discounted reward obtained by executing action $a$ in state $s$ and following the policy thereafter) via the Bellman equation: $Q(z_t, a_t) = r_t + \gamma \cdot Q(z_{t+1}, \pi(z_{t+1}))$, where $\gamma$ (discount factor) causes future rewards to decay exponentially |
| CEM planning | At each decision step, uses CEM to search for the optimal action sequence in latent space |

These three components are trained jointly: the consistency loss shapes the latent space, while the TD target trains the Q function to guide CEM search.

**The role of stop-gradient**: The `sg(z_{t+1})` in the consistency loss denotes stop-gradient. If both sides of the encoder can receive gradient updates, the model may learn an "identity function" that maps all states to a single point, driving the consistency loss to zero while being completely meaningless. Stop-gradient fixes the target side, preventing this **mode collapse** (where the model finds a degenerate solution: mapping all different inputs to the same output, minimizing the loss but producing no useful representation).

> **📖 Bellman Equation**: $Q(s_t, a_t) = r_t + \gamma \cdot \max_{a'} Q(s_{t+1}, a')$. This transforms the infinite-horizon cumulative reward problem into a form that only looks at "one-step reward + next-step Q value". **Bootstrapping**: using the model's own estimates (such as $Q(s_{t+1}, a')$) as training targets, "predicting from oneself". TD learning uses the Bellman equation for bootstrapping, allowing learning to occur at every step without waiting for an episode to end.

TD learning uses the Bellman equation to substitute "current reward + next-step Q value estimate" for a full rollout, reducing the effective planning depth from "exact model steps" to "1 step + Q function bootstrapping".

**Comparison with DreamerV3**:

| Dimension | DreamerV3 | TD-MPC2 |
|-----------|-----------|---------|
| World model form | Explicit generative (reconstructs pixels/observations) | Implicit (only guarantees accurate value prediction) |
| Planning approach | Latent space Actor-Critic | CEM + TD |
| Applicable task scope | Visually complex tasks requiring rich observations | State-observation tasks, efficient continuous control |
| Interpretability | Can visualize reconstructions | Latent space has no direct semantics |


## Comparison of Three Planning Mechanisms

| Dimension | CEM-MPC | Dreamer Actor-Critic | TD-MPC |
|-----------|---------|---------------------|--------|
| Planning approach | Random search | Policy gradient (differentiable) | Random search + TD |
| Requires pixel reconstruction | No | Yes | No |
| Long-horizon planning capability | Limited by $H$ | Relies on Critic bootstrapping | TD + MPC combined |
| Computational cost | High (large $N$) | Medium (imagined rollouts) | Low to medium |
| High-dimensional action space | Low efficiency | Gradient optimizes directly | Q function guides search |
| Model exploitation risk | Medium (myopic) | High (policy can exploit model) | Medium (TD suppresses accumulated error) |
| Typical scenario | Simple continuous control | Visually complex tasks | Efficient continuous control |


## Core Planning Checkpoint

- **Three learning paradigms** clarify what the training data can identify: observation-only data reveals visual regularities, interaction data reveals how actions change outcomes, and value-coupled training teaches which predicted outcomes matter for the task.
- **Three planning mechanisms** determine how a model is used for decision-making: CEM is the most straightforward but inefficient in high-dimensional spaces, Actor-Critic is the most elegant but carries model exploitation risk, and TD-MPC most pragmatically balances both.
- Dreamer = interaction-based paradigm + RSSM + latent Actor-Critic, and is the core reference system for this curriculum.
- TD-MPC = action-conditioned latent dynamics + CEM + TD. It serves here as the hybrid comparison point; P04 instead focuses on the separate question of replacing the RSSM backbone with a Transformer.


## Next Step

You now have enough conceptual machinery to complete [P03: Train a Dreamer Agent](../../projects/p03_dreamer_agent). After running the complete encoder, RSSM, actor, and critic loop, return to Part B to compare the RSSM against Transformer and diffusion alternatives. That ordering makes each architecture choice answer a bottleneck you have observed rather than a name you have merely encountered.


## Further Reading

Key papers covered in this lecture, listed in order of appearance:

**Foundational Architectures**
- [Ha & Schmidhuber (2018): World Models](https://arxiv.org/abs/1803.10122): original paper on the V/M/C three-module framework and training in dreams
- [Hafner et al. (2019): PlaNet / RSSM](https://arxiv.org/abs/1811.04551): deterministic + stochastic dual-path latent dynamics model
- [Hafner et al. (2019/2020/2023/2025): Dreamer V1/V2/V3/V4](https://arxiv.org/abs/1912.01603): RSSM and latent Actor-Critic series; V4 see [arxiv 2509.24527](https://arxiv.org/abs/2509.24527)

**Transformer Architectures**
- [Micheli et al. (2022): IRIS](https://arxiv.org/abs/2209.00588): VQ-VAE discretization + GPT autoregressive world model, Atari 100k 1.046 HNS
- [Zhang et al. (2023): STORM](https://arxiv.org/abs/2310.09615): categorical VAE + single-token Transformer, 126.7% HNS, 4.3h training

**Diffusion Architectures**
- [Alonso et al. (2024): Diamond](https://arxiv.org/abs/2405.12399): diffusion world model, first to achieve lower FVD than real game frames on Atari

**Planning Mechanisms**
- [Schrittwieser et al. (2020): MuZero](https://arxiv.org/abs/1911.08265): implicit world model + MCTS, superhuman performance on Go and Atari
- [Hansen et al. (2022): TD-MPC](https://arxiv.org/abs/2203.04955), [TD-MPC2 (2024)](https://arxiv.org/abs/2310.16828): CEM + TD hybrid planning

**JEPA Series**
- [Assran et al. (2023): I-JEPA](https://arxiv.org/abs/2301.08243), [Bardes et al. (2024): V-JEPA](https://arxiv.org/abs/2404.08471): semantic space prediction without pixel reconstruction

**Genie / Interactive Generation**
- [Bruce et al. (2024): Genie](https://arxiv.org/abs/2402.15391): automatically discovers latent actions from unannotated video, 11B parameters

**RWM / Robot Deployment**
- [Li et al. (2026): RWM-U](https://arxiv.org/abs/2504.16680): offline MBRL + ensemble uncertainty, validated on quadruped and humanoid robots
- [NeurIPS 2025: Self-Forcing](https://arxiv.org/abs/2506.08009): introduces self-prediction feedback during training to alleviate teacher forcing gap

**WAM / Joint Learning**
- [Bi et al. (2025): Motus](https://arxiv.org/abs/2512.13030): unified latent action world model, cross-embodiment transfer from heterogeneous video data
- [NVIDIA (2026): WAM](https://arxiv.org/abs/2509.20328): pretrained video model as zero-shot policy
- [NVIDIA (2025): Cosmos](https://arxiv.org/abs/2501.03575): general physical AI world foundation model, open-source with open weights
- [Hu et al. (2023): GAIA-1](https://arxiv.org/abs/2309.17080): generative world model for autonomous driving, joint modeling of video, text, and actions
