---
title: "Part A: CEM-MPC and Latent Actor-Critic"
description: The counterfactual paradigm and MuZero's implicit world model, CEM shooting-method random search, and Dreamer's differentiable latent Actor-Critic planning.
lecture: 3
---

# Part A: CEM-MPC and Latent Actor-Critic

Given a world model, how does an agent use it to select actions? This section is the direct prerequisite for P03 and provides the planning context used later to compare architectures. It introduces three planning mechanisms: from the most intuitive random search, to Dreamer's imagination-based training, to TD-MPC's hybrid approach.

Before the three planning mechanisms, one more world-model architecture needs to be on the table: MuZero. It is introduced beside planning because its defining feature is inseparable from tree search. Its world model is deliberately shaped to support search, so understanding the representation without its consumer would leave the system incomplete.

## MuZero and the Counterfactual Paradigm

There is a class of tasks that takes counterfactual reasoning to the extreme: the **counterfactual paradigm**, which forgoes pixel prediction entirely and instead makes accurate predictions only at the abstract level of values or rewards. MuZero ([Nature, 2020](https://arxiv.org/abs/1911.08265)) decomposes the world model into three functions:

- **Representation function** $h_\theta$: compresses past observations $o_{1:t}$ into an internal hidden state $s^0 = h_\theta(o_{1:t})$
- **Dynamics function** $g_\theta$: given the previous hidden state and a candidate action, predicts the immediate reward and the next hidden state: $r^k, s^k = g_\theta(s^{k-1}, a^k)$
- **Prediction function** $f_\theta$: predicts a policy prior and value from the hidden state: $\mathbf{p}^k, v^k = f_\theta(s^k)$

The three functions are trained jointly end-to-end, with the total loss:

$$l_t(\theta) = \sum_{k=0}^{K} \left[ l^r(u_{t+k},\, r_t^k) + l^v(z_{t+k},\, v_t^k) + l^p(\pi_{t+k},\, \mathbf{p}_t^k) \right] + c\|\theta\|^2$$

Symbol definitions: $K$ is the number of unroll steps (how many steps are unrolled per training update); $l^r$, $l^v$, $l^p$ are the loss functions for the reward, value, and policy prediction heads respectively; $u_{t+k}$ is the actual reward collected from real interactions (the training target); $r_t^k$ is the reward predicted by the dynamics function; $z_{t+k}$ is the $n$-step bootstrapped target value (constructed from real rewards plus a value estimate several steps later); $\pi_{t+k}$ is the improved policy produced by MCTS search (the visit-count distribution, used as the training target for the policy head); $c\|\theta\|^2$ is L2 regularization (weight decay, where $c$ is the regularization coefficient, preventing overfitting from excessively large parameters). **The hidden state $s^k$ has no semantic constraints**: it does not need to correspond to the true environment state, nor does it need to be able to reconstruct pixels. The only requirement is: "starting from $s^k$, accurately predict rewards, values, and policies." This is the most fundamental design difference between MuZero and PlaNet/Dreamer.

MuZero maintains three prediction heads:

| Prediction head | Prediction target | Role |
|-----------------|-------------------|------|
| reward head | immediate reward $r_t$ | evaluates the quality of the current step |
| value head | future cumulative value $V(s_t)$ | guides MCTS search direction |
| policy prior | action probability distribution $\pi(a \mid s_t)$ | reduces the number of branches MCTS needs to explore |

All three heads are trained jointly through the unrolled dynamics function on real interaction data.

<figure>
<img src="/muzero/muzero-model.png" alt="MuZero's implicit world model: three-module architecture of representation function, dynamics function, and prediction function" style="width:90%;display:block;margin:0 auto">
<figcaption>Schrittwieser et al. (2020) MuZero's three-function structure: the representation function h compresses historical observations into hidden state s; the dynamics function g simulates action transitions in hidden state space and predicts immediate rewards; the prediction function f outputs a policy prior and value estimate from the hidden state, driving MCTS search. The hidden state does not need to correspond to real pixels, only to support accurate reward and value prediction.</figcaption>
</figure>

As long as these three prediction heads are accurate, the exact form of the latent state $s_t$ does not matter. **For the agent, "faithfully reconstructing the world" is not necessarily the optimal objective.** MuZero achieves superhuman performance on Go (without being given the rules), Chess, Shogi, and 57 Atari games, while relying on no real model or environment rules.

> **📖 MCTS** (Monte Carlo Tree Search): Starting from the current state, repeatedly perform four steps: (1) **Select**: traverse down the tree, selecting the node with the highest **UCB score** (Upper Confidence Bound, $\text{UCB} = Q(s,a) + c\sqrt{\ln N / n_a}$, where $Q$ is the average value of that action, $N$ is the total visit count of the parent node, $n_a$ is the visit count of that action, and $c$ is the exploration coefficient; UCB balances "choosing known good actions" with "exploring less-visited actions"); (2) **Expand**: try a new action at a leaf node; (3) **Simulate/Evaluate**: use the neural network to estimate the value of the new node (MuZero uses the value head directly, without rollout); (4) **Backpropagate**: update the value estimate upward along the path. After hundreds of repetitions, the most-visited action is the one "deemed optimal after sufficient search." MuZero's key extension over AlphaZero: support for single-agent domains (not just two-player games) and intermediate step rewards (Atari), with value targets constructed via $n$-step bootstrapping rather than terminal win/loss.


## Mechanism 1: CEM Shooting-Method MPC

> **📖 CEM and MPC**: **CEM** (Cross-Entropy Method) is a sampling-based optimizer: sample many candidate solutions from a distribution such as a Gaussian, retain the best fraction as elite samples, refit the distribution to those samples, and repeat so that sampling increasingly concentrates on high-quality regions. Here CEM searches over action sequences, hence the name "shooting method." **MPC** (Model Predictive Control) specifies how the result is used: at each time step, predict $H$ steps ahead, select the best action sequence, execute only its first action, and then re-plan. Frequent re-planning can correct errors even when the model is imperfect.

**In one sentence**: randomly sample a batch of action sequences, "imagine" executing them in the model, select the sequence with the highest expected return, execute only the first step, and repeat.

**Algorithm steps**:

```
CEM-MPC Planning Loop (executed once per step)

Input: current state s_t, world model f, reward model r, planning steps H, refinement rounds K

1. Initialize action distribution: μ ← 0, σ ← 1

2. FOR k = 1 to K (refinement rounds):
   a. Sample N action sequences from N(μ, σ²): {a^(i)_{t:t+H}}
   b. FOR each sequence i:
        Roll out imagined trajectory: s^(i)_{t+1} = f(s_t, a^(i)_t), ..., s^(i)_{t+H}
        Compute cumulative reward: R^(i) = Σ_{h=0}^{H-1} γ^h · r(s^(i)_{t+h}, a^(i)_{t+h})
        # γ (gamma) is the discount factor, 0 < γ < 1, causing future rewards to decay exponentially
        # γ=0.99 means a reward 100 steps away is still worth 0.99^100 ≈ 0.37 of its face value
   c. Select Top-K sequences (sorted by R^(i) descending)
   d. Refit using Top-K sequences: μ ← mean(Top-K), σ ← std(Top-K)

3. Execute the first action from μ: a_t ← μ[0]
```

The first round of sampling covers a broad range with low precision, identifying roughly "where the high-return regions are." Subsequent rounds refit the distribution using elite sequences, progressively narrowing the sampling range toward high-return regions.

**Limitation**: in high-dimensional continuous action spaces (e.g., a robotic arm controlling 7 joints simultaneously), random search is extremely inefficient. This is the core problem TD-MPC addresses: guiding the search with a Q-function rather than sampling blindly.

**Advantages**: simple, gradient-free, easy to implement, with no differentiability requirements on the world model.


## Mechanism 2: Actor-Critic in Latent Space (Dreamer's Approach)

> **📖 Actor-Critic architecture**: consists of two networks. The **Actor** (policy network $\pi_\theta(a|s)$) handles "decision-making," and the **Critic** (value network $V_\phi(s)$) handles "evaluation." The baseline provided by the Critic greatly reduces the variance of gradient estimates, making training more stable.

Dreamer's core insight: rather than collecting large amounts of data in the real environment to train a policy, train inside the **imagined trajectories of the world model**, which is faster, risk-free, and differentiable.

**Training procedure**:
1. **Imagination rollout**: starting from the current latent state $z_t$, sample actions with the Actor and use RSSM to roll forward $H$ steps
2. **Critic evaluation**: compute $V(z_h)$ for each imagined state, constructing training targets with $\lambda$-return
3. **Actor optimization**: the Actor maximizes the cumulative value predicted by the Critic via backpropagation through the entire imagined trajectory
4. **World model update**: update the RSSM and encoder using real environment data (reconstruction loss + KL)

**Intuition behind $\lambda$-return**: pure Monte Carlo requires waiting until the episode ends to obtain a true return, giving high variance; pure TD looks only one step ahead, giving high bias. $\lambda$-return interpolates between the two, constructing a $k$-step return using "the first $k$ steps of real rewards plus a Critic estimate at step $k+1$," then taking a weighted average over all $k$. $\lambda \to 1$ trusts the real rollout; $\lambda \to 0$ trusts the Critic.

**Why differentiability matters**: the Actor's gradients flow directly through the differentiable dynamics of the RSSM, which is far more accurate than estimating policy gradients via Monte Carlo sampling.

**Model exploitation problem**: the policy may discover actions that yield high rewards inside the model but are invalid in the real world, such as high-frequency jitter actions that score highly in the world model but would only damage motors on a real robot. The Dreamer series addresses this by periodically updating the world model with real environment data and limiting the number of imagination rollout steps, but the problem has not been fundamentally solved.

CEM is inefficient in high-dimensional action spaces, and Actor-Critic carries model exploitation risk. TD-MPC's core value is combining both approaches to mitigate these two problems at the same time.
