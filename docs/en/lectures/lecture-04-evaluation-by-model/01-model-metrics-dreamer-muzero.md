---
title: Dreamer-Specific Metrics
description: Dedicated evaluation metrics and diagnostic rules for RNN/RSSM architectures, covering reward correlation, FID, and KL collapse warnings.
lecture: 4
---

# Dreamer-Specific Metrics

## Why "Evaluate by Model"?

**A counterexample**: using FID to evaluate MuZero. MuZero never generates pixel images; its world model is implicit, and FID is meaningless for it. Likewise, using "token prediction loss" to evaluate Dreamer only creates the false impression that Dreamer is a language model.

Different world models break down at different points:

| Architecture | Most Common Failure Point |
|---------|----------------|
| RNN/RSSM (Dreamer) | Encoder degradation, imagined reward distortion, KL collapse |
| Implicit model (MuZero) | Value estimation bias, weak representation stability, search tree degradation |
| Latent MPC (TD-MPC) | Latent representation inconsistency, representation collapse, low planning efficiency |
| Transformer dynamics (STORM) | Gap between teacher forcing and free-running, long-horizon token drift |
| Diffusion world model (Diamond) | Physical consistency collapse, objects disappearing from the scene |

## Dreamer (RNN/RSSM)

*In P03 you implemented the full Dreamer pipeline by hand: encoder → RSSM prediction → latent Actor-Critic → action execution.*

Dreamer is a reinforcement learning algorithm, not a generative model. The core evaluation question across the Dreamer V1/V2/V3 papers is singular: **can the policy achieve high reward in the real environment?** Every metric serves that question, not "how good does image reconstruction look."

### Policy Reward Curve (Episode Return)

This is the primary metric across the Dreamer series. Hafner et al. 2019 report per-task episode return on DMControl; Hafner et al. 2020 report episode return on Atari over 200M frames; Hafner et al. 2023 report episode return across 7 domains with a single set of hyperparameters. The central figure in all three papers is a **training steps vs. cumulative reward curve**, compared against model-free baselines (such as **SAC**, Soft Actor-Critic, an off-policy Actor-Critic algorithm based on the maximum-entropy framework and a strong model-free baseline on continuous control tasks; and **DrQ-v2**, Data-regularized Q, which adds data augmentation and n-step returns on top of SAC and serves as a representative baseline for pixel-input continuous control) and model-based baselines (such as **MBPO**, Model-Based Policy Optimization, which uses a learned world model to generate short synthetic rollouts to improve sample efficiency; and **PlaNet**, Planning with Latent Dynamics, the predecessor to RSSM that performs only MPC planning without an Actor-Critic).

**Diagnostic rule**: a training curve that stagnates or declines over an extended period has two possible sources. First, reward prediction distortion (the world model is lying), causing the Actor-Critic to optimize against incorrect imagined rewards. Second, RSSM dynamics prediction drift, causing imagined rollouts to diverge increasingly from the real environment distribution. Distinguishing the two requires simultaneously inspecting reward correlation (see the next section).

**How to track in P03 experiments**: pause training at fixed intervals (e.g., every 10k steps), run several complete episodes in the real environment, record the mean episode return, and plot the curve. This curve is the final criterion for whether Dreamer is training correctly.

### Reward Correlation

Dreamer rolls out trajectories in "imagination" and predicts rewards. These imagined rewards must be highly correlated with the rewards returned by the real environment for the Actor-Critic to learn a useful policy.

$$\rho = \text{Pearson}(r_{\text{imagined}},\, r_{\text{real}})$$

> **📖 Pearson correlation coefficient**: the standard metric for measuring the degree of linear correlation between two variables, with values in $[-1, 1]$. $\rho = 1$ indicates perfect positive correlation (one increases as the other does, at a fixed ratio); $\rho = 0$ indicates no linear correlation; $\rho = -1$ indicates perfect negative correlation. The formula is $\rho = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$, where $\text{Cov}$ is covariance and $\sigma$ is standard deviation. Here it measures whether the trends of the imagined reward sequence and the real reward sequence move together.

In practice, take a batch of rollouts (e.g., 1000 steps), compute the Pearson correlation coefficient between the imagined and real reward sequences, and target `ρ ≥ 0.8`.

**Diagnostic rule**: if `ρ` stays below 0.5, the stochastic state `z_t` in the RSSM is encoding insufficient reward information. Try increasing the latent dimension or extending the KL annealing schedule.

**Experimental suggestion**: visualize the reward curves of imagined rollouts vs. real rollouts. Concretely: starting from the same initial state, let Dreamer imagine a 20-step rollout while simultaneously executing the same action sequence in the real environment, then plot both reward curves on the same figure. If the two curves share a similar trend (even without perfectly overlapping), the world model is faithfully reflecting the environment. If the imagined rewards are consistently higher than the real rewards and trend in the opposite direction, the world model is lying, and the policy has learned tricks against a false optimization target. This is the root cause of the model exploitation problem.

### Encoder Health Diagnosis: Reconstruction FID (Fréchet Inception Distance)

FID is not a reported metric in the Dreamer papers. The ELBO objective in Dreamer includes a reconstruction loss, but the papers never use FID to measure policy quality. FID is used here as an **auxiliary diagnostic tool**: if the encoder degrades, image reconstruction quality drops, and FID can catch this signal early, allowing intervention before episode return collapses.

FID uses **Inception-v3** (a deep convolutional image classification network proposed by Google in 2015, pretrained on ImageNet, whose intermediate-layer feature vectors are widely used as a perceptual proxy for image quality) to extract deep features from real frames and reconstructed frames, then computes the **Fréchet distance** between the two feature distributions ([Heusel et al., 2017](https://arxiv.org/abs/1706.08500)). Lower FID is better: a high FID means the feature distribution of reconstructed frames has diverged from that of real frames, indicating encoder representation degradation.

<details>
<summary>📖 FID Calculation Details (expand)</summary>

① Use an intermediate layer of Inception-v3 to extract a feature vector from each of a large number of real images and generated images; ② fit a multivariate Gaussian distribution (mean $\mu$, covariance matrix $\Sigma$) to each set of features; ③ compute the Fréchet distance (also known as the Wasserstein-2 distance, the minimum work required to "transport" one distribution into the other, more sensitive to shape differences than KL divergence) between the two Gaussians:

$$\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}\!\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$$

where `Tr(·)` is the trace of a matrix (the sum of its diagonal elements). Inception-v3 features are used instead of pixel MSE because the feature space approximates human perceptual judgment of images.
</details>

**Diagnostic rule**: a sudden rise in FID mid-training indicates representation collapse in the encoder, where convolutional weights have degenerated to constant outputs. **Mitigation**: reduce the encoder learning rate, or add LayerNorm after the encoder.

**Normal FID does not mean Dreamer is healthy.** It is entirely possible for visual reconstruction to look acceptable while imagined rewards have quietly become distorted, because the stochastic state `z_t` in the RSSM can satisfy reconstruction quality while encoding insufficient reward information. Normal FID only confirms that the encoder has not degraded; reward correlation must also be checked together.

### Imagined Trajectory Entropy

This is an early-warning metric that is easy to overlook but highly important. The stochastic state `z_t` in the RSSM is sampled from a Gaussian distribution, and in principle each step should carry a certain amount of variance, reflecting both the intrinsic stochasticity of the environment and model uncertainty.

$$H_{\text{traj}} = \mathbb{E}_t\!\left[H\!\left(q(z_t \mid h_t, o_t)\right)\right] = \mathbb{E}_t\!\left[\tfrac{1}{2}\sum_i \left(1 + \log \sigma^2_i\right)\right]$$

**Diagnostic rule**: if the RSSM returns very similar `z_t` values for all states during imagined rollouts (all variances `σ²` approaching 0), the stochastic variable has degenerated to deterministic. This is an early signal of **KL collapse**. During KL collapse, the KL divergence term in the loss function becomes nearly zero, the encoder stops injecting any additional information into `z_t`, and the entire RSSM degrades to a pure RNN.

**Mitigation strategies**:
- Use **KL annealing** (gradually increasing the weight coefficient of the KL divergence term in the loss function from 0 to the target value during early training, e.g., from 0 to 1 over the first 10k steps, giving the encoder time to first learn reconstruction before being gradually forced to encode information into the stochastic latent variable): slowly ramp the KL weight from 0 to the target value at the start of training, giving the encoder time to first learn reconstruction and then learn to encode stochasticity
- Set **KL free bits** (when computing the KL loss, apply no gradient to dimensions whose KL value falls below a threshold $\lambda$ (e.g., 1 nat), i.e., $\max(0, \text{KL} - \lambda)$, forcing the model to retain at least $\lambda$ nats of information for each latent dimension and preventing all dimensions from collapsing to zero simultaneously): force the KL term to reach at least some minimum value, preventing premature collapse
