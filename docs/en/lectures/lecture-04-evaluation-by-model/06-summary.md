---
title: Five-Model Summary, Deployment Strategies, and Curriculum Outlook
description: A comparative summary table of metrics across five world models, three pragmatic deployment strategies, and a four-lecture curriculum summary with outlook.
lecture: 4
---

# Five-Model Summary, Deployment Strategies, and Curriculum Outlook

## Comparative Summary Table

| Model | Primary Metrics | Additional Diagnostic Metrics | Common Failure Modes | Diagnostic Approach |
|-------|----------------|-------------------------------|----------------------|---------------------|
| **Dreamer** (RSSM) | Reconstruction FID, reward correlation `ρ` | Imagined trajectory entropy (KL collapse warning) | Encoder degradation, imagined reward distortion, KL collapse | FID rising: reduce encoder LR; `ρ` dropping: increase latent dimension; entropy approaching zero: KL annealing / free bits |
| **MuZero** (implicit) | Value accuracy, MCTS visit entropy | Representation stability (cosine similarity > 0.95) | Value estimation bias, false confidence, unstable representation | Low accuracy: retrain reward model; entropy judgment requires task stochasticity context; low stability: increase network width or add contrastive loss |
| **TD-MPC** (latent MPC) | Latent consistency loss, plan efficiency | Latent space t-SNE visualization (local isometry) | Representation collapse, myopic planning | Removing sg yields lower loss: collapse; low covariance rank: collapse; low efficiency: increase elite ratio |
| **STORM** (Transformer) | Token prediction loss, long-horizon PSNR | FVD (I3D features, sequential dynamics quality) | Teacher forcing gap, autoregressive drift | PSNR sudden drop: shorten context window; use PSNR for debugging, FVD for policy evaluation |
| **Diffusion World Model** (Diamond) | FVD, physics consistency, action-conditioned fidelity | Depth violation rate (DepthAnything + DINO automated evaluation) | Object persistence loss, inverted 3D relationships | High depth violation rate: introduce 3D constraints; low fidelity: inject action information at each layer |

---

## Three Pragmatic Deployment Strategies

Depending on risk tolerance and system maturity, there are three progressive strategies for deploying world models in production:

**1. Shadow Evaluator**

The real policy executes as normal, while the world model independently predicts future outcomes in parallel and compares them against what actually occurs, without intervening in control. This approach systematically identifies "which object types, action ranges, and contact states the model is unreliable on," building a reliability map. It carries the lowest risk and is well-suited for early deployment phases.

**2. Action Filter**

The policy first proposes multiple candidate actions (such as N trajectories from MPC or K actions sampled by the Actor), then the world model predicts the consequences of each. Two classes of actions are filtered out: (a) those with predicted consequences that are clearly dangerous (such as predicted collisions or object drops), and (b) those where uncertainty exceeds a threshold (meaning the world model has low confidence in the consequences). The action with the highest predicted reward among the remaining candidates is then executed.

**3. Closed-loop Planner / Imagined Training**

The world model enters the MPC rollout or imagined rollout, used directly for online planning or offline policy training. This is the standard usage in Dreamer and TD-MPC. The potential gains are highest here (vast numbers of states can be explored in imagination without real-world interaction), but so are the risks: model exploitation, safety exploits, and distribution shift all directly affect policy quality. This strategy is recommended only after the world model has been thoroughly validated via the shadow evaluator phase.

---

## Curriculum Summary

Across four lectures, each one addressed a concrete problem:

**L01: Internal Simulation and Historical Context**
Starting from Craik's "mental models" (1943), tracing through the 1950s RNN origins, Ha and Schmidhuber's 2018 World Models paper, the end-to-end maturation of Dreamer in 2019, and finally LeCun's JEPA paradigm in 2023, this lecture built a historical intuition for how world models have evolved.

**L02: Observation Encoding and Latent Dynamics**
Part A implemented a VAE encoder: a CNN compresses 64x64 images into a latent vector `z`, with the ELBO loss (reconstruction term + KL term) regularizing the latent space. Part B started from the GRU, progressed through the MDN-RNN, and arrived at the RSSM, where the dual-track architecture of deterministic state `h_t` and stochastic state `z_t` forms the foundation of Dreamer.

**L03: Architecture Patterns, Learning Paradigms, and Planning Methods**
Using the RSSM implemented in P02 as the RNN baseline, this lecture compared seven architecture families side by side (RNN/RSSM, Transformer, Diffusion, JEPA, RWM, Genie, WAM), clarified four learning paradigms, and traced the planning chain from CEM-MPC to latent actor-critic to TD-MPC.

**L04: Evaluation by Model (this lecture)**
Evaluation is not "scoring" but "diagnosis." Each architecture has its own characteristic failure modes, and only targeted metrics can surface problems. Horizon drift is the long-horizon challenge shared by all world models; mitigating it requires short-horizon training, target networks, and continuous supplementation with real data.

### From Theory to Deployment: Outlook

World models are becoming **critical infrastructure for embodied intelligence**. Whether in game AI (MuZero conquering Go), robot manipulation (Dreamer learning to grasp), or autonomous driving (Wayve's GAIA), world models play a central role in internalizing physical world dynamics and reducing the need for real-world interaction.

But the content covered in this curriculum is primarily laboratory-scale world models. Moving from the lab to real deployment involves many unresolved engineering challenges: how to degrade safely under out-of-distribution states, how to pass uncertainty meaningfully to a controller, and how to continuously update a world model during online deployment without introducing catastrophic forgetting.

These questions have no standard answers. But you now have the tools needed to ask them correctly: understanding architectures, diagnosing failures, and selecting metrics. That is the core capability this curriculum aims to convey, not telling you what the right world model is, but teaching you how to determine where a world model is wrong.

---

## Next Lecture

L05 has no code, only debates. Is language the "opium" of world models or a necessary tool? Is the LLM a victory of the Bitter Lesson or a betrayal of it? Is AGI a legitimate research target or a false premise? These questions have no standard answers. The sharpest arguments on each side are laid out for you to judge.

If you want to see how world models compare on standardized interactive tasks beyond the metrics covered here, [World Arena](https://huggingface.co/spaces/WorldArena/WorldArena) is a community benchmark platform where models are evaluated on interactive prediction tasks head-to-head.

> **Complete P05**: Build an evaluation dashboard that displays all metrics for Dreamer, TD-MPC, and STORM side by side, translating the theory from this lecture into interactive experimental evidence. The dashboard should cover: reconstruction FID, reward correlation, consistency loss, token prediction loss, long-horizon PSNR, FVD, and a visualization of the latent drift curve.
