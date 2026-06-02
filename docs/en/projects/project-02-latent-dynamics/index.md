---
title: "P02: Build an RSSM Dynamics Model"
description: Implement GRU, MDN-RNN, and RSSM as three successive dynamics models. Compare how each handles uncertainty, and see why the RSSM's split prior/posterior design enables imagination-only rollouts.
---

# P02: Build an RSSM Dynamics Model

**Format**: Jupyter notebook (`p02_rssm_dynamics.ipynb`)  
**Prerequisites**: P01, L02 Part B  
**Builds toward**: P03 and P04 (the RSSM trained here is reused as the dynamics core)

---

## What You Will Build

Three dynamics models of increasing sophistication, all operating in the latent space produced by P01's encoder. The notebook loads the P01 encoder, collects a small trajectory dataset, then trains and compares all three models in sequence.

---

## Notebook Sections

**Section 1: Setup and Data Collection**

Load the frozen P01 encoder. Run a random policy in the same environment to collect (observation, action, next observation) tuples. Encode observations to produce a latent trajectory dataset. All of this runs inline; no separate data collection script is needed.

**Section 2: GRU Dynamics Model**

Implement a GRU that takes `(z_t, a_t)` and predicts `z_{t+1}` deterministically. Train with MSE loss. Compute 1-step and 5-step prediction error on a held-out set.

**Section 3: MDN-RNN**

Replace the output head with a mixture of K Gaussians. Train with negative log-likelihood. Compare 1-step and 5-step prediction error against the GRU using a side-by-side bar chart.

**Section 4: RSSM**

Implement the three core equations: deterministic GRU update for `h_t`, prior `p(z_t | h_t)` used during imagination, and posterior `q(z_t | h_t, o_t)` used during training. Train with reconstruction loss plus KL divergence between prior and posterior.

**Section 5: Prior vs Posterior Rollout**

Starting from a real initial state, roll forward 10 steps using only the prior (no real observations). Plot the imagined latent trajectory alongside the ground-truth posterior trajectory. Observe and annotate where they diverge.

---

## Deliverables

- Completed notebook with all cells executed
- Prediction error comparison (GRU vs MDN-RNN vs RSSM at 1-step and 5-step)
- Prior vs posterior rollout plot with divergence annotated

---

## Reference

GRU, MDN-RNN, and RSSM mechanics: L02 Part B. PlaNet ablation study (why both deterministic and stochastic paths are necessary) discussed there.
