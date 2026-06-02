---
title: "P03: Train a Dreamer Agent"
description: Assemble encoder, RSSM, and latent Actor-Critic into a full Dreamer training loop. The agent learns entirely from imagined rollouts inside the world model, touching the real environment only to collect new data.
---

# P03: Train a Dreamer Agent

**Format**: Jupyter notebook (`p03_dreamer_agent.ipynb`)  
**Prerequisites**: P02, L03 Part B  
**Builds toward**: P05 (this trained agent is one of the two models compared in the evaluation dashboard)

---

## What You Will Build

A complete Dreamer training loop in a single notebook. The agent alternates between two phases: world model update (real transitions, encoder, RSSM) and behavior learning (imagined rollouts, latent Actor-Critic). The environment is small enough that a meaningful policy emerges within a notebook session.

Use a lightweight pixel environment: CartPole with pixel observations rendered at 64×64, or DMControl Cartpole Swingup with a small image size. Keep the RSSM hidden dimension at 128-256 and the Actor/Critic as 2-layer MLPs so training completes in under 2 hours on CPU or 30 minutes on GPU.

---

## Notebook Sections

**Section 1: Setup**

Load the P01 encoder and P02 RSSM. Define the Actor (outputs action distribution from latent state) and Critic (outputs scalar value from latent state). Initialize the replay buffer.

**Section 2: World Model Update**

Implement the world model training step: encode a batch of real transitions, run the RSSM to produce `h_t` and `z_t`, compute reconstruction loss and KL divergence, update encoder and RSSM parameters.

**Section 3: Behavior Learning**

Implement the imagination rollout: starting from RSSM states in the buffer, roll forward H steps using the prior and the Actor. Compute λ-return targets and train the Critic. Train the Actor to maximize the Critic's value predictions.

**Section 4: Training Loop**

Alternate between environment interaction, world model update, and behavior learning for a fixed number of iterations. Log and plot: episode reward, reconstruction loss, KL divergence, and Actor entropy.

**Section 5: Self-Evaluation Metrics**

Compute on a held-out set: reward correlation ρ between imagined and real rewards, and imagined trajectory entropy. Plot both alongside the reward curve. These are the Dreamer-specific diagnostics from L04.

---

## Deliverables

- Completed notebook with all cells executed
- Training curves: episode reward, reconstruction loss, KL divergence
- Self-evaluation metrics: reward correlation ρ, trajectory entropy

---

## Reference

Actor-Critic design and λ-return: L03 Part B. Dreamer evaluation metrics (FID, ρ, entropy): L04.
