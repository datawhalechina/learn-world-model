---
title: Projects
description: Five progressive projects, from a VAE encoder to a three-model evaluation dashboard, building world models one piece at a time.
---

# Projects

The lectures explain how world models work; the projects get them running on your own machine. Five projects build on one another along a single reference chain: **Dreamer (RSSM) → TD-MPC → STORM**. Each project's output feeds the next: once P01's encoder is done, P02 has a latent representation to model, and so on.

**Suggested order:** P01 → P02 → P03 → P04 → P05, interleaved with the lectures: L01 → L02 → P01 → P02 → L03 → P03 → P04 → L04 → P05 → L05.

## The five projects

| # | Project | Prerequisites | Deliverable |
|---|---------|---------------|-------------|
| [P01](./project-01-vae-encoder/) | Train a VAE Encoder | [L01](../lectures/lecture-01-internal-simulation/), [L02](../lectures/lecture-02-encode-and-dynamics/) Part A | A VAE compressing 64×64 images to a latent z; reconstruction loss curve; latent slider demo |
| [P02](./project-02-latent-dynamics/) | Build a Latent Dynamics Model | P01, L02 Part B | GRU → RSSM predicting the next latent; 1-step vs 5-step prediction-error plot |
| [P03](./project-03-dreamer-pipeline/) | Full Dreamer Pipeline | P02, [L03](../lectures/lecture-03-architecture-patterns/) Part A | End-to-end: encode → RSSM → latent Actor-Critic → act; reward curve + FID/ρ/entropy self-eval |
| [P04](./project-04-td-mpc/) | Implement TD-MPC Planning | P03, L03 Part B | CEM-MPC + latent consistency loss; reward curve compared against Dreamer |
| [P05](./project-05-storm-dashboard/) | STORM + Three-Model Dashboard | P03, P04, L03, [L04](../lectures/lecture-04-evaluation-by-model/) | Swap GRU → Transformer (STORM-style); side-by-side Dreamer / TD-MPC / STORM dashboard |

## Project highlights

### [P01 · Train a VAE Encoder](./project-01-vae-encoder/)

The first step in a world model is compressing high-dimensional observations into a compact latent. You implement a VAE that encodes 64×64 images into a low-dimensional z and decodes them back, using the ELBO to constrain both reconstruction quality and latent structure. Once it runs, drag a slider over a single latent dimension and watch which visual factor it controls.

### [P02 · Build a Latent Dynamics Model](./project-02-latent-dynamics/)

With latents in hand, the next step is predicting how they evolve over time. Start from a GRU, then extend to an RSSM that models deterministic and stochastic state separately. The focus is measuring multi-step error growth: one-step predictions are usually sharp, drift sets in after five, and that drift is the question every later evaluation keeps returning to.

### [P03 · Full Dreamer Pipeline](./project-03-dreamer-pipeline/)

Wire the encoder and the dynamics model into one chain: encode observations, roll out imagination in latent space with the RSSM, train an Actor-Critic on the imagined trajectories, and output actions. You get an end-to-end Dreamer running, then score it with FID, reward correlation ρ, and visitation entropy.

### [P04 · Implement TD-MPC Planning](./project-04-td-mpc/)

Dreamer acts straight from a learned policy; TD-MPC plans online in latent space instead. You implement CEM-MPC with a latent consistency loss to keep planning stable inside imagination, then put its reward curve next to P03's Dreamer to feel the trade-off between planning and a policy.

### [P05 · STORM + Three-Model Dashboard](./project-05-storm-dashboard/)

Finally, swap the GRU inside the RSSM for a Transformer to get a STORM-style world model. You build a side-by-side dashboard that runs Dreamer, TD-MPC, and STORM on the same tasks, reading off their strengths and failure modes with the [L04](../lectures/lecture-04-evaluation-by-model/) metric vocabulary.
