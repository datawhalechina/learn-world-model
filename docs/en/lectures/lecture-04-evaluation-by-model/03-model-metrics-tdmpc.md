---
title: "Latent Dynamics and Planning Efficiency: TD-MPC"
description: Consistency loss, representation collapse diagnosis, latent space visualization, and plan efficiency evaluation for latent MPC architectures.
lecture: 4
---

# Latent Dynamics and Planning Efficiency: TD-MPC as a Worked Example

## TD-MPC (Latent MPC)

*TD-MPC is a conceptual comparison in L03, not an implementation in P04. Use it here to learn diagnostics that apply whenever planning depends on an implicit latent transition model.*

The core requirement of TD-MPC is that representations produced by the encoder `h = enc(o)` at different time steps must be **mutually consistent**, so that MPC can plan effectively in latent space.

### Latent Consistency Loss

$$\mathcal{L}_{\text{consist}} = \|\text{sg}(h_{t+1}) - f(h_t, a_t)\|^2$$

where `f` is the dynamics function and `sg` denotes stop-gradient. This loss measures the distance between "the next state predicted by the dynamics function" and "the next observation directly encoded by the encoder."

**Diagnostic rule (representation collapse):** If the consistency loss decreases further after removing `sg`, the encoder has degenerated into an identity mapping, compressing all states to a single point. Formal diagnosis: on the validation set, inspect the rank of the covariance matrix of the latent vectors. A rank close to 1 indicates collapse. (The `sg` mechanism is discussed in Lecture 3 Part A, TD-MPC section.)

**Diagnostic rule (training oscillation):** If the consistency loss oscillates during training (unstable up-and-down fluctuations) rather than decreasing monotonically, the learning rate is too high, or the gradient scales of the encoder and dynamics function are mismatched. Try reducing the learning rate by one order of magnitude, or using a separate (smaller) learning rate for the encoder.

### Latent Space Visualization

This is a powerful qualitative diagnostic tool that provides intuitive insight beyond quantitative metrics.

**Experimental procedure:** In a 2D continuous control task (such as Pendulum or HalfCheetah), collect a batch of state-action trajectories, map all observations into latent space with the encoder, then reduce the high-dimensional latent vectors to 2D with **t-SNE** (t-distributed Stochastic Neighbor Embedding, a nonlinear dimensionality reduction algorithm that projects high-dimensional vectors onto a 2D plane such that points close in high dimensions remain close in 2D, commonly used to visualize the clustering structure of high-dimensional representations), and inspect the result visually.

**A healthy TD-MPC latent space should satisfy:**
- Physically nearby states (e.g., states with similar pendulum angles) map to nearby points in latent space: local isometry.
- Trajectories originating from the same state under different actions point in consistent directions in latent space: action predictability.
- As rollout steps increase, trajectories move smoothly through latent space rather than jumping randomly.

**Diagnostic rule:** If states of the same type (e.g., "pendulum upright") are scattered throughout the t-SNE plot rather than clustered together, the geometry of the latent space is disordered and the planning performed by MPC in this space has no physical meaning.

<figure>
<img src="/tdmpc/tdmpc-overview.png" alt="TD-MPC architecture overview: encoder, implicit dynamics, Q-function, and CEM planning" style="width:90%;display:block;margin:0 auto">
<figcaption>Hansen et al. (2022) TD-MPC architecture: encoder, implicit dynamics function, Q-function, and CEM work together as four coordinated modules. The latent consistency loss (aligning sg(z_{t+1}) with d(z_t, a_t)) keeps the dynamics function and encoder mutually consistent and prevents representation collapse. This is the diagnostic starting point for TD-MPC-specific metrics.</figcaption>
</figure>

### Plan Efficiency

Defined as the number of MPC planning steps required to reach a **target reward threshold** (e.g., 80% of the optimal policy's reward), starting from a randomly initialized policy. Fewer steps indicate higher plan efficiency.

**Diagnostic rule:** Low plan efficiency (requiring many steps to converge) suggests the CEM elite ratio is set too low, or the planning horizon is too short, causing myopic planning that misses long-range rewards.
