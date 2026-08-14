---
title: From Latent Dynamics to Planning and Architecture Choices
description: Complete the agent loop with CEM-MPC, latent Actor-Critic, and TD-MPC, then compare alternative dynamics backbones and optionally survey frontier systems.
lecture: 3
difficulty: medium-high
---

# From Latent Dynamics to Planning and Architecture Choices

The RSSM you built in P02 can predict latent futures, but it does not yet tell an agent which action to execute. This lecture first completes that loop. Only after you can trace how predictions affect actions does it compare alternative model architectures.

- **Planning and Control**: CEM-MPC search, Dreamer latent Actor-Critic, and TD-MPC. Complete P03 after this module.
- **Backbone Selection**: use RSSM as the baseline, then study when a Transformer or diffusion backbone is justified. Complete P04 after this module.
- **Optional Frontier Survey**: JEPA, RWM, spatial 3D/4D models, Genie, LoopWM, WAM, system-integration patterns and VLA mechanics, and the LS-Imagine case study. These pages broaden research judgment but are not prerequisites for P03 or P04.

If your goal is to build a working system, complete Planning and Control followed by Backbone Selection. If your goal is literature orientation, continue through the Optional Frontier Survey after completing that core path.
