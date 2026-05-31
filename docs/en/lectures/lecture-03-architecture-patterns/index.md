---
title: Architecture Patterns, Learning Paradigms, and Planning Methods
description: Using the RSSM you implemented in P02 as a baseline, compare seven major world model architecture families, understand the knowledge boundaries of four learning paradigms, and learn the principles and trade-offs of CEM-MPC, latent Actor-Critic, and TD-MPC planning mechanisms.
lecture: 3
difficulty: medium-high
---

# Architecture Patterns, Learning Paradigms, and Planning Methods

The RSSM you built in P02 is the RNN baseline. This lecture takes it as the starting point and unfolds in two parts:

- **Part A: Architecture Patterns**: Principles and a selection guide for seven architecture families (RNN/RSSM, Transformer, Diffusion, JEPA, RWM, Genie, WAM), plus the knowledge boundaries of four learning paradigms
- **Part B: Planning Mechanisms**: CEM-MPC random search, Dreamer latent Actor-Critic, and TD-MPC hybrid approach. This is the direct prerequisite for P03 and P04.

The architecture section does not require implementation; that is the work of later projects. The focus is on understanding design trade-offs and building judgment about when to use what. The planning section requires careful reading.
