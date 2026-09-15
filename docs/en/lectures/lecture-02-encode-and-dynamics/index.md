---
title: State Estimation, Observation Encoding, and Latent Dynamics
description: Start from partial observability, then learn to estimate hidden state, encode high-dimensional observations, predict latent dynamics, and diagnose the gap between training and free rollouts.
lecture: 2
difficulty: intermediate
---

# Lecture 2: State Estimation, Observation Encoding, and Latent Dynamics

Dreamer's core problem is not limited to how to compress perception and predict the future. First we must ask what the model should remember when a single observation omits part of the state. Afterwards we must ask whether a model that is accurate during training can remain stable on its own predictions.

After this lecture, you should be able to distinguish observations, hidden states, and belief states, explain the division of labor between a VAE and an RSSM, and judge planning suitability from free rollouts rather than one-step loss alone.

- **Observation, State, and Belief**: partial observability, memory, prediction and correction, and their relationship to the RSSM prior and posterior
- **Observation Encoding**: the VAE encoder-decoder, ELBO, CNN encoder, and what reconstruction does and does not guarantee
- **Latent Dynamics**: from GRU and MDN-RNN to the deterministic/stochastic paths of an RSSM
- **Training Distributions and Free Rollouts**: teacher forcing, prior imagination, horizon drift, model uncertainty, and planning risk
- **Dreamer Series Evolution**: place the components back into a complete system and see which bottleneck each generation addresses

Read Observation, State, and Belief followed by Observation Encoding, then complete P01. Next read Latent Dynamics and Training Distributions and Free Rollouts before completing P02. Each project metric will then correspond to a clear modeling question.
