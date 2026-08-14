---
title: Observation Encoding and Latent Dynamics
description: Learn how to compress high-dimensional pixels into compact latent representations, and how to model the dynamics of future states in latent space.
lecture: 2
difficulty: intermediate
---

# Lecture 2: Observation Encoding and Latent Dynamics

The core problem of Dreamer breaks into two questions: **how to compress perception**, and **how to predict the future**. This lecture addresses each question in turn.

- **Observation Encoding**: why compression is necessary, the encoder-decoder structure of a VAE, intuition behind the ELBO loss, and the structure of a CNN encoder
- **Latent Dynamics**: starting from the simplest GRU, moving through MDN-RNN's uncertainty modeling, and arriving at RSSM's deterministic/stochastic dual-path design

Read Observation Encoding, complete P01, then return for Latent Dynamics and complete P02. This interleaving lets the dynamics model operate on a representation you have already trained and inspected.
