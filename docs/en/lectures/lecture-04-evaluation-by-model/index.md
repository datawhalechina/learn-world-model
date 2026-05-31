---
title: "Evaluation by Model: Metrics and Diagnostic Methods"
description: A systematic review of model-specific evaluation metrics for Dreamer, MuZero, TD-MPC, STORM, and diffusion world models. Understand horizon drift as the universal failure mode, master real-world deployment evaluation, and learn to diagnose world models by targeting the right metrics.
lecture: 4
difficulty: Advanced
---

# L04 · Evaluation by Model: Metrics and Diagnostic Methods

Many tutorials default to a universal metric checklist: FID, PSNR, reward curves, then score every model against the same rubric. This appears fair but in practice conceals the true failure modes of each architecture.

**Core principle**: *Metrics must align with the failure modes of the architecture.*

This lecture is organized into three parts:

- **Model-specific metrics**: Dreamer (FID + reward correlation), MuZero (value accuracy + visit entropy), TD-MPC (latent consistency loss), STORM (token loss + long-horizon PSNR), diffusion world models (physics consistency)
- **Universal failure mode**: horizon drift and mitigation strategies
- **Real-world deployment evaluation**: limitations of paper metrics, seven common pitfalls, three pragmatic deployment strategies

It is recommended to complete P03 through P05 before reading this lecture. Having run your own numbers makes many of the diagnostic rules immediately clear.
