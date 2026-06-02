---
title: "P04: Swap the Dynamics Backbone"
description: Replace the RSSM with a small causal Transformer (STORM-style). Train on the same task as P03, then compare the two architectures on prediction quality, training speed, and long-horizon rollout stability.
---

# P04: Swap the Dynamics Backbone

**Format**: Jupyter notebook (`p04_transformer_backbone.ipynb`)  
**Prerequisites**: P02, L03 Part A  
**Builds toward**: P05 (this Transformer world model is the second system compared in the evaluation dashboard)

---

## What You Will Build

A Transformer-based world model in a single notebook. The RSSM's GRU is replaced by a small causal Transformer. The design follows STORM: a categorical VAE compresses each frame into one discrete latent token, and a causal Transformer processes the token sequence to predict future latents. Keep the Transformer small (2-4 layers, 128 hidden dimensions) so it trains within the same time budget as P03.

Train on the same environment and data budget as P03 so the P05 comparison is controlled.

---

## Notebook Sections

**Section 1: Categorical VAE**

Implement a categorical VAE: the encoder outputs a categorical distribution over a small codebook (32 categories, each 32-dimensional). Use the straight-through estimator to pass gradients through the discrete sampling. Each frame maps to a single token index.

**Section 2: Causal Transformer**

Implement a small Transformer with causal masking. The input sequence interleaves latent tokens `z_t` and action tokens `a_t`. The Transformer predicts the next token distribution, current reward, and continuation flag.

**Section 3: Training**

Train on the same trajectory data as P03. Use the same number of gradient steps for a fair comparison. Log token prediction loss and reward prediction loss.

**Section 4: Rollout Quality Comparison**

Generate 10-step imagined rollouts from both the RSSM (load P03 checkpoint) and this Transformer model. Decode both back to pixel space. Compute PSNR at each horizon step (1, 3, 5, 10). Plot both PSNR curves on the same axes.

**Section 5: Training Efficiency**

Plot validation loss vs wall-clock time for both architectures on the same axes. Note whether the Transformer's attention cost is observable at the sequence lengths used here.

---

## Deliverables

- Completed notebook with all cells executed
- PSNR vs horizon step: Transformer vs RSSM on the same plot
- Training efficiency comparison plot
- One paragraph comparing the two architectures based on what you observed

---

## Reference

Transformer world models (IRIS and STORM), categorical VAE, and straight-through estimator: L03 Part A. Teacher forcing gap and long-horizon PSNR diagnostic: L04.
