---
title: "P05: World Model Evaluation Dashboard"
description: Load the Dreamer agent from P03 and the Transformer world model from P04. Compute per-model metrics from L04 and display them side by side in a notebook dashboard.
---

# P05: World Model Evaluation Dashboard

**Format**: Jupyter notebook (`p05_evaluation_dashboard.ipynb`)  
**Prerequisites**: P03, P04, L04

---

## What You Will Build

A diagnostic notebook that loads both trained world models, runs them on the same held-out environment episodes, computes L04 metrics for each, and renders the results side by side using inline plots. The goal is not to declare a winner but to make the characteristic failure modes of each architecture visible.

---

## Notebook Sections

**Section 1: Load Both Models**

Load the P03 Dreamer checkpoint (encoder + RSSM + Actor) and the P04 Transformer world model. Freeze both. Run a set of held-out episodes to collect ground-truth trajectories.

**Section 2: Per-Model Metrics**

Compute for each model on the held-out set:

| Metric | Model | What it measures |
|--------|-------|-----------------|
| Reward correlation ρ | Dreamer | How well imagined rewards track real rewards |
| Token prediction loss | Transformer | One-step latent prediction accuracy |
| Long-horizon PSNR (steps 1, 3, 5, 10) | Both | How fast prediction quality degrades |
| Latent drift | Both | Divergence between imagined and real latent trajectories |

**Section 3: Side-by-Side Plots**

Render in the notebook:
- PSNR vs horizon step: both models on one axes
- Latent drift curves: both models on one axes
- Decoded frame sequences: ground truth, Dreamer imagined, Transformer imagined (a grid of frames at steps 1, 5, 10)
- Summary metrics table printed inline

**Section 4: Diagnostic Summary**

For each metric where the two models differ substantially, write one markdown cell explaining the architectural reason. Use the L04 framework: which failure mode does this metric surface, and which design choice explains the gap?

---

## Deliverables

- Completed notebook with all cells executed and all plots rendered inline
- Metrics table with values filled in for both models
- PSNR and latent drift plots
- Diagnostic summary (5-10 sentences in markdown cells)

---

## Reference

Per-model metrics and interpretation: L04. Horizon drift and mitigation strategies: L04.
