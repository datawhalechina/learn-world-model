---
title: "P01: Train a VAE Encoder"
description: Train a small convolutional VAE on 64×64 pixel observations. Understand ELBO loss, the reparameterization trick, and what a well-structured latent space looks like.
---

# P01: Train a VAE Encoder

**Format**: Jupyter notebook (`p01_vae_encoder.ipynb`)  
**Prerequisites**: L02 Part A  
**Builds toward**: P02 (the trained encoder is reused as the perception module)

---

## What You Will Build

A convolutional Variational Autoencoder that compresses 64×64 RGB frames into a compact latent vector `z`, trained end-to-end with the ELBO objective. The entire project fits in a single notebook: data loading, model definition, training loop, loss curves, and latent visualization all run in sequence.

The model is intentionally small: a 4-layer CNN encoder, a 32-dimensional latent space, and a mirrored CNN decoder. It trains to a useful reconstruction quality within 20-30 minutes on CPU, or a few minutes with a GPU.

---

## Notebook Sections

**Section 1: Setup**

Install dependencies (`torch`, `torchvision`, `matplotlib`) and define the dataset. Use a small pixel-observation dataset: rendered CartPole frames at 64×64, or a downloaded set of environment screenshots.

**Section 2: Model**

Define the `Encoder` (CNN + linear head outputting μ and σ) and `Decoder` (linear + transposed CNN). Implement the reparameterization trick: `z = μ + σ * ε` where `ε ~ N(0, I)`.

**Section 3: ELBO Loss**

Implement the two-term ELBO loss: pixel reconstruction (MSE or BCE) and KL divergence between the encoder output and the standard normal prior. Verify both terms separately before combining.

**Section 4: Training Loop**

Train for a fixed number of epochs. Log and plot reconstruction loss and KL divergence separately on the same figure. Confirm both decrease.

**Section 5: Latent Space Visualization**

Load the trained encoder. Use interactive sliders (ipywidgets or a manual loop) to vary individual dimensions of `z` and display the decoded image inline. At least one dimension should correspond to a recognizable semantic factor.

---

## Deliverables

- Completed notebook with all cells executed and outputs visible
- ELBO loss curve (reconstruction and KL plotted separately)
- Latent slider visualization showing at least 3 dimensions

---

## Reference

VAE architecture and ELBO derivation: L02 Part A. Reparameterization trick and KL intuition explained there in detail.
