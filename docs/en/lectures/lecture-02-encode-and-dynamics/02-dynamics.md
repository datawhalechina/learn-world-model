---
title: "Part B: Latent Dynamics (GRU, MDN-RNN, RSSM)"
description: From GRU to MDN-RNN to RSSM, understanding how to model future-state dynamics in latent space, and the design that separates prior from posterior.
lecture: 2
---

# Part B: Latent Dynamics

## The Encoder Is Not Enough: We Need to Predict the Future

With a VAE encoder, we can compress the current frame $\mathbf{o}_t$ into $\mathbf{z}_t$. But the central task of a world model is **predicting the future**:

> In latent space, given the current state $\mathbf{z}_t$ and action $\mathbf{a}_t$, predict the next state $\mathbf{z}_{t+1}$.

This prediction capability lets the agent "simulate" the future internally, enabling planning without actually executing actions in the environment. This is the key reason world models reduce sample complexity.

---

## The Simplest Dynamics Model: GRU

The **Gated Recurrent Unit (GRU)** is a foundational tool for sequence modeling. As a dynamics model, the GRU takes $(\mathbf{z}_t, \mathbf{a}_t)$ and predicts the next latent state:

$$
\mathbf{z}_{t+1} = \text{GRU}(\mathbf{z}_t, \mathbf{a}_t; \theta)
$$

> **📖 GRU internals (brief)**: A GRU controls information flow through two gates: the **reset gate** decides "how much of the past to forget", and the **update gate** decides "how much of the old state to retain vs. how much new information to write in". Gate values lie between 0 and 1, determined jointly by the current input and the previous hidden state. This allows the GRU to selectively retain long-term dependencies while discarding irrelevant information, making it better at handling longer sequences than a plain RNN. Compared to an LSTM, the GRU has one fewer gate (no separate memory cell), fewer parameters, and trains faster.

The GRU's strengths are simplicity and stable training. Its limitation is that it produces deterministic predictions and cannot express **uncertainty**. In real environments, the same action can lead to multiple different outcomes (for example, pushing a box might succeed or might get stuck).

---

## MDN-RNN: Modeling Uncertainty

**MDN-RNN (Mixture Density Network + RNN)**, proposed in [Ha & Schmidhuber (2018)](https://arxiv.org/abs/1803.10122), models uncertainty over the next state using a **mixture of Gaussians**:

$$
p(\mathbf{z}_{t+1} | \mathbf{z}_t, \mathbf{a}_t) = \sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(\mathbf{z}_{t+1}; \mu_k, \sigma_k^2)
$$

- $K$ Gaussian components, each with its own mean $\mu_k$ (center of the distribution) and variance $\sigma_k^2$ (width of the distribution)
- **Mixture weights** $\pi_k$: the probability mass of the $k$-th Gaussian component, satisfying $\sum_{k=1}^K \pi_k = 1$, $\pi_k \geq 0$. These can be read as "the probability that the $k$-th future occurs". The network outputs $\pi_k$ values and normalizes them through softmax to ensure the weights sum to 1.

MDN-RNN can capture **multimodal distributions**: the environment may transition to several distinct next states, and the model can represent all of them.

<figure>
<img src="/worldmodels/mdn-rnn.png" alt="MDN-RNN: combination of a mixture density network and an RNN, outputting a multimodal Gaussian mixture distribution" style="width:80%;display:block;margin:0 auto">
<figcaption>MDN-RNN architecture from Ha & Schmidhuber (2018): the RNN hidden state is passed through a fully connected layer to produce K parameter groups (π_k, μ_k, σ_k), representing mixture weights, means, and variances, which together define the Gaussian mixture distribution over the next latent state.</figcaption>
</figure>

---

## RSSM: Separating Deterministic and Stochastic Components

The **RSSM (Recurrent State Space Model)** is the core innovation of the Dreamer series. It splits the state into two parts:

- **Deterministic hidden state** $\mathbf{h}_t$: maintained by an RNN, aggregating information from the historical trajectory, with no stochasticity
- **Stochastic latent state** $\mathbf{z}_t$: sampled from a distribution conditioned on $\mathbf{h}_t$, expressing current uncertainty

**Core equations of the RSSM**:

> **📖 Subscript $\phi$ (phi)**: the subscript $\phi$ in $f_\phi$, $p_\phi$, $q_\phi$ denotes "this function has parameters $\phi$", i.e., the learnable weights of the neural network. $f_\phi(\cdot)$ is read as "function $f$ parameterized by $\phi$". During training, gradient descent updates $\phi$ so that the predictions of these functions become increasingly accurate. Similarly, $\theta$ (theta) that appears later is another commonly used symbol for a distinct set of learnable parameters.

$$
\mathbf{h}_t = f_\phi(\mathbf{h}_{t-1},\ \mathbf{z}_{t-1},\ \mathbf{a}_{t-1})
\quad \text{(deterministic update, GRU/RNN)}
$$

$$
\mathbf{z}_t \sim p_\phi(\mathbf{z}_t \mid \mathbf{h}_t)
\quad \text{(prior: no access to real observations, infers current state from history } h_t \text{ alone; used for pure imagination/prediction)}
$$

$$
\mathbf{z}_t \sim q_\phi(\mathbf{z}_t \mid \mathbf{h}_t,\ \mathbf{o}_t)
\quad \text{(posterior: corrects the prior using real observation } o_t \text{; used during training)}
$$

> **📖 Prior vs. posterior**: these are fundamental concepts in Bayesian statistics. The **prior** is "belief before seeing data", the RSSM's guess about the current state $z_t$ based on historical memory $h_t$. The **posterior** is "belief updated after seeing data", refining the prior with real observation $o_t$ to obtain a more accurate estimate. During training, the posterior generates $z_t$ and the KL loss is computed (measuring the gap between prior and posterior). During inference and imagination, only the prior is available (there is no real $o_t$), so the RSSM rolls forward using the prior alone.

**Why separate them?**

| State | Role | Property |
|-------|------|----------|
| $\mathbf{h}_t$ | Memory | Deterministic, aggregates history |
| $\mathbf{z}_t$ | Perception | Stochastic, expresses uncertainty |

After separation, the model can roll forward using only the prior $p(\mathbf{z}_t | \mathbf{h}_t)$ without real observations, enabling **planning purely in imagination**. This is the fundamental reason for Dreamer's sample efficiency.

The PlaNet paper (Hafner et al., ICML 2019) verified this design through **ablation studies** (systematically removing one component of the model and observing the change in performance, thereby confirming the component's necessity): a purely stochastic path (no deterministic $h_t$) struggles to reliably retain information across multiple steps, and training optimization may fail to find solutions where some dimensions collapse to near-zero variance to store long-term information; a purely deterministic path (no stochastic $z_t$) cannot express the inherent stochasticity of the environment, and the distribution gap between imagined and real trajectories grows larger. **Both paths are indispensable.** The observation model is therefore conditioned on both $h_t$ and $z_t$: $o_t \sim p(o_t | h_t, z_t)$, with deterministic memory and stochastic perception jointly determining the reconstructed image.

---

## Comparison of Three Dynamics Models

| Model | Uncertainty Modeling | Memory Mechanism | Primary Use |
|-------|---------------------|-----------------|-------------|
| **GRU** | None (deterministic output) | Fixed-dimension hidden state $h_t$ | Simple sequence prediction, rapid prototyping |
| **MDN-RNN** | Mixture of Gaussians (multimodal) | Fixed-dimension hidden state $h_t$ | Multimodal uncertainty, Ha & Schmidhuber M-module |
| **RSSM** | Separated prior/posterior (Gaussian) | Dual-track: deterministic $h_t$ + stochastic $z_t$ | Core of Dreamer, supports pure-imagination planning |

The three form a progression: GRU establishes the foundation for sequence modeling, MDN-RNN introduces uncertainty, and RSSM further decouples "memory" from "perceptual uncertainty", enabling the model to roll forward and plan without real observations.
