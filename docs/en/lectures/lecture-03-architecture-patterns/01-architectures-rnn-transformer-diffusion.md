---
title: "Part B: Core Backbone Choices"
description: RSSM serves as the baseline for comparing the core mechanisms, learning paradigms, and applicable scenarios of Transformer-based and Diffusion-based world models.
lecture: 3
---

# Part B: Core Backbone Choices

Read this page after completing P03. You now have a working RSSM-based agent and can compare alternatives against failures you have actually observed. RNN/RSSM and Transformer are the core comparison needed for P04. The diffusion section is an extension for tasks where visual fidelity, rather than online control latency, is the dominant requirement.

## Review: You Already Have an RNN Baseline

The RSSM from P02 has two parallel paths:

- **Deterministic path** (GRU): $h_t = f_\phi(h_{t-1}, z_{t-1}, a_{t-1})$, capturing smooth dynamic trends
- **Stochastic path**: $z_t \sim q_\phi(\cdot \mid h_t, o_t)$, sampling the uncertainty at the current timestep in latent space

This design, validated in Dreamer V1/V2, achieves solid policy performance on continuous control tasks at very low computational cost. Its limitation is equally clear: **GRU memory capacity degrades as sequences grow longer**, making it inadequate for tasks requiring reasoning across hundreds of steps.

The five architecture families that follow each address this limitation, but take different directions.


## Architecture 1: RNN / RSSM (Your Baseline)

**Representative systems**: [Ha & Schmidhuber World Models (2018)](https://arxiv.org/abs/1803.10122), [Dreamer V1 (2019)](https://arxiv.org/abs/1912.01603), [Dreamer V2 (2020)](https://arxiv.org/abs/2010.02193)

The GRU incrementally updates the hidden state with **O(1)** per-step cost, independent of sequence length. RSSM builds on this by splitting out the stochastic path $z_t$, making uncertainty a first-class citizen of the model (see L02 Part B for the full mechanism).

**Learning paradigm**: Interactive. Collects $(o_t, a_t, r_t, o_{t+1})$ tuples and learns the action-conditioned transition distribution $p(s_{t+1} \mid s_t, a_t)$. The interactive paradigm can answer "what would happen if I took a different action," which observation-only paradigms (pure video) cannot.

**Applicable scenarios**: Simple to moderately complex continuous control tasks (e.g., **DMControl**, the DeepMind Control Suite, a set of standard continuous control benchmarks based on the MuJoCo physics engine, including Cheetah running, Cartpole balancing, Reacher goal reaching, and similar tasks. **Atari**, a set of classic video game benchmarks covering 57 games used to evaluate general decision-making capability), and latency-sensitive online reinforcement learning.

**Limitations**: Weak long-term memory, with the effective memory window of the GRU hidden state typically between 50-100 steps. Generation quality inferior to Diffusion. Data collection on real robots remains expensive.


## Architecture 2: Transformer-based (2022, 2023)

**Representative systems**: [IRIS (2022)](https://arxiv.org/abs/2209.00588), [STORM (2023)](https://arxiv.org/abs/2310.09615)

### Core Mechanism

Replace the GRU with a **Transformer**, tokenize the historical observation sequence $o_{1:t}$ into discrete tokens, and use **self-attention** to compute weights across the entire sequence:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

> **📖 softmax and Q, K, V in self-attention**: Softmax converts an arbitrary real-valued vector $[x_1, x_2, \ldots, x_n]$ into a probability distribution (all elements non-negative and summing to 1): $\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$. Self-attention first projects each position's vector into three roles: **Query** (Q) represents what the current position is "asking," **Key** (K) represents what other positions "offer," and **Value** (V) carries the actual information. $QK^\top$ computes pairwise relevance scores, division by $\sqrt{d_k}$ prevents growing dimensionality from making softmax overly peaked, softmax normalizes the scores into weights, and those weights produce a weighted sum of $V$.

Every position can directly "see" any historical timestep in the sequence, no longer constrained by the GRU's hidden state bottleneck.

### IRIS: Turning Images into "Sentences"

IRIS (Imagination with auto-Regression over an Inner Speech, ICLR 2023) centers on **VQ-VAE quantization**, converting continuous image frames into discrete token sequences. GPT can predict "the next word" because words are discrete and finite, and the probability distribution can be modeled precisely with softmax. By converting images into discrete units analogous to "words," one can directly apply a GPT-style autoregressive Transformer to predict "the next visual word."

> **📖 VQ** (vector quantization) works as follows: (1) the encoder maps an image patch to a continuous vector $z$. (2) the closest vector $e_k$ in the codebook is found ($k = \arg\min_j \|z - e_j\|_2$). (3) the index $k$ of $e_k$ replaces the continuous vector and is passed to the Transformer. During backpropagation, the **straight-through estimator** is used: the forward pass uses the quantized discrete vector, while the backward pass pretends the quantization operation does not exist and passes gradients straight through.

The Transformer in IRIS receives a **sequence of interleaved frame tokens and actions**: each frame is encoded by VQ-VAE into $K$ tokens (e.g., $K=16$, codebook size $N=1024$), and action $a_t$ is inserted as a separate token after each frame's tokens. The Transformer simultaneously predicts three targets: the transition distribution $\hat{z}_{t+1}$ (via cross-entropy loss), the immediate reward $\hat{r}_t$, and the episode termination flag $\hat{d}_t$. The policy is trained entirely within imagined trajectories without touching the real environment. On the Atari 100k benchmark (allowing only 100,000 environment interaction steps, roughly equivalent to 2 hours of real gameplay, to test sample efficiency), IRIS achieves an average **HNS** (Human Normalized Score, which normalizes agent performance to the interval where random policy = 0 and human = 1, with values above 1 indicating superhuman performance) of 1.046, surpassing humans on 10 out of 26 games.

IRIS processes each frame as a pipeline: VQ-VAE encodes the raw frame into a discrete token sequence, the Transformer autoregressively predicts the next-frame token sequence, and VQ-VAE decodes it back into a reconstructed image.

### STORM's Key Improvement: Single-Token Stochastic Latent Variable

STORM (Stochastic Transformer-based wORld Models, NeurIPS 2023) differs from IRIS mainly in its latent variable design. IRIS uses VQ-VAE to represent one frame as multiple discrete tokens ($4 \times 4 = 16$). STORM instead uses a **categorical VAE** to compress an entire frame into a single stochastic latent variable $z_t$ (32 categories, each 32-dimensional, with straight-through gradient estimation), then fuses $z_t$ with action $a_t$ into a **single token** $e_t$ fed into the Transformer:

$$e_t = m_\phi(z_t, a_t), \quad h_{1:T} = f_\phi(e_{1:T})$$

The Transformer processes the sequence with causal masking, and $h_t$ simultaneously predicts the current reward $\hat{r}_t$, continuation flag $\hat{c}_t$, and next-step latent distribution $\hat{\mathcal{Z}}_{t+1}$. The single-token design makes sequences 16 times shorter than IRIS, resulting in much faster training: on a single RTX 3090, using 1.85 hours of real interaction and 4.3 hours of training, STORM achieves 126.7% average human normalized score on the Atari 100k benchmark (the highest level without lookahead search).

> **📖 Teacher Forcing**: During training, the model conditions on **real historical frames** at each timestep rather than its own previous predictions. This makes training more stable and convergence faster, but creates a distribution gap: "always having correct historical frames during training, only having the model's own predicted frames during inference." For autoregressive world models, this is the most common source of error accumulation. In STORM's evaluation metrics, long-horizon PSNR is specifically designed to quantify this gap (see the STORM metrics section in L04).

Compared to DreamerV3's GRU-based RSSM, STORM's Transformer sequence model is stronger at long-sequence modeling and supports parallel training. The trade-off is the removal of RSSM's recurrent hidden state $h_t$: image reconstruction does not use recurrent hidden state information, and long-range context depends entirely on the Transformer's context window.

<figure>
<img src="/storm/storm-world-model.png" alt="STORM Transformer dynamics model architecture" style="width:90%;display:block;margin:0 auto">
<figcaption>Zhang et al. (2023) STORM architecture: a categorical VAE compresses each frame into a single stochastic latent variable z_t, which is fused with action a_t and fed into a causal-masked Transformer. The Transformer simultaneously predicts reward, continuation flag, and next-step latent distribution. The single-token design makes sequences 16 times shorter than IRIS.</figcaption>
</figure>

**Learning paradigm**: Interactive (action-conditioned). Action $a_t$ is concatenated into the token sequence, and the model predicts the future latent distribution conditioned on actions.

**Applicable scenarios**: Complex games (long Atari games, strategy games), tasks requiring multi-step planning. The preferred choice when sufficient compute and data are available.

**Limitations**: Computation scales quadratically with sequence length ($O(T^2)$). Inference latency is higher than RNN. Requires more data to converge.


## Architecture 3: Diffusion-based (2023, 2024)

**Representative systems**: [Diamond (2024)](https://arxiv.org/abs/2405.12399), [GameNGen (Google, 2024)](https://arxiv.org/abs/2408.14837)

### Core Mechanism

Diffusion models generate outputs through **iterative denoising**: Gaussian noise is added to real frames, and the network is trained to predict the noise:

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1};\, \mu_\theta(x_t, t),\, \sigma_t^2 I)$$

In the world model setting, conditioned on historical frames and actions, the diffusion model iteratively "denoises" the next frame. Each denoising step is a full forward pass through the neural network, guided by "action conditioning" to determine "where to remove noise."

> **📖 U-Net**: A convolutional neural network with an encoder-decoder structure, named for its "U" shape. The encoder progressively reduces spatial resolution (extracting features), and the decoder progressively restores resolution (recovering details), with skip connections that pass features from each encoder layer directly to the corresponding decoder layer to preserve high-frequency detail. The **bottleneck** is the lowest layer of the U-shape, where resolution is minimal and information is highly compressed before being gradually expanded. Diffusion world models use U-Net at each denoising step to process images and predict progressively clearer frames.

GameNGen (2024) is the first system to run a complete game engine **in real time** using a neural network, simulating DOOM at 20fps. **The model itself is the game engine.** Generating each frame requires 10-100 denoising iterations, each a full U-Net forward pass, making diffusion world models extremely expensive in **online RL training loops**.

### Diamond: A World Model Combining the Diffusion Process with the RL Training Loop

Diamond (NeurIPS 2024) directly integrates the diffusion process with the reinforcement learning training loop. Conditioned on a number of past frames and the current action, it uses a U-Net to denoise and generate the next frame, with the full generation chain serving as an environment simulator for policy training.

Diamond's key design decision: action information is injected via **cross-attention** (a variant of self-attention where the Query comes from one sequence and the Key and Value come from another, aligning two different sources of information. Here used to let image features "query" action information) into **every resolution layer** of the U-Net, rather than only into the bottleneck (the lowest-resolution layer at the bottom of the U-Net), which tightly aligns the generated frames with the action instructions. On the Atari 100k benchmark, Diamond achieves an average HNS of 1.46, surpassing all prior world model methods while maintaining excellent visual generation quality.

The inherent challenge for diffusion world models is **object persistence**: each frame is denoised independently, and the model does not maintain explicit object state, causing the identity, position, and occlusion relationships of objects to quietly drift in long sequences. Diamond mitigates this by limiting the number of rollout steps and adding a depth consistency penalty to the loss (for more diagnostic methods, see L04).

**Learning paradigm**: Interactive (Diamond is action-conditioned) or observation-only (pure video diffusion models). Observation-only diffusion models are trained on large-scale internet video, learning the visual regularities of the world without action conditioning, and cannot answer "what would happen if I took a different action."

**Applicable scenarios**: Offline video prediction, high-fidelity simulators, film and game content generation. Not suitable for RL scenarios requiring real-time closed-loop control.

**Limitations**: Slow inference (10-100 denoising steps). Difficult to interface directly with policy optimization (the sampling process is non-differentiable). Object persistence is hard to maintain. Training and inference costs are substantial.


## Core Path Checkpoint

For P04, the actionable comparison is RSSM versus Transformer: recurrent state gives low-latency online updates, while attention trades additional compute for access to a longer context. Complete [P04: Swap the Dynamics Backbone](../../projects/p04_transformer_backbone) before continuing. The remaining Part C pages are optional and survey research directions that are not implemented by the project sequence.
