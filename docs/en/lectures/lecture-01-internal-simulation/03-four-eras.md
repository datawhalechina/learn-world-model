---
title: Four Eras
description: From the 1950s RNN origins, to Ha & Schmidhuber's dream-based learning, to Dreamer's latent space, to JEPA's video-as-world paradigm.
lecture: 1
---

# Four Eras

### Era One: Theoretical Foundations (1950s–2017)

Recurrent neural networks (RNNs), Kalman filters, hidden Markov models: over seven decades, researchers across different fields independently built tools for "predicting future states," but this work was scattered across corners of control theory, speech recognition, and robotics, never unified under the name "world model."

### Era Two: Ha & Schmidhuber's "Learning in Dreams" (2018)

In 2018, David Ha and Jürgen Schmidhuber published the now widely cited paper [World Models](https://arxiv.org/abs/1803.10122).

They unified these scattered ideas with an elegant three-module framework:

- **V (Vision module)**: a CNN encoder that compresses each game frame into a low-dimensional vector $z$
- **M (Memory module)**: an MDN-RNN that receives $z$ and the previous action, then predicts the next $z$; this is the "world model" of the system, responsible for modeling the future
- **C (Controller)**: an extremely simple linear layer that takes the current $z$ and M's hidden state as input and outputs an action; this is the policy network, responsible for decision-making

The three modules are V (Vision encoder), M (Memory/MDN-RNN), and C (Controller). V compresses each frame into a latent vector, M maintains a predictive model of the world given past latents and actions, and C maps the current latent and M's hidden state directly to an action.

The most compelling aspect of their experiment was placing controller C inside a **virtual environment** hallucinated by memory module M, training there, then transferring the policy to the real game. On the Car Racing task (an OpenAI Gym 2D racing environment with a top-down camera view, where the goal is to complete a randomly generated track), a policy trained purely in dreams could achieve solid results in the real environment. The VizDoom task (an RL research environment based on the first-person shooter Doom, with a first-person 3D perspective and significantly higher task complexity than Car Racing) exposed a more fundamental problem: the controller learned to exploit errors in the world model to manufacture artificially high scores (model exploitation), "cheating" in the dream rather than learning genuine skills. They ultimately needed to introduce a temperature parameter to increase dream diversity before transfer became viable. This "cheating" problem later became one of the central challenges in the world model field.

Ha & Schmidhuber's framing (train entirely inside a hallucinated environment, then transfer to the real one) brought the world model idea into mainstream awareness for the first time.

<figure>
<img src="/worldmodels/world-models-card.png" alt="Ha & Schmidhuber (2018) World Models experiment results: Car Racing and VizDoom side by side" style="width:90%;display:block;margin:0 auto">
<figcaption>Ha & Schmidhuber (2018) experiment overview. Left: the agent navigating the Car Racing track after training inside the M-module dream, showing that a policy trained purely in imagination can transfer to the real environment. Right: the VizDoom task, where the controller learned to exploit world model errors to manufacture artificially high scores (model exploitation), requiring temperature-based dream diversification before transfer became viable.</figcaption>
</figure>

### Era Three: Dreamer and Latent Space (2019)

In 2019, Danijar Hafner and colleagues released [Dreamer V1](https://arxiv.org/abs/1912.01603), introducing **RSSM** (Recurrent State Space Model). The full mechanism is covered in Lecture 2; the core idea is to split the state into two parallel paths: a deterministic history memory and a stochastic uncertainty component.

> **📖 Latent space**: after an encoder compresses high-dimensional raw data (such as the tens of thousands of pixels in an image) into a low-dimensional vector, the space that vector lives in is called the latent space. "Latent" means this representation does not directly correspond to raw pixels; instead, it captures the semantic structure of the data. Operating in latent space is far more efficient than operating in pixel space, because the dimensionality is lower and irrelevant information has been filtered out.

Unlike Ha & Schmidhuber's approach, Dreamer no longer needs to reconstruct images in pixel space. It does everything directly in **latent space**: prediction, planning, and reward learning.

Dreamer substantially outperformed prior model-free methods on Atari games and continuous control tasks, demonstrating that latent space learning is a viable and efficient path.

### Era Four: Video as World (2023+)

Around 2023, two parallel lines of research converged on the same question: **can the physical laws of the world be learned from video alone?**

- **JEPA** (Joint Embedding Predictive Architecture, LeCun's team, [2022](https://openreview.net/forum?id=BZ5a1r-kVsf)): abandons pixel reconstruction and makes predictions purely in a semantic embedding space. "I don't need to draw your face; I just need to know who you are."

The evolutionary logic across the four eras is clear: from "how to predict states in a sequence" (Era 1), to "how to train a policy in dreams" (Era 2), to "how to compress perception in latent space" (Era 3), to "how to retain only semantics and discard noise" (Era 4). Each step is a direct response to the bottleneck of the previous one.

The next page discusses why this evolution suddenly accelerated around 2024.
