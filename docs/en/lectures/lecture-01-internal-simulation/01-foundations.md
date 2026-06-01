---
title: "Intellectual Foundations: Craik, Predictive Coding, and the Internal Model Principle"
description: Starting from Craik's 1943 miniature model, understand predictive coding and the Internal Model Principle, and clarify the conceptual boundary between broad and narrow world models.
lecture: 1
---

# Intellectual Foundations

## The First Foundation: Craik's Miniature Model (1943)

British psychologist **Kenneth Craik** wrote a slim book during World War II, *The Nature of Explanation* [1]. He proposed an idea far ahead of its time:

> *"If the organism carries a 'small-scale model' of external reality and of its own possible actions within its head, it is able to try out various alternatives, conclude which is the best of them, react to future situations before they arise."*

Craik argued that the brain is not a passive black box that receives stimuli and emits responses. Instead, it actively maintains an **internal simulator**. This simulator can fast-forward into the future and replay the past, allowing a creature to filter out the best action before any real cost is incurred.

Perception feeds the internal model, which generates predictions to guide decisions. Prediction errors flow back to correct the model.

Tragically, Craik died in a bicycle accident in 1945 at only 31 years old. His ideas lay dormant for decades, only to be rediscovered with the rise of cognitive science and neuroscience.

---

## The Brain's Prediction Mechanism: Predictive Coding (1990s)

In the 1990s, neuroscientists began using **[predictive coding](https://pubmed.ncbi.nlm.nih.gov/10195184/)** to explain how the brain works.

The core idea is surprisingly simple.

The brain does not "see" the world; it **predicts** the world and then only processes the parts it got wrong.

The visual cortex does not faithfully relay every pixel from the eyes to higher brain areas, which would be far too costly energetically. Instead, higher-level areas continuously send predictions downward to lower-level areas, which only need to propagate the **error between the prediction and the actual sensory signal** back up.

When you walk into a familiar room, the brain hardly needs to process anything because everything is within expectation. But if a chair has been moved, that misaligned signal is immediately amplified and draws attention.

This mechanism explains why we are so sensitive to change and so oblivious to familiar backgrounds: **the accurately predicted parts are compressed away, and only errors are worth transmitting**.

---

## The Insight from Control Theory: The Internal Model Principle (1960s)

Around the same time, the field of control engineering independently arrived at a similar insight. In the 1960s, the **[Internal Model Principle](https://doi.org/10.1016/0005-1098(76)90006-6)** was formally stated:

> *To achieve perfect control of a system, the controller must contain an internal model of that system.*

This sounds like engineering jargon, but the intuition is clear: a self-driving car maintaining its lane through a curve must have an algorithm that "knows" how the vehicle behaves dynamically in a curve, not through reaction, but through **anticipation**.

This principle appears throughout robotics, spacecraft control, and economic modeling, and it became the theoretical foundation for model-based methods in reinforcement learning. To control something, you must first understand it. The Internal Model Principle turns this commonsense observation into a mathematical necessity.

---

## A Common Confusion: Broad vs. Narrow World Models

Before entering the historical narrative, one conceptual boundary must be established clearly, because every example that follows touches it: **the term "world model" does not mean the same thing in every context.**

### Broad World Models: Any Predictive Model Qualifies

Broadly speaking, any model that can predict "what happens next" can be called a world model.

- A language model predicting the next **token** (the basic unit a language model processes, which can be a word, a character, or a sub-word fragment) is a broad world model.
- A video generation model predicting the next frame is a broad world model.
- A weather forecasting model predicting tomorrow's temperature is a broad world model.

Under this definition, Veo, Genie, and Cosmos all fit under the "world model" umbrella. They genuinely learn statistical regularities of the world: how light and shadow change, how objects move, how scenes evolve.

### Narrow World Models: Must Be Action-Conditioned

In robotics and reinforcement learning (RL), "world model" carries a stricter meaning: **it must be conditioned on actions**.

It is not merely "what does the next frame look like," but "**what happens to the world after I take this action**." Formally:

$$p(o_{t+1} \mid o_t, a_t)$$

> **📖 Subscript convention**: The subscript `t` in formulas denotes the time step, a discrete counter: `t=0` is the first step, `t=1` is the second, and so on. $o_t$ is read as "the observation at time t," $a_t$ as "the action at time t," and $o_{t+1}$ as "the observation at the next time step." This subscript notation runs throughout the entire curriculum: any variable with a `t` subscript refers to its value at that time step; a `t+1` subscript refers to the next step's value.

Here `a_t` is the action the agent executes at time `t`. The presence of this one conditioning variable transforms the world model from a "bystander" into a "participant": it can tell you not only how the world will evolve, but what consequences your choices will bring.

A broad world model is a prophet, telling you "what will happen." A narrow world model is an advisor, telling you "what will happen if you do this." Robots need advisors, not only prophets.

### Three Practical Classification Questions

When encountering a specific model, three questions quickly reveal which kind of world model it is:

| Dimension | Options | Representative Systems |
|-----------|---------|----------------------|
| **What does it predict?** | Pixels / raw frames | Video diffusion models |
| | Latent vectors (low-dimensional compressed representations inside the network) | Dreamer (an RL system that trains a policy in latent space, covered in L02-L03), RSSM (the dynamics core of Dreamer, covered in L02) |
| | Structured state (no pixels, only information needed for decisions) | MuZero, TD-MPC |
| | Actions themselves (latent actions inferred automatically from video) | Genie |
| **Does it accept actions?** | No, passive video prediction | Veo |
| | Yes, given actions, controllable simulation | Dreamer, world model robots |
| | Learns its own actions, latent action | Genie |
| **What purpose does it serve?** | Generating content (video, images) | Veo |
| | Evaluating policies / counterfactual simulation | Autonomous driving testing |
| | Training a policy inside a "dream" | Dreamer, Ha & Schmidhuber |
| | Understanding physics, transferring knowledge | JEPA, foundation world models |

This curriculum focuses on **narrow world models**: action-conditioned dynamics models that can be used for planning and policy learning.

<figure>
<img src="/worldmodels/world-model-schematic.png" alt="World model three-module schematic: V visual encoder, M dynamics predictor, C controller" style="width:80%;display:block;margin:0 auto">
<figcaption>The three-module world model structure from Ha & Schmidhuber (2018): V compresses high-dimensional pixels into a low-dimensional latent vector z, M conditions on z and action a to predict the next z, and C directly outputs actions from z and M's hidden state. This framework cleanly separates the responsibilities of perception, prediction, and decision-making. Later architectures including RSSM and Dreamer both build on this foundation.</figcaption>
</figure>
