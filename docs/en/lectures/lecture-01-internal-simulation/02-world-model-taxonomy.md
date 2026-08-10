---
title: A Rigorous Taxonomy of World Models
---

# A Rigorous Taxonomy of World Models: From Technical Labels to Cognitive Function

## The clean separation: history is not taxonomy

A rigorous tutorial should not confuse two useful but different questions:

1. **Historical lineage**: how did world-model research evolve?
2. **Functional capability**: what can a given world model actually do?

The “four eras” framing belongs to the first question. It is a narrative arc: early recurrent prediction, the Ha and Schmidhuber formulation, Dreamer-style latent imagination, and JEPA-style representation/prediction. This is useful for orientation, but it is not the deepest classification of world models.

The L1-L5 ladder belongs to the second question. It classifies a model by its operational capability: whether it merely compresses the present, predicts the future, supports action-conditioned counterfactuals, couples prediction to value and planning, or improves itself through error-driven interaction.

So the tutorial uses a two-axis view:

| Axis | What it answers | Role in the course |
| --- | --- | --- |
| Historical eras | “How did the field get here?” | Pedagogical framing |
| L1-L5 capability ladder | “What kind of world model is this?” | Main taxonomy |

This avoids a common mistake: treating every famous technique as a separate kind of world model. DINO, MAE, JEPA, NeRF, video prediction, Dreamer, MuZero, and simulators are not peers in a flat list. They occupy different capability levels and often solve different parts of the world-model problem.

> **📖 A quick preview of the recurring names**: this taxonomy uses a handful of systems as running examples before the curriculum builds them from scratch. **JEPA** (Joint Embedding Predictive Architecture) predicts future representations in semantic space instead of pixels (full mechanism in L03). **Dreamer** and its dynamics core **RSSM** (Recurrent State Space Model) compress observations into a latent state and predict forward in that latent space (built by hand in L02, compared against other architectures in L03). **MuZero** predicts rewards and values directly from an internal state with no pixel reconstruction at all, using tree search to plan (covered in L03 Part B). **TD-MPC** combines a learned Q-function with search-based planning (also L03 Part B). None of these need to be understood in depth yet: what matters here is only which capability level each one reaches.

> **📖 DINO, MAE, CLIP, ViT**: these four names recur throughout this taxonomy as examples of L1 representation learning, so it helps to fix them before going further. **ViT** (Vision Transformer, [Dosovitskiy et al., 2021](https://arxiv.org/abs/2010.11929)) applies the Transformer self-attention mechanism (introduced in L03) directly to image patches instead of convolutions: an image is cut into fixed-size patches, each patch is treated as a token, and self-attention runs over the patch sequence. **DINO** (self-DIstillation with NO labels, [Caron et al., 2021](https://arxiv.org/abs/2104.14294), Meta AI) trains a ViT with self-supervision alone, no labels: a student network is trained to match the output of a slowly-updated teacher network (via EMA, the same mechanism used later in JEPA) on different augmented views of the same image, and the resulting features cluster by object semantics without ever being told what the objects are. **CLIP** (Contrastive Language-Image Pretraining, [Radford et al., 2021](https://arxiv.org/abs/2103.00020), OpenAI) trains an image encoder and a text encoder jointly, pulling the representations of matching image-caption pairs together and pushing mismatched pairs apart, producing representations aligned across the visual and language modalities. **MAE** (Masked Autoencoder, [He et al., 2021](https://arxiv.org/abs/2111.06377)) is trained by masking a large fraction of an image's patches (typically 75%) and reconstructing the missing pixels from the visible ones, similar in spirit to how BERT masks words in text. All four are representation learners: they produce a compressed, semantically structured encoding of an image, which is why they sit at L1 in this taxonomy rather than higher, they do not by themselves predict the future or condition on actions.

## The Four Eras, Briefly

**Era One: Theoretical Foundations (1950s-2017)**. Recurrent neural networks, Kalman filters, hidden Markov models: over seven decades, researchers across control theory, speech recognition, and robotics independently built tools for predicting future states, but this work was never unified under the name "world model."

**Era Two: Ha and Schmidhuber's "Learning in Dreams" (2018)**. Ha and Schmidhuber's [World Models](https://arxiv.org/abs/1803.10122) unified these scattered ideas with a three-module framework: a **V**ision encoder compresses each frame into a latent vector, an **M**emory module (MDN-RNN) predicts how that vector evolves given past latents and actions, and a **C**ontroller maps the current latent and the memory module's hidden state directly to an action. Training the controller entirely inside a hallucinated environment produced by the memory module, then transferring the policy to the real game, brought the world-model idea into mainstream awareness for the first time.

**Era Three: Dreamer and Latent Space (2019)**. Hafner et al.'s [Dreamer V1](https://arxiv.org/abs/1912.01603) introduced RSSM (Recurrent State Space Model, full mechanism in Lecture 2), splitting state into a deterministic history path and a stochastic uncertainty path. Unlike Ha and Schmidhuber's approach, Dreamer never reconstructs images in pixel space: prediction, planning, and reward learning all happen directly in latent space, substantially outperforming prior model-free methods on Atari and continuous control.

**Era Four: Video as World (2023+)**. JEPA (Joint Embedding Predictive Architecture, LeCun's team, [2022](https://openreview.net/forum?id=BZ5a1r-kVsf)) abandons pixel reconstruction entirely and predicts purely in a semantic embedding space: "I don't need to draw your face; I just need to know who you are."

The evolutionary logic across the four eras: from "how to predict states in a sequence" (Era 1), to "how to train a policy in dreams" (Era 2), to "how to compress perception in latent space" (Era 3), to "how to retain only semantics and discard noise" (Era 4). Each step is a direct response to the bottleneck of the previous one.

## How the Four Eras Map Onto L1-L5

The historical eras and the capability ladder can be combined without contradiction:

| Historical framing | Typical contribution | Capability interpretation |
| --- | --- | --- |
| Early recurrent prediction | Learn compact hidden states and predict sequences | L1-L2 |
| Ha and Schmidhuber world models | Separate representation, memory/dynamics, and controller | L1-L3, with an agent interface |
| Dreamer-style latent imagination | Plan and learn policies inside a learned latent dynamics model | L3-L4 |
| JEPA-style representation/prediction | Learn abstract predictive representations without reconstructing every pixel | L1-L2, potentially a foundation for L3-L5 |

This table also shows why a single method should not be overclaimed. JEPA is extremely important, but by itself it is not automatically an agentic world model. Dreamer is closer to the agentic sense because its learned dynamics model is used for policy learning. A simulator such as MuJoCo is runnable, but it is not a learned internal world model unless the agent internalizes or approximates its dynamics.

In this course, the phrase **world model** is used strictly when a system contains an internal model that supports prediction, counterfactual evaluation, planning, or self-correction. Systems that only provide representations are treated as **world-model components** unless they are integrated into a larger predictive or agentic loop.

The term “world model” is used very broadly. Self-supervised vision, video generation, 3D reconstruction, physical simulation, reinforcement learning, and embodied AI all sometimes claim to build world models. To avoid conceptual inflation, this course uses a stricter framework.

The closer a system is to a complete world model, the more it should answer three questions:

1. **Does it learn or maintain an internal representation of world state?**
2. **Can it predict future states from the current state and possible actions?**
3. **Can those predictions be used by an agent for planning, control, or decision-making?**

These questions define three increasingly strong meanings of “world model”. They clean up many confusions in the literature, but they are still not the final taxonomy we want. A more powerful taxonomy should not only ask which paper family a model belongs to. It should ask: **what operational capability does this model give the agent?**

## L1-L5: Classification by Operational Capability

“Reconstruct the world,” “predict the next step,” and “it runs” are useful intuitive entry points, but they still describe surface behavior. A stronger taxonomy should start from what the agent can actually do. This course uses a five-level capability ladder as its main taxonomy. Each level below gives its core question, a formal expression, typical examples, and its limits.

### L1 Compression Models: What is here?

Compression models turn high-dimensional observations into computable, memorable, comparable internal states. They answer "what can the world I currently see be represented as." They typically do not unroll the future or serve action directly, so they are better described as world-model components rather than complete world models.

Formally:

$$z_t = \text{Encoder}(o_t)$$

where $o_t$ is the high-dimensional observation at time $t$ (pixels, point clouds, etc.) and $z_t$ is the compressed internal state.

Typical examples:

- DINO / MAE / CLIP-style representation
- autoencoder / VAE encoder
- object-centric representation

Core capabilities: pixels to latent state, local observations to stable objects, noisy detail to task-relevant variables.

Limits: it knows "what is here," but not necessarily "what happens next."

### L2 Dynamics Models: What happens next?

Dynamics models do more than represent the current world; they learn how state changes over time, predicting "if the world keeps evolving, what happens next." Prediction can happen in pixel space, feature space, object space, or 3D space.

Formally:

$$z_{t+1} = \text{Predictor}(z_{\le t})$$

JEPA is the canonical example at this level, and it further restricts prediction to a mapping between visible and masked patches: $z_{\text{masked}} = \text{Predictor}(z_{\text{visible}}, \Delta)$, where $\Delta$ encodes the position of the masked region. Video world models take the pixel-space version of the same idea: $I_{t+1} = \text{Generator}(I_{\le t}, c)$, generating the next frame directly from history frames $I_{\le t}$ and an optional semantic prompt $c$.

Typical examples:

- JEPA / latent dynamics
- video prediction / video diffusion
- scene flow / object dynamics

Core capabilities: temporal prediction, latent rollout, uncertainty over futures.

Limits: it can predict the future, but not necessarily "what my action would change."

### L3 Action-Conditioned Models: What if I act?

Action-conditioned models fold the agent's action into world evolution. They do not merely predict what naturally happens next; they predict "what happens if I take this action."

Formally:

$$s_{t+1} = f(s_t, a_t)$$

where $a_t$ is the action the agent takes at time $t$. This single condition turns the world model from a bystander into a participant.

Typical examples:

- model-based RL dynamics model
- robotics forward model
- controllable video generation
- action-conditioned latent transition

Core capabilities: action-conditioned counterfactual prediction, imagined trajectories under candidate policies.

Limits: it can answer single-step or short-horizon counterfactuals, but not necessarily long-horizon planning, and it does not necessarily know which consequences are worth pursuing.

### L4 Value-Coupled Models: What matters?

Value-coupled models bind world prediction to goals, rewards, preferences, or survival constraints, answering "which futures are better, which are more dangerous." This is the strictest sense of an agentic world model used in this course: prediction is used directly for planning, control, and decision-making.

Formally:

$$a^* = \arg\max_{a} \; \text{Value}\big(\text{Rollout}_{\text{WM}}(s, a)\big)$$

That is, the world model rolls out multiple candidate actions in parallel imagination, a value function or evaluator scores them, and the best action is selected. Dreamer does this with a learned actor-critic inside latent imagination; MuZero does the same thing with search.

Typical examples:

- Dreamer-style actor-critic in imagination
- MuZero-style reward/value prediction
- learned cost models for control
- preference-conditioned world models

Core capabilities: reward / value prediction, planning over imagined futures, credit assignment through latent rollouts.

Limits: it knows which futures are more valuable, but does not necessarily keep revising its own world assumptions, and errors in the model itself can be amplified by the planning process.

### L5 Self-Correcting Models: How do I improve my model of the world?

Self-correcting models close the loop between prediction error, exploration, and model update. They do not just use a world model; they actively improve it: detecting model error, choosing experiments that reduce uncertainty, updating beliefs after intervention, and maintaining a growing world model across tasks. This is a higher-order world model, one that not only simulates the world but also notices where its simulation is unreliable.

Typical examples:

- active inference
- curiosity-driven model learning
- uncertainty-guided exploration
- lifelong world-model learning
- scientific discovery agents

Core capabilities: detect model error, choose experiments that reduce uncertainty, update beliefs after intervention, maintain a growing world model across tasks.

## The Final Ability Ladder

This course does not use a binary label such as “is this a world model or not?” Instead, it uses an ability ladder:

| Level | Core Question | Formalization | Typical Examples | Strict Term |
| --- | --- | --- | --- | --- |
| L1 Compression | What is here? | $z_t = \text{Encoder}(o_t)$ | DINO, MAE, NeRF encoder | world-model component |
| L2 Dynamics | What happens next? | $z_{t+1} = \text{Predictor}(z_{\le t})$ | JEPA, video prediction, scene flow | predictive world model |
| L3 Action-Conditioning | What if I act? | $s_{t+1} = f(s_t, a_t)$ | robotics forward model, action-conditioned dynamics | controllable world model |
| L4 Value-Coupling | What matters? | $a^* = \arg\max_a \text{Value}(\text{Rollout}_{\text{WM}}(s,a))$ | Dreamer, MuZero | agentic world model |
| L5 Self-Correction | How do I improve? | Active exploration and model update (no single closed form) | active inference, curiosity, lifelong agents | self-improving world model |

This taxonomy is stronger than the common “reconstruct / predict / run the world” grid because it classifies models by agent capability rather than surface behavior. A system can implement these abilities over pixels, objects, 3D scenes, language, physical states, or abstract latent variables. The modality is not the essence. **Operational counterfactual ability is the essence.**

There is also a class of systems commonly mislabeled as world models: physics simulators such as MuJoCo, Brax, and Isaac Gym; rule-based or procedurally generated environments such as Atari, Snake, and Minecraft; and game engines in general. They genuinely "contain a world," and they matter a great deal for training world models because they supply data and evaluation environments. But they are usually not internal models the agent has learned itself. Unless the agent has internalized their regularities into its own internal model, they count as **external simulators** and sit outside the L1-L5 ladder.

## Two-Dimensional Classification: Object × Capability

A popular meme grid, the "world model nine-grid" (reconstruction / predict-next-step / runnable-simulation on one axis, features-latents / objects-3D / pixels-video on the other), captures exactly the intuition in this section: DINO, JEPA, and Dreamer each occupy one cell; NeRF, scene flow, and MuJoCo occupy others. This is the same idea as the three-question table in [Foundations](./01-foundations): "what does it predict / does it condition on actions / what purpose does it serve," laid out differently. It arranges models along two orthogonal axes:

- **Horizontal: modeling space**, what the representation is over: features/latents, objects/3D, pixels/video, or physical state.
- **Vertical: capability level**, what the model can do in that space: reconstruction, prediction, or action-conditioned closure, corresponding to L1-L4 above.

| Representation object | Reconstruction (L1) | Prediction (L2) | Action closure (L3-L4) | Representative formula |
| --- | --- | --- | --- | --- |
| Features / latents | MAE, autoencoder | JEPA, latent dynamics | Dreamer latent imagination | $z_{\text{masked}} = \text{Predictor}(z_{\text{visible}}, \Delta)$ |
| Objects / 3D | NeRF, 3D Gaussian Splatting | scene flow, object dynamics | model-based manipulation | $I = \text{Renderer}(\Theta, c)$ |
| Pixels / video | image/video reconstruction | video diffusion, video prediction | visual model predictive control | $I_{t+1} = \text{Generator}(I_{\le t}, c)$ |
| State / physical quantities | state estimator | learned dynamics | MPC, model-based RL | $s_{t+1} = f(s_t, a_t)$ |

In the NeRF / 3D Gaussian Splatting formula, $\Theta$ is the scene representation (NeRF weights or a 3DGS Gaussian set), $c$ is the query condition (viewpoint, timestamp, or pose), and $I$ is the rendered image.

The key point of this table is: **the object axis does not determine whether something is a complete world model; the capability axis determines the strict level**. NeRF can be an excellent 3D world representation, but if it is only static reconstruction, it stays at the L1 representational level. Dreamer is closer to a complete world model precisely because it connects latent prediction to action learning, reaching L3-L4. This is also why DINO and MAE usually stay at L1, JEPA and video prediction reach L2, action-conditioned dynamics reach L3, Dreamer and MuZero reach L4, and only agents that actively design experiments and revise their own assumptions reach L5, a level with no corresponding cell in the nine-grid, since it goes beyond what the "modeling space" axis alone can describe.

## Application: From the L1-L5 Ladder to Seven Planning Architectures

The ladder above answers "what kind of world-modeling capability does this system have." For people building planning pipelines for robots or autonomous vehicles, the more practical question is "how does a world model actually get wired into planning." The seven paths below, drawn from Zhijian Qiao's (HKUST UAV Group) survey of current literature, each map back to a level on the L1-L5 ladder, connecting the abstract ladder to concrete architecture choices.

| Path | One-line summary | Corresponding capability level |
| --- | --- | --- |
| 1. Representation pretraining | Pretrain an encoder on unlabeled video, then attach only the encoder to a planner at inference | L1, the foundation for higher levels |
| 2. Slow-fast hierarchy | A VLM issues high-level instructions at low frequency while an end-to-end planner executes at high frequency | L1-L3 combined, with the instruction as an external condition rather than an internal prediction |
| 3. VLA with Chain-of-Causality (CoC) | A single autoregressive stream outputs a chain of thought before the action, supervising *how* vision maps to action, not just the outcome | L3, folding both action-conditioned prediction and the decision process itself into the training signal |
| 4. Dense supervision | Train a future-scene-prediction branch and an action/trajectory branch jointly; only the action branch runs at inference | L2's dense gradient feeds back into L3's planning head |
| 5. Generative goal / visual trace | First “imagine” a goal image or visual trace, then use an inverse dynamics model to turn the imagined trace into actions | L2 produces a visual chain of thought, L3 converts it into action |
| 6. Action selection via world model (evaluator) | Treat the world model as a black-box simulator, roll out multiple candidate actions in parallel, score them with pixels, point clouds, latents, or a VLM, then execute the best one under a receding horizon | L4, directly matching the $a^* = \arg\max_a \text{Value}(\text{Rollout}_{\text{WM}}(s,a))$ formula from the previous section |
| 7. Closed-loop simulation | Use the world model offline to manufacture out-of-distribution scenarios: for imitation learning, perturb trajectories and let an expert planner correct them to augment the dataset; for reinforcement learning, sample at high throughput inside imagination to train the policy | L4-L5, the world model continuously improves training data and the policy itself |

A few things worth noting:

- Paths one through three are really the same idea at three levels of coupling. Path one swaps out the prediction head after pretraining; path two keeps two separate systems that cooperate; path three fuses prediction and decision-making into a single autoregressive stream. Tighter coupling means training-time and test-time behavior match more closely, but engineering complexity rises accordingly.
- Paths four and five both provide “dense supervision,” but the form of the supervision differs. Path four directly predicts the future scene itself; path five first generates a visual goal or trace and hands it to an inverse dynamics model to convert into actions. Both aim to mitigate the sparsity of action labels.
- The key difference between paths six and seven is **online** versus **offline** use of the world model. Path six calls the world model at deployment time to evaluate candidate actions in real time. Path seven uses the world model offline, before training, to generate or augment data; the world model does not participate in inference once training is done.
- Whether a language world model (closer to paths two and three) or a video world model (closer to path five) is better suited for “imagination” is still an open question. Language models are information-dense but depend on large amounts of manually labeled causal chains and may filter out details a planner actually needs. Video models preserve raw pixels and are cheaper to label, but cost more to compute and can more easily latch onto background noise that has nothing to do with planning.

Together, these seven paths make one point: **a world model's value for planning is not just “predicting accurately,” but whether it can be properly wired into one of three places: the training signal, action selection, or data generation.** This is also why L4 (value-coupling) and L5 (self-correction) are rarely a property of a single model in practice. They are usually “an L1-L3 world model plus a way of wiring it in.”

## Course Definition

In this course, “world model” by default means **an agent-internal, predictive model of world dynamics that can be used for action selection**. In broader discussions, we acknowledge that representation models, reconstruction models, video prediction models, and external simulators are all related to world models, but we distinguish them carefully:

- **world-model component**: learns some representation or local regularity of the world.
- **predictive world model**: predicts future states or observations internally.
- **controllable world model**: predicts action-conditioned counterfactual consequences.
- **agentic world model**: uses internal prediction for planning, control, and decision-making.
- **self-improving world model**: improves itself through exploration, prediction error, and model revision.
- **external simulator**: provides an interactive world but is not an internal model learned by the agent.

So claims such as “DINO is a world model,” “NeRF is a world model,” “MuJoCo is a world model,” or “Sora is a world model” are only valid in a broad sense. More strictly, DINO and NeRF are usually world-model components, MuJoCo is an external simulator, a video diffusion model such as Sora is at most an L2 predictive world model (it is not conditioned on discrete actions and does not serve planning), and systems such as Dreamer or MuZero are closer to the core meaning of world model in this course.
