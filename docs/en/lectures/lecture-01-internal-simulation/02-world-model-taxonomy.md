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

The strict taxonomy of this course is therefore:

```text
History explains origin.
Capability explains essence.
```

## How the four eras map onto L1-L5

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

## Level 1: Representational World-Model Components

Representational models compress high-dimensional observations into computable, memorable, comparable internal states. They answer the question: “What is here?”

Typical examples:

- **DINO / CLIP-like visual representations**: learn semantic and object-level visual structure.
- **MAE / masked reconstruction**: learn latent variables by reconstructing missing image or video content.
- **Autoencoder / VAE encoder**: compress high-dimensional observations into latent state.
- **Object-centric representation**: decompose scenes into persistent objects and attributes.

These models answer the first question: they have internal state representations. But without explicit dynamics and action interfaces, they are better described as world-model components rather than complete world models.

## Level 2: Predictive World Models

Predictive models represent the current world and predict what will happen next. Prediction may happen in pixel space, feature space, object space, 3D space, or physical state space.

Typical examples:

- **JEPA**: predicts target representations in latent space rather than reconstructing every pixel.
- **Video diffusion / video prediction**: predicts future frames or possible visual trajectories.
- **Scene flow / optical flow models**: predict motion of 3D points, pixels, or objects.
- **Dynamics model**: learns a transition function such as `s_{t+1} = f(s_t, a_t)`.

These models answer the first two questions: they represent state and predict future state. But if prediction is not connected to action selection or planning, they remain predictive world models rather than agentic world models.

## Level 3: Action-Conditioned World Models

Action-conditioned models include the agent’s action in world evolution. They do not merely predict what will naturally happen next. They predict: “What if I act?”

Typical examples:

- **Model-based RL dynamics models**
- **Robotics forward models**
- **Controllable video generation**
- **Action-conditioned latent transitions**

Core capabilities:

- `s_{t+1} = f(s_t, a_t)`
- action-conditioned counterfactual prediction
- imagined trajectories under candidate policies

These models can answer local counterfactuals, but they do not necessarily support long-horizon planning or value-aware decision-making.

## Level 4: Value-Coupled World Models

Value-coupled models bind world prediction to goals, rewards, preferences, costs, or survival constraints. They answer: “What matters?”

Typical examples:

- **Dreamer-style actor-critic learning in imagination**
- **MuZero-style reward and value prediction**
- **Learned cost models for control**
- **Preference-conditioned world models**

Core capabilities:

- reward and value prediction
- planning over imagined futures
- credit assignment through latent rollouts

These models know which imagined futures are better or worse. They are agentic world models because prediction is used for planning, control, and decision-making.

## Level 5: Self-Correcting World Models

Self-correcting models close the loop between prediction error, exploration, and model update. They do not just use a world model. They actively improve it.

Typical examples:

- **Active inference**
- **Curiosity-driven model learning**
- **Uncertainty-guided exploration**
- **Lifelong world-model learning**
- **Scientific discovery agents**

Core capabilities:

- detect model error
- choose experiments that reduce uncertainty
- update beliefs after intervention
- maintain a growing world model across tasks

This is a higher-order world model. It not only simulates the world, but also notices where its simulation is unreliable and actively collects information to repair it.

## The Final Ability Ladder

This course does not use a binary label such as “is this a world model or not?” Instead, it uses an ability ladder:

| Level | Core Question | Minimal Capability | Typical Examples | Strict Term |
| --- | --- | --- | --- | --- |
| L1 Compression | What is here? | Represent current state | DINO, MAE, NeRF encoder | world-model component |
| L2 Dynamics | What happens next? | Predict future state | JEPA, video prediction, scene flow | predictive world model |
| L3 Action-Conditioning | What if I act? | Predict consequences of actions | robotics forward model, action-conditioned dynamics | controllable world model |
| L4 Value-Coupling | What matters? | Use prediction for planning and decision-making | Dreamer, MuZero | agentic world model |
| L5 Self-Correction | How do I improve? | Explore, detect error, and update the model | active inference, curiosity, lifelong agents | self-improving world model |

This taxonomy is stronger than the common “reconstruct / predict / run the world” grid because it classifies models by agent capability rather than surface behavior. A system can implement these abilities over pixels, objects, 3D scenes, language, physical states, or abstract latent variables. The modality is not the essence. **Operational counterfactual ability is the essence.**

## Relation to the Common Grid

Common grids arrange models by axes such as “reconstruction, prediction, runnable simulation” and “features, objects, pixels, 3D.” This intuition is useful, but it mixes two different questions:

- What space is being modeled? Features, objects, 3D, pixels, physical states.
- What can the model do in that space? Compress, predict, answer action-conditioned counterfactuals, plan, self-correct.

This course keeps “modeling space” as a horizontal dimension, but uses the ability ladder as the main vertical axis. This explains why:

- DINO and MAE are important, but usually remain at L1.
- JEPA and video prediction enter L2.
- Action-conditioned dynamics enter L3.
- Dreamer and MuZero enter L4.
- Agents that actively design experiments and revise their own assumptions enter L5.
- MuJoCo and game engines are external simulators; they are not automatically internal world models learned by the agent.

## Course Definition

In this course, “world model” by default means **an agent-internal, predictive model of world dynamics that can be used for action selection**. In broader discussions, we acknowledge that representation models, reconstruction models, video prediction models, and external simulators are all related to world models, but we distinguish them carefully:

- **world-model component**: learns some representation or local regularity of the world.
- **predictive world model**: predicts future states or observations internally.
- **controllable world model**: predicts action-conditioned counterfactual consequences.
- **agentic world model**: uses internal prediction for planning, control, and decision-making.
- **self-improving world model**: improves itself through exploration, prediction error, and model revision.
- **external simulator**: provides an interactive world but is not an internal model learned by the agent.

So claims such as “DINO is a world model,” “NeRF is a world model,” or “MuJoCo is a world model” are only valid in a broad sense. More strictly, DINO and NeRF are usually world-model components, MuJoCo is an external simulator, and systems such as Dreamer or MuZero are closer to the core meaning of world model in this course.
