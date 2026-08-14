---
title: Seven World-Model Integration Patterns
description: A system-level comparison of where learned prediction enters an agent, from representation pretraining and dense supervision to online action selection and offline data generation, including how a VLA's direct observation-to-action mapping relates to world-model prediction.
lecture: 3
difficulty: advanced
---

# Seven World-Model Integration Patterns

The L1-L5 ladder in Lecture 1 classifies capability. The earlier pages in this lecture classify dynamics architectures and explain three concrete planning mechanisms. A complete system needs one more design decision: **where does learned prediction enter the agent's pipeline?**

This is not another list of model architectures. The same RSSM, Transformer, or video model can participate in several patterns below. The patterns describe system integration, while the architecture pages in Parts B and C describe model structure.

## Four Places Prediction Can Enter

A learned predictor can shape a system in four places:

1. It can pretrain the representation consumed by a policy.
2. It can add a dense training signal to a policy or planner.
3. It can generate candidate futures used to select the next action online.
4. It can generate training data offline, after which deployment may no longer call the model.

The seven patterns below are finer variations of these four roles.

## Where a VLA Fits Next to a World Model

Two of the seven patterns below route a world model's prediction into a **VLA** (Vision-Language-Action model, [Brohan et al., 2023](https://arxiv.org/abs/2307.15818)): a model that maps an observation and a language instruction directly to an action, typically built by adding an action output head to a pretrained vision-language model and training it with **behavior cloning** (supervised learning on observation-instruction-action demonstrations). A bare VLA never predicts a future state. It decides what to do right now, with no world model anywhere in its loop. This is exactly why it needs distinguishing from the architectures in this lecture: everything on this page so far predicts a state, and a VLA is the part of a complete system that does not.

Patterns 3 and 5 below describe two different ways a world model's prediction can still be spliced into that direct mapping, without turning the VLA itself into a predictive model.

## The Seven Patterns

| Pattern | How prediction is used | Capability connection |
| --- | --- | --- |
| 1. Representation pretraining | Pretrain an encoder on unlabeled video, then give its representation to a policy or planner | L1 supplies the foundation for higher levels |
| 2. Slow-fast hierarchy | A vision-language model issues occasional high-level instructions while a fast policy executes low-level actions | L1-L3 components cooperate without becoming one model |
| 3. VLA with an explicit causal trace | A Vision-Language-Action model predicts an intermediate explanation or causal trace before its action | L3 action prediction receives supervision on parts of the decision process |
| 4. Dense predictive supervision | Train future-scene and action branches together, but retain only the action branch at deployment | L2 prediction supplies a denser gradient to an L3 policy |
| 5. Generative goal or visual trace | Imagine a goal image or visual trajectory, then use an inverse dynamics model to infer the actions that connect consecutive imagined states | L2 generates a target future and L3 converts it into actions |
| 6. Online action evaluator | Roll out candidate actions in a learned model, score their consequences, execute the first action, then plan again from the new observation | L4 uses prediction directly for receding-horizon control |
| 7. Offline simulation and data generation | Generate difficult or out-of-distribution experience before policy training, then train the deployed policy on the augmented data | L4-L5 improve the policy or dataset without requiring the model online |

> **📖 Terms used in the table**: A **VLA** maps vision and language instructions to actions. A **causal trace** is an intermediate prediction intended to expose why an action follows from the observation. An **inverse dynamics model** predicts the action that could transform one state into another. **Receding-horizon control** plans several steps ahead, executes only the first action, observes the real result, and replans. These definitions describe interfaces, not guarantees: an intermediate verbal trace does not by itself prove that the model learned the true causal mechanism.

## Three Comparisons That Matter

Patterns one through three differ mainly in coupling. Representation pretraining passes features to a separate policy. A slow-fast hierarchy keeps two cooperating modules at deployment. A unified VLA trains perception, intermediate prediction, and action generation in one sequence. Greater coupling can align training and deployment behavior, but it also makes failures harder to localize.

Patterns four and five both make supervision denser. Pattern four uses future prediction as an auxiliary training objective. Pattern five exposes an imagined target to an inverse model at inference. The former can discard the predictor after training. The latter still depends on generation quality when acting.

Patterns six and seven differ in when the world model runs. Online evaluation can respond to the current observation but adds latency and exposes the planner to model exploitation. Offline generation avoids inference-time cost, but the deployed policy cannot ask the model to reconsider an unfamiliar situation.

## Connection to the Core Planning Mechanisms

CEM-MPC and TD-MPC are instances of pattern six because they score candidate action sequences online and replan after each real transition. Dreamer spans two roles: imagined rollouts provide a dense training signal for the actor and critic, while the trained actor can act without running a search at every environment step. This is why capability is a property of the complete loop, not merely the dynamics network.

The practical question is therefore not “Which pattern is the world model?” Ask instead: **which component predicts, what consumes that prediction, when does prediction run, and can its error change the executed action?** Those four questions make superficially similar systems comparable.

## Further Reading

- Review the [L1-L5 capability ladder](../lecture-01-internal-simulation/02-world-model-taxonomy) before assigning a capability level to any complete system.
- Review [CEM-MPC and latent Actor-Critic](./05-planning-cem-ac) and [TD-MPC](./06-planning-tdmpc) for concrete examples of online and training-time integration.
