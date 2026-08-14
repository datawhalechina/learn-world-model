---
title: "A Philosophical Coda: Enactive Cognition and What World Models Still Miss"
description: Rafiee and Sutton's 2025 paper introduces enactive cognition as a framework for evaluating current AI. This coda examines what it means for world models specifically.
lecture: 5
---

# A Philosophical Coda: Enactive Cognition and What World Models Still Miss

The debates in this lecture have mostly been conducted inside the field: language camp vs. world-model camp, JEPA vs. WAM, scaling vs. structure. But there is a critique that cuts across all of them from the outside.

In 2025, Banafsheh Rafiee and Richard Sutton published a paper, *Toward Enactive Artificial Intelligence* ([arXiv:2605.24238](https://arxiv.org/abs/2605.24238)), that takes stock of where all current AI, including LLMs, vision models, and world models, actually stands relative to the kind of cognition that biological agents exhibit. Their framework comes from cognitive science, not AI research. That outside vantage point makes some things visible that internal debates tend to obscure.


## The Passive Representation Problem

The dominant paradigm in AI, whether symbolic or neural, follows a common architecture: receive input, build an internal representation, reason over the representation, output an action. Intelligence, in this view, is the quality of the internal representation. A better model of the world produces better behavior.

Rafiee and Sutton call this **representationalism**, and they argue it has a structural ceiling.

The world is open, dynamic, and infinitely complex. No finite internal model can fully capture it. Rodney Brooks, the roboticist, put the same point more bluntly in 1991: "The world is its own best model." This is a claim about completeness, not about necessity: Lecture 1 already established that a general, goal-directed agent cannot avoid learning some internal model. Brooks' point here is sharper and survives that result intact: whatever model such an agent learns, it will always be a lossy compression, and the most current, richest, most accurate information about any situation remains in the world itself, not in any internal copy of it.

This is not a new observation. What Rafiee and Sutton add is a systematic alternative framework: **enactive cognition**. Its central claim is that cognition is not the processing of pre-formed representations. It is generated in the ongoing interaction between an embodied agent and its environment. Perception, cognition, and action are not a pipeline. They are mutually constitutive, inseparable in practice.

To understand what this demands, it helps to look at what enactive cognition actually requires.


## Four Pillars

The framework rests on four properties that jointly characterize genuinely enactive intelligence.

**Experience**: not data, but the agent's own history of action and consequence. A supervised model learns from the traces of others' experiences, compressed into a dataset. Enactive experience requires the agent itself to act, observe outcomes, fail, and revise. Reinforcement learning is the closest existing paradigm here, but even RL usually depends on reward functions designed by external engineers rather than arising from the agent's own self-maintenance.

**Perception-action inseparability**: perception is not input that precedes action. It is itself a form of action. Humans do not passively receive visual input. We move our eyes, heads, and bodies to actively reveal environmental structure. A system that only predicts what it would observe, without being able to change what it observes through movement, has a fundamentally impoverished relationship to the world.

**Autonomy**: the agent is not a stimulus-response machine but a self-organizing system. Objects in the environment have meaning because they bear on the agent's own goals and continued existence. A truly autonomous system generates its own criteria for success and failure from its internal dynamics, rather than having them specified by external labels or reward functions.

**Embodiment**: the body is not a platform for executing plans computed elsewhere. The specific shape, sensor placement, and motor capacities of the body determine what the environment means and what affordances are available. The same chair is "sittable" for a human, an obstacle for an ant, and a function of joint geometry and control bandwidth for a robot. Intelligence is not substrate-independent in the way classical AI assumes.


## Where World Models Stand

Against these four pillars, world models look significantly better than LLMs, but still short in important ways.

The Dreamer series explicitly uses the world model to simulate consequences of actions in latent space before acting. That is a form of predictive foresight that purely reactive systems lack. It addresses some of the perception-action gap: the policy learns inside a "dream" that includes action as a first-class input, not just an afterthought.

But the gap remains on the other three pillars.

On **experience**: world models are still trained on offline datasets or carefully managed simulation environments. The data is designed by engineers, not generated by an agent pursuing its own survival. The world model learns the dynamics from fixed trajectories. It does not acquire them through open-ended exploration driven by its own needs.

On **autonomy**: the reward function in Dreamer and TD-MPC is provided externally. The agent does not decide what is worth caring about. A human specifies the objective, and the agent optimizes it. Intrinsic motivation research exists at the edges of RL, but it has not been integrated into any of the major world model architectures at scale.

On **embodiment**: the world models in this curriculum operate mostly on pixel observations from fixed-viewpoint cameras or standardized sensors. The specific structure of a body, the way it can be moved to reveal new information, the affordances that follow from its particular configuration, these are not part of what current world models learn to exploit.


## What the Framework Reveals

The enactive cognition framework does not settle the architectural debates in L05, but it shifts what those debates are really about.

The JEPA vs. WAM question asks where prediction should happen: pixel space or representation space. From an enactive standpoint, this is a second-order question. What matters more is whether the agent's predictive capability is coupled to its own action history and self-generated goals, or whether it is a passive observer of pre-collected data. A world model trained entirely on offline trajectories, regardless of architecture, is still largely passive in the enactive sense. The architecture question is real, but it sits downstream of a more fundamental one about how the learning is grounded.

The language camp vs. world model camp debate looks different too. Both sides, as typically framed, are asking how to build better internal representations. Enactive cognition asks whether representations, however accurate, can ever be sufficient on their own. The Brooks quote is not an argument for worse representations. It is an argument that the relationship between agent and world cannot be fully captured inside any model.

Where Rafiee and Sutton and the world model camp converge: the path forward runs through action-grounded learning in the physical world, not through accumulating more passive observations.

Where they diverge: world model researchers focus on architecture and scale. Rafiee and Sutton argue that the missing ingredient is not architectural but relational, a different kind of coupling between agent, body, and environment than any current system instantiates.


## For You to Consider

Rafiee and Sutton argue that even reinforcement learning, the paradigm closest to enactive cognition, still falls short on autonomy and embodiment. If they are right, what would a genuinely enactive AI system look like at the engineering level? What would have to change in how we collect training data, design reward signals, or build robot bodies?

And a harder question: if cognition is fundamentally enactive, if it is generated in action rather than stored in representation, does the concept of a "world model" as a separable internal module even make sense? Or is the world model, in the end, not a thing you build but a property you acquire through the right kind of living in the world?


## Further Reading

- Rafiee, B. & Sutton, R. S. (2025). [Toward Enactive Artificial Intelligence](https://arxiv.org/abs/2605.24238): the source paper for this coda
- Brooks, R. (1991). "Intelligence Without Representation." *Artificial Intelligence* 47(1-3): 139-159. The foundational argument that the world is its own best model
