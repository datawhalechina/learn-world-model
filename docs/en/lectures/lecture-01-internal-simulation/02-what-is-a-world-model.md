---
title: "What Is a World Model: Interfaces, Rendering, Simulation, and Planning"
description: "Establish a working definition through state estimation, transition, emission, and prediction queries, then distinguish the functions of renderers, simulators, and planners."
lecture: 1
---

# What Is a World Model: Interfaces, Rendering, Simulation, and Planning

If a model generates realistic video, is it a world model? If a robot policy outputs actions without predicting their consequences, does it contain a world model? Where does a physics engine belong if it was never trained from data but simulates collisions accurately?

Model names cannot answer these questions. We need to inspect what a system represents, what it predicts, whether actions condition those predictions, and how a decision-maker consumes them. The renderer-simulator-planner distinction from World Labs provides one functional view. This page adds an interface view that can be traced into code throughout the course.

## A Working Contract for the Course

The POMDP framework from reinforcement learning provides the basic loop: an agent acts, the action changes a hidden world state, the state produces an observation, and the observation informs the next action. A **state** contains environment variables that determine future evolution. An **observation** is incomplete evidence supplied to the agent by its sensors. L02 develops this distinction in detail.

This course treats a world model as a set of interfaces that can be inspected and replaced:

| Interface | Input and output | Question answered |
| --- | --- | --- |
| State estimation or encoding | Observation and history → latent state | What state might the world be in now? |
| Transition model | Latent state and action → next-state distribution | What happens if this action is taken? |
| Emission or decoding | Latent state → observable quantity | What would this state look like? |
| Prediction query | History and candidate actions → future trajectory or task signal | Which possible futures matter for this decision? |

Not every system implements all four interfaces explicitly. A JEPA need not reconstruct pixels, MuZero predicts only quantities needed for planning, and a physics simulator obtains its transition from equations rather than a neural network. The contract is not meant to exclude such systems. It makes comparison concrete: what did the system omit or replace, and how does the downstream component use its predictions?

## Common Misconceptions

Three misreadings of the contract above show up often enough to name directly.

**"A world model is just a video predictor."** Video prediction is one possible emission interface, not the whole system. A model that only predicts pixels, without ever conditioning on actions or supporting a transition model that lets you ask "what if," has built a renderer but not the transition and query interfaces this course cares about. The root cause is treating the most visible output, a video, as the definition, when the definition concerns which interfaces exist and what predictions they support.

**"If a system does not reconstruct observations, it has no world model."** The interface table's second column already answers this: a JEPA that never decodes back to pixels can still hold a state estimator and a transition model, and MuZero predicts only the value and reward signals a planner needs, never an image. Reconstruction is one possible emission interface among several, not a requirement.

**"A policy that outputs actions without an explicit prediction step cannot contain a world model."** This confuses architecture with function. Whether the transition model is a labeled module, a hidden layer inside an end-to-end policy, or a term implicit in a loss function, the question is whether the system predicts consequences and conditions its action on that prediction. LeCun's critique of VLA architectures targets exactly this gap: they have plenty of parameters but no internal transition model to reason about a genuinely novel situation, so they fall back on memorized pattern matching.

All three myths share a root: mistaking a visible implementation detail, pixels in, pixels out, a module with a specific name, for the functional contract itself. The rest of this page traces that contract through three concrete functional types.

## Three Functional Types of World Models

Computer vision, robotics, reinforcement learning, and generative AI all claim to be developing "world models," but each field is pointing at something different. The root cause is ambiguity about what "world" means.

Every system currently called a "world model" is, at bottom, producing a different output from this loop. Based on this, the World Labs article distinguishes three functional types.

### Renderer: Produces Observations a Human Can Read

The renderer's job is to output observations, typically in pixel form. The primary measure of its quality is visual fidelity: how convincing the image looks.

Text-to-video models (Veo, Sora) are renderers. Interactive generation systems (Genie, World Labs' RTFM) are renderers. Their shared characteristic is that they have no explicit understanding of three-dimensional structure: they produce what things look like, not what they actually are. This is why an AI-generated city can look perfect from above but reveal collapsing buildings and physically impossible street geometry when viewed from street level.

### Simulator: Produces States That Obey Physical Laws

The simulator outputs world states that are faithful to reality in geometry, physics, or dynamics. Where the renderer only needs visual plausibility, the simulator must satisfy stricter structural constraints: geometric relationships must hold under scrutiny, physical processes must obey Newton's laws, and dynamic behavior must respect causal structure.

Simulators serve two audiences: professionals such as architects, engineers, and game developers who need accuracy beyond visual realism. And computational systems such as RL agents, robot controllers, and autonomous driving pipelines that need to test dangerous or expensive scenarios safely at scale.

The Dreamer series (V1–V4) trains its policy inside a "dream" that functions as an implicit simulator: it maintains a state representation in latent space, rolls forward through actions to predict the next state, and the policy learns entirely from this internal simulation before transferring to the real environment.

### Planner: Produces the Action the Agent Should Take

The planner outputs actions: given current observations and a goal, what should the agent do next? In a sense the planner is the inverse of the renderer. The renderer takes actions as input and converts them to observations. The planner takes observations as input and produces actions, closing the perception-action loop.

VLA (Vision-Language-Action) models, which take visual observations and language instructions as input and output robot actions directly, are planners. CEM-MPC and TD-MPC (two planning algorithms built on top of world models, covered in detail in L03) are planners. The latent Actor-Critic inside Dreamer is a planner. The planner is the hardest of the three to get right. The impressive-looking robot demonstrations of recent years are almost uniformly confined to tightly controlled laboratory settings. The gap between a demo video and a robot that reliably works in a real kitchen, warehouse, or operating room remains large.


## Why the Simulator Is the Missing Link

Though the three types can be defined separately, they share a common root: a deep understanding of how the world works, its geometry, physics, and dynamics. A model that genuinely understands the world should be able to do all three: render what a cup looks like from any angle, simulate what happens when the cup is pushed, and plan how a hand should reach out to grasp it.

Of the three, the simulator receives the least commercial attention but is the most functionally critical. The reason is directional.

The renderer optimizes for visual plausibility without requiring physical accuracy. That ceiling is real: renderer outputs are beautiful enough for content generation but not accurate enough for robot training or engineering design.

The planner is the most attractive target, but without an internal model of how the world actually works, a planner can only rely on memorized situations and pattern matching to produce actions. This is the core of LeCun's critique of VLA architectures: they memorize enormous catalogs of driving situations but have no internal causal model. When they encounter a genuinely novel situation, they have no way to reason about consequences.

The simulator is the bridge between the two. If language is an abstraction of the world and pixels are a projection of the world, then geometry, physics, and dynamics are the world itself. The simulator operates at that level, providing the structural skeleton from which visual representations can be derived for human consumption and action consequences can be derived for agent use.


## The Frontier: Boundaries Dissolving

The World Labs article notes something worth carrying into the rest of this curriculum: the most interesting current research is deliberately blurring the lines between the three categories.

World Labs' Marble, a generative model that reconstructs three-dimensional scenes from one or a few images, already outputs Gaussian splats (for rendering) and collision meshes (for physical simulation) from a single model: one output serves vision, the other serves a physics engine. Work from several robotics labs has shown that pretrained video renderers can serve directly as backbones for action prediction, merging the renderer and planner into a single model.

Both lines point in the same direction: one model that can render, simulate, and plan, switching output depending on what the downstream task needs.


## A Philosophical Aside

The World Labs three-part taxonomy is not coincidental. It maps onto a classical triangle in philosophy of knowledge.

**The simulator answers the ontological question: what is the world itself.** The objective physical structure that exists independent of any observer, with its own inherent rules of space, mechanics, and causality.

**The renderer answers the phenomenological question: what does the world look like?** The appearances that reach us through our senses, including visual images and perceptual surfaces, are projections of the world onto the dimension of perception.

**The planner answers the practical question: what can the agent do.** Standing on appearances, facing the objective world, acting on it and changing it through practice.

When Craik described the human mind as a "small-scale model of external reality" in 1943, he was pointing at exactly this unified structure: a system that reasons at the ontological level, presents at the phenomenological level, and outputs action recommendations at the practical level. What Xie Saining calls "the destination everyone will reach" has, in this sense, been the destination all along. It took eighty years of engineering to catch up with the intuition.


## Further Reading

- Li, F.-F. Et al., World Labs (2025). [What Is a World Model?](https://x.com/drfeifei/status/2062247238143996275): systematic definitions of the three functional world model types
- Xie, S. (2024). [World Models, Embodied Intelligence, and AMI Labs](https://www.youtube.com/watch?v=rIwgZWzUKm8): Saining Xie's full elaboration on why world models are "the destination everyone will reach"
- Ha & Schmidhuber (2018): World Models (see L01 Further Reading): the earliest engineering framework to cleanly separate rendering (V), simulation (M), and planning (C)
