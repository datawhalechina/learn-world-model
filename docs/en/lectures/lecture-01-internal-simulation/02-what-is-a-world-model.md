---
title: "What Is a World Model: Rendering, Simulation, and Planning"
description: "Fei-Fei Li and the World Labs team systematically clarify the conceptual confusion around \"world model\": three functional definitions (renderer, simulator, planner) and why the simulator is the missing link between the other two."
lecture: 1
---

# What Is a World Model: Rendering, Simulation, and Planning

"World models are the destination everyone will reach. I've gone all-in on this path. Are you with me?"

Saining Xie's statement carries an implicit premise: the "world model" he is talking about is not necessarily the same thing most people mean when they use the term. This is not a terminological quibble; it is a real conceptual fracture. In 2025, Fei-Fei Li and the World Labs team published an article that systematically clarified this confusion.

---

## Three Functional Types of World Models

Computer vision, robotics, reinforcement learning, and generative AI all claim to be developing "world models," but each field is pointing at something different. The root cause is ambiguity about what "world" means.

The reinforcement learning framework of partially observable Markov decision processes (POMDPs) offers a useful baseline: the agent takes an action, the action changes the world state, an observation is produced, and that observation drives the next action. A key distinction runs through this loop. **State** refers to the complete description of the world at a given moment: all objects, positions, velocities, and properties. **Observation** is what the agent actually perceives: an incomplete projection of the state, typically images or video frames.

Every system currently called a "world model" is, at bottom, producing a different output from this loop. Based on this, the World Labs article distinguishes three functional types.

### Renderer: Produces Observations a Human Can Read

The renderer's job is to output observations, typically in pixel form. The primary measure of its quality is visual fidelity: how convincing the image looks.

Text-to-video models (Veo, Sora) are renderers. Interactive generation systems (Genie, World Labs' RTFM) are renderers. Their shared characteristic is that they have no explicit understanding of three-dimensional structure: they produce what things look like, not what they actually are. This is why an AI-generated city can look perfect from above but reveal collapsing buildings and physically impossible street geometry when viewed from street level.

### Simulator: Produces States That Obey Physical Laws

The simulator outputs world states that are faithful to reality in geometry, physics, or dynamics. Where the renderer only needs visual plausibility, the simulator must satisfy stricter structural constraints: geometric relationships must hold under scrutiny, physical processes must obey Newton's laws, and dynamic behavior must respect causal structure.

Simulators serve two audiences: professionals such as architects, engineers, and game developers who need accuracy beyond visual realism; and computational systems such as RL agents, robot controllers, and autonomous driving pipelines that need to test dangerous or expensive scenarios safely at scale.

The Dreamer series (V1–V4) trains its policy inside a "dream" that functions as an implicit simulator: it maintains a state representation in latent space, rolls forward through actions to predict the next state, and the policy learns entirely from this internal simulation before transferring to the real environment.

### Planner: Produces the Action the Agent Should Take

The planner outputs actions — given current observations and a goal, what should the agent do next. In a sense the planner is the inverse of the renderer: the renderer takes actions as input and converts them to observations; the planner takes observations as input and produces actions, closing the perception-action loop.

VLA (Vision-Language-Action) models, which take visual observations and language instructions as input and output robot actions directly, are planners. CEM-MPC and TD-MPC (two planning algorithms built on top of world models, covered in detail in L03) are planners. The latent Actor-Critic inside Dreamer is a planner. The planner is the hardest of the three to get right. The impressive-looking robot demonstrations of recent years are almost uniformly confined to tightly controlled laboratory settings; the gap between a demo video and a robot that reliably works in a real kitchen, warehouse, or operating room remains large.

---

## Why the Simulator Is the Missing Link

Though the three types can be defined separately, they share a common root: a deep understanding of how the world works, its geometry, physics, and dynamics. A model that genuinely understands the world should be able to do all three: render what a cup looks like from any angle, simulate what happens when the cup is pushed, and plan how a hand should reach out to grasp it.

Of the three, the simulator receives the least commercial attention but is the most functionally critical. The reason is directional.

The renderer optimizes for visual plausibility without requiring physical accuracy. That ceiling is real: renderer outputs are beautiful enough for content generation but not accurate enough for robot training or engineering design.

The planner is the most attractive target, but without an internal model of how the world actually works, a planner can only rely on memorized situations and pattern matching to produce actions. This is the core of LeCun's critique of VLA architectures: they memorize enormous catalogs of driving situations but have no internal causal model. When they encounter a genuinely novel situation, they have no way to reason about consequences.

The simulator is the bridge between the two. If language is an abstraction of the world and pixels are a projection of the world, then geometry, physics, and dynamics are the world itself. The simulator operates at that level, providing the structural skeleton from which visual representations can be derived for human consumption and action consequences can be derived for agent use.

---

## The Frontier: Boundaries Dissolving

The World Labs article notes something worth carrying into the rest of this curriculum: the most interesting current research is deliberately blurring the lines between the three categories.

World Labs' Marble, a generative model that reconstructs three-dimensional scenes from one or a few images, already outputs Gaussian splats (for rendering) and collision meshes (for physical simulation) from a single model: one output serves vision, the other serves a physics engine. Work from several robotics labs has shown that pretrained video renderers can serve directly as backbones for action prediction, merging the renderer and planner into a single model.

Both lines point in the same direction: one model that can render, simulate, and plan, switching output depending on what the downstream task needs.

---

## A Philosophical Aside

The World Labs three-part taxonomy is not coincidental. It maps onto a classical triangle in philosophy of knowledge.

**The simulator answers the ontological question: what is the world itself.** The objective physical structure that exists independent of any observer, with its own inherent rules of space, mechanics, and causality.

**The renderer answers the phenomenological question: what does the world look like.** The appearances that reach us through our senses — visual images, perceptual surfaces — everything we see is a projection of the world onto the dimension of perception.

**The planner answers the practical question: what can the agent do.** Standing on appearances, facing the objective world, acting on it and changing it through practice.

When Craik described the human mind as a "small-scale model of external reality" in 1943, he was pointing at exactly this unified structure: a system that reasons at the ontological level, presents at the phenomenological level, and outputs action recommendations at the practical level. What Xie Saining calls "the destination everyone will reach" has, in this sense, been the destination all along. It took eighty years of engineering to catch up with the intuition.

---

## Further Reading

- Li, F.-F. et al., World Labs (2025). [What Is a World Model?](https://x.com/drfeifei/status/2062247238143996275): systematic definitions of the three functional world model types
- Xie, S. (2024). [World Models, Embodied Intelligence, and AMI Labs](https://www.youtube.com/watch?v=rIwgZWzUKm8): Saining Xie's full elaboration on why world models are "the destination everyone will reach"
- Ha & Schmidhuber (2018): World Models (see L01 Further Reading): the earliest engineering framework to cleanly separate rendering (V), simulation (M), and planning (C)
