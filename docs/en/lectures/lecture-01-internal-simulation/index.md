---
title: "Working Definition, History, and Intuition of World Models"
description: "Start from everyday intuition, establish a working definition and interfaces for world models, then trace the idea from Craik, predictive coding, and the internal model principle to modern learned systems."
lecture: 1
difficulty: Introductory
---

# Working Definition, History, and Intuition of World Models

A friend throws a tennis ball from three meters away. Before you have even registered the ball's spin, your hand has already moved to the right position. This is neither reflex nor luck: **your brain simulated the ball's flight path in that fraction of a second**, predicted where it would land, and directed your muscles to act in advance.

A world model is the computational version of this mechanism: a sketch of "how the world works" maintained internally by a brain or an AI, allowing an agent to mentally rehearse outcomes before acting in reality.

After this lecture, you should be able to identify what a system predicts and how its predictions are consumed, distinguish rendering, simulation, and planning, and use a compact capability ladder to compare different systems.

The lecture follows the order in which its questions arise:

- **Working definition**: distinguish renderers, simulators, and planners, then describe a world model through state estimation, transition, emission, and prediction-query interfaces
- **Origins of the idea**: move from Craik (1943) through predictive coding and the internal model principle to see why acting after internal rehearsal is not a new idea
- **Value and current context**: ask what world models solve and why video generation, embodied intelligence, and autonomous driving are converging now
- **Taxonomy and roadmap**: use the four eras and the L1-L5 capability ladder to organize systems already encountered, then enter the project's build path
