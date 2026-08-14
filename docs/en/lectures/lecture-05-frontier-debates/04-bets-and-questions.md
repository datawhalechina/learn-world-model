---
title: Core Bets of Each Route and Closing Questions
description: The core assumptions each of the nine architectural routes is betting on, the CWM special case, and three questions to carry forward.
lecture: 5
---

# Core Bets of Each Route and Closing Questions

## Core Bets of Each Route

Lecture 3 already compared the trade-offs of the nine major architectural families from an engineering-selection perspective. Here the lens shifts: what assumption is each route betting on, and what does it mean if that assumption holds?

| Architectural Route | Core Bet | If the Bet Holds |
|---------------------|----------|-----------------|
| RNN/RSSM | The key dynamics of the physical world can be represented with a compact recurrent state, without pixel-level reconstruction | Dreamer-style sample-efficient RL can scale to complex real-world tasks |
| Transformer | The core of intelligence is long-range sequential dependency, and a unified attention mechanism is more powerful than manually designed state separation | The STORM/Dreamer V4 line will converge toward a general-purpose world model backbone |
| Diffusion | The regularities of the physical world are embedded in pixel distributions, and high-fidelity generation is itself understanding | A sufficiently large diffusion model will automatically develop physical reasoning, without explicit modeling |
| JEPA | Pixels are noise, semantics are signal, and prediction should happen at the level of abstract representations rather than perception | Models that do not generate pixels will achieve physical understanding faster than generative models |
| RWM | Long-horizon rollout stability and policy-exploitation resistance are the real bottleneck for robotics, not raw generation quality | Uncertainty-aware and self-forcing training will make world models deployable on real hardware before larger backbones do |
| Spatial 3D/4D | Object permanence and multi-view consistency require explicit 3D geometry, not just a large enough flat latent | Scene-level 3D representations will become the default state for embodied and driving world models |
| Genie | Actions can be discovered from unlabeled video rather than requiring paired action labels | Internet-scale video, not teleoperation data, will be the dominant source of interactive world models |
| LoopWM | A dynamics model's effective depth can be decoupled from its parameter count via looped, spectrally-stabilized computation | Iterative latent depth becomes a third scaling axis alongside model size and data volume |
| WAM | World models and policies should not be two separate modules. Video itself is the supervisory signal for action learning | Joint training will break the division of labor between foundation modeling and policy learning, producing new emergent capabilities |
| CWM | Code execution space is a kind of "world" that can be explicitly modeled. An LLM genuinely understands code only after learning to predict program state transitions | The world model approach can transfer to digital space, not just physical perception |

These bets are not mutually exclusive, but their answers to "what is the core of intelligence" conflict with one another. The RNN camp holds that state representation is the core. The Transformer camp holds that sequence modeling is the core. The Diffusion camp holds that generative fidelity is the core. The JEPA camp holds that semantic abstraction is the core. The RWM camp holds that deployment reliability is the core. The Spatial 3D/4D camp holds that explicit geometry is the core. The Genie camp holds that action discovery from unlabeled video is the core. The LoopWM camp holds that iterative depth is the core. The WAM camp holds that joint modeling of action and perception is the core. And CWM raises a question that straddles the language camp and the world model camp: if the "world" is a Python interpreter, and an LLM learns to make predictions inside it, which side does it belong to?

This debate will not be settled in papers. It will be forced toward an answer by benchmarks over the next several years.


## CWM: World Models for Code Execution Space

CWM is a domain extension discussed for comparison, not a tenth dynamics-backbone family in Lecture 3. The nine-family survey classifies architectures used for physical or visual prediction. CWM asks whether the same capability criteria transfer when the modeled environment is a program interpreter.

Physical-world world models predict pixels, joint angles, and sensor readings. But the "world" need not be physical. Meta's 2024 **CWM (Code World Model, [arXiv:2510.02387](https://arxiv.org/abs/2510.02387))** extends this idea to code execution space: the Python interpreter is itself a deterministic dynamical system. Each line of code executed applies an "action" to the "current program state" and produces the "next program state."

<figure>
<img src="/cwm/cwm-data-overview.png" alt="CWM training and inference types: Python execution trajectories and agent-environment interactions" style="width:90%;display:block;margin:0 auto">
<figcaption>Meta (2024) CWM data and inference overview: training data consists of Python interpreter execution trajectories (action = code statement, observation = local variable state) and Docker agent interaction trajectories (action = shell command / code edit, observation = environment response). At inference time, CWM can simulate Python execution line by line, predicting the program state after each step, rather than merely generating syntactically correct code text.</figcaption>
</figure>

CWM is a 32-billion-parameter open-source LLM that underwent mid-training after pretraining, using two types of execution trajectories:

- **Python execution trajectories**: action = one Python statement, observation = the full local variable state after execution (variable names, types, values). The training objective is to teach the model "what is in memory after this line runs," not merely "whether this line is syntactically correct."
- **ForagerAgent trajectories**: an agent that autonomously executes software engineering tasks inside Docker containers, repeatedly editing code, observing error output, and editing again. An action is a shell command or code edit. An observation is the terminal response.

This design maps almost one-to-one onto the RSSM framework from physical-world models: an encoder compresses program state into a representation, a dynamics function predicts the next state, with the only substitution being that the "physics engine" is replaced by the "Python interpreter."

**Why is this a boundary question?** The existence of CWM blurs the boundary between the language camp and the world model camp. It uses a Transformer architecture (the language camp's primary tool), trains on natural language text plus code (the language camp's data), yet the training objective is to predict the dynamic changes in program execution state (the world model camp's core claim). If CWM ultimately demonstrates that "understanding code = being able to make predictions inside an interpreter," the next question becomes: does understanding physics also mean "being able to make predictions inside a physics engine"? The answer to that question bears on the future of both camps.


## The Bet of the Unpopular Side

Xie Saining knows that what he is doing is not mainstream:

> "You don't have to believe us. We'll see. I'm all-in on this path now. Are you coming?"

Hinton in 2012 had the same tone. So did Sutton (Richard Sutton, a founding figure in reinforcement learning, author of *Reinforcement Learning: An Introduction*, the same researcher who wrote the Bitter Lesson) in 2016, persisting with reinforcement learning when most people did not believe in it, going all-in on a direction the majority rejected.

LeCun's optimism is broader in scope, but consistent in direction:

> "This is exactly like what happened with deep learning and neural networks before. There is always a small group of people who can see clearly where the world is heading."

Placed side by side, these two statements are both a manifesto and a risk disclosure. History does show that small groups have been right. But it also shows that more small groups have been right never saw the turning point they were waiting for.

The world model researcher's bet is: language is not the substrate of thought. Predicting and understanding the physical world is the core of intelligence. If they are right, the center of AI over the next decade will not be data centers in Silicon Valley, but sensor networks in factories, hospitals, and farms.

If they are wrong, scaling laws will continue to hold, and LLMs will gradually approach physical understanding through more data and larger models: not via world models, but via language.


## Three Questions to Carry Forward

**Question 1**: Do you think language is a "shortcut" or a "fast path" to a world model?

A "shortcut" means you took a route that bypasses the real challenge and will eventually reach a dead end. A "fast path" means you took a more efficient route that still arrives at the same destination. The difference between these two determines where the ceiling of LLMs lies.

**Question 2**: If Sutton's Bitter Lesson is correct, will world models have their own Bitter Lesson moment?

At some point, will a "simpler and more general" method emerge that overtakes the carefully designed architectures of the world model camp in one move, just as AlphaZero surpassed Deep Blue, just as Transformer surpassed LSTM? Can the Bitter Lesson itself have a Bitter Lesson?

**Question 3**: Are world models the destination that everyone will eventually reach, or one fork in the road?

Perhaps the final answer is not "who won," but that different application domains converge on different technical routes: language generation and code assistants follow the LLM route. Robotics, industrial control, and autonomous driving follow the world model route. And certain tasks, perhaps the ones you and I use most, will remain permanently in the gray zone between the two.

If that is the case, the significance of this debate lies not in determining a winner, but in helping us understand more clearly: **what problem are we actually trying to solve, and where does the path we are on lead.**
