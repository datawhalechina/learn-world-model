---
title: Observation, State, and Belief
description: Start from partial observability to understand why an agent must estimate hidden state from history, and how this process leads to the RSSM prior and posterior.
lecture: 2
---

# Observation, State, and Belief

Look at a photograph of a ball in the middle of a table. You cannot tell whether it will move left or right next. Two very different trajectories can produce the same image. A single frame tells us where the ball is, but not its velocity, whether it just collided with something, or whether someone outside the frame is pushing it.

This example exposes the first modeling problem for a world model: **the model receives observations, but it must infer a state sufficient for predicting the future.**

After this page, you should be able to distinguish observations, environment states, and belief states, explain why memory is part of state estimation, and map the classical predict-correct cycle onto the RSSM prior and posterior.

## The Same Observation Can Lead to Different Futures

Let $s_t$ denote the complete environment state at time $t$. It may include object positions, velocities, masses, contact relations, and variables the agent cannot directly see. The agent receives an observation $o_t$, such as an image or a set of sensor readings. An action $a_t$ changes the environment state, which then produces another observation:

$$
s_{t+1} \sim p_{\mathrm{env}}(s_{t+1}\mid s_t,a_t),
\qquad
o_t \sim p_{\mathrm{obs}}(o_t\mid s_t)
$$

These two distributions describe how the world changes and how the world is seen. An observation can have more numerical dimensions than the state, as an image may contain thousands of pixels, while still omitting variables needed for prediction. High-dimensional does not mean complete.

| Variable | Meaning | Example in a pixel-control task |
| --- | --- | --- |
| $s_t$ | Hidden environment state | Position, velocity, contact state |
| $o_t$ | Observation supplied by sensors | One RGB frame |
| $a_t$ | Action applied by the agent | Push left or right |
| $h_t$ | Compressed memory of history | Recent motion trend and collision clues |

If earlier history no longer improves the prediction of $s_{t+1}$ once $s_t$ and $a_t$ are given, then $s_t$ has the Markov property. Raw observations usually do not. Velocity remains unknown when only one image is available.

## A Belief State Estimates the Hidden World

In a partially observable environment, the agent cannot access $s_t$ directly. It instead maintains a belief state from its observation and action history:

$$
b_t(s)=p(s_t=s\mid o_{1:t},a_{1:t-1})
$$

$b_t$ is not a single certain answer. It is the model's probability judgment over possible current states. The same still image may come from a ball moving left or right. More history usually concentrates this distribution. When sensors are noisy or objects are occluded, retaining a distribution is more honest than retaining only a point estimate.

A belief state provides three immediate benefits:

1. **Recover hidden variables**: infer velocity, direction, and contact from a sequence of observations.
2. **Carry uncertainty**: distinguish "I know where it is" from "either of two positions is plausible."
3. **Support decisions**: avoid costly actions in highly uncertain states or act to gather information.

## Prediction and Correction

State estimation repeatedly performs two steps. First, the model uses the previous belief and action to predict the current state:

$$
b_t^-(s_t)=\int p(s_t\mid s_{t-1},a_{t-1})b_{t-1}(s_{t-1})\,ds_{t-1}
$$

When a new observation arrives, the model corrects that prediction:

$$
b_t(s_t)\propto p(o_t\mid s_t)b_t^-(s_t)
$$

The prediction step asks where the system should now be according to its dynamics. The correction step asks which possible states remain credible after seeing new evidence. Kalman filters, particle filters, and modern recurrent state-space models differ in implementation, but they solve this shared problem.

## From Belief States to the RSSM

An RSSM does not store the true environment state or evaluate the Bayesian integrals above exactly. It learns a neural approximation:

- The deterministic hidden state $h_t$ summarizes past states, actions, and observations.
- The prior $p(z_t\mid h_t)$ predicts the latent state without seeing the current observation.
- The posterior $q(z_t\mid h_t,o_t)$ corrects that prediction with the current observation.

When later pages say that the prior supports imagination and the posterior supports training, this should not look like an isolated neural-network trick. It is the predict-correct cycle implemented inside a deep world model.

Stacking the most recent $K$ frames supplies more evidence, but it does not automatically produce a Markov state. Sufficiency still has to be tested: given this representation and the action, does earlier history significantly improve future prediction?

## Consequences for P01 and P02

The VAE in P01 only guarantees that its latent variables help reconstruct the current observation. It does not guarantee that they contain velocity or form a state suitable for prediction. When inspecting P01, look beyond reconstructed images. Ask whether position, motion direction, and task-relevant variables can be read reliably from the latent representation.

The RSSM in P02 introduces time. It must use memory to recover information missing from a single frame and learn to continue predicting without new observations. P01 answers how to encode this frame. P02 begins to answer what hidden state the world is in and what happens next.

## Closing the Loop on the Photograph

Every device introduced above, the belief distribution, the predict-correct cycle, the RSSM's prior and posterior, is a different answer to the question this page opened with: given only a still photograph of the ball, what is still missing, and how does a model recover it. The next two pages return to the same photograph to show where that missing information can be compressed away by an encoder, and where a dynamics model must reconstruct it from history alone.

## Check Your Understanding

1. Why can two identical pixel observations correspond to different environment states? Name one hidden variable.
2. In an RSSM, what roles do $h_t$, the prior, and the posterior play in the state-estimation cycle?
3. Design a small test for whether P01's latent representation preserves object velocity rather than only position.
