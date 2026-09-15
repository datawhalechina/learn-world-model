---
title: Curriculum Roadmap
description: "The interleaved learning path through five lectures and six projects, from intuition and latent dynamics to planning, evaluation, and frontier debates."
lecture: 1
---

# Curriculum Roadmap

## One Path Through Lectures and Projects

The lectures and projects are one curriculum, not two separate tracks. Stop and complete each project when it appears in the path. The resulting checkpoint becomes the concrete object used in the next stage, so later ideas attach to a system you have already inspected and trained.

| Stage | Read | Then practice | What you should be able to explain afterward |
| --- | --- | --- | --- |
| State estimation and representation | L01, then L02 Observation, State, and Belief plus Observation Encoding | [P01: Train a VAE Encoder](../../projects/p01_vae_encoder) | Why an observation is not a state, and what information an encoder keeps and discards |
| Dynamics and rollout | L02 Latent Dynamics, Training Distributions and Free Rollouts, and the Dreamer series | [P02: Build an RSSM](../../projects/p02_rssm_dynamics) | Why a useful latent state needs memory and uncertainty, and why one-step accuracy does not guarantee long-horizon stability |
| Control | L03 Planning and Control | [P03: Train a Dreamer Agent](../../projects/p03_dreamer_agent) | How imagined trajectories train an actor and critic |
| Alternatives | L03 Backbone Selection | [P04: Swap the Dynamics Backbone](../../projects/p04_transformer_backbone) | Which bottleneck justifies replacing RSSM with another backbone |
| Evaluation | L04 | [P05: Build an Evaluation Dashboard](../../projects/p05_evaluation_dashboard) | Which metric diagnoses each representation, rollout, or planning failure |
| Causality | Revisit the L1-L5 ladder after L04 | [P06: Test Counterfactual Fidelity](../../projects/p06_counterfactual_world_model) | Whether actions causally change predicted futures rather than merely correlate with them |
| Frontier | L05 | No required project | Which open questions are empirical, architectural, or philosophical |


## Next Lecture

L02 starts with a question that comes before compression: **when a 64×64 image omits velocity and occluded information, how does a model estimate the current state?** After distinguishing observations, states, and beliefs, we use a Variational Autoencoder (VAE) to learn a compact representation.

Complete P01 after state estimation and encoding rather than waiting until the end of L02. Then return to latent dynamics and free rollouts, connect the learned representation to an RSSM, and complete P02. By that point you will have built the course's two central predictive components and observed how one-step errors change over time under the model's own input distribution.


*L01 requires no coding and treats its mathematical callouts as optional. L02 assumes fundamental deep learning knowledge and introduces the additional machinery when it is first needed.*


## Further Reading

- Craik, K.J.W. *The Nature of Explanation*. Cambridge University Press, 1943.
- [Ha & Schmidhuber (2018): World Models](https://arxiv.org/abs/1803.10122): the V/M/C three-module framework and the original paper on training in dreams
- [Hafner et al. (2019): Dream to Control (Dreamer V1)](https://arxiv.org/abs/1912.01603): the first end-to-end implementation of RSSM and latent actor-critic
- [LeCun (2022): A Path Towards Autonomous Machine Intelligence](https://arxiv.org/abs/2306.15364): the JEPA framework and the argument for world models as a cognitive core
