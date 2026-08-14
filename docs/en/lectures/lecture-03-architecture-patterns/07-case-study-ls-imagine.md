---
title: "Part C, Optional Case Study: LS-Imagine"
description: "ICLR 2025 Oral. A long short-term world model with affordance-map-driven jump-style state transitions, letting a Minecraft agent imagine directly toward distant, sparsely distributed goals."
lecture: 3
---

# Part C, Optional Case Study: LS-Imagine

Every planning mechanism in this lecture shares one assumption: the world model rolls forward one step at a time. CEM-MPC samples full action sequences step by step. Dreamer's Actor-Critic imagines $H$ steps ahead one transition at a time. TD-MPC bootstraps through the Bellman equation one step at a time. This works well when the reward that matters is close by. It breaks down in tasks like Minecraft, where a useful outcome (finding a village, reaching an ore vein) may be hundreds of steps away and only weakly signposted by nearby rewards. An agent that only ever imagines a few steps ahead is myopic: it optimizes what it can see and stays blind to what pays off later.

**LS-Imagine** ([Li, Wang, Wang et al., ICLR 2025 Oral](https://openreview.net/pdf?id=vzItLaEoDa)) addresses this myopia directly, not by making the imagination horizon longer step by step, but by letting the world model occasionally jump.

## The Core Idea: Two Kinds of State Transition

LS-Imagine trains a world model with two transition branches instead of one:

- **Immediate transitions**: the ordinary one-step prediction used throughout this lecture, $\hat{s}_{t+1} = f(s_t, a_t)$.
- **Jump-style transitions**: a single imagined step that skips directly to a distant, task-relevant future state $\hat{s}_{t+H}$, bypassing the intermediate states entirely.

A **jump predictor** decides, from the current state, which branch to use. When the target is far away and only vaguely indicated, jumping lets the agent imagine "arriving near the goal" without having to roll forward through every intervening step, expanding the effective imagination horizon without deepening the rollout.

## The Chicken-and-Egg Problem: Affordance Maps

Training a jump predictor to jump toward a goal requires knowing, from the current frame, which regions are worth jumping toward. But there is no real trajectory of the agent having already reached a sparse, distant goal to learn this from.

LS-Imagine's answer is a synthetic exploration signal called an **affordance map**. For a given observation, a sliding window scans across the image. At each window position, the region is cropped and zoomed to simulate what the agent would see if it moved toward that patch. A pretrained video-text alignment model, **MineCLIP** ([Fan et al., 2022](https://arxiv.org/abs/2206.08853), a CLIP-style model, see L01 for CLIP, pretrained on Minecraft gameplay video paired with narration), scores how well each simulated approach matches the task's language description (such as "cut a tree"). Stitching these per-window scores together produces a heatmap over the image: the affordance map, marking which regions are worth heading toward. Because computing this densely is expensive, a smaller network is trained to approximate it directly from a single frame plus the language instruction, so it can run at every step during actual training and inference.

The affordance map does double duty:

- It becomes an **intrinsic reward**, encouraging the agent to move high-value regions toward the center of its view.
- Its concentration (how sharply the high-value mass clusters in one place) becomes the signal the jump predictor uses to decide whether an immediate or jump-style transition is appropriate: a sharply peaked map means a distant goal is worth jumping toward.

## Why This Matters for the Course

LS-Imagine is a concrete answer to a limitation that runs through every planning mechanism covered so far: fixed-depth, step-by-step imagination cannot see far enough when rewards are sparse and distant. Its solution, an alternate transition branch driven by a learned relevance signal, is one instance of a more general pattern worth remembering: when a single mechanism cannot span both short-range and long-range reasoning well, letting the model choose between two specialized branches can outperform forcing one branch to do both jobs.

On Minecraft tasks with sparse, distant goals (finding water, mining ore), LS-Imagine outperforms DreamerV3 and prior model-free and video-pretrained baselines by a wide margin, with the gap widest specifically on the sparsest-goal tasks, evidence that the jump mechanism is doing what it was designed to do rather than just adding capacity.

## Further Reading

- [Li, Wang, Wang et al. (2025): LS-Imagine](https://openreview.net/pdf?id=vzItLaEoDa): full method, affordance map computation, and Minecraft benchmark results. [Project page](https://qiwang067.github.io/ls-imagine) and [code](https://github.com/qiwang067/LS-Imagine).
- [Fan et al. (2022): MineCLIP](https://arxiv.org/abs/2206.08853): the CLIP-style video-text alignment model used to compute affordance maps.
- [Hafner et al. (2022): Director](https://arxiv.org/abs/2206.04114): a hierarchical Dreamer extension that addresses the same long-horizon myopia problem through manager/worker subgoals rather than jump-style transitions, one of LS-Imagine's baselines.
