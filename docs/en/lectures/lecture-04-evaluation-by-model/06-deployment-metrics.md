---
title: "Real-World Deployment Evaluation: Beyond Paper Metrics"
description: The limitations of paper metrics in real deployment, and the dynamics quality, uncertainty, policy transfer, and system performance metrics that should be recorded during deployment.
lecture: 4
---

# Real-World Deployment Evaluation: Beyond Paper Metrics

The metric frameworks for the five models covered earlier were all designed for controlled laboratory settings, where you have clean datasets, reproducible simulation environments, and sufficient compute to run controlled experiments repeatedly. When a world model enters real deployment, everything becomes significantly more complex.

## Why Paper Metrics Are Not Enough

**FID** (Fréchet Inception Distance, measures image feature distribution distance, lower is better), **FVD** (Fréchet Video Distance, measures video sequence dynamics quality, lower is better), and **PSNR** (Peak Signal-to-Noise Ratio, higher is better) tell you whether a model "predicts accurately," but they cannot answer the following questions:

- Can the actions learned by a policy inside the world model actually be executed by real robot hardware?
- Will sensor latency and asynchrony invalidate the temporal assumptions of the world model?
- When the world model is uncertain about a given state, can the system detect this and safely request human intervention?

In real deployment, the world model is just one link in a long chain:

The full control chain runs through six stages: Sensors feed State Estimation, which feeds the World Model, which feeds the Planner or Policy, which feeds Low-Level Control, which drives the Actuators. Paper metrics measure only the World Model box; failures at any other stage cause system failure regardless of model quality.

Failure at any link in this chain causes system failure. Paper metrics measure only the input-output quality of the "world model" box, not the reliability of the entire chain.

## What to Record and Evaluate in Real Deployment

**Dynamics Quality**

- **One-step prediction error**: whether short-horizon dynamics are accurate
- **Multi-step rollout error**: whether long-horizon predictions drift (5 / 10 / 20 steps)
- **Contact event accuracy**: whether the model correctly predicts contact, sliding, dropping, and jamming

**Uncertainty and Reliability**

- **Uncertainty calibration**: whether high uncertainty truly corresponds to high error, measured by Expected Calibration Error (ECE).

> **📖 Calibration**: When a model says "I am 80% confident," does the true accuracy also approach 80%? A well-calibrated model has confidence equal to actual accuracy. ECE is the weighted average of the difference between confidence and actual accuracy within each confidence bucket, lower is better.

**Policy Transfer**

- **Policy transfer gap**: the cumulative reward loss when a policy learned inside the world model is transferred to a real robot (sim-to-real gap)

**Human-Robot Collaboration**

- **Intervention rate**: how many human takeovers are required per hour
- **Failure recovery rate**: whether the system can recover from intermediate failure states

**System Performance**

- **Latency**: whether the observation-to-action cycle meets the required control frequency (real-time factor: sim\_speed / real\_speed ≥ 1)
