---
title: Seven Common Pitfalls, Deployment Strategies, and Curriculum Summary
description: The seven most common failure modes in real deployment, three progressive deployment strategies, a five-model summary comparison table, and a four-lecture curriculum summary with outlook.
lecture: 4
---

# Seven Common Pitfalls, Deployment Strategies, and Curriculum Summary

## The Seven Most Common Pitfalls in Real Deployment

### Action Semantic Mismatch

In simulation, one `action` may be an ideal joint target (assigned directly to the joint angle), while on the real robot it must pass through a **PD controller** (Proportional-Derivative controller, a classical closed-loop controller that outputs a control signal based on the magnitude of the current error (proportional term P) and the rate of change of the error (derivative term D) to track the target joint angle. Virtually all real robots include this low-level control layer, which is often simplified or omitted in simulation), hardware limits, velocity constraints, and motor response delays. If a world model is trained with "ideal actions," the dynamics it learns describe a non-existent "perfect robot," and the action sequences the policy learns may not be executable by real hardware.

### Timing Delays and Asynchronous Sensors

Cameras (typically 30 Hz or 60 Hz), force sensors (typically 1 kHz), joint states (250 Hz or higher), and control commands (variable frequency) are often not synchronized. The world model assumes that `o_t` and `a_t` are collected at the same moment, but in practice they may differ by tens or even hundreds of milliseconds. For high-speed locomotion or contact-rich manipulation, this time gap is enough to invalidate predictions: the robot may predict "the leg just touched down" when the leg has already lifted off again.

### Invisible Contact State

A visual appearance of contact does not mean force has been transmitted. A visual appearance of no motion does not mean the object has not micro-slipped. This is the largest blind spot of visual world models on manipulation tasks: grasping, peg insertion, cap tightening, and drawer pulling all depend heavily on invisible contact variables (normal force, tangential force, contact area). The prediction ceiling of world models that rely solely on RGB input is far below human expectations for this class of tasks.

### Long-Horizon Drift

Video world models look good on short rollouts (1 to 5 steps), but as time extends, object identity (a red ball becomes a blue ball), geometric relations (the relative positions of two objects flip), and contact state ("object in hand" becomes "object floating") all quietly degrade. Representation-space prediction (TD-MPC style), self-forcing training (STORM style), and 3D explicit representations (**NeRF**, Neural Radiance Field, a neural network that implicitly represents a 3D scene and can render images from arbitrary viewpoints. **3DGS**, 3D Gaussian Splatting, an explicit scene representation using large numbers of 3D Gaussians that renders much faster than NeRF. Both maintain explicit 3D geometry, which helps preserve object persistence across frames) all mitigate this problem, but as of now there is no complete solution.

### Policy Exploiting Model Vulnerabilities (Model Exploitation)

A policy is an optimizer, and it will find actions that yield high reward inside the world model but do not hold in the real world. This is not the policy's fault. It is the nature of optimization. A canonical example: inside a learned simulator, the policy discovers a trick of "rapidly oscillating joints by small amounts to obtain high reward." This action pattern circumvents all physical constraints within the model, but on the real robot it will only damage the motors or trigger an emergency stop.

Detection method: periodically execute the high-reward action sequences learned by the policy in the real environment and check whether "model-effective but real-robot-ineffective" actions exist. If the proportion exceeds 20%, adversarial training or systematic patching of the world model's vulnerabilities is needed.

<figure>
<img src="/rwmu/rwmu-diagram.png" alt="RWM-U ensemble uncertainty architecture: ensemble variance across multiple world models quantifies epistemic uncertainty" style="width:100%;display:block;margin:0 auto">
<figcaption>Li et al. (2026) Overall framework of RWM-U: N independently initialized autoregressive world models are trained simultaneously. Ensemble variance (the degree of disagreement among model predictions) quantifies epistemic uncertainty and is propagated temporally and consistently along the full rollout trajectory. During policy optimization, high-uncertainty regions are penalized, keeping the policy within the state distribution where the model is reliable, directly addressing the core pitfall of "uncertainty not entering control decisions."</figcaption>
</figure>

### Uncertainty Not Entering Control Decisions

Many world models produce a seemingly reasonable prediction of the future but do not inform the downstream policy that "I have actually never seen a similar state here." This silent failure mode is more dangerous than an obvious prediction error: the policy believes it is navigating familiar terrain while it has already entered an out-of-distribution region.

Real deployment must let uncertainty participate in planning: when uncertain, slow down, choose a more conservative action, or actively request human intervention. A simple implementation: maintain a density estimator over training data in the world model's latent space (such as **kernel density estimation**, a non-parametric method that estimates probability density by placing Gaussian kernels around training data points, where low density indicates the current state is far from the training distribution. Or a **normalizing flow**, a reversible neural network model that can exactly compute the probability density of any input point under the learned distribution). When the density of a new observation falls below a threshold, a "high uncertainty" flag is triggered.

### Safety Cannot Be Fully Solved by Reward Shaping Alone

Robots in homes and factories require **hard safety layers**: joint velocity limits, end-effector force limits, workspace collision detection, emergency stops, and human takeover protocols. These cannot rely entirely on the "safety awareness" learned by the world model, because the world model itself can be wrong.

The world model can play the role of **risk prediction** ("if this action is executed, there is a 40% probability of a collision within the next 3 steps"), but the final hard safety guarantees must come from an independent, learning-free control layer. Safety constraints are a software engineering problem, not only an ML training problem.


## Further Reading

- Dreamer series papers: see L01 Further Reading (Dreamer V1) and L02 Further Reading (V2/V3/V4)
- MuZero: see L03 Further Reading (Schrittwieser et al., 2020)
- TD-MPC: see L03 Further Reading (Hansen et al., 2022)
- STORM: see L04 STORM metrics page Further Reading (Zhang et al., 2023)
- [Alonso et al. (2024): Diamond](https://arxiv.org/abs/2405.12399): diffusion world model, NeurIPS 2024
- [Heusel et al. (2017): FID](https://arxiv.org/abs/1706.08500): Fréchet Inception Distance original paper
