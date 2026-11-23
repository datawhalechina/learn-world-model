---
title: "Lecture 6. Does a robot need to imagine before it acts?"
description: "Lecture 6. Does a robot need to imagine before it acts?"
lecture: 6
---

# Lecture 6. Does a robot need to imagine before it acts?

A gripper touches the slider on a resealable bag. Pulling along the top might open it or drag the entire bag across the table. A robot could learn a useful pull from recorded demonstrations. It could also propose several pulls, predict what would happen and choose one. All five papers use video generation, but the predicted future enters the decision at a different point in each one.

## Follow an action from observation to execution

| Reading order | How a command is produced |
| --- | --- |
| [Fast-WAM](./01-fast-wam.md) | Encode the current image once and generate actions directly. Future video supplies a training target. |
| [Motus](./02-motus.md) | Learn latent actions from optical flow. The model can generate actions directly or sample paired video and actions. |
| [LingBot-VA](./03-lingbot-va.md) | Predict visual change and infer controls. When running asynchronously, estimate where the action in flight will end. |
| [DreamZero](./04-dreamzero.md) | Denoise video and action jointly within each chunk. Replace the forecast with camera feedback for the next round. |
| [Cosmos Policy](./05-cosmos-policy.md) | Execute the policy output directly, or generate multiple proposals and rank their predicted endpoints by value. |

## Three clocks in one control loop

Let $K$ count denoising iterations, $H$ the number of generated control steps and $h$ the number executed before the policy updates. Thirty commands at 30 Hz cover one second. If the controller executes six before reading a new image, feedback can influence control after 0.2 seconds, provided the model can finish its next prediction in time.

Write $C$ for the information available now, $Z$ for future visual latents and $A$ for an action chunk. The factorization $p(A,Z\mid C)=p(Z\mid C)\,p(A\mid C,Z)$ admits both joint sampling and a visual prediction followed by inverse dynamics. To understand deployment cost and feedback, inspect which inputs attention can read, how many denoising iterations each modality needs and when real observations re-enter the model.

## Return to the moment the gripper touches the slider

In Fast-WAM, video prediction has already shaped the weights that encode the first observation. Motus can choose whether to generate an explicit visual future at this point. DreamZero refines the image and action together during each round of denoising.

Both LingBot-VA and Cosmos Policy predict an endpoint. Whose action does it belong to? Asynchronous LingBot-VA estimates where an **already issued action** will leave the slider so the next chunk starts in the right place. Cosmos Policy estimates where **several actions still under consideration** might leave it so the controller can choose which command to issue.

[Pairwise reading guide and papers](./06-comparison.md)
