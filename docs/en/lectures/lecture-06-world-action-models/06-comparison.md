---
title: "Pairwise reading guide and papers"
description: "Open a pair to compare where its predictions enter the control loop."
lecture: 6
---

# Pairwise reading guide and papers

Open a pair to compare where its predictions enter the control loop.

## Compare two methods

<details>
<summary>Fast-WAM · Motus</summary>

Fast-WAM removes the future-video inputs at inference through its attention mask. Motus can output actions alone by holding the missing video at pure noise and advancing only the action schedule.

</details>

<details>
<summary>Fast-WAM · LingBot-VA</summary>

Fast-WAM shortens the computation after receiving a camera image. LingBot-VA overlaps computation with an action already in flight, using feedback to correct the estimated starting point of the next chunk.

</details>

<details>
<summary>DreamZero · Fast-WAM</summary>

Fast-WAM can discard future video tokens because actions attend only to the first frame for visual information. DreamZero updates the current chunk's video and action together at each denoising iteration.

</details>

<details>
<summary>Cosmos Policy · Fast-WAM</summary>

Fast-WAM skips future video calculation before issuing an action. Cosmos Policy can spend that time comparing the consequences of several proposals while the choice is still open.

</details>

<details>
<summary>LingBot-VA · Motus</summary>

In Motus's inverse-dynamics mode the visual sequence is supplied. LingBot-VA also predicts the next visual segment and uses recent feedback between chunks to close the control loop.

</details>

<details>
<summary>DreamZero · Motus</summary>

Motus extracts a latent action from optical flow in unlabeled footage. DreamZero uses an extra human or cross-robot clip for visual prediction and keeps action supervision on the target robot's labeled trajectories.

</details>

<details>
<summary>Cosmos Policy · Motus</summary>

Both can predict the consequences of a given action. Cosmos Policy passes the predicted endpoint to a value function and compares multiple proposals from the same observation.

</details>

<details>
<summary>DreamZero · LingBot-VA</summary>

Both policies can update context from a camera. LingBot-VA predicts a visual change before inferring controls. DreamZero refines video and action together in each denoising iteration.

</details>

<details>
<summary>Cosmos Policy · LingBot-VA</summary>

LingBot-VA uses feedback to predict where the ongoing motion will end. Cosmos Policy predicts endpoints for alternative motions before choosing which one to execute.

</details>

<details>
<summary>Cosmos Policy · DreamZero</summary>

DreamZero can execute the action from one joint video-action sample. Cosmos Policy's planning mode proposes several actions, predicts an outcome for each and ranks them by value.

</details>

## Papers

- [Fast-WAM: Do World Action Models Need Test-time Future Imagination?](https://arxiv.org/abs/2603.16666)
- [Motus: A Unified Latent Action World Model](https://arxiv.org/abs/2512.13030)
- [Causal World Modeling for Robot Control](https://arxiv.org/abs/2601.21998)
- [World Action Models are Zero-shot Policies](https://arxiv.org/abs/2602.15922)
- [Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning](https://arxiv.org/abs/2601.16163)

The reading-guide structure draws on [Nemo’s Diffusion RL guide](https://knightnemo.github.io/blog/posts/diffusion-rl-zh/).
