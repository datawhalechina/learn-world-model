---
title: "Part A (cont. 2): Genie"
description: Genie's latent action discovery mechanism, bridging observation-only pretraining and interactive generation without action labels.
lecture: 3
---

# Part A (cont. 2): Genie

## Genie: Discovering Actions Implicitly from Video

**Representative systems**: Genie (Google DeepMind, 2024), Genie 2 (2024)

The first five architecture families share a common assumption: training data either includes action labels (interactive) or requires no actions at all (observation-only). Genie breaks this dichotomy by **automatically discovering implicit latent actions from unannotated internet video**.

Training data consists of large collections of video clips showing humans playing games and manipulating objects, with no action labels of any kind. Genie jointly trains three modules: a video tokenizer (**ST-ViT**, Spatiotemporal Vision Transformer, which applies patch-based encoding simultaneously along both the spatial and temporal dimensions to produce spatiotemporal discrete tokens) that compresses frame sequences into spatiotemporal discrete tokens; a latent action model (**LAM**, which learns to infer the type of change between adjacent frames) that infers discrete latent action codes from consecutive frame pairs; and a dynamics model that predicts the next frame token sequence conditioned on the latent action. At inference time, a user can specify a latent action and the model generates the next frame accordingly, making the entire process fully interactive.

> **📖 latent action**: Not a keyboard input like "move left" or a joint-space torque, but a discrete code derived purely from differences between video frames. It captures "what type of change occurred between adjacent frames," not a concrete physical action. Two video clips with similar scene-transition patterns (such as "an object moving to the right") should share the same latent action code, regardless of whether the footage shows a game or a robot manipulation task.

<figure>
<img src="/genie/genie-architecture.png" alt="Genie architecture: ST-ViT tokenizer, LAM latent action model, and MaskGIT dynamics model" style="width:100%;display:block;margin:0 auto">
<figcaption>Bruce et al. (2024) Genie's three-module design: ST-ViT encodes video frame sequences into spatiotemporal discrete tokens; LAM infers discrete latent action codes from consecutive frame pairs (no action annotations required); the dynamics model is conditioned on the latent action and uses MaskGIT to autoregressively predict the next frame token sequence.</figcaption>
</figure>

Genie was trained on 30,000 hours of platformer game video (no action annotations) with 11B parameters. The paper measures generation quality degradation using $\Delta_t\text{PSNR}$ (the drop in PSNR at inference time relative to a teacher forcing baseline) as a proxy for latent action alignment. Genie's significance lies in bypassing the "action annotation" bottleneck: the internet contains vast quantities of video, but almost none of it comes with paired robot action labels. Genie 2 extends the approach to 3D scenes, generating fully interactive 3D worlds from a single input image. Bi et al. released [Motus](https://arxiv.org/abs/2512.13030) (A Unified Latent Action World Model) in 2025, validating a similar idea on embodied manipulation tasks: a unified latent action representation extracts action knowledge from heterogeneous video data, with a small amount of labeled data used to align it to real control signals, enabling cross-embodiment transfer.

**Learning paradigm**: sits between observation-only and interactive. Training uses only video (observation-only), but inference supports action-conditioned generation (interactive). This idea directly inspired the subsequent WAM family, covered next.

**Limitations**: latent actions are induced automatically and are not aligned with real physical actions, so they cannot be used directly for robot control. An additional alignment step is still required to go from latent actions to a real policy.


## Next: Two More Architecture Families

Genie's latent-action trick raises a question the next page answers from two different directions: LoopWM asks whether a dynamics model's depth can be decoupled from its parameter count, and WAM asks whether the world model and the policy need to be separate modules at all. Both build directly on ideas introduced here, so it is worth pausing before continuing.
