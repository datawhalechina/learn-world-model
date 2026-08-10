<div align="center">
  <img src="./docs/public/preface.png" width="100%" alt="Learn World Models Banner">
  <br>

[English](./README.md) · [中文](./README-CN.md)

# Learn World Models（⚠️ Alpha Preview）

[![Read Online](https://img.shields.io/badge/Read-Online-blue?style=for-the-badge&logo=github)](https://datawhalechina.github.io/learn-world-model)
[![GitHub Stars](https://img.shields.io/github/stars/datawhalechina/learn-world-model?style=for-the-badge&logo=github)](https://github.com/datawhalechina/learn-world-model/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](https://github.com/datawhalechina/learn-world-model/blob/main/LICENSE)

> **Learn world models by building them: from the intuition behind latent dynamics to a working simulation, planning, and evaluation system.**

</div>

> [!CAUTION]
> ⚠️ **Alpha Preview**: This is an early build. Content is still being completed and revised: sections, examples, and wording may continue to change. Feedback via Issues is welcome.

---

## ✨ Preview

### 🏠 Course Home
> Structured learning path with lecture and project cards.

![Course home](./docs/public/screenshots/readme/en-home.png)

### 📖 Lecture Pages
> Concept-first explanations with mermaid diagrams and background callouts for deep-learning readers.

![Lecture page](./docs/public/screenshots/readme/en-lecture-01.png)

### 🗂️ Architecture Deep Dive
> Seven architecture families, three planning mechanisms, side-by-side comparison tables.

![Architecture lecture](./docs/public/screenshots/readme/en-lecture-03.png)

---

## What this course covers

Five lectures and six projects that take you from the intuition behind world models to training, evaluating, and causally probing modern world-model systems.

| # | Type | Title | Core Topics |
|---|------|-------|-------------|
| L01 | Lecture | Internal Simulation & Historical Context | Craik's mental models, predictive coding, four eras of world model evolution |
| L02 | Lecture | Observation Encoding & Latent Dynamics | VAE, CNN encoder, ELBO, GRU → MDN-RNN → RSSM |
| L03 | Lecture | Architecture Patterns, Learning Paradigms & Planning | Seven architecture families, CEM-MPC, latent Actor-Critic, TD-MPC |
| L04 | Lecture | Evaluation by World Model | FID, reward correlation, consistency loss, PSNR, horizon drift |
| L05 | Lecture | Frontier Debates | Language vs physical grounding, Bitter Lesson, AGI as a research target |
| P01 | Project | Train a VAE Encoder | Small CNN VAE on 64×64 pixels; ELBO loss curve; latent slider visualization |
| P02 | Project | Build an RSSM Dynamics Model | GRU, MDN-RNN, and RSSM compared; prior vs posterior rollout plots |
| P03 | Project | Train a Dreamer Agent | Full training loop: encoder + RSSM + latent Actor-Critic on a small pixel env |
| P04 | Project | Swap the Dynamics Backbone | Replace RSSM with a small causal Transformer (STORM-style); architecture comparison |
| P05 | Project | World Model Evaluation Dashboard | Per-model metrics side by side: FID, reward correlation, PSNR, latent drift |
| P06 | Project | Counterfactual Action-Conditioned World Model | Interventional and counterfactual rollouts, inverse-dynamics regularization, action-influence metric |

---

## Curriculum flow

```mermaid
flowchart TD
    L01["L01 History and Intuition"] --> L02A
    L02A["L02 Part A: VAE Encoder"] --> P01["P01 Train VAE, visualize latent space"]
    L02A --> L02B["L02 Part B: GRU to RSSM"]
    L02B --> P02["P02 Build RSSM, compare prior vs posterior"]
    L02B --> L03A["L03 Part A: Architecture Patterns"]
    L03A --> L03B["L03 Part B: Planning mechanisms"]
    L03B --> P03["P03 Train Dreamer agent"]
    P02 --> P04["P04 Swap RSSM for Transformer backbone"]
    L03A --> P04
    P03 & P04 --> L04["L04 Evaluation metrics"]
    L04 --> P05["P05 Evaluation dashboard"]
    P03 & P04 --> P06["P06 Counterfactual action-conditioned WM"]
    P05 --> L05["L05 Frontier Debates"]
    P06 --> L05
```

Suggested path: L01, L02, P01, P02, L03, P03, P04, L04, P05, P06, L05

You do not need to finish all theory before starting a project. Build, then come back with questions.

---

## Quick start

```sh
npm install
npm run docs:dev        # dev server with hot reload
npm run docs:build      # production build
npm run docs:preview    # preview built site
```

To refresh the README screenshots after a build:

```sh
npm run docs:build
npm run screenshots:readme
```

---

## Repo structure

```
learn-world-model/
├── docs/                                  # VitePress documentation site
│   ├── .vitepress/config.mts             # nav and sidebar (EN + ZH)
│   ├── en/lectures/                       # 5 English lecture pages
│   ├── zh/lectures/                       # 5 Chinese lecture pages
│   ├── en/projects/                       # 6 English project pages
│   └── zh/projects/                       # 6 Chinese project pages
├── external/world-model-tutorial/         # PyTorch source referenced by projects
│   └── references.md                      # four-era history and architecture survey
├── scripts/                               # build utilities (screenshots, PDF)
└── package.json
```

---

## Community

Scan the QR code to join the WeChat discussion group (微信交流群):

<div align="center">
  <img src="./docs/public/wechat.png" width="300" alt="WeChat Group QR Code">
</div>

---

## Contributing

Contributions are welcome. Before submitting a pull request, read [CLAUDE.md](./CLAUDE.md) for the writing style rules that apply to all lecture and project files (no em dashes, no linear mermaid diagrams, no arrow-chain prose, EN/ZH sync, and others). Content that does not follow those rules will be asked to revise before merging.

---

## Contributors

| Name | Role | Affiliation | GitHub |
| ---- | ---- | ----------- | ------ |
| Zhimin Zhao | Project Lead | Queen's University | [@zhimin-z](https://github.com/zhimin-z) |
| Qi Wang | Project Lead | Chinese Academy of Sciences | [@qiwang067](https://github.com/qiwang067) |
