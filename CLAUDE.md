# CLAUDE.md

This file guides Claude Code when working in this repository.

## Repository Purpose

This repository is a bilingual VitePress curriculum for learning world models through paired lectures and hands-on projects. It serves readers with fundamental deep learning and reinforcement learning knowledge, then progressively develops latent dynamics, planning, evaluation, and frontier research judgment.

English and Chinese are parallel editions of the same curriculum. Keep them structurally and semantically aligned unless the user explicitly requests divergence.

## Canonical Learning Path

Lectures and projects form one interleaved path. Do not present them as independent tracks.

| Stage | Conceptual goal | Reading | Practice |
| --- | --- | --- | --- |
| Foundations | Define world models, their scope, and the L1-L5 capability ladder | L01 | None |
| Representation | Compress observations into useful latent states | L02 Part A | P01, VAE encoder |
| Dynamics | Predict latent state transitions with memory and uncertainty | L02 Part B | P02, RSSM dynamics |
| Control | Use predicted futures to select actions | L03 Part A | P03, Dreamer agent |
| Backbone choice | Compare RSSM with Transformer and diffusion alternatives | L03 Part B | P04, Transformer backbone |
| Frontier survey | Build research orientation beyond the implementation path | L03 Part C, optional | None |
| Diagnosis | Evaluate representations, dynamics, planning, rollouts, and deployment | L04 | P05, evaluation dashboard; P06, counterfactual fidelity |
| Open questions | Examine unresolved technical and philosophical debates | L05 | None |

Readers should encounter a working mechanism before a broad catalogue of alternatives. Preserve the planning-first order in L03 and the model-independent diagnostic framework at the start of L04.

## Content Architecture

### Lecture 1: Scope and Capability

L01 establishes motivation, vocabulary, historical context, and the L1-L5 capability ladder. Keep it accessible to readers who have not studied model-based RL. Advanced system-integration details belong later in the curriculum.

### Lecture 2: Representation and Dynamics

L02 builds the technical foundation incrementally. Part A leads directly to P01. Part B then develops GRU, MDN-RNN, RSSM, and the Dreamer series before P02.

### Lecture 3: Planning, Backbones, and Frontier Systems

L03 has three distinct layers:

- **Part A, core planning**: CEM-MPC, latent Actor-Critic, and TD-MPC. This is the prerequisite for P03.
- **Part B, core backbone comparison**: RNN/RSSM, Transformer, and diffusion. This supports P04.
- **Part C, optional frontier survey**: JEPA, RWM, Genie, LoopWM, WAM, system-integration patterns, and LS-Imagine.

The architecture survey contains eight families: RNN/RSSM, Transformer, Diffusion, JEPA, RWM, Genie, LoopWM, and WAM. The seven system-integration patterns describe where prediction enters a complete agent; they are not additional architecture families. CWM is a domain extension into code execution space, not a ninth L03 backbone.

### Lecture 4: Diagnostic Framework

L04 is organized by transferable diagnostic interfaces, with named models used as worked examples. Preserve this diagnostic order:

1. Representation quality
2. One-step dynamics
3. Long-horizon rollout
4. Task signals such as reward and value
5. Planner or policy behavior
6. Deployment-loop reliability

Dreamer illustrates representation and task-signal diagnosis. TD-MPC illustrates latent consistency and planning efficiency. MuZero illustrates value and search quality. STORM illustrates autoregressive rollout quality. Diamond illustrates physical consistency and horizon drift.

### Lecture 5: Frontier Debates

L05 is a synthesis and debate lecture. New systems introduced here should be framed as extensions or open questions, not silently added to the core architecture taxonomy.

## Project Map

Project notebooks pass trained checkpoints forward through the curriculum.

| Project | Prerequisites | Purpose |
| --- | --- | --- |
| P01 | L02 Part A | Train a VAE encoder |
| P02 | P01 and L02 Part B | Build and inspect RSSM dynamics |
| P03 | P02 and L03 Part A | Train the complete Dreamer loop |
| P04 | P03 and L03 Part B | Replace the RSSM backbone with a STORM-style Transformer |
| P05 | P03, P04, and L04 | Compare trained systems with a diagnostic dashboard |
| P06 | P03 and P04; use the L04 framework | Test action-conditioned and counterfactual fidelity |

P04 does not implement TD-MPC. P05 applies L04 and is not a prerequisite for reading it.

## Repository Layout

- `docs/` contains the VitePress site.
- `docs/.vitepress/config.mts` defines bilingual navigation and sidebars.
- `docs/en/` and `docs/zh/` contain the parallel language editions.
- `docs/*/lectures/` contains lecture indexes and section pages.
- `docs/*/projects/` contains project notebooks and generated Markdown pages.
- `external/world-model-tutorial/` contains reference code and notes.
- `scripts/build-notebook-pages.ts` regenerates project Markdown from notebooks.

## Editing Invariants

- Read the surrounding section before editing and match its depth, terminology, and tone.
- Mirror structural and prose changes across English and Chinese unless the user asks otherwise.
- Update `docs/.vitepress/config.mts` whenever pages are added, removed, renamed, or reordered.
- Explain a concept when it first becomes necessary. If a name appears earlier as orientation, give a short preview and clearly defer the full mechanism.
- Keep prerequisites truthful. A page must not claim that a project implements a model or metric it does not contain.
- Separate core prerequisites from optional frontier material explicitly.
- Keep pages reasonably short. Split a long page at a clear conceptual boundary and update both sidebars.
- Do not place two `> **📖` learning-note blocks consecutively. Merge closely related definitions into one note; when a passage carries the main explanation, present it as ordinary prose instead of using a note as decoration.
- Do not use em dashes.
- Do not use arrow-chain prose such as `A -> B -> C` or `A → B → C`. Use sentences, lists, or tables.
- Do not use ASCII diagrams.
- Use Mermaid only when visual structure materially improves understanding. Avoid trivial linear diagrams.
- Do not place a table immediately next to a Mermaid block or figure. Insert explanatory prose between them.

## Projects and Notebooks

- The `.ipynb` file is the source of truth for each project page. Its matching `.md` page is generated.
- `docs:dev` and `docs:build` run `scripts/build-notebook-pages.ts` before VitePress.
- Edit notebook narrative cells instead of generated project Markdown.
- Do not modify notebook code cells unless the user explicitly requests code changes.
- Keep generated project Markdown in git.
- Keep `docs/*/projects/notebook-assets/` ignored.
- Generated project pages should contain narrative text and code blocks only, without rendered outputs, plots, or tables.

## Verification Workflow

Run checks proportional to the change. Structural curriculum edits should receive the full sequence:

```sh
git diff --check
npm run docs:build
```

Before finishing a structural edit, also verify:

- Every English lecture page has a Chinese counterpart.
- Every sidebar target resolves to a page.
- Relative Markdown links resolve.
- Lecture indexes, roadmap, homepage, project prerequisites, and summaries describe the same order.
- Generated project Markdown was not edited directly or unintentionally changed.

## Commands

```sh
npm install
npm run docs:dev
npm run docs:build
npm run docs:preview
```
