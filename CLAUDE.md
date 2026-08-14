# CLAUDE.md

This file guides Claude Code when working in this repository.

## Repository Purpose

A bilingual VitePress curriculum teaching world models through paired lectures and hands-on projects, for readers with fundamental deep learning and RL knowledge. English and Chinese are parallel editions: keep them structurally and semantically aligned unless the user asks otherwise.

## Canonical Learning Path

Lectures and projects are one interleaved path, not independent tracks. Readers should see a working mechanism before a catalogue of alternatives.

| Stage | Goal | Reading | Practice |
| --- | --- | --- | --- |
| Foundations | World models, scope, L1-L5 capability ladder | L01 | None |
| Representation | Compress observations into latent states | L02: Observation Encoding | P01, VAE encoder |
| Dynamics | Predict latent transitions with memory and uncertainty | L02: Latent Dynamics | P02, RSSM dynamics |
| Control | Use predicted futures to select actions | L03: Planning and Control | P03, Dreamer agent |
| Backbone choice | RSSM vs. Transformer vs. diffusion | L03: Backbone Selection | P04, Transformer backbone |
| Frontier survey | Research orientation beyond the build path | L03: Optional Frontier Survey | None |
| Diagnosis | Evaluate representations, dynamics, planning, rollouts, deployment | L04 | P05, evaluation dashboard; P06, counterfactual fidelity |
| Open questions | Unresolved technical and philosophical debates | L05 | None |

Preserve the planning-first order in L03 and the model-independent diagnostic framework at the start of L04.

## Content Architecture

**L01**: motivation, vocabulary, history, the L1-L5 capability ladder. Accessible without prior model-based RL. Advanced system-integration detail belongs later.

**L02**: Observation Encoding leads to P01. Latent Dynamics (GRU, MDN-RNN, RSSM, Dreamer series) leads to P02.

**L03**, three named modules:
- Planning and Control (prerequisite for P03): CEM-MPC, latent Actor-Critic, TD-MPC.
- Backbone Selection (supports P04): RNN/RSSM, Transformer, diffusion.
- Optional Frontier Survey: JEPA, RWM, Spatial 3D/4D, Genie, LoopWM, WAM, system-integration patterns, LS-Imagine.

The architecture survey has nine families: RNN/RSSM, Transformer, Diffusion, JEPA, RWM, Spatial 3D/4D, Genie, LoopWM, WAM. The seven system-integration patterns describe where prediction enters an agent; they are not additional families. The Integration Patterns page briefly distinguishes a VLA (Vision-Language-Action model) from a world model, since two patterns route a world model's prediction into one — VLA-internal mechanics (action tokenization, action chunking, embodiment gap) are out of scope and must not be added. CWM is a domain extension into code execution space, not a tenth backbone.

**L04**: organized by diagnostic interface, not by model. Order: representation quality, one-step dynamics, long-horizon rollout, task signals, planner/policy behavior, deployment-loop reliability. Worked examples: Dreamer (representation, task signal), TD-MPC (latent consistency, planning efficiency), MuZero (value, search), STORM (autoregressive rollout), Diamond (physical consistency, horizon drift).

**L05**: synthesis and debate. New systems here are extensions or open questions, not silent additions to the core taxonomy.

## Project Map

Projects pass trained checkpoints forward.

| Project | Prerequisites | Purpose |
| --- | --- | --- |
| P01 | L02: Observation Encoding | Train a VAE encoder |
| P02 | P01, L02: Latent Dynamics | Build and inspect RSSM dynamics |
| P03 | P02, L03: Planning and Control | Train the complete Dreamer loop |
| P04 | P03, L03: Backbone Selection | Replace the RSSM backbone with a STORM-style Transformer |
| P05 | P03, P04, L04 | Compare trained systems with a diagnostic dashboard |
| P06 | P03, P04; use the L04 framework | Test action-conditioned and counterfactual fidelity |

P04 does not implement TD-MPC. P05 applies L04 but is not a prerequisite for reading it.

## Repository Layout

- `docs/` — VitePress site. `docs/.vitepress/config.mts` defines bilingual navigation/sidebars.
- `docs/en/`, `docs/zh/` — parallel editions. `docs/*/lectures/`, `docs/*/projects/`.
- `external/world-model-tutorial/` — reference code and notes.
- `scripts/build-notebook-pages.ts` — regenerates project Markdown from notebooks.

## Editing Invariants

- Read the surrounding section first; match its depth, terminology, tone.
- Mirror structural and prose changes across EN/ZH unless told otherwise.
- Update `config.mts` whenever pages are added, removed, renamed, or reordered.
- Explain a concept when it first becomes necessary; a name used earlier gets a short preview, not the full mechanism.
- Keep prerequisites truthful: a page must not claim a project implements a model or metric it doesn't contain.
- Separate core prerequisites from optional frontier material explicitly.
- Split long pages at a clear conceptual boundary and update both sidebars.
- No two consecutive `> **📖` learning-note blocks; merge related definitions, or use prose if the passage is the main explanation.
- No em dashes. No arrow-chain prose (`A -> B -> C`). No ASCII diagrams.
- Use Mermaid only when it materially improves understanding, never for trivial linear flows.
- Don't place a table immediately next to a Mermaid block or figure; put prose between them.

## Projects and Notebooks

- The `.ipynb` is the source of truth; its `.md` page is generated (`docs:dev`/`docs:build` run `scripts/build-notebook-pages.ts` first).
- Edit notebook narrative cells, not generated Markdown. Don't touch notebook code cells unless asked.
- Keep generated Markdown in git; keep `docs/*/projects/notebook-assets/` ignored.
- Generated project pages: narrative text and code blocks only, no rendered outputs/plots/tables.

## Verification Workflow

Structural edits get the full sequence:

```sh
git diff --check
npm run docs:build
```

Also verify: every EN lecture page has a ZH counterpart; every sidebar target resolves; relative links resolve; lecture indexes/roadmap/homepage/project prerequisites describe the same order; generated project Markdown wasn't edited directly.

## Commands

```sh
npm install
npm run docs:dev
npm run docs:build
npm run docs:preview
```
