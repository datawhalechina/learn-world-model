# CLAUDE.md

This file guides Claude Code when working in this repository.

## Repository Purpose

A bilingual VitePress curriculum teaching world models through paired lectures and hands-on projects, for readers with fundamental deep learning and RL knowledge. English and Chinese are parallel editions: keep them structurally and semantically aligned unless the user asks otherwise.

## Canonical Learning Path

Lectures and projects are one interleaved path, not independent tracks. Readers should see a working mechanism before a catalogue of alternatives. Projects pass trained checkpoints forward.

| Stage | Reading | Practice | Notes |
| --- | --- | --- | --- |
| Foundations: world models, scope, L1-L5 capability ladder | L01 | None | Accessible without prior model-based RL |
| Representation: compress observations into latent states | L02: Observation Encoding | P01, VAE encoder | |
| Dynamics: predict latent transitions with memory and uncertainty | L02: Latent Dynamics (GRU, MDN-RNN, RSSM, Dreamer series) | P02, RSSM dynamics | Builds on P01's checkpoint |
| Control: use predicted futures to select actions | L03: Planning and Control (CEM-MPC, latent Actor-Critic, TD-MPC) | P03, Dreamer agent | Prerequisite for P03 |
| Backbone choice: RSSM vs. Transformer vs. diffusion | L03: Backbone Selection | P04, Transformer backbone (STORM-style) | P04 does not implement TD-MPC |
| Frontier survey: research orientation beyond the build path | L03: Optional Frontier Survey (JEPA, RWM, Spatial 3D/4D, Genie, LoopWM, WAM, system-integration patterns, LS-Imagine) | None | Nine architecture families total (RNN/RSSM, Transformer, Diffusion, JEPA, RWM, Spatial 3D/4D, Genie, LoopWM, WAM); the seven integration patterns describe where prediction enters an agent, not additional families; CWM is a domain extension, not a tenth family |
| Diagnosis: evaluate representations, dynamics, planning, rollouts, deployment | L04, organized by diagnostic interface not by model (representation quality, one-step dynamics, long-horizon rollout, task signals, planner/policy behavior, deployment-loop reliability) | P05, evaluation dashboard (applies L04, not a prerequisite for reading it); P06, counterfactual fidelity | Worked examples: Dreamer, TD-MPC, MuZero, STORM, Diamond |
| Open questions: unresolved technical and philosophical debates | L05 | None | Extensions or open questions only, not silent additions to the core taxonomy |

Preserve the planning-first order in L03 and the model-independent diagnostic framework at the start of L04. Keep prerequisites truthful: a page must not claim a project implements a model or metric it doesn't contain, and core prerequisites must stay separate from optional frontier material.

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
