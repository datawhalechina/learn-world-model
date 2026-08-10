# CLAUDE.md

This file guides Claude Code when working in this repository.

## Overview

This repo is a bilingual VitePress documentation site for a world-models curriculum. English and Chinese content should stay structurally aligned unless the user explicitly wants divergence.

## Curriculum Structure

The lectures and projects form one interleaved learning path, not two independent tracks. Preserve this progression when changing content, prerequisites, navigation, or summaries:

1. L01 establishes motivation, vocabulary, scope, and the L1-L5 capability ladder.
2. L02 Part A introduces observation encoding, followed immediately by P01.
3. L02 Part B introduces latent dynamics and RSSM, followed immediately by P02.
4. L03 Part A completes the planning loop with CEM-MPC, latent Actor-Critic, and TD-MPC, followed by P03.
5. L03 Part B compares core dynamics backbones, especially RSSM and Transformer, followed by P04.
6. L03 Part C is an optional frontier survey covering JEPA, RWM, Genie, LoopWM, WAM, system-integration patterns, and LS-Imagine. It is not a prerequisite for P03 or P04.
7. L04 teaches a model-independent diagnostic framework and uses named models as worked examples, followed by P05 and P06.
8. L05 contains frontier debates and the philosophical coda.

Keep these conceptual distinctions consistent:

- L03 surveys eight model architecture families: RNN/RSSM, Transformer, Diffusion, JEPA, RWM, Genie, LoopWM, and WAM.
- The seven system-integration patterns describe where prediction enters a complete agent. They are not seven additional architecture families.
- CWM is a domain extension into code execution space, not a ninth L03 dynamics-backbone family.
- L04 evaluation begins by locating the failed interface: representation, one-step dynamics, long-horizon rollout, task signal, planner or policy, or deployment loop. Model-specific metrics illustrate this framework rather than define the lecture order.
- P04 replaces the RSSM backbone with a STORM-style Transformer. It does not implement TD-MPC.
- P05 evaluates the trained Dreamer and Transformer systems. It is an application of L04, not a prerequisite for reading L04.

## Commands

```sh
npm install
npm run docs:dev
npm run docs:build
npm run docs:preview
```

## Important Paths

- `docs/` - site content
- `docs/.vitepress/config.mts` - nav and sidebar config for both locales
- `docs/en/` and `docs/zh/` - English and Chinese docs
- `docs/*/lectures/` - lecture pages, typically `index.md` plus numbered sub-pages
- `docs/*/projects/` - project notebooks and generated markdown summaries
- `external/world-model-tutorial/` - reference code and notes
- `scripts/build-notebook-pages.ts` - regenerates project markdown from notebooks

## Editing Rules

- Keep new text consistent with the surrounding section. Read nearby paragraphs before editing.
- Mirror any structural or prose change across `docs/en/` and `docs/zh/` unless the user asks otherwise.
- Do not use em dashes.
- Do not use arrow-chain prose such as `A -> B -> C` or `A → B → C`. Rewrite as sentences, lists, or tables.
- Use Mermaid only when the visual structure adds real value. Avoid linear chains, trivial diagrams, and anything that prose already explains well.
- Do not use ASCII diagrams.
- Do not place a table immediately next to a Mermaid block or figure. Insert prose between them.
- Keep pages reasonably short. If a markdown file becomes too long, split it at a clear conceptual boundary and update `config.mts`.

## Projects And Notebooks

- For project pages, the `.ipynb` file is the source of truth. The matching `.md` page is generated.
- `docs:dev` and `docs:build` run `scripts/build-notebook-pages.ts`, which regenerates project markdown, derives the title from the first heading, and inserts a notebook-source link.
- Prefer editing the notebook content rather than the generated project markdown.
- Keep project markdown pages in git. Keep `docs/*/projects/notebook-assets/` ignored.
- Project markdown should contain narrative text and code blocks only. Do not add rendered outputs, plots, or tables there.
- Do not modify notebook code cells unless the user explicitly asks for notebook code changes.
