# CLAUDE.md

This file guides Claude Code when working in this repository.

## Overview

This repo is a bilingual VitePress documentation site for a world-models curriculum. English and Chinese content should stay structurally aligned unless the user explicitly wants divergence.

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
