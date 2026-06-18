/// <reference types="node" />
/**
 * Convert project Jupyter notebooks into VitePress markdown pages.
 *
 * VitePress only routes `.md` files, so a linked `.ipynb` 404s. This script
 * renders each `docs/{en,zh}/projects/*.ipynb` into a sibling `.md` page:
 *   - markdown cells pass through as-is
 *   - code cells become ```python fenced blocks
 *   - notebook outputs are intentionally omitted from the markdown pages
 *   - output artifacts stay in the source `.ipynb` files
 *
 * The generated `.md` files are what the sidebar and project index link to.
 * Run automatically before `docs:build` and `docs:dev`; safe to re-run.
 */
import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(SCRIPT_DIR, "..");
const PROJECT_DIRS = [
  path.join(ROOT, "docs", "en", "projects"),
  path.join(ROOT, "docs", "zh", "projects"),
];

interface NotebookCell {
  cell_type: "markdown" | "code" | "raw";
  source: string[] | string;
}

interface Notebook {
  cells: NotebookCell[];
  metadata?: Record<string, unknown>;
}

const joinSource = (src: string[] | string): string =>
  Array.isArray(src) ? src.join("") : src;

const fence = (lang: string, body: string): string =>
  "```" + lang + "\n" + body.replace(/\n+$/, "") + "\n```";

function firstHeadingTitle(cells: NotebookCell[], fallback: string): string {
  for (const cell of cells) {
    if (cell.cell_type !== "markdown") continue;
    const text = joinSource(cell.source);
    const m = text.match(/^\s*#{1,6}\s+(.+?)\s*$/m);
    if (m) return m[1].replace(/[:"]/g, "").trim();
  }
  return fallback;
}

async function convertNotebook(nbPath: string): Promise<void> {
  const raw = await fs.readFile(nbPath, "utf8");
  const nb = JSON.parse(raw) as Notebook;
  const base = path.basename(nbPath, ".ipynb");

  const title = firstHeadingTitle(nb.cells, base);
  const parts: string[] = [`---\ntitle: ${title}\n---\n`];

  for (const cell of nb.cells) {
    if (cell.cell_type === "markdown") {
      parts.push(joinSource(cell.source).replace(/\n+$/, "") + "\n");
      continue;
    }
    if (cell.cell_type === "raw") {
      continue;
    }

    // code cell
    const code = joinSource(cell.source).replace(/\n+$/, "");
    if (code.trim().length > 0) {
      parts.push(fence("python", code));
    }
  }

  const mdPath = path.join(path.dirname(nbPath), `${base}.md`);
  await fs.writeFile(mdPath, parts.join("\n") + "\n", "utf8");
  const rel = path.relative(ROOT, mdPath);
  console.log(`  rendered ${rel}`);
}

async function main(): Promise<void> {
  for (const dir of PROJECT_DIRS) {
    let entries: string[];
    try {
      entries = (await fs.readdir(dir)).filter((f) => f.endsWith(".ipynb"));
    } catch {
      continue;
    }
    if (entries.length === 0) continue;

    console.log(`Converting ${entries.length} notebook(s) in ${path.relative(ROOT, dir)}:`);
    for (const file of entries.sort()) {
      await convertNotebook(path.join(dir, file));
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
