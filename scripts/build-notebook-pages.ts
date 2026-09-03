/// <reference types="node" />
/**
 * Convert project Jupyter notebooks into VitePress markdown pages.
 *
 * VitePress only routes `.md` files, so a linked `.ipynb` 404s. This script
 * renders each `docs/{en,zh,ko}/projects/*.ipynb` into a sibling `.md` page:
 *   - markdown cells pass through as-is
 *   - code cells become ```python fenced blocks
 *   - notebook outputs are intentionally omitted from the markdown pages
 *   - output artifacts stay in the source `.ipynb` files
 *
 * The generated `.md` files are what the sidebar and project index link to.
 * The script also keeps a `project_url` field in notebook metadata so the
 * notebook itself knows which docs page it belongs to.
 *
 * Run automatically before `docs:build` and `docs:dev`; safe to re-run.
 */
import { promises as fs, watch as watchFs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(SCRIPT_DIR, "..");
const PROJECT_DIRS = [
  path.join(ROOT, "docs", "en", "projects"),
  path.join(ROOT, "docs", "zh", "projects"),
  path.join(ROOT, "docs", "ko", "projects"),
];
const GITHUB_BLOB_BASE =
  "https://github.com/datawhalechina/learn-world-model/blob/main/";

interface NotebookCell {
  cell_type: "markdown" | "code" | "raw";
  source: string[] | string;
}

interface Notebook {
  cells: NotebookCell[];
  metadata?: {
    project_url?: string;
    [key: string]: unknown;
  };
}

const joinSource = (src: string[] | string): string =>
  Array.isArray(src) ? src.join("") : src;

const fence = (lang: string, body: string): string =>
  "```" + lang + "\n" + body.replace(/\n+$/, "") + "\n```";

function codeFenceLanguage(code: string): string {
  const firstLine = code.trimStart().split(/\r?\n/, 1)[0]?.trim() ?? "";
  if (/^%%(?:bash|sh)\b/.test(firstLine)) return "bash";
  return "python";
}

function projectPageUrl(nbPath: string): string {
  const relPath = path.relative(ROOT, nbPath).split(path.sep).join("/");
  const match = relPath.match(/^docs\/(en|zh|ko)\/projects\/([^/]+)\.ipynb$/);
  if (!match) {
    throw new Error(`Unexpected notebook path: ${nbPath}`);
  }
  return `/${match[1]}/projects/${match[2]}/`;
}

function sourceNotebookUrl(nbPath: string): string {
  const relPath = path.relative(ROOT, nbPath).split(path.sep).join("/");
  return `${GITHUB_BLOB_BASE}${relPath}`;
}

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
  const relPath = path.relative(ROOT, nbPath).split(path.sep).join("/");
  const locale = relPath.startsWith("docs/zh/") ? "zh" : relPath.startsWith("docs/ko/") ? "ko" : "en";
  const sourceLabel = locale === "zh" ? "Notebook 源文件" : locale === "ko" ? "Notebook 원본" : "Notebook source";
  const projectLabel = locale === "zh" ? "项目页面" : locale === "ko" ? "프로젝트 페이지" : "Project page";
  const projectUrl = projectPageUrl(nbPath);
  const notebookUrl = sourceNotebookUrl(nbPath);

  // Ensure notebook metadata has project_url
  const metadata = nb.metadata ?? {};
  let nbDirty = metadata.project_url !== projectUrl;
  if (nbDirty) {
    nb.metadata = { ...metadata, project_url: projectUrl };
  }

  // Ensure a GitHub source link cell exists in the notebook itself (after the first markdown cell)
  const linkLine = `> ${sourceLabel}: [${base}.ipynb](${notebookUrl})`;
  const hasLinkCell = nb.cells.some(
    (c) => c.cell_type === "markdown" && joinSource(c.source).includes(notebookUrl)
  );
  if (!hasLinkCell) {
    const firstMdIdx = nb.cells.findIndex((c) => c.cell_type === "markdown");
    const insertAt = firstMdIdx >= 0 ? firstMdIdx + 1 : 0;
    nb.cells.splice(insertAt, 0, {
      cell_type: "markdown",
      id: "notebook-source-link",
      metadata: {},
      source: [linkLine],
    } as unknown as NotebookCell);
    nbDirty = true;
  }

  if (nbDirty) {
    await fs.writeFile(nbPath, JSON.stringify(nb, null, 1) + "\n", "utf8");
  }

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
      parts.push(fence(codeFenceLanguage(code), code));
    }
  }

  const mdPath = path.join(path.dirname(nbPath), `${base}.md`);
  await fs.writeFile(mdPath, parts.join("\n") + "\n", "utf8");
  const rel = path.relative(ROOT, mdPath);
  console.log(`  rendered ${rel}`);
}

async function renderAll(): Promise<void> {
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

function parseWatchFlag(): boolean {
  return process.argv.includes("--watch");
}

async function watch(): Promise<void> {
  let timer: NodeJS.Timeout | null = null;
  let rerendering = false;
  let rerunRequested = false;

  const schedule = () => {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => {
      timer = null;
      void run();
    }, 150);
  };

  const run = async () => {
    if (rerendering) {
      rerunRequested = true;
      return;
    }
    rerendering = true;
    try {
      await renderAll();
    } catch (error) {
      console.error(error);
    } finally {
      rerendering = false;
      if (rerunRequested) {
        rerunRequested = false;
        schedule();
      }
    }
  };

  await renderAll();
  for (const dir of PROJECT_DIRS) {
    watchFs(dir, { persistent: true }, (eventType, filename) => {
      if (!filename || !filename.endsWith(".ipynb")) {
        return;
      }
      console.log(`File change detected in ${path.relative(ROOT, dir)}: ${String(filename)} (${eventType})`);
      schedule();
    });
  }

  await new Promise<void>(() => {});
}

async function main(): Promise<void> {
  if (parseWatchFlag()) {
    await watch();
    return;
  }
  await renderAll();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
