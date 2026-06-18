import { spawn } from "node:child_process";

function start(command: string, args: string[]) {
  const child = spawn(command, args, {
    stdio: "inherit",
    env: process.env,
  });
  child.on("exit", (code, signal) => {
    stopAll(signal ? 1 : code ?? 0);
  });
  return child;
}

let notebookWatcher: ReturnType<typeof spawn> | null = null;
let vitepressDev: ReturnType<typeof spawn> | null = null;
let shuttingDown = false;

function stopAll(code: number) {
  if (shuttingDown) {
    return;
  }
  shuttingDown = true;
  notebookWatcher?.kill("SIGTERM");
  vitepressDev?.kill("SIGTERM");
  process.exit(code);
}

async function main() {
  notebookWatcher = start("tsx", ["scripts/build-notebook-pages.ts", "--watch"]);
  vitepressDev = start("vitepress", ["dev", "docs"]);

  const shutdown = () => {
    stopAll(0);
  };

  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
