// Renders the README figures (light + dark variants) from artifacts/*/history.csv.
// Usage, from the repository root:  node docs/charts/export.js

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { parseHistory, renderArchitecture, renderLossFigure } from "./charts.js";

const root = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const out = join(root, "docs", "figures");
mkdirSync(out, { recursive: true });

const runs = ["vae", "cnn"].map((id) => ({
  id,
  rows: parseHistory(readFileSync(join(root, "artifacts", id, "history.csv"), "utf8")),
}));

const figures = {
  "loss-total": (theme) =>
    renderLossFigure({
      runs,
      theme,
      metrics: ["total"],
      title: "BCE + KL objective on MNIST",
      subtitle: "Per-sample loss in nats · lower is better · y-axis starts after warm-up epochs",
    }),
  "loss-components": (theme) =>
    renderLossFigure({
      runs,
      theme,
      metrics: ["bce", "kl"],
      title: "Loss decomposition",
      subtitle: "Reconstruction term vs. KL regularization",
    }),
  "arch-vae": (theme) => renderArchitecture({ model: "vae", theme }),
  "arch-cnn": (theme) => renderArchitecture({ model: "cnn", theme }),
};

for (const [name, render] of Object.entries(figures)) {
  for (const theme of ["light", "dark"]) {
    const file = join(out, `${name}-${theme}.svg`);
    writeFileSync(file, render(theme).svg + "\n");
    console.log("wrote", file.replace(root + "/", ""));
  }
}
