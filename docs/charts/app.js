import { METRICS, MODELS, THEMES, parseHistory, renderArchitecture, renderLossFigure } from "./charts.js";

const RUN_IDS = ["vae", "cnn"];
const SVG_NS = "http://www.w3.org/2000/svg";

const state = {
  runs: {},
  tests: {},
  metric: "total",
  show: { train: true, val: true },
  themeChoice: "auto",
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

const darkQuery = matchMedia("(prefers-color-scheme: dark)");
const theme = () => (state.themeChoice === "auto" ? (darkQuery.matches ? "dark" : "light") : state.themeChoice);

function readStoredTheme() {
  try {
    return localStorage.getItem("vae-comp-theme") ?? "auto";
  } catch {
    return "auto";
  }
}

function storeTheme(value) {
  try {
    localStorage.setItem("vae-comp-theme", value);
  } catch {
    /* storage unavailable: theme simply isn't remembered */
  }
}

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

async function fetchText(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.text();
}

async function loadAll() {
  try {
    await Promise.all(
      RUN_IDS.map(async (id) => {
        state.runs[id] = parseHistory(await fetchText(`../../artifacts/${id}/history.csv`));
        state.tests[id] = parseHistory(await fetchText(`../../artifacts/${id}/test.csv`))[0];
      }),
    );
  } catch (err) {
    console.warn("Could not fetch CSVs:", err);
    $("#fallback").hidden = false;
    $("#loss-chart").innerHTML = `<div class="placeholder">No data loaded yet — see below.</div>`;
  }
  renderAll();
}

for (const input of $$("[data-file]")) {
  input.addEventListener("change", async () => {
    const file = input.files[0];
    if (!file) return;
    state.runs[input.dataset.file] = parseHistory(await file.text());
    renderAll();
  });
}

const ready = () => RUN_IDS.every((id) => state.runs[id]?.length);

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function renderLoss() {
  if (!ready()) return;
  const metrics = state.metric === "all" ? ["total", "bce", "kl"] : [state.metric];
  const figure = renderLossFigure({
    runs: RUN_IDS.map((id) => ({ id, rows: state.runs[id] })),
    metrics,
    show: state.show,
    theme: theme(),
    width: metrics.length > 1 ? 1080 : 880,
    title: metrics.length > 1 ? "Loss decomposition" : METRICS[state.metric].label,
    subtitle: "Per-sample loss in nats · lower is better",
  });
  const host = $("#loss-chart");
  host.innerHTML = figure.svg;
  attachHover(host.querySelector("svg"), figure.panels);
}

function renderTable() {
  const body = $("#test-table tbody");
  const rows = RUN_IDS.filter((id) => state.tests[id]);
  if (!rows.length) return;

  const best = (key) => Math.min(...rows.map((id) => state.tests[id][key]));
  const t = THEMES[theme()];
  body.innerHTML = rows
    .map((id) => {
      const test = state.tests[id];
      const history = state.runs[id] ?? [];
      const steady = history.filter((r) => r.epoch > 1);
      const secs = steady.reduce((acc, r) => acc + r.train_seconds, 0) / Math.max(steady.length, 1);
      const cell = (key) =>
        `<td class="${test[key] === best(key) ? "best" : ""}">${test[key].toFixed(2)}</td>`;
      return `<tr>
        <td><span class="dot" style="background:${t[id]}"></span>${MODELS[id].label}</td>
        <td>${test.best_epoch}</td>
        ${cell("total")}${cell("bce")}${cell("kl")}
        <td>${secs.toFixed(2)}</td>
      </tr>`;
    })
    .join("");
}

function renderArchitectures() {
  for (const id of RUN_IDS) {
    $(`#arch-${id}`).innerHTML = renderArchitecture({ model: id, theme: theme() }).svg;
  }
}

function renderAll() {
  if (state.themeChoice === "auto") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = state.themeChoice;
  renderLoss();
  renderTable();
  renderArchitectures();
}

// ---------------------------------------------------------------------------
// Hover: crosshair + tooltip with every series at the hovered epoch
// ---------------------------------------------------------------------------

function attachHover(svg, panels) {
  const tooltip = $("#tooltip");
  const t = THEMES[theme()];

  const crosshairs = panels.map((p) => {
    const line = document.createElementNS(SVG_NS, "line");
    line.setAttribute("y1", p.py);
    line.setAttribute("y2", p.py + p.ph);
    line.setAttribute("stroke", t.muted);
    line.setAttribute("stroke-width", "1");
    line.setAttribute("visibility", "hidden");
    svg.appendChild(line);
    return line;
  });

  const hide = () => {
    tooltip.hidden = true;
    crosshairs.forEach((l) => l.setAttribute("visibility", "hidden"));
  };

  svg.addEventListener("pointerleave", hide);
  svg.addEventListener("pointermove", (event) => {
    const pt = new DOMPoint(event.clientX, event.clientY).matrixTransform(svg.getScreenCTM().inverse());
    const panel = panels.find((p) => pt.x >= p.px - 6 && pt.x <= p.px + p.pw + 6 && pt.y >= p.py && pt.y <= p.py + p.ph);
    if (!panel) return hide();

    const frac = Math.min(Math.max((pt.x - panel.px) / panel.pw, 0), 1);
    const epoch = Math.round(1 + frac * (panel.maxEpoch - 1));
    const x = panel.px + ((epoch - 1) / (panel.maxEpoch - 1)) * panel.pw;
    crosshairs.forEach((line, i) => {
      const p = panels[i];
      const xx = p.px + ((epoch - 1) / (p.maxEpoch - 1)) * p.pw;
      line.setAttribute("x1", xx);
      line.setAttribute("x2", xx);
      line.setAttribute("visibility", "visible");
    });

    const metric = panel.metric;
    const rows = RUN_IDS.map((id) => {
      const row = state.runs[id].find((r) => r.epoch === epoch);
      if (!row) return "";
      const cells = [];
      if (state.show.val) cells.push(`val <b>${row[`val_${metric}`].toFixed(2)}</b>`);
      if (state.show.train) cells.push(`train <b>${row[`train_${metric}`].toFixed(2)}</b>`);
      return `<div class="tt-row"><span class="swatch" style="background:${t[id]}"></span>${MODELS[id].label}${cells
        .map((c) => `<span>${c}</span>`)
        .join("")}</div>`;
    }).join("");
    const lr = state.runs.vae.find((r) => r.epoch === epoch);
    tooltip.innerHTML = `<div class="tt-head">Epoch ${epoch} · ${METRICS[metric].short}</div>${rows}${
      lr ? `<div class="tt-row" style="margin-top:4px"><span></span>β = ${lr.beta.toFixed(2)}</div>` : ""
    }`;
    tooltip.hidden = false;

    // Position relative to the section, flipping left near the right edge.
    const section = svg.closest("section").getBoundingClientRect();
    const screen = new DOMPoint(x, panel.py).matrixTransform(svg.getScreenCTM());
    const left = screen.x - section.left + 14;
    const flip = left + tooltip.offsetWidth > section.width;
    tooltip.style.left = `${flip ? left - tooltip.offsetWidth - 28 : left}px`;
    tooltip.style.top = `${event.clientY - section.top - tooltip.offsetHeight / 2}px`;
  });
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

function press(buttons, active) {
  buttons.forEach((b) => b.setAttribute("aria-pressed", String(b === active)));
}

for (const button of $$("[data-metric]")) {
  button.addEventListener("click", () => {
    state.metric = button.dataset.metric;
    press($$("[data-metric]"), button);
    renderLoss();
  });
}

for (const box of $$("[data-split]")) {
  box.addEventListener("change", () => {
    state.show[box.dataset.split] = box.checked;
    renderLoss();
  });
}

for (const button of $$("[data-theme-choice]")) {
  button.addEventListener("click", () => {
    state.themeChoice = button.dataset.themeChoice;
    storeTheme(state.themeChoice);
    press($$("[data-theme-choice]"), button);
    renderAll();
  });
}

darkQuery.addEventListener("change", () => state.themeChoice === "auto" && renderAll());

function download(name, svgText) {
  const blob = new Blob([svgText], { type: "image/svg+xml" });
  const url = URL.createObjectURL(blob);
  const a = Object.assign(document.createElement("a"), { href: url, download: `${name}-${theme()}.svg` });
  a.click();
  URL.revokeObjectURL(url);
}

for (const button of $$("[data-download]")) {
  button.addEventListener("click", () => {
    const target = button.dataset.download;
    const host = target === "loss" ? $("#loss-chart") : $(`#${target}`);
    const svg = host.querySelector("svg")?.cloneNode(true);
    if (!svg) return;
    svg.querySelectorAll("line[visibility]").forEach((l) => l.remove());
    download(target === "loss" ? `loss-${state.metric}` : target, svg.outerHTML);
  });
}

state.themeChoice = readStoredTheme();
press($$("[data-theme-choice]"), $(`[data-theme-choice="${state.themeChoice}"]`));
loadAll();
