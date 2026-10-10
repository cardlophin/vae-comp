// Pure SVG renderers shared by the interactive template (index.html) and the
// static exporter (export.js). No DOM access here: every function returns an
// SVG string plus the layout needed to attach hover behaviour.

export const THEMES = {
  light: {
    surface: "#fcfcfb",
    ink: "#0b0b0b",
    secondary: "#52514e",
    muted: "#898781",
    grid: "#e1e0d9",
    axis: "#c3c2b7",
    border: "rgba(11,11,11,0.10)",
    band: "rgba(11,11,11,0.035)",
    node: "#ffffff",
    vae: "#2a78d6",
    cnn: "#eb6834",
  },
  dark: {
    surface: "#1a1a19",
    ink: "#ffffff",
    secondary: "#c3c2b7",
    muted: "#898781",
    grid: "#2c2c2a",
    axis: "#383835",
    border: "rgba(255,255,255,0.10)",
    band: "rgba(255,255,255,0.04)",
    node: "#232322",
    vae: "#3987e5",
    cnn: "#d95926",
  },
};

export const MODELS = {
  vae: { label: "Dense VAE" },
  cnn: { label: "CNN-VAE" },
};

export const METRICS = {
  total: { label: "Total loss (BCE + KL)", short: "Total", key: "total" },
  bce: { label: "Reconstruction (BCE)", short: "BCE", key: "bce" },
  kl: { label: "KL divergence", short: "KL", key: "kl" },
};

const FONT = `system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif`;

export function parseHistory(text) {
  const [header, ...lines] = text.trim().split(/\r?\n/);
  const keys = header.split(",");
  return lines.map((line) => {
    const values = line.split(",").map(Number);
    return Object.fromEntries(keys.map((key, i) => [key, values[i]]));
  });
}

const esc = (s) =>
  String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

const fmt = (v, digits = 1) => v.toFixed(digits);

function niceTicks(min, max, count = 5) {
  const span = max - min;
  const raw = span / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const start = Math.floor(min / step) * step;
  const end = Math.ceil(max / step) * step;
  const ticks = [];
  for (let v = start; v <= end + step / 2; v += step) ticks.push(+v.toFixed(10));
  return ticks;
}

function linePath(points) {
  return points.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join("");
}

// Auto y-domain ignoring the first epochs of β warm-up, where the loss is
// several times larger and would flatten everything else.
function autoDomain(runs, metric, skip) {
  const values = runs.flatMap((r) =>
    r.rows.filter((row) => row.epoch > skip).flatMap((row) => [row[`train_${metric}`], row[`val_${metric}`]]),
  );
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const pad = (hi - lo) * 0.06;
  return [lo - pad, hi + pad];
}

/**
 * One loss panel. Returns { svg, layout } where svg is a <g> fragment.
 * runs: [{ id: "vae" | "cnn", rows }]
 */
function lossPanel({ runs, metric, theme, x, y, width, height, title, show, warmup = 5, skip = 2 }) {
  const t = THEMES[theme];
  const m = { top: 30, right: 74, bottom: 38, left: 48 };
  const pw = width - m.left - m.right;
  const ph = height - m.top - m.bottom;
  const px = x + m.left;
  const py = y + m.top;

  const maxEpoch = Math.max(...runs.map((r) => r.rows.at(-1).epoch));
  const [lo, hi] = autoDomain(runs, metric, skip);
  const yTicks = niceTicks(lo, hi, 5).filter((v) => v >= lo && v <= hi);
  const xTicks = niceTicks(0, maxEpoch, 5).filter((v) => v <= maxEpoch);

  const sx = (e) => px + ((e - 1) / (maxEpoch - 1)) * pw;
  const sy = (v) => py + ph - ((v - lo) / (hi - lo)) * ph;
  const clipId = `clip-${metric}-${Math.round(x)}-${Math.round(y)}`;

  const parts = [];
  if (title) parts.push(`<text x="${px}" y="${y + 16}" font-size="13" font-weight="600" fill="${t.ink}">${esc(title)}</text>`);

  // β warm-up band: the loss is not comparable to later epochs there.
  const bandRight = sx(warmup);
  parts.push(`<rect x="${px}" y="${py}" width="${(bandRight - px).toFixed(1)}" height="${ph}" fill="${t.band}"/>`);
  parts.push(
    `<text x="${bandRight + 6}" y="${py + 14}" font-size="10" fill="${t.muted}">← β warm-up</text>`,
  );

  for (const v of yTicks) {
    const yy = sy(v).toFixed(1);
    parts.push(`<line x1="${px}" x2="${px + pw}" y1="${yy}" y2="${yy}" stroke="${t.grid}" stroke-width="1"/>`);
    parts.push(
      `<text x="${px - 8}" y="${yy}" dy="0.32em" font-size="11" text-anchor="end" fill="${t.muted}" style="font-variant-numeric:tabular-nums">${v}</text>`,
    );
  }
  parts.push(`<line x1="${px}" x2="${px + pw}" y1="${py + ph}" y2="${py + ph}" stroke="${t.axis}" stroke-width="1"/>`);
  for (const v of xTicks) {
    const xx = sx(Math.max(v, 1)).toFixed(1);
    parts.push(
      `<text x="${xx}" y="${py + ph + 16}" font-size="11" text-anchor="middle" fill="${t.muted}" style="font-variant-numeric:tabular-nums">${Math.max(v, 1)}</text>`,
    );
  }
  parts.push(
    `<text x="${px + pw / 2}" y="${py + ph + 32}" font-size="11" text-anchor="middle" fill="${t.secondary}">epoch</text>`,
  );

  parts.push(`<clipPath id="${clipId}"><rect x="${px}" y="${py}" width="${pw}" height="${ph}"/></clipPath>`);
  parts.push(`<g clip-path="url(#${clipId})" fill="none" stroke-linejoin="round" stroke-linecap="round">`);
  for (const run of runs) {
    const color = t[run.id];
    const train = run.rows.map((r) => [sx(r.epoch), sy(r[`train_${metric}`])]);
    const val = run.rows.map((r) => [sx(r.epoch), sy(r[`val_${metric}`])]);
    if (show.train) parts.push(`<path d="${linePath(train)}" stroke="${color}" stroke-width="1.5" stroke-dasharray="4 3" opacity="0.75"/>`);
    if (show.val) parts.push(`<path d="${linePath(val)}" stroke="${color}" stroke-width="2"/>`);
  }
  parts.push(`</g>`);

  // Direct labels at the end of each validation curve, nudged apart.
  const ends = runs
    .map((run) => {
      const last = run.rows.at(-1);
      return { run, value: last[`val_${metric}`], y: sy(last[`val_${metric}`]) };
    })
    .sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) {
    if (ends[i].y - ends[i - 1].y < 34) ends[i].y = ends[i - 1].y + 34;
  }
  for (const end of ends) {
    const color = t[end.run.id];
    const lx = px + pw + 8;
    parts.push(`<circle cx="${px + pw}" cy="${sy(end.value).toFixed(1)}" r="3.5" fill="${color}" stroke="${t.surface}" stroke-width="2"/>`);
    parts.push(
      `<text x="${lx}" y="${end.y.toFixed(1)}" dy="-0.1em" font-size="11" font-weight="600" fill="${t.ink}">${MODELS[end.run.id].label.replace("Dense ", "")}</text>`,
    );
    parts.push(
      `<text x="${lx}" y="${end.y.toFixed(1)}" dy="1.15em" font-size="11" fill="${t.secondary}" style="font-variant-numeric:tabular-nums">${fmt(end.value, 2)}</text>`,
    );
  }

  return {
    svg: parts.join(""),
    layout: { metric, px, py, pw, ph, maxEpoch, lo, hi },
  };
}

function legend(t, x, y, ids) {
  const items = [];
  let cx = x;
  for (const id of ids) {
    items.push(`<line x1="${cx}" x2="${cx + 18}" y1="${y}" y2="${y}" stroke="${t[id]}" stroke-width="2.5" stroke-linecap="round"/>`);
    items.push(`<text x="${cx + 24}" y="${y}" dy="0.32em" font-size="12" fill="${t.ink}">${MODELS[id].label}</text>`);
    cx += 24 + MODELS[id].label.length * 7 + 22;
  }
  cx += 8;
  items.push(`<line x1="${cx}" x2="${cx + 18}" y1="${y}" y2="${y}" stroke="${t.secondary}" stroke-width="2"/>`);
  items.push(`<text x="${cx + 24}" y="${y}" dy="0.32em" font-size="12" fill="${t.secondary}">validation</text>`);
  cx += 24 + 62 + 16;
  items.push(`<line x1="${cx}" x2="${cx + 18}" y1="${y}" y2="${y}" stroke="${t.secondary}" stroke-width="1.5" stroke-dasharray="4 3"/>`);
  items.push(`<text x="${cx + 24}" y="${y}" dy="0.32em" font-size="12" fill="${t.secondary}">train</text>`);
  return items.join("");
}

function frame(t, width, height, body, label) {
  return [
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" font-family='${FONT}' role="img" aria-label="${esc(label)}">`,
    `<rect x="0.5" y="0.5" width="${width - 1}" height="${height - 1}" rx="10" fill="${t.surface}" stroke="${t.border}"/>`,
    body,
    `</svg>`,
  ].join("");
}

/**
 * Loss figure with one panel per metric, laid out in a single row.
 * Returns { svg, panels } — panels carries scales for hover overlays.
 */
export function renderLossFigure({ runs, metrics = ["total"], show = { train: true, val: true }, theme = "light", width = 880, panelHeight = 320, title, subtitle }) {
  const t = THEMES[theme];
  const head = title ? 64 : 34;
  const height = head + panelHeight + 8;
  const pad = 16;
  const panelWidth = (width - pad * 2) / metrics.length;

  const parts = [];
  if (title) {
    parts.push(`<text x="${pad + 8}" y="28" font-size="16" font-weight="650" fill="${t.ink}">${esc(title)}</text>`);
    if (subtitle) parts.push(`<text x="${pad + 8}" y="47" font-size="12" fill="${t.secondary}">${esc(subtitle)}</text>`);
  }
  parts.push(legend(t, width - pad - 400, title ? 28 : 20, runs.map((r) => r.id)));

  const panels = metrics.map((metric, i) => {
    const panel = lossPanel({
      runs,
      metric,
      theme,
      x: pad + i * panelWidth,
      y: head - 8,
      width: panelWidth,
      height: panelHeight,
      // A single panel is already named by the figure title.
      title: metrics.length > 1 || !title ? METRICS[metric].label : null,
      show,
    });
    parts.push(panel.svg);
    return panel.layout;
  });

  return { svg: frame(t, width, height, parts.join(""), title ?? "Loss curves"), panels, width, height };
}

// ---------------------------------------------------------------------------
// Architecture diagrams
// ---------------------------------------------------------------------------

export const ARCHITECTURES = {
  vae: {
    title: "Dense VAE",
    params: "829,232 parameters",
    encoder: [
      { kind: "io", lines: ["Input x", "784 (28×28 flat)"] },
      { kind: "layer", lines: ["Linear 784→512", "ReLU"] },
    ],
    decoder: [
      { kind: "layer", lines: ["Linear 16→512", "ReLU"] },
      { kind: "layer", lines: ["Linear 512→784", "logits"] },
      { kind: "io", lines: ["Reconstruction", "sigmoid · 784"] },
    ],
    heads: ["Linear 512→16", "Linear 512→16"],
  },
  cnn: {
    title: "CNN-VAE",
    params: "220,385 parameters",
    encoder: [
      { kind: "io", lines: ["Input x", "1×28×28"] },
      { kind: "layer", lines: ["Conv 4×4, s2", "32×14×14 · ReLU"] },
      { kind: "layer", lines: ["Conv 4×4, s2", "64×7×7 · ReLU"] },
    ],
    decoder: [
      { kind: "layer", lines: ["Linear 16→3136", "ReLU · 64×7×7"] },
      { kind: "layer", lines: ["ConvT 4×4, s2", "32×14×14 · ReLU"] },
      { kind: "layer", lines: ["ConvT 4×4, s2", "1×28×28 logits"] },
      { kind: "io", lines: ["Reconstruction", "sigmoid · 1×28×28"] },
    ],
    heads: ["Linear 3136→16", "Linear 3136→16"],
  },
};

function arrow(t, x1, y1, x2, y2, id) {
  return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${t.muted}" stroke-width="1.5" marker-end="url(#${id})"/>`;
}

function nodeBox(t, accent, node, x, y, w, h) {
  const io = node.kind === "io";
  return [
    `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="${io ? t.surface : t.node}" stroke="${io ? t.axis : t.border}" stroke-width="1" ${io ? 'stroke-dasharray="4 3"' : ""}/>`,
    io ? "" : `<rect x="${x + 10}" y="${y}" width="${w - 20}" height="3" rx="1.5" fill="${accent}"/>`,
    `<text x="${x + w / 2}" y="${y + h / 2 - 3}" font-size="13" font-weight="600" text-anchor="middle" fill="${t.ink}">${esc(node.lines[0])}</text>`,
    `<text x="${x + w / 2}" y="${y + h / 2 + 15}" font-size="11" text-anchor="middle" fill="${t.secondary}">${esc(node.lines[1])}</text>`,
  ].join("");
}

/**
 * U-shaped block diagram: encoder left→right on top, latent sample on the
 * right, decoder right→left underneath.
 */
export function renderArchitecture({ model, theme = "light", latent = 16 }) {
  const t = THEMES[theme];
  const arch = ARCHITECTURES[model];
  const accent = t[model];
  const mk = `arrow-${model}`;

  // Fixed width so every architecture renders at the same scale; boxes stretch.
  const width = 958, h = 64, gap = 34, pad = 24;
  const encCols = arch.encoder.length + 1; // + heads
  const cols = Math.max(encCols, arch.decoder.length) + 1; // + latent column
  const w = (width - pad * 2 - (cols - 1) * gap) / cols;
  const topY = 74, botY = 74 + h + 84;
  const height = botY + h + 48;
  const colX = (c) => pad + c * (w + gap);

  const parts = [
    `<defs><marker id="arrow-${model}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${t.muted}"/></marker></defs>`,
    `<circle cx="${pad + 5}" cy="25" r="5" fill="${accent}"/>`,
    `<text x="${pad + 18}" y="30" font-size="16" font-weight="650" fill="${t.ink}">${esc(arch.title)}</text>`,
    `<text x="${width - pad}" y="30" font-size="12" text-anchor="end" fill="${t.secondary}">${esc(arch.params)} · latent d = ${latent}</text>`,
    `<text x="${pad}" y="${topY - 12}" font-size="11" letter-spacing="0.06em" fill="${t.muted}">ENCODER  q(z | x)</text>`,
    `<text x="${pad}" y="${botY + h + 26}" font-size="11" letter-spacing="0.06em" fill="${t.muted}">DECODER  p(x | z)</text>`,
  ];

  // Encoder row.
  arch.encoder.forEach((node, i) => {
    parts.push(nodeBox(t, accent, node, colX(i), topY, w, h));
    parts.push(arrow(t, colX(i) + w, topY + h / 2, colX(i + 1) - 3, topY + h / 2, mk));
  });

  // Gaussian heads: two stacked boxes μ and log σ².
  const hx = colX(arch.encoder.length);
  const hh = (h - 6) / 2;
  [["μ", arch.heads[0]], ["log σ²", arch.heads[1]]].forEach(([name, sub], i) => {
    const yy = topY + i * (hh + 6);
    parts.push(`<rect x="${hx}" y="${yy}" width="${w}" height="${hh}" rx="6" fill="${t.node}" stroke="${accent}" stroke-width="1.25"/>`);
    parts.push(`<text x="${hx + 12}" y="${yy + hh / 2}" dy="0.35em" font-size="13" font-weight="600" fill="${t.ink}">${name}</text>`);
    parts.push(`<text x="${hx + w - 10}" y="${yy + hh / 2}" dy="0.35em" font-size="10" text-anchor="end" fill="${t.secondary}">${esc(sub)}</text>`);
  });

  // Latent sample (reparameterisation trick).
  const zx = colX(cols - 1);
  const zy = (topY + botY) / 2;
  parts.push(`<rect x="${zx}" y="${zy}" width="${w}" height="${h}" rx="${h / 2}" fill="${accent}" fill-opacity="0.12" stroke="${accent}" stroke-width="1.5"/>`);
  parts.push(`<text x="${zx + w / 2}" y="${zy + h / 2 - 3}" font-size="13" font-weight="650" text-anchor="middle" fill="${t.ink}">z = μ + σ ⊙ ε</text>`);
  parts.push(`<text x="${zx + w / 2}" y="${zy + h / 2 + 15}" font-size="11" text-anchor="middle" fill="${t.secondary}">ε ~ N(0, I) · d = ${latent}</text>`);
  parts.push(arrow(t, hx + w, topY + h / 2, zx + w / 2 - 6, zy - 3, mk));

  // Decoder row, right to left, starting under the heads column.
  const decStart = cols - 2;
  arch.decoder.forEach((node, i) => {
    const c = decStart - i;
    parts.push(nodeBox(t, accent, node, colX(c), botY, w, h));
    if (i < arch.decoder.length - 1) parts.push(arrow(t, colX(c), botY + h / 2, colX(c - 1) + w + 3, botY + h / 2, mk));
  });
  parts.push(arrow(t, zx + w / 2 - 6, zy + h, colX(decStart) + w + 3, botY + h / 2, mk));

  return { svg: frame(t, width, height, parts.join(""), `${arch.title} architecture`), width, height };
}
