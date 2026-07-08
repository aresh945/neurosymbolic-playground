/**
 * Live, schematic visualizations of the three neurosymbolic categories for the
 * digit-sum demo. Each draws the SAME pair (user-picked or held-out) so the
 * user can watch one example flow through the chosen architecture as it
 * trains. On first draw (or whenever the pair/pattern changes) the diagram
 * reveals itself in STAGES that match the pattern's true data-flow order —
 * see drawSerial/drawGuided/drawTight below. Every subsequent training tick
 * only calls the returned handle's cheap `update()`, which pushes new numbers
 * into the already-built DOM without replaying the staged intro.
 *
 * Kept deliberately small and self-contained (plain D3 + SVG, no layout libs).
 */

import * as d3 from "d3";
import type { DigitSumModel } from "../neural/digitSum";

const POS = "#0877bd"; // blue  — the neurosymbolic approach
const NEG = "#f59322"; // orange
const PUR = "#8e44ad"; // purple — perception / symbolic accent
const INK = "#183d4e";
const MUTED = "#777f86";
const LINE = "#d9dde1";

const W = 820;
const H = 280;

type Svg = d3.Selection<SVGSVGElement, unknown, null, undefined>;
type G = d3.Selection<SVGGElement, unknown, null, undefined>;

export interface VizHandle {
  /** Push new numbers into the already-built DOM — no rebuild, no replay. */
  update(model: DigitSumModel, pair: [number, number]): void;
  /** Cancel any pending stage/pulse timers (call before discarding/rebuilding). */
  destroy(): void;
}

function freshSvg(container: HTMLElement): Svg {
  d3.select(container).selectAll("*").remove();
  const svg = d3
    .select(container)
    .append("svg")
    .attr("viewBox", `0 0 ${W} ${H}`)
    .attr("preserveAspectRatio", "xMidYMid meet")
    .style("width", "100%")
    .style("height", "auto")
    .style("max-height", "290px") as Svg;

  const defs = svg.append("defs");
  for (const [id, color] of [
    ["arrow", MUTED],
    ["arrowPos", POS],
  ] as const) {
    defs
      .append("marker")
      .attr("id", id)
      .attr("viewBox", "0 0 10 10")
      .attr("refX", 9)
      .attr("refY", 5)
      .attr("markerWidth", 7)
      .attr("markerHeight", 7)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,0 L10,5 L0,10 z")
      .attr("fill", color);
  }
  return svg;
}

function label(g: G, x: number, y: number, text: string, color = MUTED, size = 11): SVGTextElement {
  return g
    .append("text")
    .attr("x", x)
    .attr("y", y)
    .attr("fill", color)
    .attr("font-size", size)
    .attr("font-family", "system-ui, sans-serif")
    .text(text)
    .node()!;
}

function arrow(g: G, x1: number, y1: number, x2: number, y2: number, pos = false): void {
  g.append("line")
    .attr("x1", x1)
    .attr("y1", y1)
    .attr("x2", x2)
    .attr("y2", y2)
    .attr("stroke", pos ? POS : MUTED)
    .attr("stroke-width", 1.5)
    .attr("marker-end", `url(#${pos ? "arrowPos" : "arrow"})`);
}

function box(
  g: G,
  x: number,
  y: number,
  w: number,
  h: number,
  fill: string,
  stroke: string,
  dashed = false
): void {
  const r = g
    .append("rect")
    .attr("x", x)
    .attr("y", y)
    .attr("width", w)
    .attr("height", h)
    .attr("rx", 8)
    .attr("fill", fill)
    .attr("stroke", stroke)
    .attr("stroke-width", 1.4);
  if (dashed) r.attr("stroke-dasharray", "4 3");
}

/**
 * Build a fixed-size row of `n` bars once; returns an updater that adjusts
 * height/fill in place so live training numbers refresh with no DOM churn.
 * `labelStep` thins tick labels for wide distributions (e.g. the 19-wide
 * P(sum) chart) so index numbers don't collide.
 */
function barGroup(
  g: G,
  x: number,
  y: number,
  w: number,
  h: number,
  color: string,
  n: number,
  labelStep = 1
): (dist: number[], highlight: number) => void {
  const bw = w / n;
  const rects: SVGRectElement[] = [];
  for (let i = 0; i < n; i++) {
    const r = g
      .append("rect")
      .attr("x", x + i * bw + 1)
      .attr("y", y + h)
      .attr("width", Math.max(1, bw - 2))
      .attr("height", 0)
      .attr("rx", 1.5)
      .attr("fill", LINE)
      .node();
    if (r) rects.push(r);
    if (i % labelStep === 0) label(g, x + i * bw + bw / 2 - 3, y + h + 11, String(i), MUTED, 9);
  }
  return (dist, highlight) => {
    const max = Math.max(...dist, 0.001);
    for (let i = 0; i < n; i++) {
      const bh = (dist[i] / max) * h;
      d3.select(rects[i])
        .attr("y", y + h - bh)
        .attr("height", bh)
        .attr("fill", i === highlight ? color : LINE);
    }
  };
}

/** Small local timer chain — drives stage order, independent of any other
 * timer in the app (never shares state with digitSumResults.ts's training loop). */
function sequence(steps: Array<() => void>, gapMs: number): { destroy: () => void } {
  let i = 0;
  let t: ReturnType<typeof setTimeout> | null = null;
  const next = () => {
    if (i >= steps.length) return;
    steps[i++]();
    t = setTimeout(next, gapMs);
  };
  next();
  return {
    destroy: () => {
      if (t !== null) clearTimeout(t);
      t = null;
    },
  };
}

function fadeIn(sel: G, duration = 280, target = 1): void {
  sel.transition().duration(duration).attr("opacity", target);
}

/** Flash the true-sum diagonal's stroke width a couple of times — a visual
 * stand-in for "gradients flow back through this path into perception". */
function pulseDiagonal(cells: SVGRectElement[][], trueSum: number, times: number, duration: number): void {
  const diag: SVGRectElement[] = [];
  for (let i = 0; i < cells.length; i++) {
    const j = trueSum - i;
    if (j >= 0 && j < cells.length) diag.push(cells[i][j]);
  }
  const sel = d3.selectAll<SVGRectElement, unknown>(diag);
  let n = 0;
  const step = () => {
    if (n >= times * 2) return;
    sel.transition().duration(duration).attr("stroke-width", n % 2 === 0 ? 3.5 : 2);
    n++;
    setTimeout(step, duration);
  };
  step();
}

/** Category 1 — Neural → Symbolic: perceive each digit, then add by logic. */
function drawSerial(svg: Svg, model: DigitSumModel, pair: [number, number]): VizHandle {
  const g = svg.append("g") as unknown as G;

  const gPerc = g.append("g").attr("opacity", 0) as unknown as G;
  const gSymbolic = g.append("g").attr("opacity", 0) as unknown as G;
  const gVerdict = g.append("g").attr("opacity", 0) as unknown as G;

  const rowsY = [28, 128];
  const captions: SVGTextElement[] = [];
  const barUpdaters: Array<(dist: number[], hi: number) => void> = [];
  const readTexts: SVGTextElement[] = [];

  rowsY.forEach((y) => {
    captions.push(label(gPerc, 0, y - 6, "", INK, 11));
    barUpdaters.push(barGroup(gPerc, 0, y, 150, 44, POS, model.K));
    arrow(gPerc, 158, y + 22, 196, y + 22);
    box(gPerc, 206, y + 4, 40, 36, "#eef5fb", POS);
    readTexts.push(label(gPerc, 219, y + 28, "", POS, 18));
    arrow(gPerc, 250, y + 22, 286, y + 22);
  });

  box(gSymbolic, 300, 60, 150, 70, "#f6f1fb", PUR);
  label(gSymbolic, 318, 90, "symbolic add", PUR, 12);
  const sumText = label(gSymbolic, 318, 116, "", INK, 16);

  const verdictText = label(gVerdict, 470, 99, "", POS, 12);
  label(gVerdict, 470, 118, "cost: needs digit labels to train perception", MUTED, 10);

  function paint(m: DigitSumModel, pr: [number, number]): void {
    const preds: number[] = [];
    pr.forEach((d, idx) => {
      const p = m.readDigit(d);
      const read = p.indexOf(Math.max(...p));
      preds.push(read);
      captions[idx].textContent = `digit ${idx + 1} (true ${d}, noisy)`;
      barUpdaters[idx](p, read);
      readTexts[idx].textContent = String(read);
    });
    const pred = preds[0] + preds[1];
    const ok = pred === pr[0] + pr[1];
    sumText.textContent = `${preds[0]} + ${preds[1]} = ${pred}`;
    verdictText.textContent = ok ? "✓ exact & generalizes" : "✗ perception slip";
    verdictText.setAttribute("fill", ok ? POS : NEG);
  }

  paint(model, pair);

  const seq = sequence([() => fadeIn(gPerc), () => fadeIn(gSymbolic), () => fadeIn(gVerdict)], 600);

  return {
    update(m, pr) {
      paint(m, pr);
    },
    destroy() {
      seq.destroy();
    },
  };
}

/**
 * Category 2 — Symbolic → Neural. Staged honestly: the symbolic rule only
 * ever guides TRAINING (see trainCat2/forwardAdder in digitSum.ts) — at
 * inference the neural adder runs alone. So the live sequence is perception
 * → neural adder → P(sum); the dashed rule box appears last, dimmed, clearly
 * an annotation about training history rather than a step in the live trace.
 */
function drawGuided(svg: Svg, model: DigitSumModel, pair: [number, number]): VizHandle {
  const g = svg.append("g") as unknown as G;

  const gPerc = g.append("g").attr("opacity", 0) as unknown as G;
  const gInfer = g.append("g").attr("opacity", 0) as unknown as G;
  const gRule = g.append("g").attr("opacity", 0) as unknown as G;

  const capA = label(gPerc, 0, 16, "", INK, 11);
  const updateA = barGroup(gPerc, 0, 22, 130, 40, POS, model.K);
  const capB = label(gPerc, 0, 92, "", INK, 11);
  const updateB = barGroup(gPerc, 0, 98, 130, 40, POS, model.K);
  arrow(gPerc, 138, 80, 176, 80);

  box(gInfer, 188, 52, 150, 56, "#eef5fb", POS);
  label(gInfer, 204, 78, "neural adder", POS, 12);
  label(gInfer, 204, 96, "(MLP, runs at test)", MUTED, 10);
  arrow(gInfer, 344, 80, 382, 80);
  label(gInfer, 396, 30, "P(sum)", INK, 11);
  const updateSum = barGroup(gInfer, 396, 40, 280, 86, POS, model.SUMS, 2);
  const verdict = label(gInfer, 396, 150, "", POS, 12);

  box(gRule, 150, 168, 200, 46, "#f6f1fb", PUR, true);
  label(gRule, 166, 188, "symbolic add rule", PUR, 11);
  label(gRule, 166, 204, "guides training only — gone at test", MUTED, 9);
  gRule
    .append("line")
    .attr("x1", 250)
    .attr("y1", 168)
    .attr("x2", 250)
    .attr("y2", 140)
    .attr("stroke", PUR)
    .attr("stroke-width", 1.4)
    .attr("stroke-dasharray", "4 3")
    .attr("marker-end", "url(#arrow)");

  function paint(m: DigitSumModel, [a, b]: [number, number]): void {
    const p1 = m.readDigit(a);
    const p2 = m.readDigit(b);
    const ps = m.approachSumDist(a, b);
    const pred = ps.indexOf(Math.max(...ps));
    const trueSum = a + b;
    capA.textContent = `reads a (a = ${a})`;
    capB.textContent = `reads b (b = ${b})`;
    updateA(p1, -1);
    updateB(p2, -1);
    updateSum(ps, trueSum);
    const ok = pred === trueSum;
    verdict.textContent = ok ? `prediction ${pred} ✓` : `predicts ${pred}, truth ${trueSum} ✗`;
    verdict.setAttribute("fill", ok ? POS : NEG);
  }

  paint(model, pair);

  const seq = sequence(
    [() => fadeIn(gPerc), () => fadeIn(gInfer), () => fadeIn(gRule, 280, 0.75)],
    600
  );

  return {
    update(m, pr) {
      paint(m, pr);
    },
    destroy() {
      seq.destroy();
    },
  };
}

/** Category 3 — Neural ↔ Symbolic: perception → product grid → P(sum), with
 * a closing pulse selling the tightly-coupled, gradients-flow-back trait. */
function drawTight(svg: Svg, model: DigitSumModel, pair: [number, number]): VizHandle {
  const g = svg.append("g") as unknown as G;
  const K = model.K;
  const cell = 17;
  const gx = 196;
  const gy = 40;
  const sx = gx + K * cell + 60;

  const gPerc = g.append("g").attr("opacity", 0) as unknown as G;
  const gGrid = g.append("g").attr("opacity", 0) as unknown as G;
  const gSum = g.append("g").attr("opacity", 0) as unknown as G;
  const gPulse = g.append("g").attr("opacity", 0) as unknown as G;

  const capA = label(gPerc, 0, 16, "", INK, 11);
  const updateA = barGroup(gPerc, 0, 22, 150, 44, POS, K);
  const capB = label(gPerc, 0, 96, "", INK, 11);
  const updateB = barGroup(gPerc, 0, 102, 150, 44, POS, K);
  arrow(gPerc, 158, 90, gx - 8, 90);

  label(gGrid, gx, 30, "P(a=i) · P(b=j)", INK, 11);
  const cells: SVGRectElement[][] = [];
  for (let i = 0; i < K; i++) {
    const row: SVGRectElement[] = [];
    for (let j = 0; j < K; j++) {
      const c = gGrid
        .append("rect")
        .attr("x", gx + j * cell)
        .attr("y", gy + i * cell)
        .attr("width", cell - 2)
        .attr("height", cell - 2)
        .attr("rx", 2)
        .attr("fill", POS)
        .node();
      if (c) row.push(c);
    }
    cells.push(row);
  }
  label(gGrid, gx, gy + K * cell + 16, "highlighted diagonal: a + b = sum", PUR, 10);
  arrow(gGrid, gx + K * cell + 8, 90, sx - 14, 90, true);

  label(gSum, sx, 30, "P(sum)", INK, 11);
  const updateSum = barGroup(gSum, sx, 40, Math.max(120, W - sx - 20), 90, POS, model.SUMS, 2);
  label(gSum, sx, 175, "gradients flow back ← through the grid into perception", MUTED, 10);

  label(gPulse, gx, gy + K * cell + 32, "▲ signal flowing back through the diagonal", PUR, 10);

  function paint(m: DigitSumModel, [a, b]: [number, number]): number {
    const p1 = m.readDigit(a);
    const p2 = m.readDigit(b);
    const grid = m.outerGrid(p1, p2);
    const ps = m.approachSumDist(a, b);
    const trueSum = a + b;
    capA.textContent = `reads digit a (a = ${a})`;
    capB.textContent = `reads digit b (b = ${b})`;
    updateA(p1, -1);
    updateB(p2, -1);
    for (let i = 0; i < K; i++)
      for (let j = 0; j < K; j++) {
        d3.select(cells[i][j])
          .attr("fill-opacity", Math.min(1, grid[i][j] * 1.6))
          .attr("stroke", i + j === trueSum ? PUR : LINE)
          .attr("stroke-width", i + j === trueSum ? 2 : 0.5);
      }
    updateSum(ps, trueSum);
    return trueSum;
  }

  let lastTrueSum = paint(model, pair);

  const seq = sequence(
    [
      () => fadeIn(gPerc),
      () => fadeIn(gGrid),
      () => fadeIn(gSum),
      () => {
        fadeIn(gPulse, 280, 0.9);
        pulseDiagonal(cells, lastTrueSum, 2, 420);
      },
    ],
    600
  );

  return {
    update(m, pr) {
      lastTrueSum = paint(m, pr);
    },
    destroy() {
      seq.destroy();
    },
  };
}

/** Draw the live data-flow for the model's category onto `container`. */
export function drawDigitSumViz(
  container: HTMLElement,
  model: DigitSumModel,
  pair: [number, number]
): VizHandle {
  const svg = freshSvg(container);
  switch (model.category) {
    case "learning-for-reasoning":
      return drawSerial(svg, model, pair);
    case "reasoning-for-learning":
      return drawGuided(svg, model, pair);
    default:
      return drawTight(svg, model, pair);
  }
}
