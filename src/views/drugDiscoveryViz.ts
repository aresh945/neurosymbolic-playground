/**
 * Live, schematic visualization of the drug-discovery pipeline: a noisy
 * sensor reading flows through the frozen perception network into three
 * rule atoms, which combine through the same nested-AND structure as
 * RULE_TREE in compoundSafety.ts, ending in a single verdict box.
 *
 * Unlike digitSumViz.ts there is no live training loop to feed — every
 * compound pick or "read again" click produces one fresh noisy reading, so
 * this file only ever does a full staged (re)draw, never a cheap update().
 *
 * Kept deliberately small and self-contained (plain D3 + SVG, no layout libs),
 * mirroring digitSumViz.ts's conventions (colors, sequence(), freshSvg()).
 */

import * as d3 from "d3";
import type { CompoundReading } from "../neural/compoundSafety";
import { TOX_CUTOFF, BIND_CUTOFF } from "../neural/compoundSafety";

const POS = "#0877bd"; // blue — satisfied / forward flow
const NEG = "#f59322"; // orange — violated
const INK = "#183d4e";
const MUTED = "#777f86";
const LINE = "#d9dde1";

const W = 920;
const H = 260;

type Svg = d3.Selection<SVGSVGElement, unknown, null, undefined>;
type G = d3.Selection<SVGGElement, unknown, null, undefined>;

export interface VizHandle {
  /** Cancel any pending stage timers (call before discarding/rebuilding). */
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
    .style("max-height", "270px") as Svg;

  const defs = svg.append("defs");
  for (const [id, color] of [
    ["arrow", MUTED],
    ["arrowPos", POS],
    ["arrowNeg", NEG],
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

function label(
  g: G,
  x: number,
  y: number,
  text: string,
  color = MUTED,
  size = 11,
  anchor: "start" | "middle" | "end" = "start"
): void {
  g.append("text")
    .attr("x", x)
    .attr("y", y)
    .attr("fill", color)
    .attr("font-size", size)
    .attr("font-family", "system-ui, sans-serif")
    .attr("text-anchor", anchor)
    .text(text);
}

function arrow(g: G, x1: number, y1: number, x2: number, y2: number, satisfied?: boolean): void {
  const color = satisfied === undefined ? MUTED : satisfied ? POS : NEG;
  const marker = satisfied === undefined ? "arrow" : satisfied ? "arrowPos" : "arrowNeg";
  g.append("line")
    .attr("x1", x1)
    .attr("y1", y1)
    .attr("x2", x2)
    .attr("y2", y2)
    .attr("stroke", color)
    .attr("stroke-width", 1.5)
    .attr("marker-end", `url(#${marker})`);
}

function box(g: G, x: number, y: number, w: number, h: number, fill: string, stroke: string, strokeWidth = 1.4): void {
  g.append("rect")
    .attr("x", x)
    .attr("y", y)
    .attr("width", w)
    .attr("height", h)
    .attr("rx", 8)
    .attr("fill", fill)
    .attr("stroke", stroke)
    .attr("stroke-width", strokeWidth);
}

/** A 0..max number line with a dashed cutoff tick and a colored dot marking
 * the perceived value — satisfied (below cutoff) is blue, violated orange. */
function gauge(g: G, x: number, y: number, w: number, max: number, cutoff: number, value: number, satisfied: boolean): void {
  g.append("line")
    .attr("x1", x)
    .attr("y1", y)
    .attr("x2", x + w)
    .attr("y2", y)
    .attr("stroke", LINE)
    .attr("stroke-width", 4)
    .attr("stroke-linecap", "round");
  const cx = x + (cutoff / max) * w;
  g.append("line")
    .attr("x1", cx)
    .attr("y1", y - 9)
    .attr("x2", cx)
    .attr("y2", y + 9)
    .attr("stroke", MUTED)
    .attr("stroke-width", 1.4)
    .attr("stroke-dasharray", "3 2");
  label(g, cx, y - 13, `cutoff ${cutoff}`, MUTED, 9, "middle");
  const vx = x + (Math.min(value, max) / max) * w;
  g.append("circle")
    .attr("cx", vx)
    .attr("cy", y)
    .attr("r", 6)
    .attr("fill", satisfied ? POS : NEG)
    .attr("stroke", "#fff")
    .attr("stroke-width", 1.5);
}

/** A zero-centered horizontal bar for one raw sensor channel — these are
 * noisy mixed values (not 0..10 real-unit readings), so unlike gauge() this
 * has no cutoff, just a magnitude either side of a zero line. */
function sensorBar(g: G, x: number, y: number, w: number, value: number): void {
  const max = 2; // generous fixed scale — actual channels typically fall in ~[-1, 1.5]
  const mid = x + w / 2;
  g.append("line")
    .attr("x1", x)
    .attr("y1", y)
    .attr("x2", x + w)
    .attr("y2", y)
    .attr("stroke", LINE)
    .attr("stroke-width", 3)
    .attr("stroke-linecap", "round");
  g.append("line")
    .attr("x1", mid)
    .attr("y1", y - 5)
    .attr("x2", mid)
    .attr("y2", y + 5)
    .attr("stroke", MUTED)
    .attr("stroke-width", 1);
  const half = (w / 2) * Math.max(-1, Math.min(1, value / max));
  g.append("rect")
    .attr("x", half >= 0 ? mid : mid + half)
    .attr("y", y - 3)
    .attr("width", Math.abs(half))
    .attr("height", 6)
    .attr("rx", 3)
    .attr("fill", MUTED);
}

/** A probability bar (0-100%), for the hydrogen-bond atom which has no
 * real-unit scale, just a presence probability. */
function probBar(g: G, x: number, y: number, w: number, value: number, satisfied: boolean): void {
  g.append("rect").attr("x", x).attr("y", y - 4).attr("width", w).attr("height", 8).attr("rx", 4).attr("fill", LINE);
  g.append("rect")
    .attr("x", x)
    .attr("y", y - 4)
    .attr("width", Math.max(2, value * w))
    .attr("height", 8)
    .attr("rx", 4)
    .attr("fill", satisfied ? POS : NEG);
}

/** A labeled box with a satisfaction bar + percentage — used for every rule
 * node (atoms and AND nodes alike), so the tree's shape reads visually. */
function ruleBox(g: G, x: number, y: number, w: number, h: number, labelText: string, value: number, satisfied: boolean, big = false): void {
  box(g, x, y, w, h, satisfied ? "#eef5fb" : "#fdf1e6", satisfied ? POS : NEG, big ? 1.8 : 1.4);
  label(g, x + 10, y + (big ? 22 : 18), labelText, INK, big ? 12 : 11);
  const barY = y + h - (big ? 22 : 16);
  const barW = w - 20;
  g.append("rect").attr("x", x + 10).attr("y", barY).attr("width", barW).attr("height", 7).attr("rx", 3.5).attr("fill", LINE);
  g.append("rect")
    .attr("x", x + 10)
    .attr("y", barY)
    .attr("width", Math.max(2, value * barW))
    .attr("height", 7)
    .attr("rx", 3.5)
    .attr("fill", satisfied ? POS : NEG);
  label(g, x + w - 10, y + (big ? 22 : 18), `${Math.round(value * 100)}%`, satisfied ? POS : NEG, big ? 13 : 11, "end");
}

/** Small local timer chain — drives stage order, independent of any other
 * timer in the app (mirrors digitSumViz.ts's sequence()). */
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

function fadeIn(sel: G, duration = 280): void {
  sel.transition().duration(duration).attr("opacity", 1);
}

/** Row y-centers shared by the perception and atom columns. */
const ROW_Y = [50, 130, 210];

export function drawDrugDiscoveryViz(container: HTMLElement, reading: CompoundReading): VizHandle {
  const svg = freshSvg(container);
  const g = svg.append("g") as unknown as G;

  const gSensor = g.append("g").attr("opacity", 0) as unknown as G;
  const gNet = g.append("g").attr("opacity", 0) as unknown as G;
  const gPerc = g.append("g").attr("opacity", 0) as unknown as G;
  const gAtoms = g.append("g").attr("opacity", 0) as unknown as G;
  const gMid = g.append("g").attr("opacity", 0) as unknown as G;
  const gTop = g.append("g").attr("opacity", 0) as unknown as G;

  const sensorX = 0;
  const sensorW = 110;
  const netX = 130;
  const netW = 130;
  const percX = 280;
  const percW = 140;
  const atomX = 440;
  const atomW = 140;
  const midX = 600;
  const midW = 150;
  const topX = 770;
  const topW = 140;

  // RULE_TREE shape is fixed: AND(atom(low toxicity), AND(atom(strong
  // binding), atom(hydrogen bond present))) — read the trace generically by
  // walking to the leaves rather than assuming array order, so this stays
  // correct even if RULE_TREE's structure changes.
  const top = reading.trace; // "recommend"
  const [atomLow, mid] = top.children; // atom(low toxicity), AND("interaction likely")
  const [atomStrong, atomHbond] = mid.children;

  const lowSat = atomLow.value > 0.5;
  const strongSat = atomStrong.value > 0.5;
  const hbondSat = atomHbond.value > 0.5;
  const midSat = mid.value > 0.5;
  const topSat = top.value > 0.5;

  // ---- raw sensor: the noisy 6-dim signal the network actually reads ----
  label(gSensor, sensorX, 8, "noisy sensor (6ch)", INK, 11);
  const sensorRowY = [30, 62, 94, 126, 158, 190];
  reading.sensor.forEach((v, i) => {
    sensorBar(gSensor, sensorX, sensorRowY[i], sensorW - 34, v);
    label(gSensor, sensorX + sensorW - 30, sensorRowY[i] + 3, v.toFixed(2), MUTED, 9);
  });
  label(gSensor, sensorX, 214, "true values are never shown to it —", MUTED, 9);
  label(gSensor, sensorX, 226, "only this mixed, noisy signal.", MUTED, 9);

  // ---- perception network: the actual neural computation -----------------
  box(gNet, netX, 10, netW, 240, "#eef5fb", POS, 1.6);
  label(gNet, netX + 12, 32, "perception", POS, 12);
  label(gNet, netX + 12, 48, "network", POS, 12);
  label(gNet, netX + 12, 74, "(3×6 weights,", MUTED, 9);
  label(gNet, netX + 12, 86, "frozen — trained", MUTED, 9);
  label(gNet, netX + 12, 98, "offline once)", MUTED, 9);
  arrow(gSensor, sensorX + sensorW + 6, 110, netX - 6, 110);
  arrow(gNet, netX + netW + 6, 110, percX - 6, 110);

  // ---- perception readout: the network's 3 outputs -----------------------
  label(gPerc, percX, ROW_Y[0] - 28, `toxicity — read ${reading.toxHat.toFixed(1)}`, INK, 11);
  gauge(gPerc, percX, ROW_Y[0], percW, 10, TOX_CUTOFF, reading.toxHat, lowSat);
  label(gPerc, percX, ROW_Y[1] - 28, `binding energy — read ${reading.bindHat.toFixed(1)}`, INK, 11);
  gauge(gPerc, percX, ROW_Y[1], percW, 10, BIND_CUTOFF, reading.bindHat, strongSat);
  label(gPerc, percX, ROW_Y[2] - 28, `hydrogen bond — P(present)`, INK, 11);
  probBar(gPerc, percX, ROW_Y[2], percW, reading.hbondHat, hbondSat);

  // ---- atoms ------------------------------------------------------------
  ROW_Y.forEach((y) => arrow(gPerc, percX + percW + 8, y, atomX - 8, y));
  ruleBox(gAtoms, atomX, ROW_Y[0] - 30, atomW, 60, "low toxicity", atomLow.value, lowSat);
  ruleBox(gAtoms, atomX, ROW_Y[1] - 30, atomW, 60, "strong binding", atomStrong.value, strongSat);
  ruleBox(gAtoms, atomX, ROW_Y[2] - 30, atomW, 60, "hydrogen bond present", atomHbond.value, hbondSat);

  // ---- mid AND: "interaction likely" ------------------------------------
  const midY = (ROW_Y[1] + ROW_Y[2]) / 2;
  arrow(gAtoms, atomX + atomW + 8, ROW_Y[1], midX - 8, midY - 8, strongSat);
  arrow(gAtoms, atomX + atomW + 8, ROW_Y[2], midX - 8, midY + 8, hbondSat);
  ruleBox(gMid, midX, midY - 32, midW, 64, "interaction likely", mid.value, midSat);

  // ---- top AND: "recommend" (the verdict) --------------------------------
  const topY = (ROW_Y[0] + midY) / 2;
  arrow(gMid, atomX + atomW + 8, ROW_Y[0], topX - 8, topY - 10, lowSat);
  arrow(gMid, midX + midW + 8, midY, topX - 8, topY + 10, midSat);
  ruleBox(gTop, topX, topY - 38, topW, 76, "recommend", top.value, topSat, true);
  label(
    gTop,
    topX + topW / 2,
    topY + 76 - 38 + 16,
    topSat ? "✓ pursue" : "✗ do not pursue",
    topSat ? POS : NEG,
    12,
    "middle"
  );

  const seq = sequence(
    [
      () => fadeIn(gSensor),
      () => fadeIn(gNet),
      () => fadeIn(gPerc),
      () => fadeIn(gAtoms),
      () => fadeIn(gMid),
      () => fadeIn(gTop),
    ],
    480
  );

  return {
    destroy() {
      seq.destroy();
    },
  };
}
