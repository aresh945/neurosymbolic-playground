/**
 * Drug Discovery Screening — a live demonstration of the Logic Tensor
 * Network mechanism from Hossain & Chen's healthcare survey (§III.C /
 * Table 7): a neural network extracts features from a noisy signal, and a
 * differentiable rule engine (logicRules.ts, verified against the real
 * LTNtorch library) reasons over them to produce an EXPLAINABLE verdict —
 * not just "Active = 96%", but "Active because: low toxicity (satisfied),
 * strong binding (satisfied), hydrogen bond present (satisfied)".
 *
 * Unlike digitSum.ts, this demo's perception network is PRETRAINED OFFLINE
 * (via real gradient descent, in a throwaway script — see project memory
 * for the exact training run) rather than trained live in the browser. The
 * network itself is completely real: genuine matrix multiplication through
 * genuinely-learned weights. Only WHEN the learning happened differs — the
 * same pattern real production ML systems use (train once, freeze, deploy),
 * and it sidesteps live-training convergence risk for a demo whose whole
 * point is showing the reasoning trace clearly, not a training curve.
 *
 * No baseline comparison and no stacking-pattern taxonomy here (that's
 * digitSum.ts's job) — this is deliberately just ONE pipeline, matching the
 * paper's own diagram: drug -> neural feature extraction -> rule engine
 * checks each rule -> explainable prediction.
 *
 * Synthetic, illustrative data — not real chemistry, no real datasets.
 */

import { rule, evaluate, smoothLessThan } from "./logicRules";
import type { RuleNode, RuleTrace } from "./logicRules";

// ---- frozen, pretrained-offline perception weights -----------------------
// Trained via real gradient descent (MSE regression against known ground
// truth toxicity/binding/hydrogen-bond values, noisy 6-dim sensor input,
// 8000 epochs) in a throwaway script, then frozen here. Verified accuracy on
// 500 held-out random samples: mean |toxicity error| = 0.61 and mean
// |binding error| = 0.68 (both on a 0..10 scale), hydrogen-bond accuracy
// 100%. No training happens at runtime — these are the final weights.

const SENSOR_DIM = 6;
const TOX_SCALE = 10;
const BIND_SCALE = 10;
/** Real-unit thresholds (0..10 scale) — used for display and the rule. */
export const TOX_CUTOFF = 5;
export const BIND_CUTOFF = 4;
const TOX_CUTOFF_NORM = TOX_CUTOFF / TOX_SCALE;
const BIND_CUTOFF_NORM = BIND_CUTOFF / BIND_SCALE;
/** Sigmoid steepness for the smooth threshold atoms, in normalized space —
 * see logicRules.ts's doc comment for why ~2 suits a 0..10-ish domain. */
const K_STEEPNESS_NORM = 2 * TOX_SCALE;
/** Noise std matching what the perception net was actually trained under —
 * keeps "live" reads in the same regime the frozen weights were tuned for. */
const NOISE_STD = 0.12;

const SENSOR_MIX: number[][] = [
  [0.907692595018662, -0.12161519455346573, -0.11556871939526808],
  [1.106520535722342, -0.056565086958376115, 0.13943909382324138],
  [0.1403962906113312, 0.9006578003952648, 0.25692978680653056],
  [0.06421120520056088, 1.0085217675024647, -0.4235912900851946],
  [-0.15638288339056947, 0.055110133816771247, 0.9479745095598955],
  [0.1199492764224761, 0.22854658708793926, 1.1532546744215597],
];
const Wp: number[][] = [
  [2.027, 2.4222, 0.2554, 0.0805, -0.3389, 0.2536],
  [-0.2457, -0.1898, 2.1624, 2.5065, 0.0072, 0.3937],
  [-0.9374, 0.1576, 0.8515, -3.1239, 4.8884, 5.5234],
];
const bp: number[] = [-2.347, -2.3444, -5.3763];

function randn(): number {
  let u = 0;
  let v = 0;
  while (!u) u = Math.random();
  while (!v) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function sigmoid(u: number): number {
  return u >= 0 ? 1 / (1 + Math.exp(-u)) : Math.exp(u) / (1 + Math.exp(u));
}

export interface DrugCompound {
  id: string;
  name: string;
  toxicity: number;
  binding: number;
  hydrogenBond: boolean;
  note: string;
}

/** Five hand-picked synthetic compounds spanning clear-good, clear-bad,
 * mixed, and a genuinely borderline case (right at the toxicity threshold,
 * so its perceived reading shows visibly graded — not just true/false —
 * rule satisfaction). Illustrative only, not real chemistry. */
export const COMPOUNDS: DrugCompound[] = [
  { id: "A", name: "Compound A", toxicity: 2.0, binding: 2.5, hydrogenBond: true, note: "good candidate" },
  { id: "B", name: "Compound B", toxicity: 7.5, binding: 2.0, hydrogenBond: true, note: "toxic despite strong binding" },
  { id: "C", name: "Compound C", toxicity: 1.5, binding: 8.0, hydrogenBond: false, note: "safe but weak binder" },
  { id: "D", name: "Compound D", toxicity: 4.7, binding: 3.8, hydrogenBond: true, note: "borderline case" },
  { id: "E", name: "Compound E", toxicity: 8.5, binding: 7.0, hydrogenBond: false, note: "fails on every count" },
];

/** Fixed, well-conditioned random mix of [toxicity, binding, hydrogenBond]
 * plus noise — the network never sees the true values directly, only this. */
function noisySensor(c: DrugCompound): number[] {
  const target = [c.toxicity / TOX_SCALE, c.binding / BIND_SCALE, c.hydrogenBond ? 1 : 0];
  const clean = new Array<number>(SENSOR_DIM);
  for (let o = 0; o < SENSOR_DIM; o++) {
    let s = 0;
    for (let i = 0; i < 3; i++) s += SENSOR_MIX[o][i] * target[i];
    clean[o] = s;
  }
  return clean.map((x) => x + randn() * NOISE_STD);
}

/** The frozen perception network's forward pass — genuine matrix multiply
 * through genuinely pretrained weights, bounded to (0,1) per output. */
function perceive(sensor: number[]): [number, number, number] {
  const out: [number, number, number] = [0, 0, 0];
  for (let o = 0; o < 3; o++) {
    let s = bp[o];
    for (let i = 0; i < SENSOR_DIM; i++) s += Wp[o][i] * sensor[i];
    out[o] = sigmoid(s);
  }
  return out;
}

/** The one rule this demo reasons over — a genuine 3-leaf, 2-level tree,
 * matching the paper's two named conditions ("low toxicity" and "binding
 * energy < threshold AND hydrogen bond exists"). */
export const RULE_TREE: RuleNode = rule.and(
  [
    rule.atom(0, "low toxicity"),
    rule.and([rule.atom(1, "strong binding"), rule.atom(2, "hydrogen bond present")], "interaction likely"),
  ],
  "recommend"
);

export interface CompoundReading {
  /** The raw noisy 6-dim signal the network actually reads — exposed so the
   * viz can show what perception started from, not just what it produced. */
  sensor: number[];
  /** Real-unit (0..10) perceived toxicity/binding, rescaled for display. */
  toxHat: number;
  bindHat: number;
  /** Perceived probability a hydrogen bond is present. */
  hbondHat: number;
  /** The rule tree's overall verdict for this reading. */
  pSafe: number;
  trace: RuleTrace;
}

/** Read one compound through the frozen network and the rule engine — a
 * fresh noisy sample each call (mirrors digitSum.ts's readDigit/approachSumDist
 * pattern: noise means repeat reads legitimately vary a little). */
export function readCompound(c: DrugCompound): CompoundReading {
  const sensor = noisySensor(c);
  const [toxHatN, bindHatN, hbondHat] = perceive(sensor);
  const lowTox = smoothLessThan(toxHatN, TOX_CUTOFF_NORM, K_STEEPNESS_NORM);
  const strongBind = smoothLessThan(bindHatN, BIND_CUTOFF_NORM, K_STEEPNESS_NORM);
  const leaves = [lowTox, strongBind, hbondHat];
  const trace = evaluate(RULE_TREE, leaves);
  return {
    sensor,
    toxHat: toxHatN * TOX_SCALE,
    bindHat: bindHatN * BIND_SCALE,
    hbondHat,
    pSafe: trace.value,
    trace,
  };
}
