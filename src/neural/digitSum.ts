/**
 * Self-contained, dependency-free digit-sum engine — the real neurosymbolic
 * demonstration behind the "Handwritten Digit Sum" situation.
 *
 * Two heads train on the SAME weakly-supervised data. Every example is a pair
 * of digits in 0..K-1, and the ONLY label is their sum — never the identity of
 * either digit. Each digit is a real reference image (see digitImages.ts),
 * read as noisy pixels every time it's sampled — the model never sees a clean
 * template, so it has to learn to denoise toward one.
 *
 *   - neural (pure MLP): reads both noisy digit images at once and maps them
 *     straight to a sum. With only the sum as a signal it memorizes the pairs
 *     it sees and fails on pairs held out of training.
 *
 *   - neurosymbolic: a small perception layer reads EACH digit into P(digit),
 *     then a fixed, differentiable "add" rule combines them:
 *         P(sum = s) = Σ_{i + j = s} P(d1 = i) · P(d2 = j)
 *     Cross-entropy on the sum flows back THROUGH the add rule into perception,
 *     so the network learns to read digits from a signal that never named them.
 *     Because addition is built in, it generalizes to pairs it never trained on.
 */

/** Hidden-layer activation the baseline MLP and the Cat-2 adder can use. */
export type ActivationKind = "tanh" | "relu" | "sigmoid" | "linear";

export interface DigitSumConfig {
  /** Learning rate for both heads. */
  learningRate: number;
  /** Std-dev of Gaussian noise added to each digit's pixel vector on read. */
  noise: number;
  /** Fraction of the K×K digit pairs used for training; the rest are held out. */
  trainRatio: number;
  /** Hidden units in the baseline MLP and the Category-2 neural adder. */
  hidden: number;
  /** Activation for those hidden layers (perception always uses softmax). */
  activation: ActivationKind;
}

/**
 * The three ways neural and symbolic parts can be merged, from Yu et al.
 * (these match the app's StackingPattern ids one-for-one):
 *   - learning-for-reasoning  : Neural -> Symbolic (serial; symbolic reasons)
 *   - reasoning-for-learning  : Symbolic -> Neural (symbolic guides training)
 *   - learning-reasoning      : Neural <-> Symbolic (tight, end-to-end loop)
 */
export type NeuroCategory =
  | "learning-for-reasoning"
  | "reasoning-for-learning"
  | "learning-reasoning";

export interface HeadMetrics {
  /** Mean cross-entropy on the training pairs at the last step. */
  loss: number;
  /** Accuracy on the pairs the head trained on. */
  train: number;
  /** Accuracy on the held-out pairs (the real test of generalization). */
  test: number;
  /** Per-digit read accuracy — non-null only for the neurosymbolic head. */
  digit: number | null;
}

type Pair = [number, number];

const DEFAULTS: DigitSumConfig = {
  learningRate: 0.05,
  noise: 0.1,
  trainRatio: 0.6,
  hidden: 8,
  activation: "tanh",
};

/** Weight of the symbolic semantic-loss that guides Category-2 training. */
const SEMANTIC_REG = 1;

/** Activation function paired with its derivative (in terms of the output y). */
function activationFns(kind: ActivationKind): {
  f: (x: number) => number;
  d: (y: number) => number;
} {
  switch (kind) {
    case "relu":
      return { f: (x) => (x > 0 ? x : 0), d: (y) => (y > 0 ? 1 : 0) };
    case "sigmoid":
      return { f: (x) => 1 / (1 + Math.exp(-x)), d: (y) => y * (1 - y) };
    case "linear":
      return { f: (x) => x, d: () => 1 };
    case "tanh":
    default:
      return { f: (x) => Math.tanh(x), d: (y) => 1 - y * y };
  }
}

/** Standard normal sample (Box–Muller). */
function randn(): number {
  let u = 0;
  let v = 0;
  while (!u) u = Math.random();
  while (!v) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function vec(n: number, fill: number): number[] {
  return Array.from({ length: n }, () => fill);
}

function mat(rows: number, cols: number, fill: () => number): number[][] {
  return Array.from({ length: rows }, () => Array.from({ length: cols }, fill));
}

function softmax(z: number[]): number[] {
  const m = Math.max(...z);
  const e = z.map((x) => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
}

function argmax(a: number[]): number {
  let best = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
  return best;
}

function shuffle<T>(a: T[]): T[] {
  const out = a.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const j = (Math.random() * (i + 1)) | 0;
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

export class DigitSumModel {
  /** Number of distinct digits (0..K-1) — one per reference image. */
  readonly K = 10;
  /** Number of possible sums (0..2K-2). */
  readonly SUMS = 2 * this.K - 1;
  /** Hidden units in the pure-neural head (set from config). */
  H = 8;

  config: DigitSumConfig;
  /** Which of the three integration patterns this model demonstrates. */
  readonly category: NeuroCategory;
  /** Hidden units in the Category-2 neural adder (set from config). */
  HA = 8;
  /** Activation (+derivative) for the baseline and adder hidden layers. */
  private act = activationFns("tanh");
  /** Pixel-length of each reference digit image, inferred from the data. */
  readonly PIXEL_DIM: number;
  /** Clean, zero-mean-normalized reference pixels, one vector per digit. */
  private digitPixels: number[][];

  // perception layer, shared by all three categories: K×K linear -> softmax
  private Wp: number[][] = [];
  private bp: number[] = [];
  // pure-neural baseline: 2K -> H (tanh) -> SUMS (softmax)
  private W1: number[][] = [];
  private b1: number[] = [];
  private W2: number[][] = [];
  private b2: number[] = [];
  // Category-2 neural adder over [p1, p2]: 2K -> HA (tanh) -> SUMS (softmax)
  private Wa1: number[][] = [];
  private ba1: number[] = [];
  private Wa2: number[][] = [];
  private ba2: number[] = [];

  private split: { train: Pair[]; test: Pair[] } = { train: [], test: [] };
  private _epoch = 0;
  private lossBaseline = 0;
  private lossApproach = 0;

  constructor(
    category: NeuroCategory,
    digitPixels: number[][],
    config: Partial<DigitSumConfig> = {}
  ) {
    this.category = category;
    this.digitPixels = digitPixels;
    this.PIXEL_DIM = digitPixels[0]?.length ?? 0;
    this.config = { ...DEFAULTS, ...config };
    this.H = Math.max(1, Math.round(this.config.hidden));
    this.HA = this.H;
    this.act = activationFns(this.config.activation);
    this.init();
    this.split = this.buildSplit();
  }

  get epoch(): number {
    return this._epoch;
  }

  get splitSizes(): { train: number; test: number } {
    return { train: this.split.train.length, test: this.split.test.length };
  }

  /** Re-initialize all weights and rebuild the train/held-out split. */
  reset(): void {
    this.init();
    this._epoch = 0;
    this.lossBaseline = 0;
    this.lossApproach = 0;
    this.split = this.buildSplit();
  }

  /** Recompute the train/held-out split from the current trainRatio. */
  rebuildSplit(): void {
    this.split = this.buildSplit();
  }

  /** Train one or more epochs: always the baseline, plus the chosen approach. */
  step(epochs = 1): void {
    for (let e = 0; e < epochs; e++) {
      this.lossBaseline = this.trainNeural();
      switch (this.category) {
        case "learning-for-reasoning":
          this.lossApproach = this.trainPerceptionDigits();
          break;
        case "reasoning-for-learning":
          this.lossApproach = this.trainCat2();
          break;
        case "learning-reasoning":
          this.lossApproach = this.trainNeuro();
          break;
      }
    }
    this._epoch += epochs;
  }

  /** Metrics for the pure-neural baseline (the memorizer). */
  evalBaseline(): HeadMetrics {
    return {
      loss: this.lossBaseline,
      train: this.acc(this.split.train, (a, b) => this.predictNeural(a, b)),
      test: this.acc(this.split.test, (a, b) => this.predictNeural(a, b)),
      digit: null,
    };
  }

  /** Metrics for the selected neurosymbolic category. */
  evalApproach(): HeadMetrics {
    const predict = (a: number, b: number) => this.approachPred(a, b);
    return {
      loss: this.lossApproach,
      train: this.acc(this.split.train, predict),
      test: this.acc(this.split.test, predict),
      digit: this.digitAccuracy(),
    };
  }

  /** What supervision the selected category needs — the honest cost. */
  supervision(): { label: string; strong: boolean } {
    return this.category === "learning-for-reasoning"
      ? { label: "digit labels (strong)", strong: true }
      : { label: "sum only (weak)", strong: false };
  }

  // ---- setup -------------------------------------------------------------

  private init(): void {
    const { K, SUMS, H, HA, PIXEL_DIM } = this;
    this.Wp = mat(K, PIXEL_DIM, () => randn() * 0.1);
    this.bp = vec(K, 0);
    this.W1 = mat(H, 2 * PIXEL_DIM, () => randn() * 0.4);
    this.b1 = vec(H, 0);
    this.W2 = mat(SUMS, H, () => randn() * 0.4);
    this.b2 = vec(SUMS, 0);
    this.Wa1 = mat(HA, 2 * K, () => randn() * 0.4);
    this.ba1 = vec(HA, 0);
    this.Wa2 = mat(SUMS, HA, () => randn() * 0.4);
    this.ba2 = vec(SUMS, 0);
  }

  private buildSplit(): { train: Pair[]; test: Pair[] } {
    const { K } = this;
    const pairs: Pair[] = [];
    for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) pairs.push([i, j]);
    // Deterministic shuffle so the held-out set is stable across resets.
    pairs.sort(
      (a, b) =>
        ((a[0] * 7 + a[1] * 13) % K) - ((b[0] * 7 + b[1] * 13) % K) ||
        a[0] - b[0] ||
        a[1] - b[1]
    );
    const nTrain = Math.max(1, Math.round(pairs.length * this.config.trainRatio));
    return { train: pairs.slice(0, nTrain), test: pairs.slice(nTrain) };
  }

  /** Jitter a digit's clean reference pixels — the "fuzzy" perceived input. */
  private noisyPixels(d: number): number[] {
    const clean = this.digitPixels[d];
    const v = new Array<number>(clean.length);
    for (let i = 0; i < clean.length; i++) v[i] = clean[i] + randn() * this.config.noise;
    return v;
  }

  // ---- neurosymbolic head ------------------------------------------------

  /** Read one noisy digit-pixel vector into a probability distribution over digits. */
  private perceive(x: number[]): number[] {
    const z = vec(this.K, 0);
    for (let o = 0; o < this.K; o++) {
      let s = this.bp[o];
      for (let i = 0; i < this.PIXEL_DIM; i++) s += this.Wp[o][i] * x[i];
      z[o] = s;
    }
    return softmax(z);
  }

  /** Differentiable "add" rule: P(sum=s) = Σ_{i+j=s} P(d1=i)·P(d2=j). */
  private sumDist(p1: number[], p2: number[]): number[] {
    const ps = vec(this.SUMS, 0);
    for (let i = 0; i < this.K; i++)
      for (let j = 0; j < this.K; j++) ps[i + j] += p1[i] * p2[j];
    return ps;
  }

  /** Backprop a perception-output gradient through softmax into Wp/bp. */
  private backPerceive(x: number[], p: number[], dp: number[], lr: number): void {
    let dot = 0;
    for (let k = 0; k < this.K; k++) dot += p[k] * dp[k];
    const dz = vec(this.K, 0);
    for (let o = 0; o < this.K; o++) dz[o] = p[o] * (dp[o] - dot);
    for (let o = 0; o < this.K; o++) {
      for (let i = 0; i < this.PIXEL_DIM; i++) this.Wp[o][i] -= lr * dz[o] * x[i];
      this.bp[o] -= lr * dz[o];
    }
  }

  private trainNeuro(): number {
    const lr = this.config.learningRate;
    let total = 0;
    for (const [d1, d2] of shuffle(this.split.train)) {
      const x1 = this.noisyPixels(d1);
      const x2 = this.noisyPixels(d2);
      const p1 = this.perceive(x1);
      const p2 = this.perceive(x2);
      const ps = this.sumDist(p1, p2);
      const gold = d1 + d2;
      total += -Math.log(ps[gold] + 1e-9);
      const inv = 1 / (ps[gold] + 1e-9);
      const dp1 = vec(this.K, 0);
      const dp2 = vec(this.K, 0);
      for (let i = 0; i < this.K; i++) {
        const j = gold - i;
        if (j >= 0 && j < this.K) {
          dp1[i] += -inv * p2[j];
          dp2[j] += -inv * p1[i];
        }
      }
      this.backPerceive(x1, p1, dp1, lr);
      this.backPerceive(x2, p2, dp2, lr);
    }
    return total / this.split.train.length;
  }

  private predictNeuro(d1: number, d2: number): number {
    return argmax(
      this.sumDist(this.perceive(this.noisyPixels(d1)), this.perceive(this.noisyPixels(d2)))
    );
  }

  // ---- pure-neural head --------------------------------------------------

  private forwardNeural(x: number[]): { h: number[]; p: number[] } {
    const a1 = vec(this.H, 0);
    for (let k = 0; k < this.H; k++) {
      let s = this.b1[k];
      for (let i = 0; i < 2 * this.PIXEL_DIM; i++) s += this.W1[k][i] * x[i];
      a1[k] = s;
    }
    const h = a1.map((v) => this.act.f(v));
    const z = vec(this.SUMS, 0);
    for (let o = 0; o < this.SUMS; o++) {
      let s = this.b2[o];
      for (let k = 0; k < this.H; k++) s += this.W2[o][k] * h[k];
      z[o] = s;
    }
    return { h, p: softmax(z) };
  }

  private trainNeural(): number {
    const lr = this.config.learningRate;
    let total = 0;
    for (const [d1, d2] of shuffle(this.split.train)) {
      const x = [...this.noisyPixels(d1), ...this.noisyPixels(d2)];
      const gold = d1 + d2;
      const { h, p } = this.forwardNeural(x);
      total += -Math.log(p[gold] + 1e-9);
      const dz = p.slice();
      dz[gold] -= 1;
      const dh = vec(this.H, 0);
      for (let o = 0; o < this.SUMS; o++) {
        for (let k = 0; k < this.H; k++) {
          dh[k] += this.W2[o][k] * dz[o];
          this.W2[o][k] -= lr * dz[o] * h[k];
        }
        this.b2[o] -= lr * dz[o];
      }
      for (let k = 0; k < this.H; k++) {
        const da = dh[k] * this.act.d(h[k]);
        for (let i = 0; i < 2 * this.PIXEL_DIM; i++) this.W1[k][i] -= lr * da * x[i];
        this.b1[k] -= lr * da;
      }
    }
    return total / this.split.train.length;
  }

  private predictNeural(d1: number, d2: number): number {
    return argmax(this.forwardNeural([...this.noisyPixels(d1), ...this.noisyPixels(d2)]).p);
  }

  // ---- Category 1: learning for reasoning (Neural -> Symbolic) -----------

  /** Train perception directly on digit labels, then add with exact logic. */
  private trainPerceptionDigits(): number {
    const lr = this.config.learningRate;
    const digits: number[] = [];
    for (const [d1, d2] of this.split.train) digits.push(d1, d2);
    let total = 0;
    for (const d of shuffle(digits)) {
      const x = this.noisyPixels(d);
      const p = this.perceive(x);
      total += -Math.log(p[d] + 1e-9);
      const dz = p.slice();
      dz[d] -= 1;
      for (let o = 0; o < this.K; o++) {
        for (let i = 0; i < this.PIXEL_DIM; i++) this.Wp[o][i] -= lr * dz[o] * x[i];
        this.bp[o] -= lr * dz[o];
      }
    }
    return total / Math.max(1, digits.length);
  }

  private predictCat1(d1: number, d2: number): number {
    return (
      argmax(this.perceive(this.noisyPixels(d1))) +
      argmax(this.perceive(this.noisyPixels(d2)))
    );
  }

  // ---- Category 2: reasoning for learning (Symbolic -> Neural) -----------

  /** A small neural "adder" over the two perceptions — used at inference. */
  private forwardAdder(p1: number[], p2: number[]): { h: number[]; p: number[] } {
    const a = [...p1, ...p2];
    const h = vec(this.HA, 0);
    for (let k = 0; k < this.HA; k++) {
      let s = this.ba1[k];
      for (let i = 0; i < 2 * this.K; i++) s += this.Wa1[k][i] * a[i];
      h[k] = this.act.f(s);
    }
    const z = vec(this.SUMS, 0);
    for (let o = 0; o < this.SUMS; o++) {
      let s = this.ba2[o];
      for (let k = 0; k < this.HA; k++) s += this.Wa2[o][k] * h[k];
      z[o] = s;
    }
    return { h, p: softmax(z) };
  }

  /**
   * Neural main body learns the sum from its own perceptions, while the
   * symbolic add rule is injected ONLY as a training-time semantic loss.
   */
  private trainCat2(): number {
    const lr = this.config.learningRate;
    let total = 0;
    for (const [d1, d2] of shuffle(this.split.train)) {
      const x1 = this.noisyPixels(d1);
      const x2 = this.noisyPixels(d2);
      const p1 = this.perceive(x1);
      const p2 = this.perceive(x2);
      const a = [...p1, ...p2];
      const { h, p } = this.forwardAdder(p1, p2);
      const gold = d1 + d2;
      total += -Math.log(p[gold] + 1e-9);

      // cross-entropy gradient through the neural adder
      const dz = p.slice();
      dz[gold] -= 1;
      const dh = vec(this.HA, 0);
      for (let o = 0; o < this.SUMS; o++) {
        for (let k = 0; k < this.HA; k++) {
          dh[k] += this.Wa2[o][k] * dz[o];
          this.Wa2[o][k] -= lr * dz[o] * h[k];
        }
        this.ba2[o] -= lr * dz[o];
      }
      const da = vec(2 * this.K, 0);
      for (let k = 0; k < this.HA; k++) {
        const dpre = dh[k] * this.act.d(h[k]);
        for (let i = 0; i < 2 * this.K; i++) {
          da[i] += this.Wa1[k][i] * dpre;
          this.Wa1[k][i] -= lr * dpre * a[i];
        }
        this.ba1[k] -= lr * dpre;
      }
      const dp1 = da.slice(0, this.K);
      const dp2 = da.slice(this.K);

      // symbolic semantic loss guides perception (training only)
      const ps = this.sumDist(p1, p2);
      const inv = SEMANTIC_REG / (ps[gold] + 1e-9);
      for (let i = 0; i < this.K; i++) {
        const j = gold - i;
        if (j >= 0 && j < this.K) {
          dp1[i] += -inv * p2[j];
          dp2[j] += -inv * p1[i];
        }
      }
      this.backPerceive(x1, p1, dp1, lr);
      this.backPerceive(x2, p2, dp2, lr);
    }
    return total / this.split.train.length;
  }

  private predictCat2(d1: number, d2: number): number {
    const p1 = this.perceive(this.noisyPixels(d1));
    const p2 = this.perceive(this.noisyPixels(d2));
    return argmax(this.forwardAdder(p1, p2).p);
  }

  // ---- inference + visual helpers shared by the views --------------------

  /** Predict the sum using whichever approach this model represents. */
  approachPred(d1: number, d2: number): number {
    switch (this.category) {
      case "learning-for-reasoning":
        return this.predictCat1(d1, d2);
      case "reasoning-for-learning":
        return this.predictCat2(d1, d2);
      default:
        return this.predictNeuro(d1, d2);
    }
  }

  /** One noisy perception read for a digit (for the live visualization). */
  readDigit(d: number): number[] {
    return this.perceive(this.noisyPixels(d));
  }

  /** The K×K product grid P(d1=i)·P(d2=j) used by the Category-3 view. */
  outerGrid(p1: number[], p2: number[]): number[][] {
    const g = mat(this.K, this.K, () => 0);
    for (let i = 0; i < this.K; i++)
      for (let j = 0; j < this.K; j++) g[i][j] = p1[i] * p2[j];
    return g;
  }

  /** P(sum) the selected approach assigns to a pair (for the visualization). */
  approachSumDist(d1: number, d2: number): number[] {
    const p1 = this.readDigit(d1);
    const p2 = this.readDigit(d2);
    if (this.category === "learning-reasoning") return this.sumDist(p1, p2);
    if (this.category === "reasoning-for-learning") return this.forwardAdder(p1, p2).p;
    const ps = vec(this.SUMS, 0);
    ps[Math.min(this.SUMS - 1, argmax(p1) + argmax(p2))] = 1;
    return ps;
  }

  /** A held-out pair the model never trained on (falls back to any pair). */
  sampleHeldout(): Pair {
    const t = this.split.test;
    if (t.length) return t[(Math.random() * t.length) | 0];
    const tr = this.split.train;
    return tr.length ? tr[(Math.random() * tr.length) | 0] : [0, 0];
  }

  /** Whether a specific ordered pair was in the training split (vs. held-out). */
  isTrainPair(d1: number, d2: number): boolean {
    return this.split.train.some(([a, b]) => a === d1 && b === d2);
  }

  // ---- metrics -----------------------------------------------------------

  /** Monte-Carlo accuracy over pairs (re-noised a few times each). */
  private acc(pairs: Pair[], predict: (d1: number, d2: number) => number): number {
    if (!pairs.length) return 1;
    let ok = 0;
    let n = 0;
    for (const [d1, d2] of pairs)
      for (let r = 0; r < 6; r++) {
        if (predict(d1, d2) === d1 + d2) ok++;
        n++;
      }
    return ok / n;
  }

  /** How often perception reads each digit right — emergent, never labeled. */
  private digitAccuracy(): number {
    let ok = 0;
    let n = 0;
    for (let d = 0; d < this.K; d++)
      for (let r = 0; r < 8; r++) {
        if (argmax(this.perceive(this.noisyPixels(d))) === d) ok++;
        n++;
      }
    return ok / n;
  }
}

