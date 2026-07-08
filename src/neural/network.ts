/**
 * A small controller that wraps the vendored TF Playground engine (nn.ts):
 *  - builds a network from a NeuralConfig,
 *  - trains it on a 2D dataset,
 *  - draws it with D3 in TensorFlow Playground's visual language
 *    (blue = positive weight, orange = negative, thickness = magnitude).
 */

import * as d3 from "d3";
import * as nn from "./nn";
import {
  makeDataset,
  DOMAIN_EXTENT,
  type Point2D,
  type DatasetKind,
} from "./dataset";
import type { NeuralConfig, ActivationName, RegularizationName } from "../state";

const ACTIVATIONS: Record<ActivationName, nn.ActivationFunction> = {
  tanh: nn.Activations.TANH,
  relu: nn.Activations.RELU,
  sigmoid: nn.Activations.SIGMOID,
  linear: nn.Activations.LINEAR,
};

const REG: Record<RegularizationName, nn.RegularizationFunction | null> = {
  none: null,
  l1: nn.RegularizationFunction.L1,
  l2: nn.RegularizationFunction.L2,
};

/** All selectable input features (TF Playground's feature set). */
export const ALL_FEATURES = [
  "x1",
  "x2",
  "x1Squared",
  "x2Squared",
  "x1x2",
  "sinX1",
  "sinX2",
] as const;
export type FeatureId = (typeof ALL_FEATURES)[number];

export const FEATURE_LABEL: Record<FeatureId, string> = {
  x1: "X\u2081",
  x2: "X\u2082",
  x1Squared: "X\u2081\u00B2",
  x2Squared: "X\u2082\u00B2",
  x1x2: "X\u2081X\u2082",
  sinX1: "sin(X\u2081)",
  sinX2: "sin(X\u2082)",
};

function featureValue(id: FeatureId, x: number, y: number): number {
  switch (id) {
    case "x1":
      return x;
    case "x2":
      return y;
    case "x1Squared":
      return x * x;
    case "x2Squared":
      return y * y;
    case "x1x2":
      return x * y;
    case "sinX1":
      return Math.sin(x);
    case "sinX2":
      return Math.sin(y);
  }
}

/** Color scale shared by node activations and link weights. */
const weightColor = d3
  .scaleLinear<string>()
  .domain([-1, 0, 1])
  .range(["#f59322", "#e8eaeb", "#0877bd"])
  .clamp(true);

export class NeuralController {
  network: nn.Node[][] = [];
  config: NeuralConfig;
  dataset: DatasetKind;
  /** Full generated sample, then split into train/test by `trainRatio`. */
  allData: Point2D[] = [];
  trainData: Point2D[] = [];
  testData: Point2D[] = [];
  /** Dataset knobs, mirroring TensorFlow Playground's DATA column. */
  noise = 0;
  trainRatio = 0.5;
  iter = 0;
  lossTrain = 0;
  lossTest = 0;

  constructor(config: NeuralConfig, dataset: DatasetKind) {
    this.config = config;
    this.dataset = dataset;
    this.noise = config.noise ?? 0;
    this.buildData();
    this.rebuild(); // builds the network, then computes loss
  }

  /** Shuffle in place (Fisher–Yates). */
  private static shuffle<T>(arr: T[]): void {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  }

  /** Sample a fresh dataset and split it into train/test (no loss yet). */
  private buildData(): void {
    this.allData = makeDataset(this.dataset, 300, this.noise);
    NeuralController.shuffle(this.allData);
    const split = Math.max(
      1,
      Math.floor(this.allData.length * this.trainRatio)
    );
    this.trainData = this.allData.slice(0, split);
    this.testData = this.allData.slice(split);
  }

  /** Regenerate the sample from the current dataset/noise and re-split it. */
  regenerate(): void {
    this.buildData();
    this.recomputeLoss();
  }

  setNoise(noise: number): void {
    this.noise = noise;
    this.regenerate();
  }

  setRatio(ratio: number): void {
    this.trainRatio = ratio;
    this.regenerate();
  }

  private recomputeLoss(): void {
    this.lossTrain = this.loss(this.trainData);
    this.lossTest = this.testData.length ? this.loss(this.testData) : 0;
  }

  /** (Re)build the network from the current config. Resets weights. */
  rebuild(): void {
    const features = this.config.features as FeatureId[];
    const shape = [features.length, ...this.config.hiddenLayers, 1];
    this.network = nn.buildNetwork(
      shape,
      ACTIVATIONS[this.config.activation],
      nn.Activations.TANH,
      REG[this.config.regularization],
      features.slice()
    );
    this.iter = 0;
    this.recomputeLoss();
  }

  /** Regenerate the dataset (e.g. when the situation changes). */
  setDataset(dataset: DatasetKind): void {
    this.dataset = dataset;
    this.buildData();
    this.rebuild();
  }

  private inputsFor(p: { x: number; y: number }): number[] {
    return (this.config.features as FeatureId[]).map((f) =>
      featureValue(f, p.x, p.y)
    );
  }

  /** One training epoch over the data in mini-batches. */
  step(epochs = 1): void {
    for (let e = 0; e < epochs; e++) {
      let i = 0;
      for (const p of this.trainData) {
        const inputs = this.inputsFor(p);
        nn.forwardProp(this.network, inputs);
        nn.backProp(this.network, p.label, nn.Errors.SQUARE);
        if (++i % this.config.batchSize === 0) {
          nn.updateWeights(
            this.network,
            this.config.learningRate,
            this.config.regularizationRate
          );
        }
      }
      nn.updateWeights(
        this.network,
        this.config.learningRate,
        this.config.regularizationRate
      );
      this.iter++;
    }
    this.recomputeLoss();
  }

  predict(x: number, y: number): number {
    return nn.forwardProp(this.network, this.inputsFor({ x, y }));
  }

  loss(data: Point2D[]): number {
    if (data.length === 0) return 0;
    let total = 0;
    for (const p of data) {
      const out = this.predict(p.x, p.y);
      total += nn.Errors.SQUARE.error(out, p.label);
    }
    return total / data.length;
  }

  accuracy(data: Point2D[]): number {
    let correct = 0;
    for (const p of data) {
      const out = this.predict(p.x, p.y);
      if (Math.sign(out) === Math.sign(p.label)) correct++;
    }
    return correct / data.length;
  }

  /**
   * Sample the input space on a `density x density` grid and record, for every
   * node, its activation at each point. A single forward pass updates all node
   * outputs, so one grid sweep yields the decision boundary of the whole
   * network at once. Returned arrays are indexed `[col][row]` (col = x left to
   * right, row = y top to bottom) to match the HeatMap renderer.
   */
  computeBoundaries(density: number): {
    output: number[][];
    nodes: Map<string, number[][]>;
  } {
    const ext = DOMAIN_EXTENT;
    const nodes = new Map<string, number[][]>();
    const blank = () =>
      Array.from({ length: density }, () => new Array<number>(density));
    for (const layer of this.network)
      for (const node of layer) nodes.set(node.id, blank());
    const output = blank();

    for (let c = 0; c < density; c++) {
      const x = -ext + (c / (density - 1)) * (2 * ext);
      for (let r = 0; r < density; r++) {
        const y = ext - (r / (density - 1)) * (2 * ext);
        output[c][r] = this.predict(x, y);
        for (const layer of this.network)
          for (const node of layer) nodes.get(node.id)![c][r] = node.output;
      }
    }
    return { output, nodes };
  }

  /**
   * The raw field of a single input feature over the grid, normalized to
   * [-1, 1]. Used to draw the feature thumbnails in the FEATURES column,
   * including features that are not currently wired into the network.
   */
  featureField(id: FeatureId, density: number): number[][] {
    const ext = DOMAIN_EXTENT;
    const arr = Array.from({ length: density }, () =>
      new Array<number>(density)
    );
    let maxAbs = 1e-6;
    for (let c = 0; c < density; c++) {
      const x = -ext + (c / (density - 1)) * (2 * ext);
      for (let r = 0; r < density; r++) {
        const y = ext - (r / (density - 1)) * (2 * ext);
        const v = featureValue(id, x, y);
        arr[c][r] = v;
        if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
      }
    }
    for (let c = 0; c < density; c++)
      for (let r = 0; r < density; r++) arr[c][r] /= maxAbs;
    return arr;
  }

  /** Draw the network into the given SVG element in TF Playground style. */
  draw(svgEl: SVGSVGElement): void {
    const svg = d3.select(svgEl);
    svg.selectAll("*").remove();

    const width = svgEl.clientWidth || 640;
    const height = svgEl.clientHeight || 360;
    const padX = 70;
    const usableW = width - padX * 2;
    const nodeSize = 26;

    const numLayers = this.network.length;
    const layerX = (i: number) =>
      padX + (numLayers === 1 ? usableW / 2 : (usableW * i) / (numLayers - 1));

    const nodeY = (count: number, idx: number) => {
      const gap = Math.min(56, (height - 40) / Math.max(count, 1));
      const top = height / 2 - (gap * (count - 1)) / 2;
      return top + gap * idx;
    };

    const linkLayer = svg.append("g");
    const nodeLayer = svg.append("g");

    // ----- links -----
    for (let l = 1; l < numLayers; l++) {
      const layer = this.network[l];
      for (let j = 0; j < layer.length; j++) {
        const node = layer[j];
        const x2 = layerX(l) - nodeSize / 2;
        const y2 = nodeY(layer.length, j);
        for (const link of node.inputLinks) {
          const srcLayer = this.network[l - 1];
          const srcIdx = srcLayer.indexOf(link.source);
          const x1 = layerX(l - 1) + nodeSize / 2;
          const y1 = nodeY(srcLayer.length, srcIdx);
          linkLayer
            .append("line")
            .attr("x1", x1)
            .attr("y1", y1)
            .attr("x2", x2)
            .attr("y2", y2)
            .attr("stroke", weightColor(link.weight))
            .attr(
              "stroke-width",
              Math.max(0.8, Math.min(6, Math.abs(link.weight) * 3))
            )
            .attr("opacity", 0.85);
        }
      }
    }

    // ----- nodes -----
    for (let l = 0; l < numLayers; l++) {
      const layer = this.network[l];
      for (let j = 0; j < layer.length; j++) {
        const node = layer[j];
        const cx = layerX(l);
        const cy = nodeY(layer.length, j);
        const g = nodeLayer
          .append("g")
          .attr("transform", `translate(${cx},${cy})`);
        g.append("rect")
          .attr("x", -nodeSize / 2)
          .attr("y", -nodeSize / 2)
          .attr("width", nodeSize)
          .attr("height", nodeSize)
          .attr("rx", 5)
          .attr("class", "node-box")
          .attr("fill", weightColor(node.output ?? 0));

        // labels: inputs on the far left, output on the far right
        if (l === 0) {
          const fid = this.config.features[j] as FeatureId;
          g.append("text")
            .attr("x", -nodeSize / 2 - 8)
            .attr("y", 4)
            .attr("text-anchor", "end")
            .attr("font-size", 12)
            .attr("fill", "#555")
            .text(FEATURE_LABEL[fid] ?? fid);
        } else if (l === numLayers - 1) {
          g.append("text")
            .attr("x", nodeSize / 2 + 8)
            .attr("y", 4)
            .attr("text-anchor", "start")
            .attr("font-size", 12)
            .attr("fill", "#555")
            .text("output");
        }
      }
    }
  }
}

