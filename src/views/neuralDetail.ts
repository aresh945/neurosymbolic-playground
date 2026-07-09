/**
 * "Expand the neural block" — embeds the real TensorFlow Playground.
 *
 * Instead of re-implementing the playground, we vendor Google's compiled,
 * Apache-2.0 build (served from public/tfpg/) and load it in an <iframe>, so it
 * is pixel-identical to playground.tensorflow.org. The current situation and the
 * stored neural config are passed straight into the playground's own URL-hash
 * state (see their state.ts) so the embedded tool opens on the right problem.
 */

import { store } from "../state";
import { el } from "../dom";
import { getSituation } from "../data/situations";
import type { NeuralConfig } from "../state";

/** Our regularization names -> TF Playground's keys. */
const REG_KEY: Record<NeuralConfig["regularization"], string> = {
  none: "none",
  l1: "L1",
  l2: "L2",
};

/** Our feature ids -> TF Playground's input toggle keys. */
const FEATURE_KEY: Record<string, string> = {
  x1: "x",
  x2: "y",
  x1Squared: "xSquared",
  x2Squared: "ySquared",
  x1x2: "xTimesY",
  sinX1: "sinX",
  sinX2: "sinY",
};
const ALL_INPUT_KEYS = ["x", "y", "xSquared", "ySquared", "xTimesY", "sinX", "sinY"];

/** TF Playground's regularization keys -> ours (inverse of REG_KEY). */
const REG_KEY_REVERSE: Record<string, NeuralConfig["regularization"]> = {
  none: "none",
  L1: "l1",
  L2: "l2",
};

/** TF Playground's input toggle keys -> our feature ids (inverse of FEATURE_KEY). */
const FEATURE_KEY_REVERSE: Record<string, string> = Object.fromEntries(
  Object.entries(FEATURE_KEY).map(([ours, theirs]) => [theirs, ours])
);

/**
 * Build the TF Playground URL hash that seeds it from our app state. Their
 * deserializer splits the hash on "&"/"=" without URL-decoding, so values stay
 * unescaped (plain numbers, words, and comma-separated arrays).
 */
function playgroundHash(): string {
  const st = store.get();
  const sit = getSituation(st.situationId);
  const n = st.neural;

  // The dataset keys (circle/xor/gauss/spiral) map 1:1 to TF Playground's.
  const dataset = sit?.dataset ?? "circle";

  const params: Record<string, string> = {
    dataset,
    problem: "classification",
    activation: n.activation,
    regularization: REG_KEY[n.regularization],
    learningRate: String(n.learningRate),
    regularizationRate: String(n.regularizationRate),
    noise: String(Math.round((n.noise ?? 0) * 100)),
    batchSize: String(n.batchSize),
    networkShape: (n.hiddenLayers ?? []).join(","),
    percTrainData: String(Math.round((n.trainRatio ?? 0.6) * 100)),
    discretize: "true",
    showTestData: "false",
    initZero: "false",
    hideText: "true",
  };

  // Feature toggles: turn on the inputs the user selected, explicitly turn the
  // rest off (TF treats any present non-"false" value as true).
  const on = new Set((n.features ?? []).map((f) => FEATURE_KEY[f]).filter(Boolean));
  for (const key of ALL_INPUT_KEYS) {
    params[key] = on.has(key) ? "true" : "false";
  }

  return Object.entries(params)
    .map(([k, v]) => `${k}=${v}`)
    .join("&");
}

/**
 * Parse a live TF Playground hash (read from inside the iframe after the user
 * has been adjusting it) back into a NeuralConfig patch — the inverse of
 * playgroundHash(). Deliberately ignores `dataset`: that's tied to the chosen
 * situation, not something the embedded Playground should be able to override.
 */
function parsePlaygroundHash(hash: string): Partial<NeuralConfig> {
  const params: Record<string, string> = {};
  for (const pair of hash.split("&")) {
    const eq = pair.indexOf("=");
    if (eq < 0) continue;
    params[pair.slice(0, eq)] = pair.slice(eq + 1);
  }

  const patch: Partial<NeuralConfig> = {};
  if (params.activation) patch.activation = params.activation as NeuralConfig["activation"];
  if (params.regularization) patch.regularization = REG_KEY_REVERSE[params.regularization] ?? "none";
  if (params.learningRate) patch.learningRate = parseFloat(params.learningRate);
  if (params.regularizationRate) patch.regularizationRate = parseFloat(params.regularizationRate);
  if (params.noise) patch.noise = parseFloat(params.noise) / 100;
  if (params.percTrainData) patch.trainRatio = parseFloat(params.percTrainData) / 100;
  if (params.batchSize) patch.batchSize = parseInt(params.batchSize, 10);
  if (params.networkShape !== undefined) {
    patch.hiddenLayers = params.networkShape
      .split(",")
      .map((n) => parseInt(n, 10))
      .filter((n) => Number.isFinite(n) && n > 0);
  }

  const features: string[] = [];
  for (const key of ALL_INPUT_KEYS) {
    if (params[key] === "true") {
      const ours = FEATURE_KEY_REVERSE[key];
      if (ours) features.push(ours);
    }
  }
  patch.features = features;

  return patch;
}

export function openNeuralModal(app: HTMLElement): void {
  const sit = getSituation(store.get().situationId);

  const frame = el("iframe", {
    class: "tf-frame",
    src: `/tfpg/index.html#${playgroundHash()}`,
    title: "TensorFlow Playground",
  });

  const backdrop = el(
    "div",
    {
      class: "modal-backdrop",
      onclick: (e: Event) => {
        if (e.target === backdrop) close();
      },
    },
    el(
      "div",
      { class: "modal tf-modal" },
      el(
        "div",
        { class: "modal-head" },
        el("span", { style: { fontSize: "20px" } }, "\uD83E\uDDE0"),
        el(
          "h3",
          {},
          "Neural network",
          sit ? ` \u2014 ${sit.title}` : ""
        ),
        el("span", { class: "tf-sub" }, "live TensorFlow Playground"),
        el("button", { class: "close", onclick: () => close() }, "\u00D7")
      ),
      frame
    )
  );

  function close(): void {
    // Pull whatever the user landed on inside the Playground — same-origin
    // iframe, so its live location.hash is readable — and apply it as this
    // demo's actual neural config. Sync happens HERE, on close, not live:
    // writing to the store re-renders the whole app, which would recreate
    // this very iframe (via a fresh openNeuralModal call) mid-interaction.
    const liveHash = frame.contentWindow?.location.hash.slice(1) ?? "";
    const patch = parsePlaygroundHash(liveHash);
    backdrop.remove();
    store.set({ view: "builder", neural: { ...store.get().neural, ...patch } });
  }

  app.append(backdrop);
}

