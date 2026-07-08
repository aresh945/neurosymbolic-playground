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
    percTrainData: "50",
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
    backdrop.remove();
    store.set({ view: "builder" });
  }

  app.append(backdrop);
}

