/**
 * Step 3: results. Trains the real neural network the user configured, then
 * contrasts three approaches — neural only, symbolic only, and the chosen
 * neurosymbolic pattern — across performance, explainability, and robustness.
 */

import { store } from "../state";
import { el } from "../dom";
import { getSituation } from "../data/situations";
import { PATTERNS } from "../data/patterns";
import { getController } from "../runtime";
import { renderDigitSumResults } from "./digitSumResults";

interface Scores {
  performance: number;
  explainability: number;
  robustness: number;
}

function meter(label: string, value: number, color: string): HTMLElement {
  return el(
    "div",
    { style: { marginBottom: "10px" } },
    el(
      "div",
      {
        style: {
          display: "flex",
          justifyContent: "space-between",
          fontSize: "12px",
          marginBottom: "4px",
        },
      },
      el("span", { class: "muted" }, label),
      el("b", {}, Math.round(value * 100) + "%")
    ),
    el(
      "div",
      { class: "meter" },
      el("span", { style: { width: `${value * 100}%`, background: color } })
    )
  );
}

function scoreCard(title: string, subtitle: string, s: Scores): HTMLElement {
  return el(
    "div",
    { class: "metric-card" },
    el("div", { class: "label" }, title),
    el(
      "div",
      { style: { fontSize: "13px", color: "var(--muted)", margin: "2px 0 14px" } },
      subtitle
    ),
    meter("Performance", s.performance, "#0877bd"),
    meter("Explainability", s.explainability, "#8e44ad"),
    meter("Robustness", s.robustness, "#f59322")
  );
}

export function renderResults(root: HTMLElement): void {
  const st = store.get();

  // The digit-sum situation has a real, live neurosymbolic demo of its own;
  // every other situation uses the illustrative trade-off cards below.
  if (st.situationId === "digit-sum") {
    renderDigitSumResults(root);
    return;
  }

  const sit = getSituation(st.situationId);
  const pattern = PATTERNS[st.pattern ?? "learning-reasoning"];

  // Train the user's actual network a bit so the neural number is real.
  const controller = getController();
  controller.config = st.neural;
  controller.rebuild();
  controller.step(60);
  const neuralAcc = controller.accuracy(controller.trainData);

  // Illustrative numbers for the other two approaches.
  const neuralOnly: Scores = {
    performance: neuralAcc,
    explainability: 0.2,
    robustness: 0.85,
  };
  const symbolicOnly: Scores = {
    performance: 0.55, // can't perceive raw input on its own
    explainability: 0.95,
    robustness: 0.3,
  };
  const combinedExplain =
    pattern.mainBody === "symbolic"
      ? 0.85
      : pattern.mainBody === "neural"
      ? 0.55
      : 0.72;
  const neurosymbolic: Scores = {
    performance: Math.min(0.99, neuralAcc + (1 - neuralAcc) * 0.6),
    explainability: combinedExplain,
    robustness: 0.82,
  };

  const head = el(
    "div",
    { class: "view-head" },
    el("h2", {}, "Results & trade-offs"),
    el(
      "p",
      {},
      sit
        ? `${sit.icon} ${sit.title} \u2014 pattern: ${pattern.taxonomy} (${pattern.flow})`
        : `Pattern: ${pattern.taxonomy}`
    )
  );

  const grid = el(
    "div",
    { class: "results-grid" },
    scoreCard("Neural only", "Perception, no rules", neuralOnly),
    scoreCard("Symbolic only", "Rules, no perception", symbolicOnly),
    scoreCard(
      "Neurosymbolic",
      `${pattern.flow}`,
      neurosymbolic
    )
  );

  const note = el(
    "div",
    { class: "note" },
    el("b", {}, "Why this pattern: "),
    pattern.tradeoff,
    sit
      ? el(
          "div",
          { style: { marginTop: "8px" } },
          el("b", {}, "Combined strength: "),
          sit.combinedStrength
        )
      : ""
  );

  const actions = el(
    "div",
    { class: "btn-row" },
    el(
      "button",
      { class: "btn", onclick: () => store.set({ view: "builder", mode: "customize" }) },
      "\u2190 Back to builder"
    ),
    el(
      "button",
      {
        class: "btn",
        onclick: () => store.set({ view: "proportion", mode: "proportion" }),
      },
      "Try proportion mode \u2696\uFE0F"
    ),
    el(
      "button",
      { class: "btn", onclick: () => store.set({ view: "situations" }) },
      "New situation"
    )
  );

  root.append(
    head,
    grid,
    note,
    el(
      "p",
      { class: "muted", style: { fontSize: "12px", marginTop: "16px" } },
      "Performance for the neural column is measured on your trained network; " +
        "the other figures are illustrative to show the trade-off shape."
    ),
    actions
  );
}

