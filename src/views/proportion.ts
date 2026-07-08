/**
 * Proportion mode — intentionally SEPARATE from the customize flow (per mentor
 * feedback). Here you cannot edit the internals; you only slide the balance
 * between neural and symbolic and watch the trade-offs move.
 */

import { store } from "../state";
import { el } from "../dom";
import { getSituation } from "../data/situations";

interface Tradeoffs {
  performance: number;
  explainability: number;
  robustness: number;
}

/** Illustrative trade-off curves as a function of symbolic proportion p (0..1). */
function tradeoffs(p: number): Tradeoffs {
  const explainability = 0.15 + 0.8 * p; // symbolic = white box
  const robustness = 0.9 - 0.55 * p; // neural copes better with messy input
  const generalization = 1 - 1.7 * (p - 0.62) ** 2; // best when symbolic-leaning
  return {
    performance: Math.max(0.05, Math.min(1, generalization)),
    explainability: Math.max(0.05, Math.min(1, explainability)),
    robustness: Math.max(0.05, Math.min(1, robustness)),
  };
}

function interpret(p: number): string {
  if (p < 0.25)
    return "Mostly neural: strong on messy, raw input, but the decision is a black box.";
  if (p < 0.45)
    return "Neural-leaning: good perception with a little symbolic structure for sanity.";
  if (p <= 0.55)
    return "Balanced: learned perception feeding crisp reasoning \u2014 often the sweet spot.";
  if (p < 0.8)
    return "Symbolic-leaning: crisp, explainable reasoning backed by learned perception.";
  return "Mostly symbolic: fully auditable and great at generalization, but fragile if perception is noisy.";
}

export function renderProportion(root: HTMLElement): void {
  const st = store.get();
  const sit = getSituation(st.situationId);

  const head = el(
    "div",
    { class: "view-head" },
    el("h2", {}, "\u2696\uFE0F Proportion mode"),
    el(
      "p",
      {},
      "A separate mode: slide the balance between neural and symbolic and watch " +
        "the output trade-offs. (Internals are locked here on purpose.)"
    )
  );

  const segNeural = el("div", { class: "seg-neural" });
  const segSymbolic = el("div", { class: "seg-symbolic" });
  const bar = el("div", { class: "prop-bar" }, segNeural, segSymbolic);

  const perfSpan = el("span", { style: { background: "#0877bd" } });
  const explSpan = el("span", { style: { background: "#8e44ad" } });
  const robuSpan = el("span", { style: { background: "#f59322" } });
  const perfVal = el("b", {}, "");
  const explVal = el("b", {}, "");
  const robuVal = el("b", {}, "");
  const interpretEl = el("div", { class: "note" }, "");

  function meterRow(
    label: string,
    span: HTMLElement,
    val: HTMLElement
  ): HTMLElement {
    return el(
      "div",
      { style: { marginBottom: "14px" } },
      el(
        "div",
        {
          style: {
            display: "flex",
            justifyContent: "space-between",
            fontSize: "13px",
            marginBottom: "5px",
          },
        },
        el("span", {}, label),
        val
      ),
      el("div", { class: "meter" }, span)
    );
  }

  function refresh(): void {
    const p = store.get().proportion;
    const t = tradeoffs(p);
    const symPct = Math.round(p * 100);
    segNeural.style.width = `${100 - symPct}%`;
    segSymbolic.style.width = `${symPct}%`;
    segNeural.textContent = symPct <= 88 ? `Neural ${100 - symPct}%` : "";
    segSymbolic.textContent = symPct >= 12 ? `Symbolic ${symPct}%` : "";
    perfSpan.style.width = `${t.performance * 100}%`;
    explSpan.style.width = `${t.explainability * 100}%`;
    robuSpan.style.width = `${t.robustness * 100}%`;
    perfVal.textContent = Math.round(t.performance * 100) + "%";
    explVal.textContent = Math.round(t.explainability * 100) + "%";
    robuVal.textContent = Math.round(t.robustness * 100) + "%";
    interpretEl.textContent = interpret(p);
  }

  const slider = el("input", {
    type: "range",
    min: "0",
    max: "1",
    step: "0.01",
    value: String(st.proportion),
    style: { width: "100%" },
    oninput: (e: Event) => {
      store.get().proportion = parseFloat((e.target as HTMLInputElement).value);
      refresh();
    },
  });

  const wrap = el(
    "div",
    { class: "prop-wrap" },
    sit
      ? el(
          "p",
          { class: "muted", style: { marginTop: "0" } },
          `${sit.icon} ${sit.title}`
        )
      : "",
    bar,
    el(
      "div",
      { class: "prop-scale" },
      el("span", {}, "\uD83E\uDDE0 All neural"),
      el("span", {}, "All symbolic \uD83D\uDD23")
    ),
    slider,
    el("div", { style: { marginTop: "24px" } },
      meterRow("Task performance", perfSpan, perfVal),
      meterRow("Explainability", explSpan, explVal),
      meterRow("Robustness to messy input", robuSpan, robuVal)
    ),
    interpretEl
  );

  const actions = el(
    "div",
    { class: "btn-row" },
    el(
      "button",
      { class: "btn", onclick: () => store.set({ view: "builder", mode: "customize" }) },
      "\u2190 Back to builder"
    )
  );

  root.append(head, wrap, actions);
  refresh();
}

