import "./style.css";
import { store, type ViewName } from "./state";
import { el, clear } from "./dom";
import { getSituation } from "./data/situations";
import { renderSituations } from "./views/situations";
import { renderBuilder } from "./views/builder";
import { openNeuralModal } from "./views/neuralDetail";
import { openSymbolicModal } from "./views/symbolicDetail";
import { renderProportion } from "./views/proportion";
import { renderResults } from "./views/results";

const app = document.querySelector<HTMLDivElement>("#app")!;

interface StepDef {
  view: ViewName;
  num: number;
  label: string;
}

const STEPS: StepDef[] = [
  { view: "situations", num: 1, label: "Situation" },
  { view: "builder", num: 2, label: "Architecture" },
  { view: "results", num: 3, label: "Results" },
];

function canVisit(view: ViewName): boolean {
  const st = store.get();
  if (view === "situations") return true;
  // everything past step 1 needs a chosen situation
  return st.situationId != null;
}

function topbar(): HTMLElement {
  const st = store.get();

  const steps = STEPS.flatMap((s, i) => {
    const active =
      st.view === s.view ||
      (s.view === "builder" && (st.view === "neural" || st.view === "symbolic"));
    const stepEl = el(
      "div",
      {
        class: "step" + (active ? " active" : ""),
        onclick: () => {
          if (canVisit(s.view)) store.set({ view: s.view, mode: "customize" });
        },
      },
      el("span", { class: "num" }, String(s.num)),
      el("span", {}, s.label)
    );
    const sep =
      i < STEPS.length - 1 ? el("span", { class: "sep" }, "\u203A") : null;
    return sep ? [stepEl, sep] : [stepEl];
  });

  const propPill = el(
    "div",
    {
      class: "step" + (st.view === "proportion" ? " active" : ""),
      style: { marginLeft: "8px" },
      title: "A separate mode: blend neural and symbolic by proportion",
      onclick: () => {
        if (canVisit("builder"))
          store.set({ view: "proportion", mode: "proportion" });
      },
    },
    el("span", {}, "\u2696\uFE0F Proportion mode")
  );

  return el(
    "header",
    { class: "topbar" },
    el("span", { class: "brand-mark" }, "\u25C8"),
    el("h1", {}, "Neurosymbolic Playground"),
    el("div", { class: "stepper" }, ...steps),
    el("div", { class: "spacer" }),
    propPill
  );
}

function render(): void {
  const st = store.get();

  // Friendly guard: if a situation is required but missing, fall back.
  if (st.view !== "situations" && !getSituation(st.situationId)) {
    store.set({ view: "situations" });
    return;
  }

  clear(app);
  app.append(topbar());

  const container = el("main", { class: "view" });
  app.append(container);

  switch (st.view) {
    case "situations":
      renderSituations(container);
      break;
    case "builder":
      renderBuilder(container);
      break;
    case "neural":
      renderBuilder(container);
      openNeuralModal(app);
      break;
    case "symbolic":
      renderBuilder(container);
      openSymbolicModal(app);
      break;
    case "proportion":
      renderProportion(container);
      break;
    case "results":
      renderResults(container);
      break;
  }
}

store.subscribe(render);
render();

