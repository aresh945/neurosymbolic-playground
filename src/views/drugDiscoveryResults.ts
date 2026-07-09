/**
 * Drug Discovery Screening results — a single live pipeline (no baseline
 * comparison, no stacking-pattern taxonomy; that's digit-sum's job). A
 * frozen, pretrained-offline network reads a noisy sensor signal, and the
 * generic rule engine (logicRules.ts) reasons over the reading to produce
 * an EXPLAINABLE verdict: not just a probability, but exactly which named
 * conditions were satisfied or violated.
 *
 * There is no training loop here — every compound pick or "read again"
 * click takes one fresh noisy sample through the frozen network and stages
 * a fresh reveal via drugDiscoveryViz.ts. `teardown()` only needs to cancel
 * that staged-reveal timer, not a training loop, before a fresh mount.
 */

import { store } from "../state";
import { el, clear } from "../dom";
import { getSituation } from "../data/situations";
import { COMPOUNDS, readCompound } from "../neural/compoundSafety";
import type { DrugCompound, CompoundReading } from "../neural/compoundSafety";
import type { RuleTrace } from "../neural/logicRules";
import { drawDrugDiscoveryViz, type VizHandle } from "./drugDiscoveryViz";

const POS = "var(--pos)";
const NEG = "var(--neg)";

let activeViz: VizHandle | null = null;
function teardown(): void {
  activeViz?.destroy();
  activeViz = null;
}

/** Walk the trace generically (no domain-specific tree-shape assumption) to
 * collect every named atom-level condition and whether it was satisfied. */
function collectAtoms(trace: RuleTrace): { label: string; value: number }[] {
  const atoms: { label: string; value: number }[] = [];
  const walk = (t: RuleTrace): void => {
    if (t.node.kind === "atom") atoms.push({ label: t.node.label ?? "condition", value: t.value });
    t.children.forEach(walk);
  };
  walk(trace);
  return atoms;
}

function explainReading(compound: DrugCompound, reading: CompoundReading): string {
  const atoms = collectAtoms(reading.trace);
  const satisfied = atoms.filter((a) => a.value > 0.5);
  const violated = atoms.filter((a) => a.value <= 0.5);
  const pct = Math.round(reading.pSafe * 100);
  const recommend = reading.pSafe > 0.5;
  const fmt = (a: { label: string; value: number }) => `${a.label} (${Math.round(a.value * 100)}%)`;
  const clauses: string[] = [];
  if (satisfied.length) clauses.push("satisfied: " + satisfied.map(fmt).join(", "));
  if (violated.length) clauses.push("violated: " + violated.map(fmt).join(", "));
  return (
    `<b>${compound.name}</b> — <b style="color:${recommend ? POS : NEG}">` +
    `${recommend ? "pursue" : "do not pursue"} (${pct}%)</b> because ${clauses.join("; ")}.`
  );
}

function compoundPicker(
  active: string,
  onPick: (c: DrugCompound) => void
): { row: HTMLElement; setActive: (id: string) => void } {
  const buttons = COMPOUNDS.map((c) =>
    el(
      "button",
      {
        class: "compound-token" + (c.id === active ? " active" : ""),
        title: c.note,
        onclick: () => onPick(c),
      },
      el("span", { class: "name" }, c.name),
      el("span", { class: "note" }, c.note)
    )
  );
  const row = el("div", { class: "compound-picker" }, ...buttons);
  return {
    row,
    setActive: (id: string) => buttons.forEach((b, i) => b.classList.toggle("active", COMPOUNDS[i].id === id)),
  };
}

export function renderDrugDiscoveryResults(root: HTMLElement): void {
  teardown();
  clear(root);

  const sit = getSituation("drug-discovery");
  let current: DrugCompound = COMPOUNDS[0];

  const head = el(
    "div",
    { class: "view-head" },
    el("h2", {}, "Drug Discovery Screening — perception + explainable rules"),
    el(
      "p",
      {},
      `${sit?.icon ?? "⚗️"} A pretrained network reads a noisy sensor signal; a Logic Tensor ` +
        "Network-style rule engine reasons over the reading to explain — not just predict — " +
        "whether the compound is worth pursuing."
    )
  );

  const readAgainBtn = el("button", { class: "btn" }, "↻ Read again (new noise)");
  const picker = compoundPicker(current.id, (c) => {
    current = c;
    picker.setActive(current.id);
    reveal();
  });

  const pickerCard = el(
    "div",
    { class: "metric-card" },
    el("div", { class: "label" }, "Pick a compound"),
    picker.row,
    el("div", { style: { marginTop: "10px" } }, readAgainBtn)
  );

  const vizBox = el("div", { style: { minHeight: "220px" } });
  const vizCard = el(
    "div",
    { class: "metric-card" },
    el("div", { class: "label" }, "Live data flow"),
    vizBox
  );

  const verdict = el("div", { class: "note" });

  const actions = el(
    "div",
    { class: "btn-row" },
    el(
      "button",
      {
        class: "btn",
        onclick: () => {
          teardown();
          store.set({ view: "builder", mode: "customize" });
        },
      },
      "← Back to builder"
    ),
    el(
      "button",
      {
        class: "btn",
        onclick: () => {
          teardown();
          store.set({ view: "situations" });
        },
      },
      "New situation"
    )
  );

  function reveal(): void {
    const reading = readCompound(current);
    teardown();
    activeViz = drawDrugDiscoveryViz(vizBox, reading);
    verdict.innerHTML = explainReading(current, reading);
  }

  readAgainBtn.onclick = () => reveal();

  root.append(head, pickerCard, vizCard, verdict, actions);
  reveal();
}
