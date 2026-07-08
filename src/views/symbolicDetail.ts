/**
 * The "expand the symbolic block" modal. A dropdown chooses the symbolic engine
 * (knowledge graph / rule-based / forward chaining) and the panel shows a small,
 * method-specific illustration tied to the current situation.
 */

import { store, type SymbolicMethod } from "../state";
import { el } from "../dom";
import { getSituation } from "../data/situations";
import { METHODS } from "../data/patterns";

function svgEl(tag: string, attrs: Record<string, string>): SVGElement {
  const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
}

/** A tiny knowledge-graph illustration: two facts + one inferred (dashed) link. */
function kgVisual(): SVGElement {
  const svg = svgEl("svg", { viewBox: "0 0 420 150", width: "100%", height: "150" });
  const node = (x: number, y: number, label: string) => {
    const g = svgEl("g", { transform: `translate(${x},${y})` });
    g.appendChild(
      svgEl("circle", { r: "22", fill: "#f3ecf8", stroke: "#8e44ad", "stroke-width": "2" })
    );
    const t = svgEl("text", {
      "text-anchor": "middle",
      y: "4",
      "font-size": "12",
      fill: "#5b2c70",
    });
    t.textContent = label;
    g.appendChild(t);
    return g;
  };
  const edge = (
    x1: number,
    y1: number,
    x2: number,
    y2: number,
    label: string,
    dashed = false
  ) => {
    const g = svgEl("g", {});
    const line = svgEl("line", {
      x1: String(x1),
      y1: String(y1),
      x2: String(x2),
      y2: String(y2),
      stroke: dashed ? "#0877bd" : "#8e44ad",
      "stroke-width": "2",
    });
    if (dashed) line.setAttribute("stroke-dasharray", "5 4");
    g.appendChild(line);
    const t = svgEl("text", {
      x: String((x1 + x2) / 2),
      y: String((y1 + y2) / 2 - 6),
      "text-anchor": "middle",
      "font-size": "11",
      fill: dashed ? "#0877bd" : "#777",
    });
    t.textContent = label;
    g.appendChild(t);
    return g;
  };

  svg.appendChild(edge(50, 40, 210, 40, "parent"));
  svg.appendChild(edge(210, 40, 370, 40, "parent"));
  svg.appendChild(edge(50, 40, 370, 40, "grandparent (inferred)", true));
  svg.appendChild(node(50, 40, "Alice"));
  svg.appendChild(node(210, 40, "Bob"));
  svg.appendChild(node(370, 40, "Carol"));
  return svg;
}

function rulesVisual(): HTMLElement {
  const rules = [
    "IF digit\u2081 = a AND digit\u2082 = b THEN sum = a + b",
    "IF shape = octagon THEN action = STOP",
    "IF attributes = {stripes, hooves} THEN species = zebra",
  ];
  return el(
    "div",
    { style: { display: "flex", flexDirection: "column", gap: "8px" } },
    ...rules.map((r) =>
      el(
        "div",
        {
          style: {
            fontFamily: "ui-monospace, Consolas, monospace",
            fontSize: "13px",
            background: "#faf7fc",
            border: "1px solid #eaddf2",
            borderRadius: "8px",
            padding: "10px 12px",
            color: "#5b2c70",
          },
        },
        r
      )
    )
  );
}

function forwardChainVisual(): HTMLElement {
  const steps = [
    { t: "fact", txt: "shape = octagon (from neural perception)" },
    { t: "rule", txt: "octagon \u2192 STOP" },
    { t: "fact", txt: "action = STOP (derived)" },
    { t: "rule", txt: "STOP \u2192 require full halt" },
    { t: "fact", txt: "behavior = full halt (derived)" },
  ];
  return el(
    "div",
    { style: { display: "flex", flexDirection: "column", gap: "6px" } },
    ...steps.map((s) =>
      el(
        "div",
        {
          style: {
            display: "flex",
            alignItems: "center",
            gap: "10px",
            fontSize: "13px",
          },
        },
        el(
          "span",
          {
            style: {
              fontSize: "10px",
              textTransform: "uppercase",
              letterSpacing: "0.5px",
              padding: "2px 8px",
              borderRadius: "10px",
              color: s.t === "rule" ? "#8e44ad" : "#0877bd",
              background: s.t === "rule" ? "#f3ecf8" : "#e7f2f9",
              minWidth: "44px",
              textAlign: "center",
            },
          },
          s.t
        ),
        el("span", {}, s.txt)
      )
    )
  );
}

function methodVisual(method: SymbolicMethod): HTMLElement | SVGElement {
  switch (method) {
    case "kg":
      return kgVisual();
    case "rules":
      return rulesVisual();
    case "forward-chaining":
      return forwardChainVisual();
  }
}

export function openSymbolicModal(app: HTMLElement): void {
  const st = store.get();
  const sit = getSituation(st.situationId);
  // The symbolic method is fixed by the situation — not user-configurable.
  const method = sit?.suggestedMethod ?? st.symbolic.method;
  const info = METHODS[method];

  const methodChip = el(
    "div",
    {
      style: {
        display: "flex",
        alignItems: "center",
        gap: "12px",
        padding: "12px 14px",
        borderRadius: "10px",
        background: "#faf7fc",
        border: "1px solid #eaddf2",
        marginBottom: "14px",
      },
    },
    el("span", { style: { fontSize: "28px" } }, info.icon),
    el(
      "div",
      {},
      el("div", { style: { fontWeight: "600", color: "#5b2c70" } }, info.label),
      el(
        "div",
        { style: { fontSize: "12px", color: "var(--muted)" } },
        `${info.io} \u00b7 best fit for this situation`
      )
    )
  );

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
      { class: "modal" },
      el(
        "div",
        { class: "modal-head" },
        el("span", { style: { fontSize: "20px" } }, "\uD83D\uDD23"),
        el("h3", {}, "Symbolic engine \u2014 reasoning"),
        el("button", { class: "close", onclick: () => close() }, "\u00D7")
      ),
      el(
        "p",
        { class: "muted", style: { fontSize: "13px", lineHeight: "1.5", marginTop: "0" } },
        "Each situation uses the one symbolic method that fits it best. " +
          "This is a read-only look at how that method reasons."
      ),
      methodChip,
      el(
        "p",
        { class: "muted", style: { fontSize: "13px", lineHeight: "1.5" } },
        info.summary
      ),
      sit
        ? el(
            "div",
            { class: "note", style: { marginBottom: "16px" } },
            el("b", {}, "In this situation: "),
            sit.reasoning
          )
        : "",
      el(
        "div",
        {
          style: {
            border: "1px solid var(--line)",
            borderRadius: "10px",
            padding: "16px",
          },
        },
        methodVisual(method)
      )
    )
  );

  function close(): void {
    backdrop.remove();
    store.set({ view: "builder" });
  }

  app.append(backdrop);
}

