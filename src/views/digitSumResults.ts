/**
 * Digit-Sum results — the live, real neurosymbolic demonstration.
 *
 * Replaces the illustrative trade-off bars (used by every other situation)
 * with two networks that actually train, side by side, on the SAME
 * weakly-supervised data: only the SUM of two digits is ever labeled. Each
 * digit is a real reference image (0-9, see neural/digitImages.ts) — the user
 * can pick any pair to watch via the digit picker below.
 *
 *   - Pure neural memorizes the pairs it sees and collapses on held-out pairs.
 *   - Neurosymbolic learns to READ each digit (a built-in add rule does the
 *     arithmetic) and so it generalizes to pairs it never trained on.
 *
 * Training runs in a bounded animation loop. Because the whole app re-renders
 * on any store change, the loop self-cancels once `root` leaves the document;
 * `teardown()` cancels both that loop and any pending staged-reveal timers
 * from the live visualization (see digitSumViz.ts) before a fresh mount.
 */

import { store } from "../state";
import { el, clear, mount } from "../dom";
import { getSituation } from "../data/situations";
import { PATTERNS, PATTERN_ORDER } from "../data/patterns";
import { DigitSumModel } from "../neural/digitSum";
import { preloadDigitImages, digitThumbnailSrc } from "../neural/digitImages";
import { drawDigitSumViz, type VizHandle } from "./digitSumViz";

const NEURAL_COLOR = "var(--neg)"; // orange
const NEURO_COLOR = "var(--pos)"; // blue
const PUR = "#8e44ad"; // perception / symbolic accent
const MAX_EPOCHS = 400;
const EPOCHS_PER_FRAME = 1; // one epoch per frame so the learning is easy to follow
const FRAME_DELAY = 45; // ms between frames — deliberately slow and visible

let timer: ReturnType<typeof setTimeout> | null = null;
function stopLoop(): void {
  if (timer !== null) {
    clearTimeout(timer);
    timer = null;
  }
}

let activeViz: VizHandle | null = null;
/** Cancel the training loop AND any pending staged-reveal timers before a
 * fresh mount or navigation away — without this, a leftover setTimeout from
 * the old visualization could fire into detached DOM. */
function teardown(): void {
  stopLoop();
  activeViz?.destroy();
  activeViz = null;
}

interface Bar {
  row: HTMLElement;
  set: (value: number) => void;
}

function bar(label: string, color: string): Bar {
  const pct = el("b", {}, "—");
  const fill = el("span", { style: { width: "0%", background: color } });
  const row = el(
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
      pct
    ),
    el("div", { class: "meter" }, fill)
  );
  return {
    row,
    set: (value: number) => {
      const p = Math.round(value * 100);
      fill.style.width = `${p}%`;
      pct.textContent = `${p}%`;
    },
  };
}

function slider(
  labelText: () => string,
  min: number,
  max: number,
  step: number,
  value: number,
  onInput: (v: number) => void
): HTMLElement {
  const caption = el("span", { class: "muted", style: { fontSize: "12px" } }, labelText());
  const input = el("input", {
    type: "range",
    min: String(min),
    max: String(max),
    step: String(step),
    value: String(value),
    style: { width: "160px" },
    oninput: (ev: Event) => {
      onInput(parseFloat((ev.target as HTMLInputElement).value));
      caption.textContent = labelText();
    },
  });
  return el(
    "label",
    { style: { display: "flex", flexDirection: "column", gap: "4px" } },
    caption,
    input
  );
}

/** A row of 10 clickable digit thumbnails; `setActive` moves the highlight
 * without touching training state — picking a digit is a display concern. */
function digitPicker(
  labelText: string,
  active: number,
  onPick: (d: number) => void
): { row: HTMLElement; setActive: (d: number) => void } {
  const buttons = Array.from({ length: 10 }, (_, d) =>
    el(
      "button",
      {
        class: "digit-thumb" + (d === active ? " active" : ""),
        title: `digit ${d}`,
        onclick: () => onPick(d),
      },
      el("img", { src: digitThumbnailSrc(d), alt: String(d) })
    )
  );
  const row = el(
    "div",
    { class: "digit-picker" },
    el("span", { class: "muted digit-picker-label" }, labelText),
    ...buttons
  );
  return {
    row,
    setActive: (d: number) => buttons.forEach((b, i) => b.classList.toggle("active", i === d)),
  };
}

export function renderDigitSumResults(root: HTMLElement): void {
  teardown();
  clear(root);
  mount(
    root,
    el(
      "div",
      { class: "view-head" },
      el("h2", {}, "Handwritten Digit Sum — three ways to merge neural + symbolic")
    ),
    el("div", { class: "note" }, "Loading digit images…")
  );

  preloadDigitImages()
    .then((digitPixels) => {
      if (!root.isConnected) return; // navigated away while images were loading
      buildAndRender(root, digitPixels);
    })
    .catch((err) => {
      if (!root.isConnected) return;
      clear(root);
      mount(
        root,
        el(
          "div",
          { class: "view-head" },
          el("h2", {}, "Handwritten Digit Sum — three ways to merge neural + symbolic")
        ),
        el(
          "div",
          { class: "note" },
          `Couldn't load the digit images (${String(err)}). Make sure ` +
            "public/mnist/0.png through public/mnist/9.png exist."
        )
      );
    });
}

function buildAndRender(root: HTMLElement, digitPixels: number[][]): void {
  const sit = getSituation("digit-sum");
  const st = store.get();
  const category = st.pattern ?? "learning-reasoning";
  const info = PATTERNS[category];
  // The neural block the user configured in the builder actually drives training:
  // its hidden width, activation, learning rate and noise all flow into the model.
  const nn = st.neural;
  const model = new DigitSumModel(category, digitPixels, {
    learningRate: nn.learningRate,
    noise: nn.noise,
    hidden: Math.max(1, nn.hiddenLayers[0] ?? 4),
    activation: nn.activation,
  });
  let sample = model.sampleHeldout();

  // ---- header ------------------------------------------------------------
  const head = el(
    "div",
    { class: "view-head" },
    el("h2", {}, "Handwritten Digit Sum — three ways to merge neural + symbolic"),
    el(
      "p",
      {},
      `${sit?.icon ?? "✍️"} Same weakly-supervised data (only the SUM is labeled). ` +
        "Pick a merging method below and watch what each one can — and can't — generalize to."
    )
  );

  // ---- category switcher -------------------------------------------------
  const switcher = el(
    "div",
    { class: "seg" },
    ...PATTERN_ORDER.map((p) =>
      el(
        "button",
        {
          class: "seg-btn" + (p === category ? " active" : ""),
          onclick: () => {
            if (p !== category) {
              teardown();
              store.set({ pattern: p });
            }
          },
        },
        el("span", { class: "seg-flow" }, PATTERNS[p].flow),
        el("span", { class: "seg-name" }, PATTERNS[p].taxonomy)
      )
    )
  );

  // ---- method explanation ------------------------------------------------
  const supervision = model.supervision();
  const teach = el(
    "div",
    { class: "note" },
    el("b", {}, info.taxonomy + " — "),
    info.summary,
    el(
      "div",
      { style: { marginTop: "8px" } },
      el("b", {}, "Supervision needed: "),
      el(
        "span",
        { class: "badge " + (supervision.strong ? "badge-strong" : "badge-weak") },
        supervision.label
      )
    )
  );

  // ---- cards: baseline vs the selected approach --------------------------
  const baseBig = el("div", { class: "value", style: { color: NEURAL_COLOR } }, "—");
  const baseTrain = bar("Train — pairs it saw", NEURAL_COLOR);
  const baseTest = bar("Held-out — new pairs", NEURAL_COLOR);
  const baseCard = el(
    "div",
    { class: "metric-card" },
    el("div", { class: "label" }, "Pure neural — baseline"),
    baseBig,
    el(
      "div",
      { style: { fontSize: "12px", color: "var(--muted)", margin: "-2px 0 14px" } },
      "held-out accuracy — it memorizes pairs"
    ),
    baseTrain.row,
    baseTest.row
  );

  const appBig = el("div", { class: "value", style: { color: NEURO_COLOR } }, "—");
  const appTrain = bar("Train — pairs it saw", NEURO_COLOR);
  const appTest = bar("Held-out — new pairs", NEURO_COLOR);
  const appDigit = bar("Digit-read — perception", PUR);
  const appCard = el(
    "div",
    { class: "metric-card" },
    el("div", { class: "label" }, info.label),
    appBig,
    el(
      "div",
      { style: { fontSize: "12px", color: "var(--muted)", margin: "-2px 0 14px" } },
      "held-out accuracy — " + info.taxonomy
    ),
    appTrain.row,
    appTest.row,
    appDigit.row
  );

  const grid = el(
    "div",
    { class: "results-grid", style: { gridTemplateColumns: "repeat(2, 1fr)" } },
    baseCard,
    appCard
  );

  // ---- digit picker + live visualization ---------------------------------
  const seenBadge = el("span", { class: "badge" }, "");

  function setSample(next: [number, number]): void {
    sample = next;
    pickerA.setActive(sample[0]);
    pickerB.setActive(sample[1]);
    refresh(true);
  }

  const pickerA = digitPicker("Digit A", sample[0], (d) => setSample([d, sample[1]]));
  const pickerB = digitPicker("Digit B", sample[1], (d) => setSample([sample[0], d]));
  const randomBtn = el(
    "button",
    { class: "btn", onclick: () => setSample(model.sampleHeldout()) },
    "🎲 Random held-out"
  );

  const pickerRow = el(
    "div",
    { style: { display: "flex", flexDirection: "column", gap: "6px", marginBottom: "10px" } },
    pickerA.row,
    pickerB.row,
    el(
      "div",
      { style: { display: "flex", alignItems: "center", gap: "10px", marginTop: "4px" } },
      randomBtn,
      seenBadge
    )
  );

  const vizSub = el(
    "div",
    { style: { fontSize: "12px", color: "var(--muted)", margin: "2px 0 8px" } },
    ""
  );
  const vizBox = el("div", { style: { minHeight: "200px" } });
  const vizCard = el(
    "div",
    { class: "metric-card" },
    el("div", { class: "label" }, "Live data flow — pick two digits"),
    pickerRow,
    vizSub,
    vizBox
  );

  // ---- live verdict ------------------------------------------------------
  const verdict = el("div", { class: "note" });

  // ---- controls ----------------------------------------------------------
  const epochLabel = el("b", {}, "0");
  const trainBtn = el("button", { class: "btn primary" }, "Train ▶");
  const resetBtn = el("button", { class: "btn" }, "Reset");

  const ratioSlider = slider(
    () => {
      const s = model.splitSizes;
      return `Train ratio — ${s.train} train / ${s.test} held-out`;
    },
    0.2,
    0.8,
    0.04,
    model.config.trainRatio,
    (v) => {
      model.config.trainRatio = v;
      restart();
    }
  );

  const noiseSlider = slider(
    () => `Noise — ${model.config.noise.toFixed(2)}`,
    0,
    0.4,
    0.02,
    model.config.noise,
    (v) => {
      model.config.noise = v;
      restart();
    }
  );

  const neuralInfo = el(
    "div",
    { class: "note", style: { margin: "0 0 14px" } },
    el("b", {}, "Your neural build is running: "),
    `${model.H} hidden unit${model.H === 1 ? "" : "s"} · ${nn.activation} · ` +
      `learning rate ${nn.learningRate} · input noise ${nn.noise.toFixed(2)}. `,
    el(
      "span",
      { class: "muted" },
      "Change these on the Architecture step (double-click the neural block)."
    )
  );

  const controls = el(
    "div",
    {
      style: {
        display: "flex",
        alignItems: "flex-end",
        flexWrap: "wrap",
        gap: "16px",
        margin: "4px 0 18px",
      },
    },
    el(
      "div",
      { class: "muted", style: { fontSize: "12px" } },
      "Epoch ",
      epochLabel
    ),
    trainBtn,
    resetBtn,
    ratioSlider,
    noiseSlider
  );

  // ---- refresh + loop ----------------------------------------------------
  const pct = (x: number) => `${Math.round(x * 100)}%`;

  function verdictText(
    n: { train: number; test: number; digit: number | null },
    s: { train: number; test: number; digit: number | null }
  ): string {
    const train = Math.round(Math.max(n.train, s.train) * 100);
    const dig = Math.round((s.digit ?? 0) * 100);
    const lead =
      `<b>After ${model.epoch} epochs</b> — both reach ~<b>${train}%</b> on pairs they ` +
      `trained on. On <b>held-out</b> pairs: pure neural ` +
      `<b style="color:${NEURAL_COLOR}">${pct(n.test)}</b>, ` +
      `this method <b style="color:${NEURO_COLOR}">${pct(s.test)}</b>. `;
    switch (category) {
      case "learning-for-reasoning":
        return (
          lead +
          `Perception was trained on <b>digit labels</b> (${dig}% digit-read), so exact ` +
          "symbolic addition carries to unseen pairs — the honest cost is needing those " +
          "digit labels."
        );
      case "reasoning-for-learning":
        return (
          lead +
          `The symbolic rule made perception strong (<b>${dig}%</b> digit-read), but the ` +
          "<b>neural adder runs alone at test</b>, so it still memorizes and slips on unseen " +
          "combinations. Knowledge helped training — not compositional generalization."
        );
      default:
        return (
          lead +
          "The reasoner stays <b>in the loop end-to-end</b>: gradients from the sum taught " +
          `perception (<b>${dig}%</b> digit-read) and the built-in add rule generalizes to ` +
          "pairs it never saw — all from weak labels."
        );
    }
  }

  function refresh(restage: boolean): void {
    const n = model.evalBaseline();
    const s = model.evalApproach();
    epochLabel.textContent = String(model.epoch);

    baseBig.textContent = pct(n.test);
    baseTrain.set(n.train);
    baseTest.set(n.test);

    appBig.textContent = pct(s.test);
    appTrain.set(s.train);
    appTest.set(s.test);
    appDigit.set(s.digit ?? 0);

    const sum = sample[0] + sample[1];
    vizSub.textContent = `digits ${sample[0]} + ${sample[1]} = ${sum}`;
    const seen = model.isTrainPair(sample[0], sample[1]);
    seenBadge.textContent = seen ? "seen during training" : "held-out — never trained on";
    seenBadge.className = "badge " + (seen ? "badge-seen" : "badge-heldout");

    if (restage) {
      activeViz?.destroy();
      activeViz = drawDigitSumViz(vizBox, model, sample);
    } else {
      activeViz?.update(model, sample);
    }

    verdict.innerHTML = verdictText(n, s);
  }

  function frame(): void {
    if (!root.isConnected) {
      teardown();
      return;
    }
    model.step(EPOCHS_PER_FRAME);
    refresh(false);
    if (model.epoch < MAX_EPOCHS) {
      timer = setTimeout(frame, FRAME_DELAY);
    } else {
      stopLoop();
      trainBtn.disabled = false;
      trainBtn.textContent = "Train again ▶";
    }
  }

  function startLoop(): void {
    stopLoop();
    trainBtn.disabled = true;
    trainBtn.textContent = "Training…";
    timer = setTimeout(frame, FRAME_DELAY);
  }

  function restart(): void {
    stopLoop();
    model.reset();
    sample = model.sampleHeldout();
    pickerA.setActive(sample[0]);
    pickerB.setActive(sample[1]);
    refresh(true);
    startLoop();
  }

  trainBtn.onclick = () => {
    if (model.epoch >= MAX_EPOCHS) model.reset();
    startLoop();
  };
  resetBtn.onclick = () => {
    stopLoop();
    model.reset();
    sample = model.sampleHeldout();
    pickerA.setActive(sample[0]);
    pickerB.setActive(sample[1]);
    refresh(true);
    trainBtn.disabled = false;
    trainBtn.textContent = "Train ▶";
  };

  // ---- actions -----------------------------------------------------------
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

  root.append(head, switcher, teach, neuralInfo, controls, grid, vizCard, verdict, actions);

  // Seed the bars + viz, then auto-run so the contrast appears immediately.
  refresh(true);
  startLoop();
}
