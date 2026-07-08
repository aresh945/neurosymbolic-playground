/**
 * Metadata for the three stacking patterns and the three symbolic methods.
 * The patterns map directly onto Yu et al., "A Survey on Neural-symbolic
 * Learning Systems" (arXiv:2111.08164).
 */

import type { StackingPattern, SymbolicMethod } from "../state";

export interface PatternInfo {
  id: StackingPattern;
  /** Short, human label used on buttons/cards. */
  label: string;
  /** The taxonomy name from the survey. */
  taxonomy: string;
  /** Compact arrow diagram, e.g. "Neural \u2192 Symbolic". */
  flow: string;
  /** One-line summary. */
  summary: string;
  /** Who drives the decision. */
  mainBody: "neural" | "symbolic" | "both";
  /** Trade-off note shown in results. */
  tradeoff: string;
  /** Representative systems from the literature. */
  examples: string[];
}

export const PATTERNS: Record<StackingPattern, PatternInfo> = {
  "learning-for-reasoning": {
    id: "learning-for-reasoning",
    label: "Neural \u2192 Symbolic",
    taxonomy: "Learning for reasoning",
    flow: "Neural \u2192 Symbolic",
    summary:
      "The neural net perceives and abstracts raw data into symbols; the " +
      "symbolic engine does the reasoning. Loosely coupled, runs in sequence.",
    mainBody: "symbolic",
    tradeoff:
      "Highly explainable (the reasoner shows its work) and great at " +
      "systematic generalization, but bounded by the quality of perception.",
    examples: ["pLogicNet", "ExpressGNN", "NS-CL"],
  },
  "reasoning-for-learning": {
    id: "reasoning-for-learning",
    label: "Symbolic \u2192 Neural",
    taxonomy: "Reasoning for learning",
    flow: "Symbolic \u2192 Neural",
    summary:
      "Symbolic knowledge guides or constrains the network's training \u2014 " +
      "as a loss term or transferred knowledge. The network is the main body.",
    mainBody: "neural",
    tradeoff:
      "Strong performance with less data and better consistency, but the " +
      "reasoning is baked into weights, so it is less directly auditable.",
    examples: ["Semantic Loss", "LENSR", "DGP", "KGTN"],
  },
  "learning-reasoning": {
    id: "learning-reasoning",
    label: "Neural \u2194 Symbolic",
    taxonomy: "Learning-reasoning",
    flow: "Neural \u2194 Symbolic",
    summary:
      "Tightly coupled, bidirectional loop: neural output feeds the reasoner " +
      "and the reasoner's signal flows back to train the network, end-to-end.",
    mainBody: "both",
    tradeoff:
      "Best of both \u2014 learns perception from weak supervision and keeps " +
      "crisp reasoning \u2014 but it is the hardest to build and train.",
    examples: ["DeepProbLog", "Abductive Learning (ABL)", "WS-NeSyL"],
  },
};

export const PATTERN_ORDER: StackingPattern[] = [
  "learning-for-reasoning",
  "reasoning-for-learning",
  "learning-reasoning",
];

export interface MethodInfo {
  id: SymbolicMethod;
  label: string;
  icon: string;
  summary: string;
  /** What the engine consumes and produces. */
  io: string;
}

export const METHODS: Record<SymbolicMethod, MethodInfo> = {
  kg: {
    id: "kg",
    label: "Knowledge Graph",
    icon: "\uD83D\uDD78\uFE0F",
    summary:
      "Facts as (subject, relation, object) triples. Reasons by traversing and " +
      "embedding the graph \u2014 good for transitivity and zero-shot transfer.",
    io: "facts \u2192 inferred links",
  },
  rules: {
    id: "rules",
    label: "Rule-Based",
    icon: "\uD83D\uDCD0",
    summary:
      "Explicit IF\u2013THEN rules and arithmetic relations. Deterministic and " +
      "fully auditable \u2014 ideal for crisp operations like addition.",
    io: "facts + rules \u2192 conclusion",
  },
  "forward-chaining": {
    id: "forward-chaining",
    label: "Forward Chaining",
    icon: "\u26D3\uFE0F",
    summary:
      "Repeatedly fires rules whose conditions are met, deriving new facts " +
      "until nothing new appears \u2014 a transparent inference chain.",
    io: "facts \u2192 derived facts (chained)",
  },
};

export const METHOD_ORDER: SymbolicMethod[] = ["kg", "rules", "forward-chaining"];

