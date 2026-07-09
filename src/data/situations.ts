/**
 * The "situations" the user picks from in step 1.
 *
 * Each situation is a small story where a PERCEPTION task (good fit for a neural
 * net) meets a REASONING task (good fit for a symbolic engine). They are chosen
 * so the neural-vs-symbolic trade-off is visible and intuitive, and so each one
 * naturally suggests a different symbolic method (KG / rules / forward-chaining).
 */

import type { SymbolicMethod, StackingPattern } from "../state";

/** A TF-Playground-style 2D dataset the neural block can actually train on. */
export type DatasetKind = "circle" | "xor" | "gauss" | "spiral";

export interface Situation {
  id: string;
  title: string;
  tagline: string;
  /** Emoji used on the situation card. */
  icon: string;
  /** Longer description shown when the card is focused. */
  description: string;

  /** What the neural net is asked to perceive (plain language). */
  perception: string;
  /** What the symbolic engine is asked to reason about (plain language). */
  reasoning: string;

  /** Why a pure neural net struggles here. */
  neuralWeakness: string;
  /** Why a pure symbolic system struggles here. */
  symbolicWeakness: string;
  /** Why combining them wins. */
  combinedStrength: string;

  /** The symbolic method this situation is designed to showcase. */
  suggestedMethod: SymbolicMethod;
  /** The stacking pattern that fits this situation best. */
  suggestedPattern: StackingPattern;
  /** The 2D dataset the neural detail view trains on (keeps it TF-identical). */
  dataset: DatasetKind;
  /** Relative difficulty, 1 (simple) .. 3 (complex). */
  complexity: 1 | 2 | 3;
}

export const SITUATIONS: Situation[] = [
  {
    id: "digit-sum",
    title: "Handwritten Digit Sum",
    tagline: "Read two digits, add them — but only the sum is ever labeled.",
    icon: "\u270D\uFE0F",
    description:
      "Two handwritten digits are shown. The only training signal is their sum " +
      "(never the identity of each digit). The neural net must learn to read " +
      "digits indirectly, while a symbolic 'add' rule combines the readings.",
    perception: "Recognize each handwritten digit from raw strokes.",
    reasoning: "Add the two recognized digits to produce the sum.",
    neuralWeakness:
      "With only the sum as a label, a pure network memorizes seen pairs and " +
      "fails on digit combinations it never saw during training.",
    symbolicWeakness:
      "Addition is trivial symbolically, but logic alone cannot read pixels.",
    combinedStrength:
      "The net learns perception from weak supervision; the symbolic rule " +
      "guarantees correct, generalizable arithmetic on any pair.",
    suggestedMethod: "rules",
    suggestedPattern: "learning-reasoning",
    dataset: "circle",
    complexity: 2,
  },
  {
    id: "drug-discovery",
    title: "Drug Discovery Screening",
    tagline: "A real network reads a noisy signal — real rules explain the verdict.",
    icon: "⚗️",
    description:
      "A synthetic compound is screened for toxicity, binding strength, and " +
      "hydrogen-bond presence. A pretrained network reads these from a noisy " +
      "sensor signal; a Logic Tensor Network-style rule engine reasons over " +
      "the readings to explain — not just predict — whether the compound is " +
      "worth pursuing. (Illustrative synthetic data, not real chemistry.)",
    perception: "Estimate toxicity, binding strength, and hydrogen-bond presence from a noisy reading.",
    reasoning:
      "Check first-order-logic rules: low toxicity AND (strong binding AND " +
      "hydrogen bond present) → recommend.",
    neuralWeakness:
      "A bare probability like “Active = 96%” gives no reason a " +
      "chemist could act on or challenge.",
    symbolicWeakness:
      "The rules are simple once you know toxicity and binding strength, but " +
      "logic alone cannot read a noisy sensor signal.",
    combinedStrength:
      "Perception supplies real readings; the rule engine reports exactly " +
      "which conditions were satisfied or violated — the explanation the " +
      "prediction alone can't give.",
    suggestedMethod: "rules",
    suggestedPattern: "learning-reasoning",
    dataset: "gauss",
    complexity: 2,
  },
];

export function getSituation(id: string | null): Situation | undefined {
  return SITUATIONS.find((s) => s.id === id);
}

