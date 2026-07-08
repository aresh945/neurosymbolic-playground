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
    id: "traffic-triage",
    title: "Traffic-Sign Triage",
    tagline: "Perceive a sign, then derive the correct action by rule.",
    icon: "\uD83D\uDEA6",
    description:
      "A camera sees a road sign under glare, rain, or odd angles. The neural " +
      "net perceives shape and color; a forward-chaining rule base derives the " +
      "required driver action, including for rare signs never seen in training.",
    perception: "Estimate the sign's shape and dominant color from pixels.",
    reasoning:
      "Chain road rules: octagon \u2192 STOP; red triangle \u2192 WARNING; " +
      "red circle \u2192 PROHIBITION.",
    neuralWeakness:
      "Rare or unseen signs are misclassified; the net has no notion of the " +
      "rulebook, so it cannot justify its decision.",
    symbolicWeakness:
      "Hand-written rules cannot cope with glare, motion blur, or worn paint.",
    combinedStrength:
      "Robust perception feeds crisp, auditable rules \u2014 rare signs are " +
      "handled by logic, messy pixels by the network.",
    suggestedMethod: "forward-chaining",
    suggestedPattern: "learning-for-reasoning",
    dataset: "xor",
    complexity: 3,
  },
  {
    id: "ancestry",
    title: "Family-Tree Reasoning",
    tagline: "Extract 'parent' links, then infer ancestors by transitivity.",
    icon: "\uD83C\uDF33",
    description:
      "From noisy records the neural net extracts direct 'parent-of' relations. " +
      "A knowledge graph then infers grandparent, ancestor, and cousin links by " +
      "following chains the network alone could never count.",
    perception: "Detect direct parent-of relations from noisy records.",
    reasoning:
      "Apply transitive rules over the knowledge graph: parent+parent \u2192 " +
      "grandparent; repeated parent \u2192 ancestor.",
    neuralWeakness:
      "Networks struggle to count multi-hop relations and generalize chains " +
      "to depths not seen in training.",
    symbolicWeakness:
      "Perfect at transitivity, but cannot read messy, inconsistent records.",
    combinedStrength:
      "The net supplies clean facts; the graph reasons over them to any depth " +
      "with full explainability.",
    suggestedMethod: "kg",
    suggestedPattern: "learning-for-reasoning",
    dataset: "gauss",
    complexity: 2,
  },
  {
    id: "zero-shot-animals",
    title: "Zero-Shot Animal ID",
    tagline: "Perceive attributes, classify species never seen in training.",
    icon: "\uD83E\uDD93",
    description:
      "The neural net perceives attributes (stripes, hooves, mane). A taxonomy " +
      "knowledge graph maps attribute patterns to species \u2014 including " +
      "animals with zero training images, using only their described traits.",
    perception: "Detect visual attributes: stripes, wings, hooves, size, color.",
    reasoning:
      "Use the taxonomy graph to map attribute sets to species, even unseen ones.",
    neuralWeakness:
      "A classifier cannot name a class it never saw a single example of.",
    symbolicWeakness:
      "The taxonomy is useless without reliable attribute detection from images.",
    combinedStrength:
      "Attribute perception + taxonomy transfer = recognition of brand-new " +
      "species from their description alone (zero-shot).",
    suggestedMethod: "kg",
    suggestedPattern: "reasoning-for-learning",
    dataset: "spiral",
    complexity: 3,
  },
];

export function getSituation(id: string | null): Situation | undefined {
  return SITUATIONS.find((s) => s.id === id);
}

