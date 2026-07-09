/**
 * Generic differentiable first-order-logic rule engine (product t-norm),
 * independent of any domain — mirrors how nn.ts/network.ts know nothing
 * about digit-sum, and digitSum.ts knows nothing about digitImages.ts.
 *
 * Formulas and the stabilization technique below are verified against the
 * official LTNtorch library (github.com/tommasocarraro/LTNtorch,
 * ltn/fuzzy_ops.py — the reference implementation from the Logic Tensor
 * Networks paper this engine is modeled on), not just hand-derived.
 *
 * Two primitives have hand-derived forward+backward math; everything else
 * is closed-form De Morgan algebra built on top of them:
 *
 *   AND(x1..xn) = Π xi                 ∂AND/∂xi = Π_{j≠i} xj
 *   NOT(x)      = 1 - x                ∂NOT/∂x  = -1
 *   OR(x1..xn)  = 1 - Π(1 - xi)        (De Morgan: NOT(AND(NOT(xi))))
 *   IMPLIES(a,b)= 1 - a + a*b          (De Morgan: OR(NOT(a), b), the
 *                                       standard "Reichenbach" implicator)
 *
 * OR/IMPLIES are implemented via these closed forms directly (not by
 * literally walking a hidden AND/NOT subtree) so each connective gets its
 * own label and satisfaction value for the visualization, while staying
 * backed by a single source of truth for the calculus.
 *
 * Stabilization: a product t-norm has a real vanishing-gradient pathology
 * that LTNtorch explicitly documents — if one AND input saturates to
 * exactly 0 (or one OR input to exactly 1), the gradient to EVERY sibling
 * input vanishes too, not just the culprit's. LTNtorch's fix, applied here
 * with their default eps: before combining, nudge every operand away from
 * the problematic boundary via a tiny affine transform:
 *   π₀(x) = (1-eps)x + eps   — used before AND (keeps x off exactly 0)
 *   π₁(x) = (1-eps)x         — used before OR  (keeps x off exactly 1)
 * IMPLIES(a,b) = 1 - π₀(a) + π₀(a)·π₁(b) — π₀ on the antecedent, π₁ on the
 * consequent, matching LTNtorch's ImpliesReichenbach. Both π₀/π₁ are affine,
 * so the extra chain-rule term is a constant (1-eps) factor on the
 * backprop'd gradient — negligible at eps=1e-4, included for exactness
 * (re-verified against finite differences after adding this).
 *
 * ATOM leaves reference an index into a caller-owned `leaves: number[]`
 * array of truth values in [0,1] — this module never knows what a leaf
 * "means" (e.g. "low toxicity"); the domain model owns that mapping.
 */

const EPS = 1e-4;
/** Nudge x away from exactly 0 — used before AND, matching LTNtorch's π₀. */
function pi0(x: number): number {
  return (1 - EPS) * x + EPS;
}
/** Nudge x away from exactly 1 — used before OR, matching LTNtorch's π₁. */
function pi1(x: number): number {
  return (1 - EPS) * x;
}

export type RuleNode =
  | { kind: "atom"; ref: number; label?: string }
  | { kind: "not"; arg: RuleNode; label?: string }
  | { kind: "and"; args: RuleNode[]; label?: string }
  | { kind: "or"; args: RuleNode[]; label?: string }
  | { kind: "implies"; a: RuleNode; b: RuleNode; label?: string };

/** Ergonomic builders — construction-time only, no eval logic here. */
export const rule = {
  atom: (ref: number, label?: string): RuleNode => ({ kind: "atom", ref, label }),
  not: (arg: RuleNode, label?: string): RuleNode => ({ kind: "not", arg, label }),
  and: (args: RuleNode[], label?: string): RuleNode => ({ kind: "and", args, label }),
  or: (args: RuleNode[], label?: string): RuleNode => ({ kind: "or", args, label }),
  implies: (a: RuleNode, b: RuleNode, label?: string): RuleNode => ({
    kind: "implies",
    a,
    b,
    label,
  }),
};

/** A forward pass, cached as a tree so backward() doesn't need to recompute. */
export interface RuleTrace {
  node: RuleNode;
  value: number;
  children: RuleTrace[];
}

export function evaluate(node: RuleNode, leaves: number[]): RuleTrace {
  switch (node.kind) {
    case "atom":
      return { node, value: leaves[node.ref], children: [] };
    case "not": {
      const c = evaluate(node.arg, leaves);
      return { node, value: 1 - c.value, children: [c] };
    }
    case "and": {
      const children = node.args.map((a) => evaluate(a, leaves));
      const value = children.reduce((p, c) => p * pi0(c.value), 1);
      return { node, value, children };
    }
    case "or": {
      const children = node.args.map((a) => evaluate(a, leaves));
      const value = 1 - children.reduce((p, c) => p * (1 - pi1(c.value)), 1);
      return { node, value, children };
    }
    case "implies": {
      const a = evaluate(node.a, leaves);
      const b = evaluate(node.b, leaves);
      const A = pi0(a.value);
      const B = pi1(b.value);
      return { node, value: 1 - A + A * B, children: [a, b] };
    }
  }
}

/**
 * Backprop dLoss/dOutput through a cached trace, ACCUMULATING into dLeaves
 * (a leaf may be referenced more than once in a tree, so callers must zero
 * dLeaves before a fresh backward pass rather than this function doing it).
 */
export function backward(trace: RuleTrace, dOut: number, dLeaves: number[]): void {
  const { node, children } = trace;
  switch (node.kind) {
    case "atom":
      dLeaves[node.ref] += dOut;
      return;
    case "not":
      backward(children[0], -dOut, dLeaves);
      return;
    case "and": {
      const stab = children.map((c) => pi0(c.value));
      children.forEach((c, i) => {
        let prod = 1;
        stab.forEach((v, j) => {
          if (j !== i) prod *= v;
        });
        backward(c, dOut * prod * (1 - EPS), dLeaves);
      });
      return;
    }
    case "or": {
      const comp = children.map((c) => 1 - pi1(c.value));
      children.forEach((c, i) => {
        let prod = 1;
        comp.forEach((v, j) => {
          if (j !== i) prod *= v;
        });
        backward(c, dOut * prod * (1 - EPS), dLeaves);
      });
      return;
    }
    case "implies": {
      const [a, b] = children;
      const A = pi0(a.value);
      const B = pi1(b.value);
      backward(a, dOut * (B - 1) * (1 - EPS), dLeaves);
      backward(b, dOut * A * (1 - EPS), dLeaves);
      return;
    }
  }
}

/** Numerically stable sigmoid (branches on sign, mirrors softmax's
 * max-subtraction trick elsewhere in this codebase as defensive practice). */
function stableSigmoid(u: number): number {
  return u >= 0 ? 1 / (1 + Math.exp(-u)) : Math.exp(u) / (1 + Math.exp(u));
}

/**
 * Generic differentiable comparator: P(score < cutoff), via a smooth
 * sigmoid threshold — domain-agnostic (any FOL-over-continuous-scores rule
 * wants this), so it lives here rather than in a specific domain model.
 * "score > cutoff" is just NOT(smoothLessThan(score, cutoff, steepness)) —
 * never needs a second sigmoid, and guarantees P(low)+P(high)=1 exactly.
 */
export function smoothLessThan(score: number, cutoff: number, steepness: number): number {
  return stableSigmoid(steepness * (cutoff - score));
}

/** d(smoothLessThan)/d(score), given the already-computed value (avoids
 * recomputing the sigmoid) — standard sigmoid derivative chain. */
export function dSmoothLessThanDScore(value: number, steepness: number): number {
  return -steepness * value * (1 - value);
}
