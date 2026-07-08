/**
 * 2D toy datasets, faithful in spirit to TensorFlow Playground's dataset.ts.
 * Points carry a label in {-1, +1}. Used by the neural detail view so the
 * "expand the neural block" experience matches the original tool.
 */

export interface Point2D {
  x: number;
  y: number;
  label: number; // -1 or +1
}

export type DatasetKind = "circle" | "xor" | "gauss" | "spiral";

const DOMAIN = 6; // points live in roughly [-6, 6] on each axis

function randUniform(a: number, b: number): number {
  return Math.random() * (b - a) + a;
}

/** Gaussian-ish noise via Box–Muller. */
function randNormal(mean = 0, variance = 1): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  const n = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  return mean + n * Math.sqrt(variance);
}

function dist(a: { x: number; y: number }, b: { x: number; y: number }): number {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

export function classifyCircle(n: number, noise: number): Point2D[] {
  const points: Point2D[] = [];
  const radius = DOMAIN;
  function make(inner: boolean) {
    const r = inner
      ? randUniform(0, radius * 0.5)
      : randUniform(radius * 0.7, radius);
    const angle = randUniform(0, 2 * Math.PI);
    const x = r * Math.sin(angle);
    const y = r * Math.cos(angle);
    const nx = randUniform(-radius, radius) * noise;
    const ny = randUniform(-radius, radius) * noise;
    const label = dist({ x: x + nx, y: y + ny }, { x: 0, y: 0 }) < radius * 0.5 ? 1 : -1;
    points.push({ x, y, label });
  }
  for (let i = 0; i < n / 2; i++) make(true);
  for (let i = 0; i < n / 2; i++) make(false);
  return points;
}

export function classifyXOR(n: number, noise: number): Point2D[] {
  const points: Point2D[] = [];
  for (let i = 0; i < n; i++) {
    let x = randUniform(-DOMAIN, DOMAIN);
    let y = randUniform(-DOMAIN, DOMAIN);
    const padding = 0.3;
    x += x > 0 ? padding : -padding;
    y += y > 0 ? padding : -padding;
    const nx = randUniform(-DOMAIN, DOMAIN) * noise;
    const ny = randUniform(-DOMAIN, DOMAIN) * noise;
    const label = (x + nx) * (y + ny) >= 0 ? 1 : -1;
    points.push({ x, y, label });
  }
  return points;
}

export function classifyGauss(n: number, noise: number): Point2D[] {
  const points: Point2D[] = [];
  const variance = 0.5 + noise * 3;
  function make(cx: number, cy: number, label: number) {
    const x = randNormal(cx, variance);
    const y = randNormal(cy, variance);
    points.push({ x, y, label });
  }
  for (let i = 0; i < n / 2; i++) make(2, 2, 1);
  for (let i = 0; i < n / 2; i++) make(-2, -2, -1);
  return points;
}

export function classifySpiral(n: number, noise: number): Point2D[] {
  const points: Point2D[] = [];
  const half = n / 2;
  function spiral(deltaT: number, label: number) {
    for (let i = 0; i < half; i++) {
      const r = (i / half) * DOMAIN;
      const t = ((1.75 * i) / half) * 2 * Math.PI + deltaT;
      const x = r * Math.sin(t) + randUniform(-1, 1) * noise;
      const y = r * Math.cos(t) + randUniform(-1, 1) * noise;
      points.push({ x, y, label });
    }
  }
  spiral(0, 1);
  spiral(Math.PI, -1);
  return points;
}

export function makeDataset(
  kind: DatasetKind,
  n = 200,
  noise = 0.1
): Point2D[] {
  switch (kind) {
    case "circle":
      return classifyCircle(n, noise);
    case "xor":
      return classifyXOR(n, noise);
    case "gauss":
      return classifyGauss(n, noise);
    case "spiral":
      return classifySpiral(n, noise);
  }
}

export const DOMAIN_EXTENT = DOMAIN;

