/**
 * Loads the 10 user-supplied reference digit images (public/mnist/0.png..9.png)
 * and decodes each to a small, zero-mean-normalized grayscale pixel vector for
 * DigitSumModel's perception layer to train on. DigitSumModel itself never
 * touches the DOM — this module is the only place that does, mirroring the
 * existing split between data (dataset.ts) and compute (network.ts/nn.ts).
 */

/** Training resolution. Small on purpose — keeps perception a cheap linear
 * layer and training fast; thumbnails shown to the user are full-resolution
 * (see digitThumbnailSrc), this only affects what the network reads. */
const RES = 16;
export const DIGIT_PIXELS = RES * RES;

export function digitThumbnailSrc(d: number): string {
  return `/mnist/${d}.png`;
}

function loadOne(d: number): Promise<number[]> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = RES;
      canvas.height = RES;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        reject(new Error("2D canvas context unavailable"));
        return;
      }
      ctx.drawImage(img, 0, 0, RES, RES);
      const { data } = ctx.getImageData(0, 0, RES, RES);
      const px = new Array<number>(DIGIT_PIXELS);
      for (let i = 0; i < DIGIT_PIXELS; i++) {
        const o = i * 4;
        px[i] = (data[o] + data[o + 1] + data[o + 2]) / (3 * 255);
      }
      const mean = px.reduce((a, b) => a + b, 0) / px.length;
      for (let i = 0; i < DIGIT_PIXELS; i++) px[i] -= mean;
      resolve(px);
    };
    img.onerror = () => reject(new Error(`Couldn't load ${digitThumbnailSrc(d)}`));
    img.src = digitThumbnailSrc(d);
  });
}

let cache: Promise<number[][]> | null = null;

/** Load + decode all 10 reference digits once; memoized across remounts. */
export function preloadDigitImages(): Promise<number[][]> {
  if (!cache) cache = Promise.all(Array.from({ length: 10 }, (_, d) => loadOne(d)));
  return cache;
}
