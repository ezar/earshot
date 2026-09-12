import { describe, expect, it } from 'vitest';
import { Fft, magnitudeBins } from '../src/dsp/fft.js';
import { createRandom } from '../src/util/math.js';

/** Naive O(n^2) DFT magnitude — slow but unambiguously correct. */
function referenceMagnitude(x: Float32Array, size: number): Float64Array {
  const bins = size / 2 + 1;
  const out = new Float64Array(bins);
  for (let k = 0; k < bins; k += 1) {
    let re = 0;
    let im = 0;
    for (let n = 0; n < size; n += 1) {
      const v = n < x.length ? x[n]! : 0;
      const a = (-2 * Math.PI * k * n) / size;
      re += v * Math.cos(a);
      im += v * Math.sin(a);
    }
    out[k] = Math.hypot(re, im);
  }
  return out;
}

describe('Fft', () => {
  it('matches a naive DFT across sizes and signals', () => {
    let worstRelative = 0;
  for (const size of [4, 8, 16, 64, 256, 512]) {
    const fft = new Fft(size);
    const got = new Float32Array(magnitudeBins(size));
    const random = createRandom(size);
    const signals: Record<string, Float32Array> = {
      impulse: Float32Array.from({ length: size }, (_, i) => (i === 0 ? 1 : 0)),
      dc: Float32Array.from({ length: size }, () => 0.5),
      nyquist: Float32Array.from({ length: size }, (_, i) => (i % 2 === 0 ? 1 : -1)),
      binAligned: Float32Array.from({ length: size }, (_, i) => Math.sin((2 * Math.PI * 3 * i) / size)),
      offBin: Float32Array.from({ length: size }, (_, i) => Math.sin((2 * Math.PI * 3.37 * i) / size)),
      noise: Float32Array.from({ length: size }, () => random() * 2 - 1),
      shortInput: Float32Array.from({ length: Math.max(1, size / 2) }, () => random() * 2 - 1),
    };
    for (const [name, signal] of Object.entries(signals)) {
      fft.realMagnitude(signal, got);
      const want = referenceMagnitude(signal, size);
      let peak = 0;
      for (let k = 0; k < want.length; k += 1) peak = Math.max(peak, want[k]!);
      const scale = Math.max(peak, 1e-9);
      for (let k = 0; k < want.length; k += 1) {
        const relative = Math.abs(got[k]! - want[k]!) / scale;
        worstRelative = Math.max(worstRelative, relative);
        expect(relative, `size=${size} ${name} bin ${k}`).toBeLessThan(1e-5);
      }
      }
    }
    // 42 signal/size combinations; the real-input path agrees with the direct
    // transform to within float32 resolution.
    expect(worstRelative).toBeLessThan(1e-6);
  });

  it('rejects sizes it cannot transform', () => {
    expect(() => new Fft(2)).toThrow(/power of two/);
    expect(() => new Fft(100)).toThrow(/power of two/);
  });
});
