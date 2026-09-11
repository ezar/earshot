/** Helpers shared by the earshot test suite. */

import { WINDOW_SAMPLES, SAMPLE_RATE_HZ } from '../src/constants.js';
import { extractFeatures } from '../src/dsp/features.js';
import { frameBuffer } from '../src/dsp/framing.js';
import { createRandom } from '../src/util/math.js';
import type { ClassScore, WindowResult } from '../src/util/types.js';

/**
 * Turn a buffer into engine-shaped {@link WindowResult}s without a Worker or a
 * model, using the real framing and feature extraction.
 *
 * @param samples - Mono audio at {@link SAMPLE_RATE_HZ}.
 * @param embed - Supplies a stand-in embedding per window; omit for none.
 * @param classes - Class scores attached to every window.
 */
export function windowsFrom(
  samples: Float32Array,
  embed?: (features: ReturnType<typeof extractFeatures>, index: number) => number[],
  classes: readonly ClassScore[] = [],
): WindowResult[] {
  return frameBuffer(samples).map((frame, index) => {
    const features = extractFeatures(frame.samples);
    return {
      t: frame.t,
      embedding: embed === undefined ? [] : embed(features, index),
      classes,
      rmsDbfs: features.rmsDbfs,
      features,
    };
  });
}

/** A deterministic unit-length vector, jittered around a per-label centre. */
export function syntheticEmbedding(centre: number, dimensions: number, seed: number, jitter = 0.15): number[] {
  const random = createRandom(seed);
  const out = new Array<number>(dimensions);
  for (let i = 0; i < dimensions; i += 1) {
    out[i] = Math.sin(centre + i * 0.37) + (random() * 2 - 1) * jitter;
  }
  let norm = 0;
  for (const value of out) norm += value * value;
  norm = Math.sqrt(norm) || 1;
  return out.map((value) => value / norm);
}

/** Number of complete windows a buffer of `seconds` produces. */
export function windowCount(seconds: number): number {
  return frameBuffer(new Float32Array(Math.round(seconds * SAMPLE_RATE_HZ))).length;
}

/** Length of one analysis window, in seconds. */
export const WINDOW_SECONDS_EXACT = WINDOW_SAMPLES / SAMPLE_RATE_HZ;
