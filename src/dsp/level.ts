/** Level measurement: RMS in dBFS and the short-time amplitude envelope. */

import { DBFS_FLOOR, ENVELOPE_HOP_MS, SAMPLE_RATE_HZ, SILENCE_EPSILON } from '../constants.js';

/**
 * Root-mean-square level of a buffer, in dBFS.
 *
 * @param samples - Mono samples in `[-1, 1]`.
 * @returns Level in dBFS, floored at {@link DBFS_FLOOR}.
 */
export function rmsDbfs(samples: ArrayLike<number>): number {
  if (samples.length === 0) return DBFS_FLOOR;
  let sum = 0;
  for (let i = 0; i < samples.length; i += 1) {
    const v = samples[i] as number;
    sum += v * v;
  }
  const rms = Math.sqrt(sum / samples.length);
  if (rms < SILENCE_EPSILON) return DBFS_FLOOR;
  return Math.max(DBFS_FLOOR, 20 * Math.log10(rms));
}

/** Peak absolute amplitude of a buffer, in dBFS. */
export function peakDbfs(samples: ArrayLike<number>): number {
  let peak = 0;
  for (let i = 0; i < samples.length; i += 1) {
    const v = Math.abs(samples[i] as number);
    if (v > peak) peak = v;
  }
  if (peak < SILENCE_EPSILON) return DBFS_FLOOR;
  return Math.max(DBFS_FLOOR, 20 * Math.log10(peak));
}

/** A short-time amplitude envelope sampled on a regular grid. */
export interface Envelope {
  /** Per-hop RMS levels, in dBFS. */
  readonly levelsDbfs: Float32Array;
  /** Spacing between consecutive values, in seconds. */
  readonly hopSeconds: number;
}

/**
 * Compute the short-time RMS envelope used by the event segmenter.
 *
 * @param samples - Mono samples at `sampleRateHz`.
 * @param hopMs - Envelope resolution, in milliseconds. Defaults to {@link ENVELOPE_HOP_MS}.
 * @param sampleRateHz - Sample rate of `samples`, in Hz.
 */
export function amplitudeEnvelope(
  samples: Float32Array,
  hopMs: number = ENVELOPE_HOP_MS,
  sampleRateHz: number = SAMPLE_RATE_HZ,
): Envelope {
  const hop = Math.max(1, Math.round((hopMs / 1000) * sampleRateHz));
  const count = Math.max(0, Math.floor(samples.length / hop));
  const levelsDbfs = new Float32Array(count);
  for (let i = 0; i < count; i += 1) {
    levelsDbfs[i] = rmsDbfs(samples.subarray(i * hop, i * hop + hop));
  }
  return { levelsDbfs, hopSeconds: hop / sampleRateHz };
}
