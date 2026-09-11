/**
 * Onset detection from spectral flux, and the periodicity of the resulting
 * onset sequence — the signal SteadyHum uses to tell a steady motor from a
 * rhythmic knock.
 */

import { median } from '../util/math.js';
import type { Spectrogram } from './spectrum.js';

/**
 * Magnitude treated as silence when compressing the spectrum, corresponding to
 * -120 dBFS on the normalized scale {@link computeSpectrogram} produces.
 */
const MAGNITUDE_FLOOR = 1e-6;

/** An onset-strength curve sampled at the spectrogram's hop. */
export interface OnsetCurve {
  /** Half-wave-rectified spectral flux per frame. */
  readonly strength: Float32Array;
  /** Spacing between values, in seconds. */
  readonly hopSeconds: number;
}

/**
 * Half-wave-rectified spectral flux between consecutive frames.
 *
 * Magnitudes are compressed logarithmically before differencing so that the
 * curve responds to relative rather than absolute level changes. The floor is
 * -120 dBFS rather than an arbitrarily small epsilon: without it, bins that
 * hold nothing but numerical noise swing by orders of magnitude in the log
 * domain and a perfectly steady tone produces as much apparent flux as a knock.
 */
export function spectralFlux(spectrogram: Spectrogram): OnsetCurve {
  const { data, frames, bins, hopSeconds } = spectrogram;
  const strength = new Float32Array(frames);
  for (let f = 1; f < frames; f += 1) {
    let acc = 0;
    for (let b = 1; b < bins; b += 1) {
      const current = Math.log10((data[f * bins + b] as number) + MAGNITUDE_FLOOR);
      const previous = Math.log10((data[(f - 1) * bins + b] as number) + MAGNITUDE_FLOOR);
      const diff = current - previous;
      if (diff > 0) acc += diff;
    }
    strength[f] = acc / Math.max(1, bins - 1);
  }
  return { strength, hopSeconds };
}

/** Options for {@link detectOnsets}. */
export interface OnsetOptions {
  /**
   * Threshold above the local median flux, as a multiple of the local mean
   * absolute deviation. Defaults to 1.5.
   */
  readonly deltaFactor?: number;
  /** Half-width of the adaptive threshold window, in frames. Defaults to 8. */
  readonly medianRadiusFrames?: number;
  /** Minimum spacing between onsets, in seconds. Defaults to 0.05. */
  readonly minIntervalSeconds?: number;
  /**
   * Absolute floor on the onset strength, in mean log10 magnitude change per
   * bin. Defaults to 0.05, roughly 1 dB of mean per-bin change.
   *
   * The adaptive threshold alone is purely relative, so without a floor the
   * numerical wobble of a perfectly steady tone looks exactly like a sequence
   * of onsets. Because the curve is a log difference, this floor is invariant
   * to the recording's gain.
   */
  readonly minStrength?: number;
}

/**
 * Pick onsets from an onset-strength curve with an adaptive median threshold.
 *
 * @returns Onset times, in seconds from the start of the analysed buffer.
 */
export function detectOnsets(curve: OnsetCurve, options: OnsetOptions = {}): number[] {
  const deltaFactor = options.deltaFactor ?? 1.5;
  const radius = options.medianRadiusFrames ?? 8;
  const minIntervalSeconds = options.minIntervalSeconds ?? 0.05;
  const minStrength = options.minStrength ?? 0.05;
  const { strength, hopSeconds } = curve;

  const onsets: number[] = [];
  const window: number[] = [];
  let lastOnsetTime = Number.NEGATIVE_INFINITY;
  for (let f = 1; f + 1 < strength.length; f += 1) {
    const here = strength[f] as number;
    if (here < minStrength) continue;
    if (here <= (strength[f - 1] as number) || here < (strength[f + 1] as number)) continue;
    window.length = 0;
    const from = Math.max(0, f - radius);
    const to = Math.min(strength.length, f + radius + 1);
    for (let j = from; j < to; j += 1) window.push(strength[j] as number);
    const localMedian = median(window);
    let deviation = 0;
    for (const value of window) deviation += Math.abs(value - localMedian);
    deviation /= window.length;
    if (here < localMedian + deltaFactor * deviation) continue;
    const time = f * hopSeconds;
    if (time - lastOnsetTime < minIntervalSeconds) continue;
    onsets.push(time);
    lastOnsetTime = time;
  }
  return onsets;
}

/** How regular an onset sequence is. */
export interface OnsetPeriodicity {
  /** Strength of the dominant period in `[0, 1]`; 0 when undecidable. */
  readonly strength: number;
  /** The dominant period, in seconds; 0 when undecidable. */
  readonly periodSeconds: number;
}

/**
 * Measure how regular a sequence of onsets is.
 *
 * The strength is the resultant length of the onset times wrapped onto a
 * candidate period — the standard circular-statistics measure of phase
 * coherence. Perfectly even knocks score 1; onsets scattered at random score
 * around `1 / sqrt(n)`, which decays as evidence accumulates. A plain
 * coefficient of variation over inter-onset intervals would instead score a
 * burst of noise around 0.6, because the detector's own minimum spacing makes
 * random intervals look more even than they are.
 *
 * The period is sought around the median interval, so a sequence that misses
 * the occasional onset still resolves to the underlying period rather than to
 * twice it.
 *
 * @param onsets - Onset times, in seconds, ascending.
 * @returns The dominant period and how strongly the onsets adhere to it.
 */
export function onsetPeriodicity(onsets: readonly number[]): OnsetPeriodicity {
  if (onsets.length < 3) return { strength: 0, periodSeconds: 0 };
  const intervals: number[] = [];
  for (let i = 1; i < onsets.length; i += 1) {
    intervals.push((onsets[i] as number) - (onsets[i - 1] as number));
  }
  const seed = median(intervals);
  if (seed <= 0) return { strength: 0, periodSeconds: 0 };

  let bestStrength = 0;
  let bestPeriod = seed;
  // Search +/- 40 % around the median interval in 1 % steps. A wider search
  // would start locking onto harmonics of the true period.
  for (let step = -40; step <= 40; step += 1) {
    const period = seed * (1 + step / 100);
    if (period <= 0) continue;
    let sumCos = 0;
    let sumSin = 0;
    for (const onset of onsets) {
      const phase = (2 * Math.PI * onset) / period;
      sumCos += Math.cos(phase);
      sumSin += Math.sin(phase);
    }
    const strength = Math.sqrt(sumCos * sumCos + sumSin * sumSin) / onsets.length;
    if (strength > bestStrength) {
      bestStrength = strength;
      bestPeriod = period;
    }
  }
  return { strength: Math.min(1, bestStrength), periodSeconds: bestPeriod };
}
