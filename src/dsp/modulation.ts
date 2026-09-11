/**
 * Amplitude-modulation analysis.
 *
 * A failing bearing or an unbalanced fan modulates its own broadband noise at
 * a few hertz; this module reports that rate and its depth.
 */

import { Fft, magnitudeBins } from './fft.js';
import { mean } from '../util/math.js';
import type { Envelope } from './level.js';

/** Dominant amplitude modulation of a signal. */
export interface ModulationEstimate {
  /** Modulation rate, in Hz; 0 when nothing dominates the search range. */
  readonly rateHz: number;
  /** Modulation depth in `[0, 1]`, as the normalized spectral peak. */
  readonly depth: number;
}

/**
 * Find the dominant amplitude-modulation rate of an envelope.
 *
 * @param envelope - Short-time envelope from {@link amplitudeEnvelope}.
 * @param minRateHz - Lowest rate to consider, in Hz. Defaults to 2.
 * @param maxRateHz - Highest rate to consider, in Hz. Defaults to 20.
 */
export function modulationRate(envelope: Envelope, minRateHz = 2, maxRateHz = 20): ModulationEstimate {
  const { levelsDbfs, hopSeconds } = envelope;
  if (levelsDbfs.length < 8 || hopSeconds <= 0) return { rateHz: 0, depth: 0 };

  const envelopeRateHz = 1 / hopSeconds;
  const size = nextPowerOfTwo(levelsDbfs.length);
  const centred = new Float32Array(size);
  const offset = mean(levelsDbfs);
  let energy = 0;
  for (let i = 0; i < levelsDbfs.length; i += 1) {
    const v = (levelsDbfs[i] as number) - offset;
    // Hann taper to stop the envelope's edges from smearing across the spectrum.
    const taper = 0.5 * (1 - Math.cos((2 * Math.PI * i) / levelsDbfs.length));
    centred[i] = v * taper;
    energy += v * v;
  }
  if (energy <= 0) return { rateHz: 0, depth: 0 };

  const fft = new Fft(size);
  const magnitude = new Float32Array(magnitudeBins(size));
  fft.realMagnitude(centred, magnitude);

  let bestBin = -1;
  let bestMagnitude = 0;
  let totalMagnitude = 0;
  for (let b = 1; b < magnitude.length; b += 1) {
    const rateHz = (b * envelopeRateHz) / size;
    if (rateHz < minRateHz || rateHz > maxRateHz) continue;
    const m = magnitude[b] as number;
    totalMagnitude += m;
    if (m > bestMagnitude) {
      bestMagnitude = m;
      bestBin = b;
    }
  }
  if (bestBin < 0 || totalMagnitude <= 0) return { rateHz: 0, depth: 0 };

  return {
    rateHz: (bestBin * envelopeRateHz) / size,
    depth: Math.min(1, bestMagnitude / totalMagnitude),
  };
}

function nextPowerOfTwo(n: number): number {
  let size = 8;
  while (size < n) size <<= 1;
  return size;
}
