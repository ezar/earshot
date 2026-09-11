/**
 * YIN-style fundamental-frequency tracker.
 *
 * Meowlogue needs a pitch contour per vocalization, not a single number, so the
 * tracker reports one estimate every {@link PITCH_HOP_MS} and summarizes the
 * voiced frames into a contour slope in semitones per second.
 */

import {
  PITCH_FRAME_MS,
  PITCH_HOP_MS,
  PITCH_MAX_HZ,
  PITCH_MIN_HZ,
  SAMPLE_RATE_HZ,
} from '../constants.js';
import type { PitchFrame, PitchTrack } from '../util/types.js';
import { median, semitones } from '../util/math.js';
import { rmsDbfs } from './level.js';

/** Options for {@link trackPitch}. */
export interface PitchOptions {
  /** Lowest reportable f0, in Hz. Defaults to {@link PITCH_MIN_HZ}. */
  readonly minHz?: number;
  /** Highest reportable f0, in Hz. Defaults to {@link PITCH_MAX_HZ}. */
  readonly maxHz?: number;
  /** Analysis frame length, in milliseconds. Defaults to {@link PITCH_FRAME_MS}. */
  readonly frameMs?: number;
  /** Analysis hop, in milliseconds. Defaults to {@link PITCH_HOP_MS}. */
  readonly hopMs?: number;
  /**
   * YIN absolute threshold on the cumulative mean normalized difference.
   * Lower values are stricter about periodicity. Defaults to 0.15.
   */
  readonly threshold?: number;
  /** Frames quieter than this are forced unvoiced, in dBFS. Defaults to -60. */
  readonly silenceDbfs?: number;
  /** Sample rate of the input, in Hz. Defaults to {@link SAMPLE_RATE_HZ}. */
  readonly sampleRateHz?: number;
}

/**
 * Estimate the fundamental frequency of one frame with the YIN algorithm.
 *
 * The estimate is reported only when the frame is periodic *inside* the search
 * range: a frame whose true fundamental sits above `maxHz` is rejected rather
 * than reported as one of its subharmonics.
 *
 * @param frame - Mono samples, at least `2 * sampleRateHz / minHz` long for a reliable estimate.
 * @param options - Search range and threshold.
 * @returns The f0 in Hz and a confidence in `[0, 1]`; f0 is 0 when unvoiced.
 */
export function yin(
  frame: Float32Array,
  options: Pick<PitchOptions, 'minHz' | 'maxHz' | 'threshold' | 'sampleRateHz'> = {},
): { f0Hz: number; confidence: number } {
  const sampleRateHz = options.sampleRateHz ?? SAMPLE_RATE_HZ;
  const minHz = options.minHz ?? PITCH_MIN_HZ;
  const maxHz = options.maxHz ?? PITCH_MAX_HZ;
  const threshold = options.threshold ?? 0.15;

  const minLag = Math.max(2, Math.floor(sampleRateHz / maxHz));
  const maxLag = Math.min(Math.floor(frame.length / 2), Math.ceil(sampleRateHz / minHz));
  if (maxLag <= minLag) return { f0Hz: 0, confidence: 0 };

  // Step 1-2: squared difference function.
  const difference = new Float32Array(maxLag + 1);
  for (let lag = 1; lag <= maxLag; lag += 1) {
    let sum = 0;
    const count = frame.length - lag;
    for (let i = 0; i < count; i += 1) {
      const d = (frame[i] as number) - (frame[i + lag] as number);
      sum += d * d;
    }
    difference[lag] = sum;
  }

  // Step 3: cumulative mean normalized difference.
  const normalized = new Float32Array(maxLag + 1);
  normalized[0] = 1;
  let runningSum = 0;
  for (let lag = 1; lag <= maxLag; lag += 1) {
    runningSum += difference[lag] as number;
    normalized[lag] = runningSum > 0 ? ((difference[lag] as number) * lag) / runningSum : 1;
  }

  // A signal whose true period is shorter than `minLag` is also periodic at
  // every multiple of that period, so a lag-limited search would confidently
  // report a subharmonic: a 2 kHz kettle would come back as a 1 kHz call. If
  // any out-of-range lag explains the frame at least as well as the best
  // in-range one, the fundamental is above `maxHz` and the frame is not ours.
  let outOfRangeBest = Number.POSITIVE_INFINITY;
  for (let lag = 2; lag < minLag; lag += 1) {
    const value = normalized[lag] as number;
    if (value < outOfRangeBest) outOfRangeBest = value;
  }

  // Step 4: absolute threshold, falling back to the global minimum.
  let bestLag = -1;
  for (let lag = minLag; lag <= maxLag; lag += 1) {
    if ((normalized[lag] as number) < threshold) {
      while (lag + 1 <= maxLag && (normalized[lag + 1] as number) < (normalized[lag] as number)) lag += 1;
      bestLag = lag;
      break;
    }
  }
  if (bestLag < 0) {
    let bestValue = Number.POSITIVE_INFINITY;
    for (let lag = minLag; lag <= maxLag; lag += 1) {
      const value = normalized[lag] as number;
      if (value < bestValue) {
        bestValue = value;
        bestLag = lag;
      }
    }
    if (bestLag < 0 || bestValue > 0.6) return { f0Hz: 0, confidence: 0 };
  }

  if (outOfRangeBest <= (normalized[bestLag] as number) && outOfRangeBest < threshold) {
    return { f0Hz: 0, confidence: 0 };
  }

  // Step 5: parabolic interpolation around the chosen lag.
  const refinedLag = parabolicMinimum(normalized, bestLag, minLag, maxLag);
  const f0Hz = sampleRateHz / refinedLag;
  if (f0Hz < minHz || f0Hz > maxHz) return { f0Hz: 0, confidence: 0 };
  const confidence = Math.max(0, Math.min(1, 1 - (normalized[bestLag] as number)));
  return { f0Hz, confidence };
}

function parabolicMinimum(values: Float32Array, index: number, minIndex: number, maxIndex: number): number {
  if (index <= minIndex || index >= maxIndex) return index;
  const left = values[index - 1] as number;
  const centre = values[index] as number;
  const right = values[index + 1] as number;
  const denominator = 2 * (2 * centre - left - right);
  if (denominator === 0) return index;
  const offset = (right - left) / denominator;
  return index + Math.max(-1, Math.min(1, offset));
}

/**
 * Run the YIN tracker over a whole buffer.
 *
 * @param pcm16k - Mono samples at {@link SAMPLE_RATE_HZ} unless `options.sampleRateHz` says otherwise.
 * @param options - Search range, framing and voicing thresholds.
 * @returns Per-frame estimates plus contour summaries.
 */
export function trackPitch(pcm16k: Float32Array, options: PitchOptions = {}): PitchTrack {
  const sampleRateHz = options.sampleRateHz ?? SAMPLE_RATE_HZ;
  const frameMs = options.frameMs ?? PITCH_FRAME_MS;
  const hopMs = options.hopMs ?? PITCH_HOP_MS;
  const silenceDbfs = options.silenceDbfs ?? -60;

  // YIN needs at least two periods of the lowest searched pitch in view, so the
  // analysis frame is widened beyond `frameMs` when the range demands it.
  const minHz = options.minHz ?? PITCH_MIN_HZ;
  const requestedFrame = Math.round((frameMs / 1000) * sampleRateHz);
  const frameSamples = Math.max(requestedFrame, Math.ceil((2 * sampleRateHz) / minHz) + 2);
  const hopSamples = Math.max(1, Math.round((hopMs / 1000) * sampleRateHz));

  const frames: PitchFrame[] = [];
  for (let offset = 0; offset + frameSamples <= pcm16k.length; offset += hopSamples) {
    const slice = pcm16k.subarray(offset, offset + frameSamples);
    const level = rmsDbfs(slice);
    const t = (offset + frameSamples / 2) / sampleRateHz;
    if (level < silenceDbfs) {
      frames.push({ t, f0Hz: 0, confidence: 0, rmsDbfs: level });
      continue;
    }
    const estimate = yin(slice, {
      minHz,
      maxHz: options.maxHz ?? PITCH_MAX_HZ,
      threshold: options.threshold ?? 0.15,
      sampleRateHz,
    });
    frames.push({ t, f0Hz: estimate.f0Hz, confidence: estimate.confidence, rmsDbfs: level });
  }

  return summarize(frames);
}

/** Derive the contour summaries of a {@link PitchTrack} from its frames. */
export function summarize(frames: readonly PitchFrame[]): PitchTrack {
  const voiced = frames.filter((frame) => frame.f0Hz > 0);
  if (voiced.length === 0) {
    return {
      frames,
      medianF0Hz: 0,
      voicedFraction: 0,
      minF0Hz: 0,
      maxF0Hz: 0,
      contourSlopeSemitonesPerSecond: 0,
    };
  }
  const f0Values = voiced.map((frame) => frame.f0Hz);
  let minF0Hz = Number.POSITIVE_INFINITY;
  let maxF0Hz = 0;
  for (const f0 of f0Values) {
    if (f0 < minF0Hz) minF0Hz = f0;
    if (f0 > maxF0Hz) maxF0Hz = f0;
  }
  return {
    frames,
    medianF0Hz: median(f0Values),
    voicedFraction: frames.length === 0 ? 0 : voiced.length / frames.length,
    minF0Hz,
    maxF0Hz,
    contourSlopeSemitonesPerSecond: contourSlope(voiced),
  };
}

/**
 * Least-squares slope of the voiced f0 contour, in semitones per second.
 *
 * The fit is done in log-frequency so that a rise from 300 to 600 Hz counts the
 * same as one from 600 to 1200 Hz, which is how a listener hears it.
 */
function contourSlope(voiced: readonly PitchFrame[]): number {
  if (voiced.length < 2) return 0;
  const reference = voiced[0]?.f0Hz ?? 0;
  if (reference <= 0) return 0;
  let sumT = 0;
  let sumY = 0;
  let sumTT = 0;
  let sumTY = 0;
  for (const frame of voiced) {
    const y = semitones(reference, frame.f0Hz);
    sumT += frame.t;
    sumY += y;
    sumTT += frame.t * frame.t;
    sumTY += frame.t * y;
  }
  const n = voiced.length;
  const denominator = n * sumTT - sumT * sumT;
  if (Math.abs(denominator) < 1e-12) return 0;
  return (n * sumTY - sumT * sumY) / denominator;
}
