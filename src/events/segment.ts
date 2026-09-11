/**
 * Envelope-based segmentation at 10 ms resolution.
 *
 * YAMNet tells us *that* a window contains a cat; it does not tell us where the
 * call starts and ends. The segmenter works on the amplitude envelope with
 * hysteresis so that a call is cut at its real boundaries rather than at the
 * 487 ms window grid.
 */

import { ENVELOPE_HOP_MS, SAMPLE_RATE_HZ } from '../constants.js';
import { amplitudeEnvelope } from '../dsp/level.js';
import { percentile } from '../util/math.js';

/** A contiguous region of the envelope that is above the noise floor. */
export interface Segment {
  /** Start time, in seconds from the start of the analysed buffer. */
  readonly start: number;
  /** End time, in seconds. */
  readonly end: number;
  /** Peak level inside the segment, in dBFS. */
  readonly peakDbfs: number;
  /** Number of amplitude lobes inside the segment. */
  readonly syllables: number;
}

/** Options for {@link segmentEnvelope}. */
export interface SegmentOptions {
  /** Envelope resolution, in milliseconds. Defaults to {@link ENVELOPE_HOP_MS}. */
  readonly hopMs?: number;
  /** Level above the noise floor at which a segment opens, in dB. Defaults to 12. */
  readonly openDb?: number;
  /** Level above the noise floor at which a segment closes, in dB. Defaults to 6. */
  readonly closeDb?: number;
  /** Segments shorter than this are dropped, in seconds. Defaults to 0.04. */
  readonly minDurationSeconds?: number;
  /** Segments longer than this are truncated, in seconds. Defaults to 5. */
  readonly maxDurationSeconds?: number;
  /** Gaps shorter than this are bridged, in seconds. Defaults to 0.06. */
  readonly bridgeGapSeconds?: number;
  /** A lobe must dip this far below its peak to count as a syllable break, in dB. Defaults to 6. */
  readonly syllableDipDb?: number;
  /** Sample rate of the input, in Hz. Defaults to {@link SAMPLE_RATE_HZ}. */
  readonly sampleRateHz?: number;
}

/**
 * Cut a buffer into segments of sound separated by silence.
 *
 * The noise floor is the 20th percentile of the envelope, which tracks the room
 * rather than assuming a fixed threshold.
 *
 * @param samples - Mono buffer at `sampleRateHz`.
 * @param options - Hysteresis and duration limits.
 */
export function segmentBuffer(samples: Float32Array, options: SegmentOptions = {}): Segment[] {
  const envelope = amplitudeEnvelope(
    samples,
    options.hopMs ?? ENVELOPE_HOP_MS,
    options.sampleRateHz ?? SAMPLE_RATE_HZ,
  );
  return segmentEnvelope(envelope.levelsDbfs, envelope.hopSeconds, options);
}

/**
 * Cut a precomputed envelope into segments.
 *
 * @param levelsDbfs - Per-hop levels, in dBFS.
 * @param hopSeconds - Spacing between levels, in seconds.
 * @param options - Hysteresis and duration limits.
 */
export function segmentEnvelope(
  levelsDbfs: Float32Array,
  hopSeconds: number,
  options: SegmentOptions = {},
): Segment[] {
  const openDb = options.openDb ?? 12;
  const closeDb = options.closeDb ?? 6;
  const minDurationSeconds = options.minDurationSeconds ?? 0.04;
  const maxDurationSeconds = options.maxDurationSeconds ?? 5;
  const bridgeGapSeconds = options.bridgeGapSeconds ?? 0.06;
  const syllableDipDb = options.syllableDipDb ?? 6;
  if (levelsDbfs.length === 0 || hopSeconds <= 0) return [];

  const floor = percentile(levelsDbfs, 20);
  const openLevel = floor + openDb;
  const closeLevel = floor + closeDb;

  const raw: { from: number; to: number }[] = [];
  let start = -1;
  for (let i = 0; i < levelsDbfs.length; i += 1) {
    const level = levelsDbfs[i] as number;
    if (start < 0) {
      if (level >= openLevel) start = i;
    } else if (level < closeLevel) {
      raw.push({ from: start, to: i });
      start = -1;
    }
  }
  if (start >= 0) raw.push({ from: start, to: levelsDbfs.length });

  // Bridge short gaps so a two-lobe call stays one event.
  const bridgeHops = Math.round(bridgeGapSeconds / hopSeconds);
  const merged: { from: number; to: number }[] = [];
  for (const span of raw) {
    const previous = merged[merged.length - 1];
    if (previous !== undefined && span.from - previous.to <= bridgeHops) {
      previous.to = span.to;
    } else {
      merged.push({ ...span });
    }
  }

  const maxHops = Math.max(1, Math.round(maxDurationSeconds / hopSeconds));
  const out: Segment[] = [];
  for (const span of merged) {
    const to = Math.min(span.to, span.from + maxHops);
    const durationSeconds = (to - span.from) * hopSeconds;
    if (durationSeconds < minDurationSeconds) continue;
    let peakDbfs = Number.NEGATIVE_INFINITY;
    for (let i = span.from; i < to; i += 1) {
      const level = levelsDbfs[i] as number;
      if (level > peakDbfs) peakDbfs = level;
    }
    out.push({
      start: span.from * hopSeconds,
      end: to * hopSeconds,
      peakDbfs,
      syllables: countSyllables(levelsDbfs.subarray(span.from, to), syllableDipDb),
    });
  }
  return out;
}

/**
 * Count amplitude lobes inside a segment.
 *
 * A new syllable is counted whenever the envelope dips at least `dipDb` below
 * the running peak and then climbs `dipDb` back up. A segment always has at
 * least one syllable.
 *
 * @param levelsDbfs - Envelope of one segment, in dBFS.
 * @param dipDb - Depth a valley must reach to separate two lobes, in dB.
 */
export function countSyllables(levelsDbfs: ArrayLike<number>, dipDb = 6): number {
  if (levelsDbfs.length === 0) return 0;
  let syllables = 1;
  let peak = levelsDbfs[0] as number;
  let valley = peak;
  let rising = true;
  for (let i = 1; i < levelsDbfs.length; i += 1) {
    const level = levelsDbfs[i] as number;
    if (rising) {
      if (level > peak) peak = level;
      else if (peak - level >= dipDb) {
        rising = false;
        valley = level;
      }
    } else {
      if (level < valley) valley = level;
      else if (level - valley >= dipDb) {
        syllables += 1;
        rising = true;
        peak = level;
      }
    }
  }
  return syllables;
}
