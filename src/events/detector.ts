/**
 * Vocalization event detection.
 *
 * The detector fuses two signals: YAMNet class scores, which say *what* is in a
 * window but only at 487 ms resolution, and the amplitude envelope, which says
 * exactly *when* something happened but not what it was. A candidate window
 * opens a search region; the envelope cuts the actual event out of it.
 */

import { HOP_SECONDS, SAMPLE_RATE_HZ, WINDOW_SECONDS } from '../constants.js';
import { trackPitch } from '../dsp/pitch.js';
import { extractFeatures } from '../dsp/features.js';
import { maxScoreOf } from '../models/classifier.js';
import { humanVoiceScore, HUMAN_VOICE_CLASSES } from '../guards/index.js';
import type { ClassScore, PitchTrack, WindowFeatures, WindowResult } from '../util/types.js';
import { segmentBuffer, type Segment, type SegmentOptions } from './segment.js';

/** A detected vocalization. */
export interface VocalEvent {
  /** Event start, in seconds since capture started. */
  readonly start: number;
  /** Event end, in seconds since capture started. */
  readonly end: number;
  /** Event duration, in seconds. */
  readonly duration: number;
  /** Label of the strongest trigger class. */
  readonly type: string;
  /** Class scores of the window that triggered the event. */
  readonly classes: readonly ClassScore[];
  /** Features computed over the event's own audio. */
  readonly features: WindowFeatures;
  /** Pitch contour of the event. */
  readonly pitch: PitchTrack;
  /** Number of amplitude lobes in the event. */
  readonly syllables: number;
  /** Peak level inside the event, in dBFS. */
  readonly peakDbfs: number;
  /** True when a human-voice class was confident in the trigger window. */
  readonly possibleHuman: boolean;
  /** Confidence of the strongest trigger class, in `[0, 1]`. */
  readonly confidence: number;
}

/** Options for {@link createEventDetector}. */
export interface EventDetectorConfig {
  /**
   * Classes that can trigger an event, e.g. `['Cat', 'Meow', 'Caterwaul']`.
   * At least one is required.
   */
  readonly triggerClasses: readonly string[];
  /** Minimum trigger-class score. Defaults to 0.2. */
  readonly triggerScore?: number;
  /** Classes that mark an event as possibly human. Defaults to {@link HUMAN_VOICE_CLASSES}. */
  readonly humanClasses?: readonly string[];
  /** Human-class score above which `possibleHuman` is set. Defaults to 0.3. */
  readonly humanScore?: number;
  /** Minimum gap between two emitted events, in seconds. Defaults to 0.3. */
  readonly debounceSeconds?: number;
  /** Shortest event to emit, in seconds. Defaults to 0.05. */
  readonly minDurationSeconds?: number;
  /** Longest event to emit, in seconds. Defaults to 3. */
  readonly maxDurationSeconds?: number;
  /** Segmentation overrides. */
  readonly segment?: SegmentOptions;
  /** Sample rate of the audio pushed to the detector, in Hz. Defaults to {@link SAMPLE_RATE_HZ}. */
  readonly sampleRateHz?: number;
}

/** A stateful vocalization detector. */
export interface EventDetector {
  /**
   * Feed one analysis window together with the audio it was computed from.
   *
   * @param window - Engine result for the window.
   * @param samples - The window's own audio, at the detector's sample rate.
   * @returns Events that completed with this window; usually empty.
   */
  push(window: WindowResult, samples: Float32Array): VocalEvent[];
  /** Emit any event still pending and clear the detector's state. */
  flush(): VocalEvent[];
  /** Forget all state without emitting. */
  reset(): void;
}

/**
 * Create a vocalization detector.
 *
 * @param config - Trigger classes and timing thresholds.
 * @throws When `triggerClasses` is empty.
 */
export function createEventDetector(config: EventDetectorConfig): EventDetector {
  if (config.triggerClasses.length === 0) {
    throw new Error('earshot: createEventDetector requires at least one trigger class');
  }
  const triggerScore = config.triggerScore ?? 0.2;
  const humanClasses = config.humanClasses ?? HUMAN_VOICE_CLASSES;
  const humanScore = config.humanScore ?? 0.3;
  const debounceSeconds = config.debounceSeconds ?? 0.3;
  const minDurationSeconds = config.minDurationSeconds ?? 0.05;
  const maxDurationSeconds = config.maxDurationSeconds ?? 3;
  const sampleRateHz = config.sampleRateHz ?? SAMPLE_RATE_HZ;

  let lastEmittedEnd = Number.NEGATIVE_INFINITY;
  let pending: VocalEvent | null = null;

  const emit = (event: VocalEvent, out: VocalEvent[]): void => {
    if (event.start - lastEmittedEnd < debounceSeconds) return;
    lastEmittedEnd = event.end;
    out.push(event);
  };

  const detector: EventDetector = {
    push(window: WindowResult, samples: Float32Array): VocalEvent[] {
      const out: VocalEvent[] = [];
      const trigger = strongestTrigger(window.classes, config.triggerClasses);
      if (trigger === null || trigger.score < triggerScore) {
        if (pending !== null) {
          emit(pending, out);
          pending = null;
        }
        return out;
      }

      const segments = segmentBuffer(samples, {
        ...config.segment,
        maxDurationSeconds,
        sampleRateHz,
      });
      const best = strongestSegment(segments);
      if (best === null) return out;

      const start = window.t + best.start;
      const end = Math.min(window.t + best.end, window.t + WINDOW_SECONDS);
      if (end - start < minDurationSeconds) return out;

      const audio = sliceSeconds(samples, best.start, best.end, sampleRateHz);
      const candidate: VocalEvent = {
        start,
        end,
        duration: end - start,
        type: trigger.label,
        classes: window.classes,
        features: extractFeatures(audio, { sampleRateHz }),
        pitch: trackPitch(audio, { sampleRateHz }),
        syllables: best.syllables,
        peakDbfs: best.peakDbfs,
        possibleHuman: maxScoreOf(window.classes, humanClasses) >= humanScore,
        confidence: trigger.score,
      };

      // Consecutive windows overlap by 50 %, so the same call is seen twice.
      // Keep the more confident view and merge the time span.
      if (pending !== null && candidate.start - pending.end < HOP_SECONDS) {
        pending = mergeEvents(pending, candidate, maxDurationSeconds);
        return out;
      }
      if (pending !== null) emit(pending, out);
      pending = candidate;
      return out;
    },
    flush(): VocalEvent[] {
      const out: VocalEvent[] = [];
      if (pending !== null) {
        emit(pending, out);
        pending = null;
      }
      return out;
    },
    reset(): void {
      pending = null;
      lastEmittedEnd = Number.NEGATIVE_INFINITY;
    },
  };
  return detector;
}

function strongestTrigger(classes: readonly ClassScore[], triggers: readonly string[]): ClassScore | null {
  let best: ClassScore | null = null;
  for (const entry of classes) {
    if (!triggers.includes(entry.label)) continue;
    if (best === null || entry.score > best.score) best = entry;
  }
  return best;
}

function strongestSegment(segments: readonly Segment[]): Segment | null {
  let best: Segment | null = null;
  for (const segment of segments) {
    if (best === null || segment.peakDbfs > best.peakDbfs) best = segment;
  }
  return best;
}

function sliceSeconds(samples: Float32Array, from: number, to: number, sampleRateHz: number): Float32Array {
  const start = Math.max(0, Math.floor(from * sampleRateHz));
  const end = Math.min(samples.length, Math.ceil(to * sampleRateHz));
  return samples.slice(start, Math.max(start + 1, end));
}

/**
 * Merge two overlapping views of the same call.
 *
 * The more confident window wins every derived field, because its audio is the
 * one that actually contained the call rather than its tail.
 */
function mergeEvents(a: VocalEvent, b: VocalEvent, maxDurationSeconds: number): VocalEvent {
  const dominant = b.confidence > a.confidence ? b : a;
  const start = Math.min(a.start, b.start);
  const end = Math.min(Math.max(a.end, b.end), start + maxDurationSeconds);
  return {
    ...dominant,
    start,
    end,
    duration: end - start,
    syllables: Math.max(a.syllables, b.syllables),
    peakDbfs: Math.max(a.peakDbfs, b.peakDbfs),
    possibleHuman: a.possibleHuman || b.possibleHuman,
    confidence: Math.max(a.confidence, b.confidence),
  };
}

/**
 * Score how human a detected event looks.
 *
 * @returns The strongest human-voice class score in `[0, 1]`.
 */
export function eventHumanScore(event: VocalEvent, labels: readonly string[] = HUMAN_VOICE_CLASSES): number {
  return humanVoiceScore(event.classes, labels);
}
