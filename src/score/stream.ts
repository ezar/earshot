/**
 * Continuous scoring for SteadyHum's Watch mode.
 *
 * A check is a bounded recording; Watch mode runs for hours. The stream scorer
 * smooths window scores with an exponential moving average so a single lorry
 * driving past does not raise an alarm, and separately watches for *drift*: a
 * slow rise in the baseline that no single window would ever flag.
 */

import type { WindowResult } from '../util/types.js';
import type { Profile } from '../learn/profile.js';
import { mean } from '../util/math.js';
import { scoreWindow, statusFor, type Status, type WindowScore } from './score.js';

/** What the scorer reports after each window. */
export interface StreamUpdate {
  /** The raw window score. */
  readonly window: WindowScore;
  /** Smoothed score in `[0, 1]`. */
  readonly smoothed: number;
  /** Status of the smoothed score. */
  readonly status: Status;
  /** True on the window where the status changed. */
  readonly statusChanged: boolean;
  /** Current drift assessment. */
  readonly drift: DriftState;
}

/** Slow movement of the baseline, independent of momentary spikes. */
export interface DriftState {
  /** True when the recent baseline sits clearly above the older one. */
  readonly drifting: boolean;
  /** Difference between the recent and reference means, in score units. */
  readonly delta: number;
  /** Mean score over the reference (older) window. */
  readonly reference: number;
  /** Mean score over the recent window. */
  readonly recent: number;
}

/** Options for {@link createStreamScorer}. */
export interface StreamScorerOptions {
  /**
   * Smoothing factor of the exponential moving average, in `(0, 1]`.
   * Lower is smoother. Defaults to 0.2, roughly a 5-window memory.
   */
  readonly smoothing?: number;
  /**
   * Consecutive windows the smoothed score must hold a new status before the
   * scorer reports the change. Defaults to 3.
   */
  readonly statusHoldWindows?: number;
  /** Windows in the recent half of the drift comparison. Defaults to 60 (~30 s). */
  readonly driftRecentWindows?: number;
  /** Windows in the reference half of the drift comparison. Defaults to 240 (~2 min). */
  readonly driftReferenceWindows?: number;
  /** Score difference that counts as drift. Defaults to 0.15. */
  readonly driftDelta?: number;
}

/** A stateful scorer for a continuous stream of windows. */
export interface StreamScorer {
  /** Score one window and advance the smoothing and drift state. */
  push(window: WindowResult): StreamUpdate;
  /** Current smoothed score in `[0, 1]`. */
  readonly smoothed: number;
  /** Current reported status. */
  readonly status: Status;
  /** Current drift assessment. */
  readonly drift: DriftState;
  /** Forget all history. */
  reset(): void;
}

const NO_DRIFT: DriftState = { drifting: false, delta: 0, reference: 0, recent: 0 };

/**
 * Create a stream scorer for Watch mode.
 *
 * @param profile - The profile to score against.
 * @param options - Smoothing and drift thresholds.
 */
export function createStreamScorer(profile: Profile, options: StreamScorerOptions = {}): StreamScorer {
  const smoothing = options.smoothing ?? 0.2;
  const statusHoldWindows = options.statusHoldWindows ?? 3;
  const recentWindows = options.driftRecentWindows ?? 60;
  const referenceWindows = options.driftReferenceWindows ?? 240;
  const driftDelta = options.driftDelta ?? 0.15;
  if (smoothing <= 0 || smoothing > 1) {
    throw new Error(`earshot: smoothing must be in (0, 1], received ${smoothing}`);
  }

  const history: number[] = [];
  const capacity = recentWindows + referenceWindows;
  let smoothed = 0;
  let initialized = false;
  let status: Status = 'normal';
  let candidateStatus: Status = 'normal';
  let candidateCount = 0;
  let drift: DriftState = NO_DRIFT;

  return {
    push(window: WindowResult): StreamUpdate {
      const scored = scoreWindow(profile, window);
      smoothed = initialized ? smoothed + smoothing * (scored.score - smoothed) : scored.score;
      initialized = true;

      const observed = statusFor(smoothed, profile);
      let statusChanged = false;
      if (observed === status) {
        candidateStatus = status;
        candidateCount = 0;
      } else if (observed === candidateStatus) {
        candidateCount += 1;
        if (candidateCount >= statusHoldWindows) {
          status = observed;
          candidateCount = 0;
          statusChanged = true;
        }
      } else {
        candidateStatus = observed;
        candidateCount = 1;
        if (statusHoldWindows <= 1) {
          status = observed;
          candidateCount = 0;
          statusChanged = true;
        }
      }

      history.push(scored.score);
      if (history.length > capacity) history.splice(0, history.length - capacity);
      drift = assessDrift(history, recentWindows, referenceWindows, driftDelta);

      return { window: scored, smoothed, status, statusChanged, drift };
    },
    get smoothed(): number {
      return smoothed;
    },
    get status(): Status {
      return status;
    },
    get drift(): DriftState {
      return drift;
    },
    reset(): void {
      history.length = 0;
      smoothed = 0;
      initialized = false;
      status = 'normal';
      candidateStatus = 'normal';
      candidateCount = 0;
      drift = NO_DRIFT;
    },
  };
}

/**
 * Compare the mean score of the most recent windows with the mean of the older
 * reference windows.
 *
 * Drift is only reported once both halves are full, so a scorer that has just
 * started never claims the machine is getting worse.
 */
function assessDrift(
  history: readonly number[],
  recentWindows: number,
  referenceWindows: number,
  driftDelta: number,
): DriftState {
  if (history.length < recentWindows + referenceWindows) return NO_DRIFT;
  const split = history.length - recentWindows;
  const reference = mean(history.slice(split - referenceWindows, split));
  const recent = mean(history.slice(split));
  const delta = recent - reference;
  return { drifting: delta >= driftDelta, delta, reference, recent };
}
