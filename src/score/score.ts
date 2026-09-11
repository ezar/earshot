/**
 * Scoring a check against a learned profile.
 *
 * Every window is matched to its nearest profile state and scored by where its
 * distance falls in that state's learned distance distribution. A check score
 * is the 90th percentile of the window scores: a machine that sounds wrong for
 * a couple of seconds out of twenty should register, but one stray window
 * should not.
 */

import { percentile } from '../util/math.js';
import type { WindowResult } from '../util/types.js';
import { toVector, type Profile, type ProfileState } from '../learn/profile.js';
import { distanceScore, mahalanobisDistance } from '../learn/statistics.js';

/** How a check compares with the profile. */
export type Status = 'normal' | 'watch' | 'anomalous';

/** The score of one analysis window against a profile. */
export interface WindowScore {
  /** Window start time, in seconds. */
  readonly t: number;
  /** Anomaly score in `[0, 1]`; 0 is squarely inside the profile. */
  readonly score: number;
  /** Id of the best-matching state. */
  readonly stateId: string;
  /** Mahalanobis distance to that state, in standard deviations. */
  readonly distance: number;
}

/** The result of scoring a whole check. */
export interface CheckResult {
  /** Aggregate anomaly score in `[0, 1]`. */
  readonly score: number;
  /** Status derived from the profile's thresholds. */
  readonly status: Status;
  /** Per-window scores in input order. */
  readonly windows: readonly WindowScore[];
  /** Share of windows at or above the `anomalous` threshold, in `[0, 1]`. */
  readonly anomalousFraction: number;
  /** Id of the state that matched most windows; empty when there were none. */
  readonly dominantStateId: string;
  /** Profile revision the check was scored against. */
  readonly profileRevision: number;
}

/** Options for {@link scoreCheck}. */
export interface ScoreOptions {
  /** Percentile of window scores used as the check score. Defaults to 90. */
  readonly aggregatePercentile?: number;
}

/**
 * Match one window to its nearest state and score it.
 *
 * @param profile - A profile from {@link learnProfile}.
 * @param window - A guard-accepted window.
 */
export function scoreWindow(profile: Profile, window: WindowResult): WindowScore {
  const vector = toVector(window, profile.featureSpace);
  let bestScore = 1;
  let bestDistance = Number.POSITIVE_INFINITY;
  let bestState = '';
  for (const state of profile.states) {
    const distance = mahalanobisDistance(vector, state.model);
    const score = distanceScore(distance, state.distances);
    if (score < bestScore || bestState === '') {
      bestScore = score;
      bestDistance = distance;
      bestState = state.id;
    }
  }
  return {
    t: window.t,
    score: bestScore,
    stateId: bestState,
    distance: Number.isFinite(bestDistance) ? bestDistance : 0,
  };
}

/**
 * Score a whole check.
 *
 * @param profile - A profile from {@link learnProfile}.
 * @param windows - The check's guard-accepted windows.
 * @returns Per-window and aggregate scores plus a status.
 */
export function scoreCheck(profile: Profile, windows: readonly WindowResult[], options: ScoreOptions = {}): CheckResult {
  const aggregatePercentile = options.aggregatePercentile ?? 90;
  const scored = windows.map((window) => scoreWindow(profile, window));
  const score = scored.length === 0 ? 0 : percentile(scored.map((entry) => entry.score), aggregatePercentile);

  const counts = new Map<string, number>();
  let anomalous = 0;
  for (const entry of scored) {
    counts.set(entry.stateId, (counts.get(entry.stateId) ?? 0) + 1);
    if (entry.score >= profile.thresholds.anomalous) anomalous += 1;
  }
  let dominantStateId = '';
  let dominantCount = 0;
  for (const [stateId, count] of counts) {
    if (count > dominantCount) {
      dominantCount = count;
      dominantStateId = stateId;
    }
  }

  return {
    score,
    status: statusFor(score, profile),
    windows: scored,
    anomalousFraction: scored.length === 0 ? 0 : anomalous / scored.length,
    dominantStateId,
    profileRevision: profile.revision,
  };
}

/** Map an aggregate score onto a status using a profile's thresholds. */
export function statusFor(score: number, profile: Profile): Status {
  if (score >= profile.thresholds.anomalous) return 'anomalous';
  if (score >= profile.thresholds.watch) return 'watch';
  return 'normal';
}

/** Look up a state by id; null when the profile has no such state. */
export function findState(profile: Profile, stateId: string): ProfileState | null {
  for (const state of profile.states) {
    if (state.id === stateId) return state;
  }
  return null;
}
