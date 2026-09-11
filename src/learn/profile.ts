/**
 * Profiles: what "normal" sounds like for one machine, learned from windows
 * and refined by user verdicts.
 *
 * A profile is plain JSON. The consuming app stores it, versions it and hands
 * it back to {@link scoreCheck}; earshot keeps no state of its own.
 */

import { featureVector } from '../dsp/features.js';
import type { WindowFeatures, WindowResult } from '../util/types.js';
import { mean, percentile, standardDeviation } from '../util/math.js';
import { selectClustering, type SelectClusteringOptions } from './kmeans.js';
import {
  buildDistanceDistribution,
  distanceScore,
  fitDiagonalGaussian,
  mahalanobisDistance,
  type DiagonalGaussian,
  type DistanceDistribution,
} from './statistics.js';

/** Schema version of the {@link Profile} shape; bumped on breaking changes. */
export const PROFILE_SCHEMA_VERSION = 1;

/** Which vector space a profile clusters and scores in. */
export type FeatureSpace = 'embedding' | 'features';

/** Summary statistics of one interpretable feature over a profile's windows. */
export interface FeatureStat {
  /** Mean value over the learning windows. */
  readonly mean: number;
  /** Standard deviation over the learning windows. */
  readonly standardDeviation: number;
  /** 5th percentile. */
  readonly p05: number;
  /** 95th percentile. */
  readonly p95: number;
}

/** One learned mode of the machine. */
export interface ProfileState {
  /** Stable identifier, unique within the profile. */
  readonly id: string;
  /** Diagonal Gaussian over the profile's feature space. */
  readonly model: DiagonalGaussian;
  /** Distribution of within-state distances observed while learning. */
  readonly distances: DistanceDistribution;
  /** Per-feature statistics, keyed by the names {@link describeDifference} uses. */
  readonly featureStats: Readonly<Record<string, FeatureStat>>;
  /** Share of the learning windows that fell in this state, in `[0, 1]`. */
  readonly weight: number;
}

/** Thresholds separating normal, watch and anomalous check scores. */
export interface ProfileThresholds {
  /** Check scores at or above this are at least `watch`, in `[0, 1]`. */
  readonly watch: number;
  /** Check scores at or above this are `anomalous`, in `[0, 1]`. */
  readonly anomalous: number;
}

/** A learned "normal" profile for one machine. */
export interface Profile {
  /** Shape version; see {@link PROFILE_SCHEMA_VERSION}. */
  readonly schemaVersion: number;
  /** Monotonic revision, incremented by every {@link calibrate} call. */
  readonly revision: number;
  /** Vector space the states live in. */
  readonly featureSpace: FeatureSpace;
  /** Dimensionality of that space. */
  readonly dimensions: number;
  /** Learned states, strongest first. */
  readonly states: readonly ProfileState[];
  /** Status thresholds, adjusted by calibration. */
  readonly thresholds: ProfileThresholds;
  /** Number of windows that survived the guards and went into the fit. */
  readonly windowCount: number;
  /** Level statistics of the learning windows, in dBFS. */
  readonly levelDbfs: FeatureStat;
  /** Verdicts applied so far, oldest first. */
  readonly calibrations: readonly CalibrationRecord[];
}

/** One user verdict folded into a profile. */
export interface CalibrationRecord {
  /** What the user said about a check. */
  readonly verdict: Verdict;
  /** Check score that was being judged, in `[0, 1]`. */
  readonly score: number;
  /** Thresholds after the adjustment. */
  readonly thresholds: ProfileThresholds;
  /** Unix epoch milliseconds when the verdict was recorded. */
  readonly at: number;
}

/** The user's judgement of a check earshot already scored. */
export type Verdict = 'normal' | 'anomalous';

/** Options for {@link learnProfile}. */
export interface LearnProfileOptions extends SelectClusteringOptions {
  /** Vector space to cluster in. Defaults to `'embedding'` when embeddings are present. */
  readonly featureSpace?: FeatureSpace;
  /** Shrinkage passed to {@link fitDiagonalGaussian}. Defaults to 0.25. */
  readonly shrinkage?: number;
  /**
   * Override the derived `watch` threshold instead of measuring it. Only set
   * this when you have a reason to distrust the learning session.
   */
  readonly watchThreshold?: number;
  /** Override the derived `anomalous` threshold. See {@link watchThreshold}. */
  readonly anomalousThreshold?: number;
  /**
   * Windows per pseudo-check when deriving the thresholds. Defaults to 12,
   * about six seconds of audio.
   */
  readonly calibrationBlockWindows?: number;
  /** Percentile used to aggregate a check's window scores. Defaults to 90. */
  readonly aggregatePercentile?: number;
  /** Minimum windows required to learn anything. Defaults to 8. */
  readonly minWindows?: number;
}

/**
 * Learn a profile from windows that have already passed the guards.
 *
 * @param windows - Guard-accepted windows from one or more learning sessions.
 * @param options - Clustering and threshold overrides.
 * @returns A fresh profile at revision 1.
 * @throws When fewer than `minWindows` windows are supplied.
 */
export function learnProfile(windows: readonly WindowResult[], options: LearnProfileOptions = {}): Profile {
  const minWindows = options.minWindows ?? 8;
  if (windows.length < minWindows) {
    throw new Error(`earshot: learnProfile needs at least ${minWindows} windows, received ${windows.length}`);
  }
  const featureSpace = options.featureSpace ?? (hasEmbeddings(windows) ? 'embedding' : 'features');
  const vectors = windows.map((window) => toVector(window, featureSpace));
  const dimensions = (vectors[0] as Float32Array).length;
  if (dimensions === 0) {
    throw new Error(`earshot: learnProfile found empty vectors for featureSpace "${featureSpace}"`);
  }

  const clustering = selectClustering(vectors, options);
  const shrinkage = options.shrinkage ?? 0.25;

  const states: ProfileState[] = [];
  for (let c = 0; c < clustering.k; c += 1) {
    const indices: number[] = [];
    for (let i = 0; i < clustering.assignments.length; i += 1) {
      if (clustering.assignments[i] === c) indices.push(i);
    }
    if (indices.length === 0) continue;
    const members = indices.map((i) => vectors[i] as Float32Array);
    const model = fitDiagonalGaussian(members, shrinkage);
    const distances = members.map((vector) => mahalanobisDistance(vector, model));
    states.push({
      id: `s${c}`,
      model,
      distances: buildDistanceDistribution(distances),
      featureStats: summarizeFeatures(indices.map((i) => (windows[i] as WindowResult).features)),
      weight: indices.length / windows.length,
    });
  }
  states.sort((a, b) => b.weight - a.weight);

  const aggregatePercentile = options.aggregatePercentile ?? 90;
  const derived = deriveThresholds(
    vectors,
    clustering.assignments,
    clustering.k,
    shrinkage,
    options.calibrationBlockWindows ?? 12,
    aggregatePercentile,
  );

  return {
    schemaVersion: PROFILE_SCHEMA_VERSION,
    revision: 1,
    featureSpace,
    dimensions,
    states,
    thresholds: {
      watch: options.watchThreshold ?? derived.watch,
      anomalous: options.anomalousThreshold ?? derived.anomalous,
    },
    windowCount: windows.length,
    levelDbfs: statOf(windows.map((window) => window.rmsDbfs)),
    calibrations: [],
  };
}

/**
 * Fold a user verdict into a profile.
 *
 * A `normal` verdict on a check that earshot flagged widens the thresholds just
 * past the offending score, so the same sound stops raising an alarm. An
 * `anomalous` verdict on a check earshot let through tightens them just below
 * it. Both moves are damped, and both are clamped to a sane band so that a
 * single mistaken tap cannot destroy a profile.
 *
 * @param profile - The profile to adjust; returned unmodified.
 * @param verdict - What the user said.
 * @param windows - The windows of the check being judged; may be empty.
 * @returns A new profile at `revision + 1`.
 */
export function calibrate(
  profile: Profile,
  verdict: Verdict,
  windows: readonly WindowResult[],
  options: { readonly score?: number; readonly rate?: number; readonly now?: number } = {},
): Profile {
  const score = options.score ?? (windows.length === 0 ? profile.thresholds.watch : checkScoreOf(profile, windows));
  const rate = options.rate ?? 0.5;
  const { watch, anomalous } = profile.thresholds;

  let nextWatch = watch;
  let nextAnomalous = anomalous;
  if (verdict === 'normal' && score >= watch) {
    // Move the thresholds up towards (but not past) the score the user forgave.
    nextWatch = watch + rate * (Math.min(0.995, score + 0.01) - watch);
    nextAnomalous = Math.max(nextWatch + 0.01, anomalous + rate * (Math.min(0.999, score + 0.02) - anomalous));
  } else if (verdict === 'anomalous' && score < anomalous) {
    nextAnomalous = anomalous - rate * (anomalous - Math.max(0.5, score - 0.01));
    nextWatch = Math.min(nextAnomalous - 0.01, watch - rate * (watch - Math.max(0.4, score - 0.05)));
  }

  const thresholds: ProfileThresholds = {
    watch: clamp01(Math.min(nextWatch, nextAnomalous - 0.005), 0.4, 0.995),
    anomalous: clamp01(nextAnomalous, 0.45, 0.999),
  };
  return {
    ...profile,
    revision: profile.revision + 1,
    thresholds,
    calibrations: [
      ...profile.calibrations,
      { verdict, score, thresholds, at: options.now ?? Date.now() },
    ],
  };
}

/** Project a window into a profile's feature space. */
export function toVector(window: WindowResult, featureSpace: FeatureSpace): Float32Array {
  if (featureSpace === 'embedding') return Float32Array.from(window.embedding);
  return featureVector(window.features);
}

function hasEmbeddings(windows: readonly WindowResult[]): boolean {
  return windows.every((window) => window.embedding.length > 0);
}

function clamp01(value: number, min: number, max: number): number {
  if (Number.isNaN(value)) return min;
  return Math.min(max, Math.max(min, value));
}

/** Names of the scalar features {@link describeDifference} can talk about. */
export const DESCRIBABLE_FEATURES: readonly string[] = [
  'level',
  'spectralFlatness',
  'spectralCentroidHz',
  'spectralFlux',
  'onsetPeriodicity',
  'amplitudeModulationHz',
  'amplitudeModulationDepth',
  'peakFrequencyHz',
  'peakProminenceDb',
];

/** Extract the scalar features named in {@link DESCRIBABLE_FEATURES} from a window. */
export function describableValues(features: WindowFeatures): Record<string, number> {
  const strongestPeak = features.peaks[0];
  const values: Record<string, number> = {
    level: features.rmsDbfs,
    spectralFlatness: features.spectralFlatness,
    spectralCentroidHz: features.spectralCentroidHz,
    spectralFlux: features.spectralFlux,
    onsetPeriodicity: features.onsetPeriodicity,
    amplitudeModulationHz: features.amplitudeModulationHz,
    amplitudeModulationDepth: features.amplitudeModulationDepth,
    peakFrequencyHz: strongestPeak?.frequencyHz ?? 0,
    peakProminenceDb: strongestPeak?.prominenceDb ?? 0,
  };
  // Band values are reported relative to the window's overall level. An
  // absolute band level moves whenever the machine gets louder, so absolute
  // bands would simply restate the level change seven more times and crowd the
  // real finding out of the descriptor list.
  for (let i = 0; i < features.bands.length; i += 1) {
    const band = features.bands[i] as WindowFeatures['bands'][number];
    values[`band${Math.round(band.lowHz)}Hz`] = band.levelDbfs - features.rmsDbfs;
  }
  return values;
}

function summarizeFeatures(features: readonly WindowFeatures[]): Record<string, FeatureStat> {
  const columns = new Map<string, number[]>();
  for (const entry of features) {
    for (const [name, value] of Object.entries(describableValues(entry))) {
      const column = columns.get(name);
      if (column === undefined) columns.set(name, [value]);
      else column.push(value);
    }
  }
  const out: Record<string, FeatureStat> = {};
  for (const [name, values] of columns) out[name] = statOf(values);
  return out;
}

function statOf(values: readonly number[]): FeatureStat {
  return {
    mean: mean(values),
    standardDeviation: standardDeviation(values),
    p05: percentile(values, 5),
    p95: percentile(values, 95),
  };
}

/**
 * Derive the status thresholds from the learning session itself.
 *
 * A fixed threshold cannot work: a check score is the 90th percentile of window
 * scores, so its typical value depends on how many windows a check holds and on
 * how tightly the machine's own sound clusters. Instead the learning windows are
 * cut into contiguous pseudo-checks and each one is scored against a model
 * refitted *without* it — the same leave-one-block-out idea as cross-validation.
 * That yields the distribution of check scores this machine produces when
 * nothing is wrong, and the thresholds are placed above it.
 *
 * Blocks are contiguous rather than random because consecutive windows overlap
 * by 50 %: a random split would leak half of every held-out window back into the
 * training side and report thresholds that are far too tight.
 *
 * @returns Thresholds with the documented headroom, clamped to a sane band.
 */
function deriveThresholds(
  vectors: readonly Float32Array[],
  assignments: readonly number[],
  k: number,
  shrinkage: number,
  blockWindows: number,
  aggregatePercentile: number,
): ProfileThresholds {
  const blockCount = Math.floor(vectors.length / Math.max(1, blockWindows));
  if (blockCount < 3) {
    // Too little audio to measure anything trustworthy; fall back to a
    // deliberately permissive pair so the app raises few false alarms until the
    // user has recorded more.
    return { watch: 0.75, anomalous: 0.9 };
  }
  const size = Math.floor(vectors.length / blockCount);

  const blockScores: number[] = [];
  for (let block = 0; block < blockCount; block += 1) {
    const from = block * size;
    const to = block === blockCount - 1 ? vectors.length : from + size;

    const heldOutModels: (ProfileState | null)[] = [];
    for (let c = 0; c < k; c += 1) {
      const members: Float32Array[] = [];
      for (let i = 0; i < vectors.length; i += 1) {
        if (i >= from && i < to) continue;
        if (assignments[i] === c) members.push(vectors[i] as Float32Array);
      }
      if (members.length < 2) {
        heldOutModels.push(null);
        continue;
      }
      const model = fitDiagonalGaussian(members, shrinkage);
      heldOutModels.push({
        id: `s${c}`,
        model,
        distances: buildDistanceDistribution(members.map((v) => mahalanobisDistance(v, model))),
        featureStats: {},
        weight: 0,
      });
    }

    const scores: number[] = [];
    for (let i = from; i < to; i += 1) {
      let best = 1;
      for (const state of heldOutModels) {
        if (state === null) continue;
        const score = distanceScore(mahalanobisDistance(vectors[i] as Float32Array, state.model), state.distances);
        if (score < best) best = score;
      }
      scores.push(best);
    }
    if (scores.length > 0) blockScores.push(percentile(scores, aggregatePercentile));
  }
  if (blockScores.length === 0) return { watch: 0.75, anomalous: 0.9 };

  // `watch` sits just above the worst normal check seen, and `anomalous` a
  // clear step beyond it. The floors stop an unusually consistent learning
  // session from producing thresholds so tight that ordinary variation trips
  // them; the ceilings stop a noisy one from producing thresholds nothing can
  // ever reach.
  const worstNormal = Math.max(...blockScores);
  const typicalNormal = percentile(blockScores, 75);
  const spread = Math.max(worstNormal - typicalNormal, 0.05);
  const watch = clamp01(Math.max(0.55, worstNormal + 0.5 * spread), 0.55, 0.9);
  const anomalous = clamp01(Math.max(watch + 0.05, worstNormal + 2 * spread), watch + 0.05, 0.97);
  return { watch, anomalous };
}

/**
 * Internal helper mirroring `scoreCheck`'s aggregate, used by {@link calibrate}
 * when the caller did not pass an explicit score. Kept here so that `learn`
 * never has to import from `score`.
 */
function checkScoreOf(profile: Profile, windows: readonly WindowResult[]): number {
  const scores: number[] = [];
  for (const window of windows) {
    const vector = toVector(window, profile.featureSpace);
    let best = 1;
    for (const state of profile.states) {
      const score = distanceScore(mahalanobisDistance(vector, state.model), state.distances);
      if (score < best) best = score;
    }
    scores.push(best);
  }
  return percentile(scores, 90);
}
