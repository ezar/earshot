/**
 * Diagonal Gaussian statistics with shrinkage, and the empirical distance
 * distributions that turn a raw distance into a percentile.
 */

import { percentile, percentileRank } from '../util/math.js';

/** A diagonal-covariance Gaussian fitted to one cluster. */
export interface DiagonalGaussian {
  /** Per-dimension mean. */
  readonly mean: readonly number[];
  /** Per-dimension variance, already shrunk and floored. */
  readonly variance: readonly number[];
  /** Number of observations the fit is based on. */
  readonly count: number;
}

/**
 * Fit a diagonal Gaussian to a set of vectors, shrinking the per-dimension
 * variance towards the pooled variance.
 *
 * A learning session may hold only a few dozen windows per state, so raw
 * per-dimension variances are noisy and some come out near zero — which would
 * make the Mahalanobis distance explode on a harmless dimension. Shrinkage
 * pulls each variance towards the average variance across dimensions, by an
 * amount that fades as the sample grows.
 *
 * @param vectors - Observations of equal dimensionality; at least one.
 * @param shrinkage - Weight in `[0, 1]` given to the pooled variance at `count = 1`. Defaults to 0.25.
 * @param varianceFloor - Absolute lower bound on any variance. Defaults to 1e-6.
 */
export function fitDiagonalGaussian(
  vectors: readonly Float32Array[],
  shrinkage = 0.25,
  varianceFloor = 1e-6,
): DiagonalGaussian {
  if (vectors.length === 0) throw new Error('earshot: fitDiagonalGaussian needs at least one vector');
  const dimensions = (vectors[0] as Float32Array).length;
  const mean = new Float64Array(dimensions);
  for (const vector of vectors) {
    for (let d = 0; d < dimensions; d += 1) mean[d] = (mean[d] as number) + (vector[d] as number);
  }
  for (let d = 0; d < dimensions; d += 1) mean[d] = (mean[d] as number) / vectors.length;

  const variance = new Float64Array(dimensions);
  if (vectors.length > 1) {
    for (const vector of vectors) {
      for (let d = 0; d < dimensions; d += 1) {
        const delta = (vector[d] as number) - (mean[d] as number);
        variance[d] = (variance[d] as number) + delta * delta;
      }
    }
    for (let d = 0; d < dimensions; d += 1) {
      variance[d] = (variance[d] as number) / (vectors.length - 1);
    }
  }

  let pooled = 0;
  for (let d = 0; d < dimensions; d += 1) pooled += variance[d] as number;
  pooled = dimensions === 0 ? 0 : pooled / dimensions;

  // Shrinkage weight decays as 1 / count so that a well-sampled state keeps its
  // own variances and a thin one leans on the pooled estimate.
  const lambda = Math.min(1, shrinkage / Math.max(1, vectors.length / 8));
  const out = new Array<number>(dimensions);
  for (let d = 0; d < dimensions; d += 1) {
    out[d] = Math.max(varianceFloor, (1 - lambda) * (variance[d] as number) + lambda * pooled);
  }
  return { mean: Array.from(mean), variance: out, count: vectors.length };
}

/**
 * Mahalanobis distance under a diagonal covariance.
 *
 * @returns The distance in standard deviations, summed in quadrature over dimensions.
 */
export function mahalanobisDistance(vector: ArrayLike<number>, model: DiagonalGaussian): number {
  const dimensions = Math.min(vector.length, model.mean.length);
  let sum = 0;
  for (let d = 0; d < dimensions; d += 1) {
    const delta = (vector[d] as number) - (model.mean[d] as number);
    sum += (delta * delta) / (model.variance[d] as number);
  }
  return Math.sqrt(sum / Math.max(1, dimensions));
}

/** An empirical distribution summarized by a sorted sample and key percentiles. */
export interface DistanceDistribution {
  /** Ascending sample of observed distances, possibly thinned. */
  readonly sorted: readonly number[];
  /** 50th percentile. */
  readonly p50: number;
  /** 90th percentile. */
  readonly p90: number;
  /** 95th percentile. */
  readonly p95: number;
  /** 99th percentile. */
  readonly p99: number;
  /** Largest observed distance. */
  readonly max: number;
}

/**
 * Summarize observed distances into a persistable distribution.
 *
 * @param distances - Observed distances, any order.
 * @param maxSamples - Cap on the retained sample; larger inputs are thinned evenly. Defaults to 512.
 */
export function buildDistanceDistribution(distances: readonly number[], maxSamples = 512): DistanceDistribution {
  const sorted = Array.from(distances).sort((a, b) => a - b);
  const thinned =
    sorted.length <= maxSamples
      ? sorted
      : Array.from({ length: maxSamples }, (_, i) => sorted[Math.round((i * (sorted.length - 1)) / (maxSamples - 1))] as number);
  return {
    sorted: thinned,
    p50: percentile(sorted, 50),
    p90: percentile(sorted, 90),
    p95: percentile(sorted, 95),
    p99: percentile(sorted, 99),
    max: sorted.length === 0 ? 0 : (sorted[sorted.length - 1] as number),
  };
}

/**
 * Map a distance onto an anomaly score in `[0, 1]`.
 *
 * The learned range is deliberately compressed into `[0, 0.5]`: everything the
 * profile actually heard while learning is "normal", and the top half of the
 * scale is reserved for distances the profile has never seen. Beyond the
 * learned maximum the score approaches 1 exponentially, scaled by the spread of
 * the learned distances themselves, so a slightly unusual check and a wildly
 * unusual one stay far apart instead of both pinning at the ceiling.
 *
 * @param distance - Mahalanobis distance to a state.
 * @param distribution - That state's learned distance distribution.
 * @returns 0 for the most typical audio, 0.5 at the edge of what was learned,
 *   and approaching 1 far outside it.
 */
export function distanceScore(distance: number, distribution: DistanceDistribution): number {
  if (distribution.sorted.length === 0) return 0;
  if (distance <= distribution.max) {
    return 0.5 * percentileRank(distribution.sorted, distance);
  }
  // A robust scale for "how much further is this than normal variation?".
  const spread = Math.max(distribution.p95 - distribution.p50, distribution.max * 0.05, 1e-6);
  const excess = (distance - distribution.max) / spread;
  return 1 - 0.5 * Math.exp(-excess);
}
