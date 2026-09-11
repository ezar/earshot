/**
 * k-means with k-means++ initialization, silhouette-based K selection and a
 * minimum cluster size.
 *
 * A machine rarely has one steady sound: a fridge cycles, a pump alternates
 * between load states. Clustering the learning windows gives the profile one
 * *state* per distinguishable mode, and scoring then compares a check against
 * the nearest state rather than against a single blurred average.
 */

import { createRandom } from '../util/math.js';

/** A clustering of the input vectors. */
export interface Clustering {
  /** Cluster index per input vector, aligned with the input order. */
  readonly assignments: readonly number[];
  /** Cluster centroids, `k` vectors of the input dimensionality. */
  readonly centroids: readonly Float32Array[];
  /** Number of clusters actually produced. */
  readonly k: number;
  /** Mean silhouette coefficient in `[-1, 1]`; 0 when undefined. */
  readonly silhouette: number;
  /** Sum of squared distances from each point to its centroid. */
  readonly inertia: number;
}

/** Options for {@link kmeans} and {@link selectClustering}. */
export interface KmeansOptions {
  /** Number of clusters. Ignored by {@link selectClustering}. */
  readonly k?: number;
  /** Maximum Lloyd iterations. Defaults to 50. */
  readonly maxIterations?: number;
  /** Seed for the deterministic PRNG. Defaults to 1. */
  readonly seed?: number;
  /** Restarts kept, best inertia wins. Defaults to 4. */
  readonly restarts?: number;
}

/** Squared Euclidean distance between two equal-length vectors. */
export function squaredDistance(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = (a[i] as number) - (b[i] as number);
    sum += d * d;
  }
  return sum;
}

/**
 * Cluster `vectors` into `k` groups.
 *
 * @param vectors - Equal-length observation vectors.
 * @param options - `k` is required here; see {@link selectClustering} to choose it.
 * @throws When `vectors` is empty or `k` is outside `[1, vectors.length]`.
 */
export function kmeans(vectors: readonly Float32Array[], options: KmeansOptions & { k: number }): Clustering {
  const { k } = options;
  if (vectors.length === 0) throw new Error('earshot: kmeans needs at least one vector');
  if (k < 1 || k > vectors.length) {
    throw new Error(`earshot: kmeans k must be in [1, ${vectors.length}], received ${k}`);
  }
  const maxIterations = options.maxIterations ?? 50;
  const restarts = options.restarts ?? 4;
  const seed = options.seed ?? 1;

  let best: Clustering | null = null;
  for (let restart = 0; restart < restarts; restart += 1) {
    const candidate = runLloyd(vectors, k, maxIterations, seed + restart * 7919);
    if (best === null || candidate.inertia < best.inertia) best = candidate;
  }
  const result = best as Clustering;
  return { ...result, silhouette: silhouette(vectors, result.assignments, result.k) };
}

function runLloyd(vectors: readonly Float32Array[], k: number, maxIterations: number, seed: number): Clustering {
  const dimensions = (vectors[0] as Float32Array).length;
  const centroids = kmeansPlusPlus(vectors, k, seed);
  const assignments = new Array<number>(vectors.length).fill(0);

  for (let iteration = 0; iteration < maxIterations; iteration += 1) {
    let moved = false;
    for (let i = 0; i < vectors.length; i += 1) {
      const vector = vectors[i] as Float32Array;
      let bestCluster = 0;
      let bestDistance = Number.POSITIVE_INFINITY;
      for (let c = 0; c < k; c += 1) {
        const distance = squaredDistance(vector, centroids[c] as Float32Array);
        if (distance < bestDistance) {
          bestDistance = distance;
          bestCluster = c;
        }
      }
      if (assignments[i] !== bestCluster) {
        assignments[i] = bestCluster;
        moved = true;
      }
    }

    const sums = Array.from({ length: k }, () => new Float32Array(dimensions));
    const counts = new Array<number>(k).fill(0);
    for (let i = 0; i < vectors.length; i += 1) {
      const cluster = assignments[i] as number;
      const sum = sums[cluster] as Float32Array;
      const vector = vectors[i] as Float32Array;
      for (let d = 0; d < dimensions; d += 1) sum[d] = (sum[d] as number) + (vector[d] as number);
      counts[cluster] = (counts[cluster] as number) + 1;
    }
    for (let c = 0; c < k; c += 1) {
      const count = counts[c] as number;
      if (count === 0) continue;
      const sum = sums[c] as Float32Array;
      const centroid = centroids[c] as Float32Array;
      for (let d = 0; d < dimensions; d += 1) centroid[d] = (sum[d] as number) / count;
    }
    if (!moved) break;
  }

  let inertia = 0;
  for (let i = 0; i < vectors.length; i += 1) {
    inertia += squaredDistance(vectors[i] as Float32Array, centroids[assignments[i] as number] as Float32Array);
  }
  return { assignments, centroids, k, silhouette: 0, inertia };
}

/** k-means++ seeding: spread the initial centroids proportionally to squared distance. */
function kmeansPlusPlus(vectors: readonly Float32Array[], k: number, seed: number): Float32Array[] {
  const random = createRandom(seed);
  const centroids: Float32Array[] = [];
  const first = Math.min(vectors.length - 1, Math.floor(random() * vectors.length));
  centroids.push((vectors[first] as Float32Array).slice());

  const distances = new Float64Array(vectors.length).fill(Number.POSITIVE_INFINITY);
  while (centroids.length < k) {
    const latest = centroids[centroids.length - 1] as Float32Array;
    let total = 0;
    for (let i = 0; i < vectors.length; i += 1) {
      const distance = squaredDistance(vectors[i] as Float32Array, latest);
      if (distance < (distances[i] as number)) distances[i] = distance;
      total += distances[i] as number;
    }
    let target = random() * total;
    let chosen = vectors.length - 1;
    for (let i = 0; i < vectors.length; i += 1) {
      target -= distances[i] as number;
      if (target <= 0) {
        chosen = i;
        break;
      }
    }
    centroids.push((vectors[chosen] as Float32Array).slice());
  }
  return centroids;
}

/**
 * Mean silhouette coefficient of a clustering, in `[-1, 1]`.
 *
 * Returns 0 for a single cluster, where the coefficient is undefined.
 */
export function silhouette(vectors: readonly Float32Array[], assignments: readonly number[], k: number): number {
  if (k < 2 || vectors.length < 3) return 0;
  const members: number[][] = Array.from({ length: k }, () => []);
  for (let i = 0; i < assignments.length; i += 1) {
    (members[assignments[i] as number] as number[]).push(i);
  }

  let total = 0;
  let counted = 0;
  for (let i = 0; i < vectors.length; i += 1) {
    const own = assignments[i] as number;
    const ownMembers = members[own] as number[];
    if (ownMembers.length < 2) continue;
    const a = meanDistance(vectors, i, ownMembers, true);
    let b = Number.POSITIVE_INFINITY;
    for (let c = 0; c < k; c += 1) {
      if (c === own) continue;
      const other = members[c] as number[];
      if (other.length === 0) continue;
      const distance = meanDistance(vectors, i, other, false);
      if (distance < b) b = distance;
    }
    if (!Number.isFinite(b)) continue;
    const denominator = Math.max(a, b);
    if (denominator > 0) total += (b - a) / denominator;
    counted += 1;
  }
  return counted === 0 ? 0 : total / counted;
}

function meanDistance(
  vectors: readonly Float32Array[],
  index: number,
  members: readonly number[],
  excludeSelf: boolean,
): number {
  const vector = vectors[index] as Float32Array;
  let sum = 0;
  let count = 0;
  for (const j of members) {
    if (excludeSelf && j === index) continue;
    sum += Math.sqrt(squaredDistance(vector, vectors[j] as Float32Array));
    count += 1;
  }
  return count === 0 ? 0 : sum / count;
}

/** Options for {@link selectClustering}. */
export interface SelectClusteringOptions extends Omit<KmeansOptions, 'k'> {
  /** Smallest K to try. Defaults to 1. */
  readonly minK?: number;
  /** Largest K to try. Defaults to 5. */
  readonly maxK?: number;
  /**
   * A candidate K is rejected when any cluster holds fewer than this fraction
   * of the input. Defaults to 0.08 (8 %).
   */
  readonly minClusterFraction?: number;
  /**
   * A candidate K only beats a smaller K when it improves the silhouette by at
   * least this much. Defaults to 0.05.
   */
  readonly silhouetteMargin?: number;
  /**
   * Absolute silhouette a candidate `K > 1` must reach to be considered at all.
   * Defaults to 0.6.
   *
   * The silhouette coefficient is undefined for a single cluster, so it offers
   * no evidence for "this data has no structure" — it happily splits a uniform
   * cloud and scores the split around 0.4-0.5. Data with genuinely distinct
   * modes scores above 0.8, so this floor is what actually decides whether a
   * machine has one operating state or several.
   */
  readonly minSilhouette?: number;
}

/**
 * Choose K by silhouette, subject to a minimum cluster size.
 *
 * Both the floor and the margin bias the result towards fewer states: an extra
 * state has to earn its place, because every state dilutes the evidence behind
 * the distance distributions that scoring depends on.
 *
 * @param vectors - Observations to cluster.
 * @param options - Search range and acceptance thresholds.
 * @returns The chosen clustering; `k` is 1 when the data shows no structure.
 */
export function selectClustering(
  vectors: readonly Float32Array[],
  options: SelectClusteringOptions = {},
): Clustering {
  if (vectors.length === 0) throw new Error('earshot: selectClustering needs at least one vector');
  const minK = Math.max(1, options.minK ?? 1);
  const maxK = Math.max(minK, Math.min(options.maxK ?? 5, vectors.length));
  const minClusterFraction = options.minClusterFraction ?? 0.08;
  const margin = options.silhouetteMargin ?? 0.05;
  const minSilhouette = options.minSilhouette ?? 0.6;
  const minMembers = Math.max(2, Math.ceil(vectors.length * minClusterFraction));

  const baseOptions: Omit<KmeansOptions, 'k'> = {
    maxIterations: options.maxIterations ?? 50,
    seed: options.seed ?? 1,
    restarts: options.restarts ?? 4,
  };

  let best = kmeans(vectors, { ...baseOptions, k: minK });
  for (let k = minK + 1; k <= maxK; k += 1) {
    const candidate = kmeans(vectors, { ...baseOptions, k });
    const counts = new Array<number>(k).fill(0);
    for (const assignment of candidate.assignments) counts[assignment] = (counts[assignment] as number) + 1;
    if (counts.some((count) => count < minMembers)) continue;
    if (candidate.silhouette < minSilhouette) continue;
    if (candidate.silhouette > best.silhouette + margin) best = candidate;
  }
  return best;
}
