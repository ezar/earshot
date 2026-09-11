/** Small numeric helpers shared across earshot. Nothing here allocates in a loop. */

/** Clamp `value` into `[min, max]`. */
export function clamp(value: number, min: number, max: number): number {
  return value < min ? min : value > max ? max : value;
}

/** Arithmetic mean of `values`; 0 for an empty input. */
export function mean(values: ArrayLike<number>): number {
  if (values.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) sum += values[i] as number;
  return sum / values.length;
}

/** Population standard deviation of `values`; 0 for fewer than two samples. */
export function standardDeviation(values: ArrayLike<number>): number {
  if (values.length < 2) return 0;
  const m = mean(values);
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) {
    const d = (values[i] as number) - m;
    sum += d * d;
  }
  return Math.sqrt(sum / values.length);
}

/** Median of `values`; 0 for an empty input. Does not mutate the input. */
export function median(values: ArrayLike<number>): number {
  return percentile(values, 50);
}

/**
 * Linearly interpolated percentile of `values`.
 *
 * @param values - Samples in any order; not mutated.
 * @param p - Percentile to compute, in `[0, 100]`.
 * @returns The requested percentile, or 0 for an empty input.
 */
export function percentile(values: ArrayLike<number>, p: number): number {
  if (values.length === 0) return 0;
  const sorted = Array.from(values).sort((a, b) => a - b);
  if (sorted.length === 1) return sorted[0] as number;
  const rank = (clamp(p, 0, 100) / 100) * (sorted.length - 1);
  const low = Math.floor(rank);
  const high = Math.ceil(rank);
  const lowValue = sorted[low] as number;
  if (low === high) return lowValue;
  const highValue = sorted[high] as number;
  return lowValue + (highValue - lowValue) * (rank - low);
}

/**
 * Fraction of `sorted` that is strictly below `value`, with linear
 * interpolation inside the bracketing pair.
 *
 * @param sorted - Ascending samples.
 * @param value - Value to rank.
 * @returns The empirical cumulative probability, in `[0, 1]`.
 */
export function percentileRank(sorted: ArrayLike<number>, value: number): number {
  const n = sorted.length;
  if (n === 0) return 0;
  if (value <= (sorted[0] as number)) return 0;
  if (value >= (sorted[n - 1] as number)) return 1;
  let low = 0;
  let high = n - 1;
  while (low + 1 < high) {
    const mid = (low + high) >> 1;
    if ((sorted[mid] as number) <= value) low = mid;
    else high = mid;
  }
  const a = sorted[low] as number;
  const b = sorted[high] as number;
  const within = b > a ? (value - a) / (b - a) : 0;
  return (low + within) / (n - 1);
}

/** Convert a linear full-scale amplitude to dBFS, floored at `DBFS_FLOOR`. */
export function amplitudeToDbfs(amplitude: number, floorDb = -120): number {
  const a = Math.abs(amplitude);
  if (a <= 0) return floorDb;
  return Math.max(floorDb, 20 * Math.log10(a));
}

/** Convert a linear power ratio to dB, floored at `floorDb`. */
export function powerToDb(power: number, floorDb = -120): number {
  if (power <= 0) return floorDb;
  return Math.max(floorDb, 10 * Math.log10(power));
}

/** Ratio between two frequencies expressed in semitones. */
export function semitones(fromHz: number, toHz: number): number {
  if (fromHz <= 0 || toHz <= 0) return 0;
  return 12 * Math.log2(toHz / fromHz);
}

/** Euclidean L2 norm of `vector`. */
export function norm(vector: ArrayLike<number>): number {
  let sum = 0;
  for (let i = 0; i < vector.length; i += 1) {
    const v = vector[i] as number;
    sum += v * v;
  }
  return Math.sqrt(sum);
}

/** Dot product of two equal-length vectors. */
export function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = Math.min(a.length, b.length);
  let sum = 0;
  for (let i = 0; i < n; i += 1) sum += (a[i] as number) * (b[i] as number);
  return sum;
}

/**
 * Cosine similarity of two vectors, in `[-1, 1]`.
 * Returns 0 when either vector has zero length.
 */
export function cosineSimilarity(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const denominator = norm(a) * norm(b);
  if (denominator === 0) return 0;
  return dot(a, b) / denominator;
}

/** Cosine distance, `1 - cosineSimilarity`, in `[0, 2]`. */
export function cosineDistance(a: ArrayLike<number>, b: ArrayLike<number>): number {
  return 1 - cosineSimilarity(a, b);
}

/** Return a unit-length copy of `vector`; a zero vector is returned unchanged. */
export function l2Normalize(vector: ArrayLike<number>): Float32Array {
  const out = new Float32Array(vector.length);
  const n = norm(vector);
  const scale = n === 0 ? 0 : 1 / n;
  for (let i = 0; i < vector.length; i += 1) out[i] = (vector[i] as number) * scale;
  return out;
}

/**
 * Deterministic 32-bit pseudo-random generator (mulberry32).
 * Used wherever an algorithm needs randomness, so results are reproducible.
 *
 * @param seed - Any 32-bit integer.
 * @returns A function yielding uniform values in `[0, 1)`.
 */
export function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
