/**
 * Compact form of a {@link Profile}, for apps that store many of them.
 *
 * A profile learned in the embedding space carries a 1024-dimension mean and
 * variance per state, and those numbers dominate its size: as JSON text a
 * three-state profile runs to roughly 135 kB, almost all of it digits. The
 * compact form quantizes them to float16 and base64s the result, which is both
 * denser per value and free of the textual overhead.
 *
 * Only the embedding space is quantized. Its values are L2-normalized, so they
 * all sit in the same narrow range and float16's three significant digits are
 * far finer than the distances the profile computes. The `'features'` space is
 * the opposite on both counts: its dimensions run from negative decibels to
 * thousands of hertz, where three significant digits is a visible error, and
 * there are only about 70 of them per state, so there is nothing to save. Those
 * pass through untouched.
 */

import type { QuantizedEmbedding } from '../util/types.js';
import { dequantize, quantize } from '../util/quantize.js';
import type { DistanceDistribution } from './statistics.js';
import type { FeatureStat, Profile, ProfileState } from './profile.js';

/** Schema version of the compact shape. */
export const COMPACT_PROFILE_SCHEMA_VERSION = 1;

/**
 * A vector as stored in a compact profile: quantized when that is safe, and a
 * plain array when precision matters more than size.
 */
export type CompactVector = QuantizedEmbedding | readonly number[];

/** One state, with its Gaussian parameters quantized where it is safe to. */
export interface CompactProfileState {
  readonly id: string;
  readonly mean: CompactVector;
  readonly variance: CompactVector;
  readonly count: number;
  readonly distances: DistanceDistribution;
  readonly featureStats: Readonly<Record<string, FeatureStat>>;
  readonly weight: number;
}

/** A profile with its bulky numeric arrays quantized. */
export interface CompactProfile extends Omit<Profile, 'states'> {
  /** Shape version of the compact form itself. */
    readonly compactVersion: number;
  readonly states: readonly CompactProfileState[];
}

/**
 * Quantize a profile's Gaussian parameters for storage.
 *
 * Everything else — thresholds, feature statistics, distance distributions,
 * calibration history — is left alone: it is small, and it is what a person
 * reads when debugging a profile.
 *
 * @param profile - A profile from `learnProfile` or `calibrate`.
 * @returns A plain serializable object. An embedding-space profile shrinks
 *   several-fold as JSON; a features-space one is returned essentially as-is,
 *   because quantizing it would cost accuracy and save nothing.
 */
export function compactProfile(profile: Profile): CompactProfile {
  const { states, ...rest } = profile;
  const shrink = (values: readonly number[]): CompactVector =>
    profile.featureSpace === 'embedding' ? quantize(values, 'float16') : values;
  return {
    ...rest,
    compactVersion: COMPACT_PROFILE_SCHEMA_VERSION,
    states: states.map((state) => ({
      id: state.id,
      mean: shrink(state.model.mean),
      variance: shrink(state.model.variance),
      count: state.model.count,
      distances: state.distances,
      featureStats: state.featureStats,
      weight: state.weight,
    })),
  };
}

/**
 * Restore a profile compacted by {@link compactProfile}.
 *
 * @throws When the compact form carries a version this build does not know.
 */
export function expandProfile(compact: CompactProfile): Profile {
  if (compact.compactVersion !== COMPACT_PROFILE_SCHEMA_VERSION) {
    throw new Error(
      `earshot: compact profile version ${compact.compactVersion} is not supported by this build ` +
        `(expected ${COMPACT_PROFILE_SCHEMA_VERSION})`,
    );
  }
  const { compactVersion: _compactVersion, states, ...rest } = compact;
  const grow = (value: CompactVector): number[] =>
    Array.isArray(value) ? [...(value as readonly number[])] : Array.from(dequantize(value as QuantizedEmbedding));
  const expanded: ProfileState[] = states.map((state) => ({
    id: state.id,
    model: {
      mean: grow(state.mean),
      variance: grow(state.variance),
      count: state.count,
    },
    distances: state.distances,
    featureStats: state.featureStats,
    weight: state.weight,
  }));
  return { ...rest, states: expanded };
}
