/**
 * Turning a numeric anomaly score into something a person can act on.
 *
 * "0.93" tells a user nothing. "The 2–4 kHz band is 11 dB louder than usual"
 * tells them where to look, so `describeDifference` ranks the interpretable
 * features by how far they have moved from the matched state and hands back
 * ready-to-render descriptors.
 */

import { mean } from '../util/math.js';
import type { WindowResult } from '../util/types.js';
import { describableValues, type Profile, type ProfileState } from '../learn/profile.js';
import { findState } from './score.js';

/** One way in which a check differs from the profile. */
export interface Descriptor {
  /** Machine-readable feature key, e.g. `spectralCentroidHz`. */
  readonly feature: string;
  /** Human-readable feature name, e.g. `"spectral centroid"`. */
  readonly label: string;
  /** Direction of the change. */
  readonly direction: 'higher' | 'lower';
  /** Signed deviation, in standard deviations of the learned feature. */
  readonly zScore: number;
  /** Observed mean over the check's windows. */
  readonly value: number;
  /** Learned mean for the matched state. */
  readonly reference: number;
  /** Unit of {@link value} and {@link reference}. */
  readonly unit: string;
  /** One-sentence English summary. */
  readonly text: string;
}

/** Options for {@link describeDifference}. */
export interface DescribeOptions {
  /** Maximum descriptors to return. Defaults to 3. */
  readonly maxDescriptors?: number;
  /** Features moving less than this many standard deviations are ignored. Defaults to 1.5. */
  readonly minZScore?: number;
}

interface FeatureMeta {
  readonly label: string;
  readonly unit: string;
  readonly higher: string;
  readonly lower: string;
}

const FEATURE_META: Readonly<Record<string, FeatureMeta>> = {
  level: { label: 'level', unit: 'dBFS', higher: 'louder', lower: 'quieter' },
  spectralFlatness: { label: 'noisiness', unit: '', higher: 'noisier', lower: 'more tonal' },
  spectralCentroidHz: { label: 'spectral centroid', unit: 'Hz', higher: 'brighter', lower: 'duller' },
  spectralFlux: { label: 'spectral change', unit: '', higher: 'less steady', lower: 'steadier' },
  onsetPeriodicity: { label: 'rhythmic regularity', unit: '', higher: 'more rhythmic', lower: 'less rhythmic' },
  amplitudeModulationHz: { label: 'modulation rate', unit: 'Hz', higher: 'faster', lower: 'slower' },
  amplitudeModulationDepth: { label: 'modulation depth', unit: '', higher: 'more pulsing', lower: 'less pulsing' },
  peakFrequencyHz: { label: 'strongest tone', unit: 'Hz', higher: 'higher pitched', lower: 'lower pitched' },
  peakProminenceDb: { label: 'tone prominence', unit: 'dB', higher: 'more whiny', lower: 'less whiny' },
};

/**
 * Explain how a set of windows differs from one state of a profile.
 *
 * @param profile - The profile the check was scored against.
 * @param stateId - The matched state, usually `CheckResult.dominantStateId`.
 * @param windows - The check's windows.
 * @returns Descriptors sorted by descending absolute z-score.
 */
export function describeDifference(
  profile: Profile,
  stateId: string,
  windows: readonly WindowResult[],
  options: DescribeOptions = {},
): Descriptor[] {
  const maxDescriptors = options.maxDescriptors ?? 3;
  const minZScore = options.minZScore ?? 1.5;
  const state = findState(profile, stateId) ?? profile.states[0];
  if (state === undefined || windows.length === 0) return [];

  const observed = new Map<string, number[]>();
  for (const window of windows) {
    for (const [key, value] of Object.entries(describableValues(window.features))) {
      const column = observed.get(key);
      if (column === undefined) observed.set(key, [value]);
      else column.push(value);
    }
  }

  const descriptors: Descriptor[] = [];
  for (const [key, values] of observed) {
    const stat = state.featureStats[key];
    if (stat === undefined) continue;
    const spread = stat.standardDeviation;
    if (spread <= 0) continue;
    const value = mean(values);
    const zScore = (value - stat.mean) / spread;
    if (Math.abs(zScore) < minZScore) continue;
    descriptors.push(buildDescriptor(key, value, stat.mean, zScore, state));
  }
  descriptors.sort((a, b) => Math.abs(b.zScore) - Math.abs(a.zScore));
  return descriptors.slice(0, maxDescriptors);
}

function buildDescriptor(
  key: string,
  value: number,
  reference: number,
  zScore: number,
  _state: ProfileState,
): Descriptor {
  const meta = FEATURE_META[key] ?? bandMeta(key);
  const direction = zScore > 0 ? 'higher' : 'lower';
  const word = zScore > 0 ? meta.higher : meta.lower;
  const delta = Math.abs(value - reference);
  const amount = meta.unit === '' ? delta.toFixed(2) : `${formatNumber(delta)} ${meta.unit}`;
  return {
    feature: key,
    label: meta.label,
    direction,
    zScore,
    value,
    reference,
    unit: meta.unit,
    text: `${capitalize(meta.label)} is ${word} than usual, by ${amount}.`,
  };
}

function bandMeta(key: string): FeatureMeta {
  const match = /^band(\d+)Hz$/.exec(key);
  const hz = match?.[1];
  const label = hz === undefined ? key : `the ${formatHz(Number(hz))} band`;
  return { label, unit: 'dB', higher: 'louder', lower: 'quieter' };
}

function formatHz(hz: number): string {
  return hz >= 1000 ? `${(hz / 1000).toFixed(hz % 1000 === 0 ? 0 : 1)} kHz` : `${Math.round(hz)} Hz`;
}

function formatNumber(value: number): string {
  if (value >= 100) return value.toFixed(0);
  if (value >= 10) return value.toFixed(1);
  return value.toFixed(2);
}

function capitalize(text: string): string {
  return text.length === 0 ? text : text[0]?.toUpperCase() + text.slice(1);
}
