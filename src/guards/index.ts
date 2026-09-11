/**
 * Guards: cheap veto rules that run before learning or scoring so that a
 * window polluted by a TV, a conversation or a clipped microphone never
 * reaches a profile.
 */

import type { ClassScore, WindowResult } from '../util/types.js';
import { maxScoreOf } from '../models/classifier.js';

/** Why a window was rejected. */
export type GuardReason = 'silence' | 'too-loud' | 'interference' | 'clipping';

/** The verdict of running every guard over one window. */
export interface GuardVerdict {
  /** True when the window is usable for learning and scoring. */
  readonly accepted: boolean;
  /** Reasons the window was rejected; empty when accepted. */
  readonly reasons: readonly GuardReason[];
  /** Strongest interference class found, if any. */
  readonly interference: ClassScore | null;
}

/**
 * YAMNet classes that mean "something other than the machine is making this
 * sound". A window where any of these is confident is not evidence about the
 * machine, so SteadyHum drops it rather than learning it as normal.
 */
export const INTERFERENCE_CLASSES: readonly string[] = [
  'Speech',
  'Conversation',
  'Narration, monologue',
  'Child speech, kid speaking',
  'Shout',
  'Screaming',
  'Laughter',
  'Singing',
  'Music',
  'Musical instrument',
  'Television',
  'Radio',
  'Telephone bell ringing',
  'Dog',
  'Cat',
  'Bird',
];

/**
 * YAMNet classes that mean a human voice is present. Meowlogue uses these to
 * flag a vocalization as possibly a person imitating an animal.
 */
export const HUMAN_VOICE_CLASSES: readonly string[] = [
  'Speech',
  'Conversation',
  'Narration, monologue',
  'Child speech, kid speaking',
  'Whispering',
  'Laughter',
  'Singing',
  'Humming',
  'Whistling',
  'Shout',
  'Screaming',
  'Yell',
];

/** Thresholds for {@link createGuards}. */
export interface GuardConfig {
  /** Windows quieter than this are rejected as silence, in dBFS. Defaults to -65. */
  readonly silenceFloorDbfs?: number;
  /** Windows louder than this are rejected as overdriven, in dBFS. Defaults to -3. */
  readonly maxLevelDbfs?: number;
  /** An interference class at or above this score rejects the window. Defaults to 0.35. */
  readonly interferenceScore?: number;
  /** Classes treated as interference. Defaults to {@link INTERFERENCE_CLASSES}. */
  readonly interferenceClasses?: readonly string[];
}

/** A configured set of guards. */
export interface Guards {
  /** Run every guard over one window. */
  check(window: WindowResult): GuardVerdict;
  /** Keep only the windows every guard accepts. */
  filter(windows: readonly WindowResult[]): WindowResult[];
  /** The thresholds in force, with defaults resolved. */
  readonly config: Required<Omit<GuardConfig, 'interferenceClasses'>> & {
    readonly interferenceClasses: readonly string[];
  };
}

/**
 * Create a guard set.
 *
 * @param config - Threshold overrides; every field has a documented default.
 */
export function createGuards(config: GuardConfig = {}): Guards {
  const resolved = {
    silenceFloorDbfs: config.silenceFloorDbfs ?? -65,
    maxLevelDbfs: config.maxLevelDbfs ?? -3,
    interferenceScore: config.interferenceScore ?? 0.35,
    interferenceClasses: config.interferenceClasses ?? INTERFERENCE_CLASSES,
  };

  const check = (window: WindowResult): GuardVerdict => {
    const reasons: GuardReason[] = [];
    if (window.rmsDbfs < resolved.silenceFloorDbfs) reasons.push('silence');
    if (window.rmsDbfs > resolved.maxLevelDbfs) reasons.push('too-loud');

    let interference: ClassScore | null = null;
    for (const entry of window.classes) {
      if (!resolved.interferenceClasses.includes(entry.label)) continue;
      if (entry.score < resolved.interferenceScore) continue;
      if (interference === null || entry.score > interference.score) interference = entry;
    }
    if (interference !== null) reasons.push('interference');

    return { accepted: reasons.length === 0, reasons, interference };
  };

  return {
    check,
    filter(windows) {
      return windows.filter((window) => check(window).accepted);
    },
    config: resolved,
  };
}

/**
 * Probability that a human voice is present in a window's class scores.
 *
 * @returns The highest score among {@link HUMAN_VOICE_CLASSES}, in `[0, 1]`.
 */
export function humanVoiceScore(classes: readonly ClassScore[], labels: readonly string[] = HUMAN_VOICE_CLASSES): number {
  return maxScoreOf(classes, labels);
}

/**
 * Whether a window is likely to be a person rather than the animal.
 *
 * @param classes - Class scores for the window.
 * @param threshold - Score above which the voice guard trips. Defaults to 0.3.
 */
export function isPossiblyHuman(classes: readonly ClassScore[], threshold = 0.3): boolean {
  return humanVoiceScore(classes) >= threshold;
}
