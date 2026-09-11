import { describe, expect, it } from 'vitest';

import { learnProfile } from '../src/learn/profile.js';
import { findState, scoreCheck, scoreWindow, statusFor } from '../src/score/score.js';
import { describeDifference } from '../src/score/describe.js';
import { createStreamScorer } from '../src/score/stream.js';
import { createGuards, humanVoiceScore, isPossiblyHuman } from '../src/guards/index.js';
import {
  applyGainDb,
  injectWhine,
  knocks,
  pinkNoise,
  tone,
  whiteNoise,
} from '../fixtures/synthetic/index.js';
import { windowsFrom } from './helpers.js';

const normalAudio = (seconds: number, seed: number): Float32Array =>
  injectWhine(pinkNoise({ seconds, seed }, 0.08), 1200, 0.02);

const profile = learnProfile(windowsFrom(normalAudio(40, 1)));

describe('scoreCheck', () => {
  it('scores unseen normal audio as normal', () => {
    const result = scoreCheck(profile, windowsFrom(normalAudio(10, 99)));
    expect(result.status).toBe('normal');
    expect(result.score).toBeLessThan(profile.thresholds.watch);
    expect(result.windows.length).toBeGreaterThan(10);
    expect(result.profileRevision).toBe(profile.revision);
  });

  it('scores a new whine as anomalous', () => {
    const faulty = injectWhine(normalAudio(10, 5), 5200, 0.08);
    const result = scoreCheck(profile, windowsFrom(faulty));
    expect(result.score).toBeGreaterThan(profile.thresholds.anomalous);
    expect(result.status).toBe('anomalous');
  });

  it('scores a level jump as anomalous', () => {
    const result = scoreCheck(profile, windowsFrom(applyGainDb(normalAudio(10, 6), 18)));
    expect(result.status).toBe('anomalous');
  });

  it('scores a rhythmic knock as anomalous against a steady profile', () => {
    const result = scoreCheck(profile, windowsFrom(knocks({ seconds: 10 }, 0.5)));
    expect(result.score).toBeGreaterThan(profile.thresholds.watch);
  });

  it('is graded rather than binary about level changes', () => {
    // A decibel of drift is ordinary room variation and must not raise an
    // alarm; a big level change must. The scale in between has to be usable.
    const at = (gainDb: number): number =>
      scoreCheck(profile, windowsFrom(applyGainDb(normalAudio(10, 6), gainDb))).score;
    expect(at(1)).toBeLessThan(profile.thresholds.watch);
    expect(at(2)).toBeLessThan(at(4));
    expect(at(4)).toBeLessThan(at(8));
    expect(at(12)).toBeGreaterThan(profile.thresholds.anomalous);
  });

  it('derives thresholds that separate normal audio from anomalies', () => {
    expect(profile.thresholds.watch).toBeGreaterThan(0.55);
    expect(profile.thresholds.anomalous).toBeGreaterThan(profile.thresholds.watch);
    for (const seed of [99, 100, 101, 102]) {
      expect(scoreCheck(profile, windowsFrom(normalAudio(10, seed))).score).toBeLessThan(
        profile.thresholds.watch,
      );
    }
  });

  it('handles an empty check without throwing', () => {
    const result = scoreCheck(profile, []);
    expect(result.score).toBe(0);
    expect(result.status).toBe('normal');
    expect(result.dominantStateId).toBe('');
  });

  it('reports which state matched', () => {
    const result = scoreCheck(profile, windowsFrom(normalAudio(10, 8)));
    expect(findState(profile, result.dominantStateId)).not.toBeNull();
  });

  it('scores one window consistently with the check', () => {
    const windows = windowsFrom(normalAudio(5, 11));
    const single = scoreWindow(profile, windows[0] as (typeof windows)[number]);
    const check = scoreCheck(profile, windows);
    expect(check.windows[0]?.score).toBeCloseTo(single.score, 10);
  });

  it('maps scores onto statuses at the profile thresholds', () => {
    expect(statusFor(0, profile)).toBe('normal');
    expect(statusFor(profile.thresholds.watch, profile)).toBe('watch');
    expect(statusFor(profile.thresholds.anomalous, profile)).toBe('anomalous');
  });
});

describe('describeDifference', () => {
  it('names the level when the machine gets louder', () => {
    const windows = windowsFrom(applyGainDb(normalAudio(10, 12), 15));
    const result = scoreCheck(profile, windows);
    const descriptors = describeDifference(profile, result.dominantStateId, windows);
    expect(descriptors.length).toBeGreaterThan(0);
    // Band levels all rise with the gain, so reporting them would restate the
    // level change seven times over; the level itself is the only finding.
    expect(descriptors[0]?.feature).toBe('level');
    expect(descriptors[0]?.direction).toBe('higher');
    expect(descriptors[0]?.text).toMatch(/than usual/);
  });

  it('names brightness when the spectrum moves up', () => {
    const windows = windowsFrom(whiteNoise({ seconds: 10, seed: 4 }, 0.1));
    const descriptors = describeDifference(profile, profile.states[0]?.id ?? '', windows, {
      maxDescriptors: 5,
    });
    expect(descriptors.some((d) => d.feature === 'spectralCentroidHz' && d.direction === 'higher')).toBe(
      true,
    );
  });

  it('says nothing about audio that matches the profile', () => {
    const windows = windowsFrom(normalAudio(10, 13));
    expect(describeDifference(profile, profile.states[0]?.id ?? '', windows).length).toBe(0);
  });

  it('returns nothing for an empty check', () => {
    expect(describeDifference(profile, profile.states[0]?.id ?? '', [])).toEqual([]);
  });

  it('honours maxDescriptors', () => {
    const windows = windowsFrom(applyGainDb(whiteNoise({ seconds: 10, seed: 5 }, 0.3), 6));
    expect(describeDifference(profile, '', windows, { maxDescriptors: 2 }).length).toBeLessThanOrEqual(2);
  });

  it('sorts descriptors by how far the feature moved', () => {
    const windows = windowsFrom(applyGainDb(whiteNoise({ seconds: 10, seed: 6 }, 0.3), 6));
    const descriptors = describeDifference(profile, '', windows, { maxDescriptors: 5 });
    for (let i = 1; i < descriptors.length; i += 1) {
      const previous = Math.abs(descriptors[i - 1]?.zScore ?? 0);
      expect(Math.abs(descriptors[i]?.zScore ?? 0)).toBeLessThanOrEqual(previous + 1e-9);
    }
  });
});

describe('createStreamScorer', () => {
  it('stays normal through normal audio', () => {
    const scorer = createStreamScorer(profile);
    let anomalous = 0;
    for (const window of windowsFrom(normalAudio(30, 21))) {
      if (scorer.push(window).status === 'anomalous') anomalous += 1;
    }
    expect(anomalous).toBe(0);
  });

  it('escalates when the machine changes, after the hold period', () => {
    const scorer = createStreamScorer(profile, { statusHoldWindows: 3 });
    for (const window of windowsFrom(normalAudio(20, 22))) scorer.push(window);
    let changed = false;
    for (const window of windowsFrom(injectWhine(normalAudio(20, 23), 5200, 0.1))) {
      if (scorer.push(window).statusChanged) changed = true;
    }
    expect(changed).toBe(true);
    expect(scorer.status).not.toBe('normal');
  });

  it('does not flip status on a single stray window', () => {
    const scorer = createStreamScorer(profile, { statusHoldWindows: 5 });
    const normal = windowsFrom(normalAudio(20, 24));
    const stray = windowsFrom(applyGainDb(normalAudio(2, 25), 25));
    for (const window of normal.slice(0, 20)) scorer.push(window);
    const before = scorer.status;
    scorer.push(stray[0] as (typeof stray)[number]);
    expect(scorer.status).toBe(before);
  });

  it('detects a slow rise in the baseline as drift', () => {
    const scorer = createStreamScorer(profile, {
      driftReferenceWindows: 20,
      driftRecentWindows: 20,
      driftDelta: 0.1,
    });
    for (const window of windowsFrom(normalAudio(15, 26))) scorer.push(window);
    let drifting = false;
    for (const window of windowsFrom(injectWhine(normalAudio(15, 27), 5200, 0.1))) {
      if (scorer.push(window).drift.drifting) drifting = true;
    }
    expect(drifting).toBe(true);
  });

  it('never claims drift before both halves are full', () => {
    const scorer = createStreamScorer(profile, { driftReferenceWindows: 50, driftRecentWindows: 50 });
    for (const window of windowsFrom(normalAudio(10, 28))) {
      expect(scorer.push(window).drift.drifting).toBe(false);
    }
  });

  it('resets to a clean state', () => {
    const scorer = createStreamScorer(profile);
    for (const window of windowsFrom(normalAudio(10, 29))) scorer.push(window);
    scorer.reset();
    expect(scorer.smoothed).toBe(0);
    expect(scorer.status).toBe('normal');
    expect(scorer.drift.drifting).toBe(false);
  });

  it('rejects an out-of-range smoothing factor', () => {
    expect(() => createStreamScorer(profile, { smoothing: 0 })).toThrow(/smoothing/);
    expect(() => createStreamScorer(profile, { smoothing: 2 })).toThrow(/smoothing/);
  });
});

describe('guards', () => {
  const guards = createGuards();
  const windowOf = (rmsDbfs: number, classes: { label: string; score: number }[] = []) => ({
    t: 0,
    embedding: [],
    classes,
    rmsDbfs,
    features: (windowsFrom(tone(440, { seconds: 1.1 }, 0.2))[0] as { features: unknown })
      .features as never,
  });

  it('rejects silence', () => {
    expect(guards.check(windowOf(-90)).reasons).toContain('silence');
  });

  it('rejects an overdriven window', () => {
    expect(guards.check(windowOf(-1)).reasons).toContain('too-loud');
  });

  it('rejects confident speech', () => {
    const verdict = guards.check(windowOf(-30, [{ label: 'Speech', score: 0.8 }]));
    expect(verdict.accepted).toBe(false);
    expect(verdict.interference?.label).toBe('Speech');
  });

  it('accepts quiet background speech below the threshold', () => {
    expect(guards.check(windowOf(-30, [{ label: 'Speech', score: 0.1 }])).accepted).toBe(true);
  });

  it('accepts an ordinary machine window', () => {
    const verdict = guards.check(windowOf(-35, [{ label: 'Mechanical fan', score: 0.7 }]));
    expect(verdict.accepted).toBe(true);
    expect(verdict.reasons).toEqual([]);
  });

  it('filters a batch', () => {
    const kept = guards.filter([windowOf(-30), windowOf(-90), windowOf(-35)]);
    expect(kept.length).toBe(2);
  });

  it('scores human voice presence', () => {
    expect(humanVoiceScore([{ label: 'Speech', score: 0.6 }])).toBe(0.6);
    expect(humanVoiceScore([{ label: 'Mechanical fan', score: 0.9 }])).toBe(0);
    expect(isPossiblyHuman([{ label: 'Singing', score: 0.5 }])).toBe(true);
    expect(isPossiblyHuman([{ label: 'Singing', score: 0.1 }])).toBe(false);
  });

  it('exposes the thresholds it resolved', () => {
    expect(createGuards({ silenceFloorDbfs: -70 }).config.silenceFloorDbfs).toBe(-70);
    expect(createGuards().config.interferenceScore).toBe(0.35);
  });
});
