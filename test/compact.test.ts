import { describe, expect, it } from 'vitest';

import { compactProfile, expandProfile, COMPACT_PROFILE_SCHEMA_VERSION } from '../src/learn/compact.js';
import { learnProfile } from '../src/learn/profile.js';
import { scoreCheck } from '../src/score/score.js';
import { injectWhine, pinkNoise } from '../fixtures/synthetic/index.js';
import { windowsFrom, syntheticEmbedding } from './helpers.js';

const audio = (seconds: number, seed: number): Float32Array =>
  injectWhine(pinkNoise({ seconds, seed }, 0.08), 1200, 0.02);

/** A profile in the embedding space, which is the case worth compacting. */
const embeddingProfile = learnProfile(
  windowsFrom(audio(40, 1), (_features, index) => syntheticEmbedding(index * 0.05, 256, index)),
);
const featureProfile = learnProfile(windowsFrom(audio(30, 2)));

describe('compactProfile', () => {
  it('round-trips without changing what a check scores', () => {
    const check = windowsFrom(audio(10, 9), (_f, i) => syntheticEmbedding(i * 0.05, 256, 500 + i));
    const before = scoreCheck(embeddingProfile, check);
    const after = scoreCheck(expandProfile(compactProfile(embeddingProfile)), check);
    expect(after.score).toBeCloseTo(before.score, 3);
    expect(after.status).toBe(before.status);
    expect(after.dominantStateId).toBe(before.dominantStateId);
  });

  it('leaves a features-space profile exact, because quantizing it would cost accuracy', () => {
    // Its dimensions run from negative decibels to thousands of hertz, where
    // float16's three significant digits are a visible error — and there are
    // only ~70 of them per state, so there is nothing to save.
    const check = windowsFrom(audio(10, 10));
    const before = scoreCheck(featureProfile, check);
    const after = scoreCheck(expandProfile(compactProfile(featureProfile)), check);
    expect(after.score).toBe(before.score);
    expect(Array.isArray(compactProfile(featureProfile).states[0]?.mean)).toBe(true);
  });

  it('quantizes an embedding-space profile', () => {
    expect(Array.isArray(compactProfile(embeddingProfile).states[0]?.mean)).toBe(false);
  });

  it('is substantially smaller as JSON', () => {
    const plain = JSON.stringify(embeddingProfile).length;
    const compact = JSON.stringify(compactProfile(embeddingProfile)).length;
    expect(compact).toBeLessThan(plain / 2);
  });

  it('keeps everything a person reads when debugging', () => {
    const compact = compactProfile(embeddingProfile);
    expect(compact.thresholds).toEqual(embeddingProfile.thresholds);
    expect(compact.revision).toBe(embeddingProfile.revision);
    expect(compact.featureSpace).toBe(embeddingProfile.featureSpace);
    expect(compact.windowCount).toBe(embeddingProfile.windowCount);
    expect(compact.states[0]?.featureStats).toEqual(embeddingProfile.states[0]?.featureStats);
    expect(compact.states[0]?.distances).toEqual(embeddingProfile.states[0]?.distances);
    expect(compact.states[0]?.weight).toBe(embeddingProfile.states[0]?.weight);
  });

  it('survives a JSON round trip, which is the point of it', () => {
    const stored = JSON.parse(JSON.stringify(compactProfile(embeddingProfile))) as ReturnType<
      typeof compactProfile
    >;
    const restored = expandProfile(stored);
    expect(restored.states.length).toBe(embeddingProfile.states.length);
    expect(restored.states[0]?.model.mean.length).toBe(embeddingProfile.states[0]?.model.mean.length);
  });

  it('preserves variances well enough that no dimension collapses', () => {
    // A variance rounding to zero would make the Mahalanobis distance infinite.
    const restored = expandProfile(compactProfile(embeddingProfile));
    for (const state of restored.states) {
      expect(state.model.variance.every((v) => v > 0 && Number.isFinite(v))).toBe(true);
    }
  });

  it('refuses a compact form from a future version', () => {
    const compact = { ...compactProfile(featureProfile), compactVersion: 99 };
    expect(() => expandProfile(compact)).toThrow(/version 99 is not supported/);
  });

  it('carries its own schema version', () => {
    expect(compactProfile(featureProfile).compactVersion).toBe(COMPACT_PROFILE_SCHEMA_VERSION);
  });
});
