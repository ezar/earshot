import { describe, expect, it } from 'vitest';

import { kmeans, selectClustering, silhouette, squaredDistance } from '../src/learn/kmeans.js';
import {
  buildDistanceDistribution,
  distanceScore,
  fitDiagonalGaussian,
  mahalanobisDistance,
} from '../src/learn/statistics.js';
import { calibrate, learnProfile, PROFILE_SCHEMA_VERSION } from '../src/learn/profile.js';
import { createRandom } from '../src/util/math.js';
import { applyGainDb, injectWhine, pinkNoise, tone } from '../fixtures/synthetic/index.js';
import { windowsFrom } from './helpers.js';

function blob(centre: readonly number[], count: number, spread: number, seed: number): Float32Array[] {
  const random = createRandom(seed);
  return Array.from({ length: count }, () =>
    Float32Array.from(centre, (value) => value + (random() * 2 - 1) * spread),
  );
}

describe('kmeans', () => {
  it('separates two well-spaced blobs', () => {
    const vectors = [...blob([0, 0], 40, 0.4, 1), ...blob([10, 10], 40, 0.4, 2)];
    const clustering = kmeans(vectors, { k: 2 });
    expect(clustering.k).toBe(2);
    const first = clustering.assignments.slice(0, 40);
    const second = clustering.assignments.slice(40);
    expect(new Set(first).size).toBe(1);
    expect(new Set(second).size).toBe(1);
    expect(first[0]).not.toBe(second[0]);
    expect(clustering.silhouette).toBeGreaterThan(0.8);
  });

  it('is deterministic for a given seed', () => {
    const vectors = [...blob([0, 0], 30, 1, 1), ...blob([6, 6], 30, 1, 2)];
    const a = kmeans(vectors, { k: 2, seed: 7 });
    const b = kmeans(vectors, { k: 2, seed: 7 });
    expect(a.assignments).toEqual(b.assignments);
    expect(a.inertia).toBeCloseTo(b.inertia, 10);
  });

  it('rejects an impossible k', () => {
    expect(() => kmeans(blob([0], 3, 1, 1), { k: 5 })).toThrow(/k must be in/);
    expect(() => kmeans([], { k: 1 })).toThrow(/at least one vector/);
  });

  it('reports zero silhouette for a single cluster', () => {
    const vectors = blob([0, 0], 20, 1, 1);
    expect(silhouette(vectors, new Array<number>(20).fill(0), 1)).toBe(0);
  });

  it('computes squared distance', () => {
    expect(squaredDistance([0, 0], [3, 4])).toBe(25);
  });
});

describe('selectClustering', () => {
  it('finds three states when the data has three', () => {
    const vectors = [
      ...blob([0, 0], 40, 0.3, 1),
      ...blob([8, 0], 40, 0.3, 2),
      ...blob([4, 8], 40, 0.3, 3),
    ];
    expect(selectClustering(vectors, { maxK: 5 }).k).toBe(3);
  });

  it('stays at one state for a single blob', () => {
    expect(selectClustering(blob([0, 0], 60, 1, 5), { maxK: 5 }).k).toBe(1);
  });

  it('refuses a K that would create a tiny cluster', () => {
    // 58 points in one blob and 2 far away: splitting them off leaves a cluster
    // below the 8 % floor, so the extra state is rejected.
    const vectors = [...blob([0, 0], 58, 0.5, 1), ...blob([40, 40], 2, 0.1, 2)];
    expect(selectClustering(vectors, { maxK: 4, minClusterFraction: 0.08 }).k).toBe(1);
  });
});

describe('statistics', () => {
  it('recovers the mean and variance of a Gaussian blob', () => {
    const model = fitDiagonalGaussian(blob([5, -2], 400, 2, 9), 0);
    expect(model.mean[0]).toBeCloseTo(5, 0);
    expect(model.mean[1]).toBeCloseTo(-2, 0);
    expect(model.count).toBe(400);
  });

  it('floors the variance so a constant dimension cannot explode the distance', () => {
    const constant = Array.from({ length: 20 }, () => Float32Array.from([1, 1]));
    const model = fitDiagonalGaussian(constant);
    expect(model.variance.every((value) => value > 0)).toBe(true);
    expect(Number.isFinite(mahalanobisDistance([1, 5], model))).toBe(true);
  });

  it('shrinks a thin sample towards the pooled variance', () => {
    // One dimension is wide, the other nearly constant; shrinkage lifts the
    // narrow one towards the pooled estimate.
    const vectors = [
      Float32Array.from([0, 1]),
      Float32Array.from([10, 1.0001]),
      Float32Array.from([-10, 0.9999]),
    ];
    const shrunk = fitDiagonalGaussian(vectors, 1);
    const raw = fitDiagonalGaussian(vectors, 0);
    expect(shrunk.variance[1] as number).toBeGreaterThan(raw.variance[1] as number);
  });

  it('maps everything it learned into the bottom half of the scale', () => {
    const distribution = buildDistanceDistribution(Array.from({ length: 100 }, (_, i) => i / 100));
    expect(distanceScore(0, distribution)).toBeLessThan(0.05);
    expect(distanceScore(0.5, distribution)).toBeCloseTo(0.25, 1);
    expect(distanceScore(distribution.max, distribution)).toBeCloseTo(0.5, 6);
  });

  it('keeps distances beyond the learned range distinguishable', () => {
    const distribution = buildDistanceDistribution(Array.from({ length: 100 }, (_, i) => i / 100));
    const slightly = distanceScore(distribution.max * 1.5, distribution);
    const far = distanceScore(distribution.max * 5, distribution);
    expect(slightly).toBeGreaterThan(0.5);
    expect(far).toBeGreaterThan(slightly + 0.1);
    expect(far).toBeLessThanOrEqual(1);
  });

  it('is continuous at the learned maximum', () => {
    const distribution = buildDistanceDistribution(Array.from({ length: 100 }, (_, i) => i / 100));
    const inside = distanceScore(distribution.max - 1e-9, distribution);
    const outside = distanceScore(distribution.max + 1e-9, distribution);
    expect(Math.abs(outside - inside)).toBeLessThan(0.01);
  });

  it('thins a large sample but keeps its percentiles', () => {
    const distribution = buildDistanceDistribution(
      Array.from({ length: 5000 }, (_, i) => i / 5000),
      256,
    );
    expect(distribution.sorted.length).toBe(256);
    expect(distribution.p50).toBeCloseTo(0.5, 2);
    expect(distribution.p95).toBeCloseTo(0.95, 2);
  });
});

describe('learnProfile', () => {
  const steady = (seconds: number, seed: number): Float32Array =>
    injectWhine(pinkNoise({ seconds, seed }, 0.08), 1200, 0.02);

  it('learns a profile from steady machine noise', () => {
    const profile = learnProfile(windowsFrom(steady(30, 1)));
    expect(profile.schemaVersion).toBe(PROFILE_SCHEMA_VERSION);
    expect(profile.revision).toBe(1);
    expect(profile.featureSpace).toBe('features');
    expect(profile.states.length).toBeGreaterThanOrEqual(1);
    expect(profile.windowCount).toBeGreaterThan(30);
    const totalWeight = profile.states.reduce((sum, state) => sum + state.weight, 0);
    expect(totalWeight).toBeCloseTo(1, 6);
  });

  it('refuses to learn from too few windows', () => {
    expect(() => learnProfile(windowsFrom(steady(2, 1)))).toThrow(/at least 8 windows/);
  });

  it('records feature statistics the descriptors can use', () => {
    const profile = learnProfile(windowsFrom(steady(30, 1)));
    const stats = profile.states[0]?.featureStats;
    expect(stats?.['level']).toBeDefined();
    expect(stats?.['spectralCentroidHz']?.mean).toBeGreaterThan(0);
  });

  it('uses the embedding space when embeddings are present', () => {
    const windows = windowsFrom(steady(30, 1), (_, index) => [
      Math.sin(index * 0.1),
      Math.cos(index * 0.1),
      0.5,
    ]);
    const profile = learnProfile(windows);
    expect(profile.featureSpace).toBe('embedding');
    expect(profile.dimensions).toBe(3);
  });

  it('separates two operating modes into two states', () => {
    const quiet = windowsFrom(steady(20, 1));
    const loud = windowsFrom(applyGainDb(injectWhine(pinkNoise({ seconds: 20, seed: 2 }, 0.08), 4000, 0.05), 14));
    const profile = learnProfile([...quiet, ...loud], { maxK: 4 });
    expect(profile.states.length).toBeGreaterThanOrEqual(2);
  });
});

describe('calibrate', () => {
  const profile = learnProfile(windowsFrom(pinkNoise({ seconds: 25, seed: 3 }, 0.08)));

  it('raises the thresholds when the user forgives a flagged check', () => {
    const next = calibrate(profile, 'normal', [], { score: 0.95 });
    expect(next.revision).toBe(profile.revision + 1);
    expect(next.thresholds.watch).toBeGreaterThan(profile.thresholds.watch);
    expect(next.thresholds.anomalous).toBeGreaterThanOrEqual(profile.thresholds.anomalous);
    expect(next.calibrations.length).toBe(1);
    expect(next.calibrations[0]?.verdict).toBe('normal');
  });

  it('lowers the thresholds when the user reports a missed anomaly', () => {
    const next = calibrate(profile, 'anomalous', [], { score: 0.6 });
    expect(next.thresholds.anomalous).toBeLessThan(profile.thresholds.anomalous);
    expect(next.thresholds.watch).toBeLessThan(next.thresholds.anomalous);
  });

  it('never leaves the thresholds out of order or out of range', () => {
    let current = profile;
    for (let i = 0; i < 30; i += 1) {
      current = calibrate(current, i % 2 === 0 ? 'normal' : 'anomalous', [], { score: i % 2 === 0 ? 0.99 : 0.2 });
      expect(current.thresholds.watch).toBeLessThan(current.thresholds.anomalous);
      expect(current.thresholds.watch).toBeGreaterThanOrEqual(0.4);
      expect(current.thresholds.anomalous).toBeLessThanOrEqual(0.999);
    }
  });

  it('leaves a profile alone when the verdict agrees with it', () => {
    const next = calibrate(profile, 'normal', [], { score: 0.1 });
    expect(next.thresholds).toEqual(profile.thresholds);
  });

  it('does not mutate the input profile', () => {
    const before = JSON.stringify(profile);
    calibrate(profile, 'anomalous', [], { score: 0.5 });
    expect(JSON.stringify(profile)).toBe(before);
  });

  it('survives a JSON round trip', () => {
    const restored = JSON.parse(JSON.stringify(profile)) as typeof profile;
    expect(restored.states.length).toBe(profile.states.length);
    expect(restored.thresholds).toEqual(profile.thresholds);
  });

  it('keeps a tone profile distinguishable from a noise profile', () => {
    const tonal = learnProfile(windowsFrom(tone(800, { seconds: 25 }, 0.2)));
    expect(tonal.states[0]?.featureStats['spectralFlatness']?.mean ?? 1).toBeLessThan(
      profile.states[0]?.featureStats['spectralFlatness']?.mean ?? 0,
    );
  });
});
