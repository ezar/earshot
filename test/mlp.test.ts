import { describe, expect, it } from 'vitest';

import { createMlpHead } from '../src/classify/mlp.js';
import { createKnnClassifier, type Example } from '../src/classify/knn.js';
import { syntheticEmbedding } from './helpers.js';

/**
 * These run against the real `@tensorflow/tfjs`, not an injected stand-in.
 *
 * The engine's model path was type-checked and unit-tested for a whole release
 * before anyone asked a browser to load MediaPipe, and it turned out to be
 * broken in two separate ways. `earshot/mlp` was in the same position: typed,
 * and never once executed.
 */
const DIMENSIONS = 32;

function examplesFor<Label extends string>(labels: readonly Label[], perLabel: number): Example<Label>[] {
  const classifier = createKnnClassifier<Label>();
  labels.forEach((label, index) => {
    for (let i = 0; i < perLabel; i += 1) {
      classifier.add(label, syntheticEmbedding(index * 1.7, DIMENSIONS, index * 100 + i, 0.15));
    }
  });
  return classifier.examples();
}

describe('createMlpHead against real TensorFlow.js', () => {
  it('trains on labelled examples and separates them', async () => {
    const labels: ('luna' | 'sol')[] = ['luna', 'sol'];
    const head = await createMlpHead<'luna' | 'sol'>({
      inputDimensions: DIMENSIONS,
      labels,
      hiddenUnits: 16,
    });
    try {
      const result = await head.train(examplesFor(labels, 24), {
        epochs: 30,
        batchSize: 8,
        validationSplit: 0,
      });
      expect(Number.isFinite(result.loss)).toBe(true);
      expect(result.epochs).toBe(30);

      const prediction = await head.predict(syntheticEmbedding(0, DIMENSIONS, 9999));
      expect(prediction.label).toBe('luna');
      expect(prediction.confidence).toBeGreaterThan(0.5);
      // Scores are a softmax, so they are a distribution over the labels.
      const total = prediction.scores.reduce((sum, s) => sum + s.score, 0);
      expect(total).toBeCloseTo(1, 4);
      expect(prediction.scores.length).toBe(2);
    } finally {
      head.dispose();
    }
  }, 60000);

  it('reports a validation accuracy when a split is asked for', async () => {
    const head = await createMlpHead({ inputDimensions: DIMENSIONS, labels: ['a', 'b'], hiddenUnits: 8 });
    try {
      const result = await head.train(examplesFor(['a', 'b'], 20), {
        epochs: 10,
        batchSize: 8,
        validationSplit: 0.25,
      });
      expect(result.validationAccuracy).not.toBeNull();
      expect(result.validationAccuracy as number).toBeGreaterThanOrEqual(0);
      expect(result.validationAccuracy as number).toBeLessThanOrEqual(1);
    } finally {
      head.dispose();
    }
  }, 60000);

  it('calls back on every epoch', async () => {
    const head = await createMlpHead({ inputDimensions: DIMENSIONS, labels: ['a', 'b'], hiddenUnits: 8 });
    const seen: number[] = [];
    try {
      await head.train(examplesFor(['a', 'b'], 12), {
        epochs: 5,
        batchSize: 4,
        validationSplit: 0,
        onEpoch: (epoch, loss) => {
          seen.push(epoch);
          expect(Number.isFinite(loss)).toBe(true);
        },
      });
      expect(seen).toEqual([0, 1, 2, 3, 4]);
    } finally {
      head.dispose();
    }
  }, 60000);

  it('rejects an example carrying a label it was not built with', async () => {
    const head = await createMlpHead({ inputDimensions: DIMENSIONS, labels: ['a', 'b'] });
    try {
      await expect(head.train(examplesFor(['a', 'z'] as ('a' | 'b')[], 4))).rejects.toThrow(
        /unknown label "z"/,
      );
    } finally {
      head.dispose();
    }
  }, 60000);

  it('refuses to build a head with fewer than two labels', async () => {
    await expect(createMlpHead({ inputDimensions: DIMENSIONS, labels: ['only'] })).rejects.toThrow(
      /at least two labels/,
    );
  });

  it('refuses to train on nothing', async () => {
    const head = await createMlpHead({ inputDimensions: DIMENSIONS, labels: ['a', 'b'] });
    try {
      await expect(head.train([])).rejects.toThrow(/at least one example/);
    } finally {
      head.dispose();
    }
  }, 60000);
});
