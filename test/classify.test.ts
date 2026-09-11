import { describe, expect, it } from 'vitest';

import {
  buildPrototypes,
  createKnnClassifier,
  predictByPrototype,
  predictWith,
  KNN_SCHEMA_VERSION,
} from '../src/classify/knn.js';
import { crossValidate } from '../src/classify/validate.js';
import { dequantize, float16ToFloat32, float32ToFloat16, quantize } from '../src/util/quantize.js';
import { cosineSimilarity } from '../src/util/math.js';
import { syntheticEmbedding } from './helpers.js';

const DIMENSIONS = 64;

/** Ten examples per label, jittered around a label-specific centre. */
function population(labels: readonly string[], perLabel = 10, jitter = 0.15) {
  const classifier = createKnnClassifier<string>();
  labels.forEach((label, labelIndex) => {
    for (let i = 0; i < perLabel; i += 1) {
      classifier.add(label, syntheticEmbedding(labelIndex * 1.7, DIMENSIONS, labelIndex * 100 + i, jitter));
    }
  });
  return classifier;
}

describe('createKnnClassifier', () => {
  it('separates two well-spaced labels', () => {
    const classifier = population(['luna', 'sol']);
    const query = syntheticEmbedding(0, DIMENSIONS, 999);
    const prediction = classifier.predict(query);
    expect(prediction.label).toBe('luna');
    expect(prediction.confidence).toBeGreaterThan(0.7);
    expect(prediction.neighbours.length).toBe(5);
  });

  it('reports nothing when it has no examples', () => {
    const prediction = createKnnClassifier().predict(syntheticEmbedding(0, DIMENSIONS, 1));
    expect(prediction.label).toBeNull();
    expect(prediction.confidence).toBe(0);
  });

  it('works from a single example', () => {
    const classifier = createKnnClassifier<string>();
    classifier.add('luna', syntheticEmbedding(0, DIMENSIONS, 1));
    expect(classifier.predict(syntheticEmbedding(0, DIMENSIONS, 2)).label).toBe('luna');
  });

  it('weights a near-identical neighbour above a merely similar one', () => {
    const classifier = createKnnClassifier<string>({ k: 3 });
    const target = syntheticEmbedding(0, DIMENSIONS, 1, 0);
    classifier.add('exact', target);
    classifier.add('other', syntheticEmbedding(1.5, DIMENSIONS, 2, 0));
    classifier.add('other', syntheticEmbedding(1.6, DIMENSIONS, 3, 0));
    const prediction = classifier.predict(target);
    expect(prediction.label).toBe('exact');
  });

  it('removes examples by id and by label', () => {
    const classifier = createKnnClassifier<string>();
    const id = classifier.add('luna', syntheticEmbedding(0, DIMENSIONS, 1));
    classifier.add('sol', syntheticEmbedding(1.7, DIMENSIONS, 2));
    classifier.add('sol', syntheticEmbedding(1.7, DIMENSIONS, 3));
    expect(classifier.remove(id)).toBe(true);
    expect(classifier.remove('missing')).toBe(false);
    expect(classifier.removeLabel('sol')).toBe(2);
    expect(classifier.size).toBe(0);
  });

  it('lists its labels in insertion order', () => {
    const classifier = population(['luna', 'sol', 'nube'], 2);
    expect(classifier.labels()).toEqual(['luna', 'sol', 'nube']);
  });

  it('survives a snapshot round trip', () => {
    const classifier = population(['luna', 'sol']);
    const query = syntheticEmbedding(0, DIMENSIONS, 999);
    const before = classifier.predict(query);
    const snapshot = JSON.parse(JSON.stringify(classifier.toJSON())) as ReturnType<typeof classifier.toJSON>;
    expect(snapshot.schemaVersion).toBe(KNN_SCHEMA_VERSION);

    const restored = createKnnClassifier<string>();
    restored.load(snapshot);
    const after = restored.predict(query);
    expect(after.label).toBe(before.label);
    expect(after.confidence).toBeCloseTo(before.confidence, 10);
  });

  it('honours a distance cut-off', () => {
    const classifier = createKnnClassifier<string>({ maxDistance: 0.001 });
    classifier.add('luna', syntheticEmbedding(0, DIMENSIONS, 1, 0));
    expect(classifier.predict(syntheticEmbedding(2.5, DIMENSIONS, 2, 0)).label).toBeNull();
  });

  it('is unaffected by the scale of the query vector', () => {
    const classifier = population(['luna', 'sol']);
    const query = syntheticEmbedding(0, DIMENSIONS, 999);
    const scaled = query.map((value) => value * 17);
    expect(classifier.predict(scaled).label).toBe(classifier.predict(query).label);
  });
});

describe('prototypes', () => {
  it('averages each label into a unit vector', () => {
    const prototypes = buildPrototypes(population(['luna', 'sol']).examples());
    expect(prototypes.length).toBe(2);
    for (const prototype of prototypes) {
      expect(prototype.count).toBe(10);
      expect(cosineSimilarity(prototype.embedding, prototype.embedding)).toBeCloseTo(1, 6);
    }
  });

  it('classifies by nearest prototype', () => {
    const prototypes = buildPrototypes(population(['luna', 'sol']).examples());
    const prediction = predictByPrototype(prototypes, syntheticEmbedding(0, DIMENSIONS, 999));
    expect(prediction.label).toBe('luna');
  });

  it('reports nothing without prototypes', () => {
    expect(predictByPrototype([], syntheticEmbedding(0, DIMENSIONS, 1)).label).toBeNull();
  });
});

describe('crossValidate', () => {
  it('reports high accuracy on well-separated labels', () => {
    const report = crossValidate(population(['luna', 'sol', 'nube']).examples(), { folds: 5 });
    expect(report.accuracy).toBeGreaterThan(0.9);
    expect(report.total).toBe(30);
    expect(report.labels.length).toBe(3);
    for (const label of report.labels) {
      expect(label.support).toBe(10);
      expect(label.recall).toBeGreaterThan(0.8);
    }
  });

  it('reports low accuracy on labels that overlap completely', () => {
    const classifier = createKnnClassifier<string>();
    for (let i = 0; i < 12; i += 1) {
      classifier.add(i % 2 === 0 ? 'a' : 'b', syntheticEmbedding(0, DIMENSIONS, i, 1.2));
    }
    expect(crossValidate(classifier.examples(), { folds: 3 }).accuracy).toBeLessThan(0.8);
  });

  it('is honest about a single label', () => {
    const report = crossValidate(population(['luna']).examples());
    expect(report.accuracy).toBe(0);
    expect(report.folds).toBe(0);
    expect(report.labels[0]?.support).toBe(10);
  });

  it('caps the folds at the smallest label support', () => {
    const classifier = createKnnClassifier<string>();
    for (let i = 0; i < 10; i += 1) classifier.add('many', syntheticEmbedding(0, DIMENSIONS, i));
    classifier.add('few', syntheticEmbedding(1.7, DIMENSIONS, 50));
    classifier.add('few', syntheticEmbedding(1.7, DIMENSIONS, 51));
    expect(crossValidate(classifier.examples(), { folds: 5 }).folds).toBe(2);
  });

  it('builds a confusion matrix that accounts for every example', () => {
    const report = crossValidate(population(['luna', 'sol']).examples(), { folds: 5 });
    let counted = 0;
    for (const row of Object.values(report.confusion)) {
      for (const count of Object.values(row)) counted += count;
    }
    expect(counted).toBe(report.total);
  });

  it('is deterministic for a given seed', () => {
    const examples = population(['luna', 'sol', 'nube']).examples();
    expect(crossValidate(examples, { seed: 3 })).toEqual(crossValidate(examples, { seed: 3 }));
  });
});

describe('predictWith', () => {
  it('matches the classifier it backs', () => {
    const classifier = population(['luna', 'sol']);
    const query = syntheticEmbedding(0, DIMENSIONS, 999);
    expect(predictWith(classifier.examples(), query, 5).label).toBe(classifier.predict(query).label);
  });
});

describe('quantize', () => {
  const vector = Array.from({ length: 128 }, (_, i) => Math.sin(i * 0.31) * 0.7);

  it('round-trips float16 within half-precision resolution', () => {
    const restored = dequantize(quantize(vector, 'float16'));
    expect(restored.length).toBe(vector.length);
    for (let i = 0; i < vector.length; i += 1) {
      expect(restored[i] as number).toBeCloseTo(vector[i] as number, 3);
    }
  });

  it('round-trips int8 closely enough for cosine distance', () => {
    const restored = dequantize(quantize(vector, 'int8'));
    expect(cosineSimilarity(vector, restored)).toBeGreaterThan(0.9999);
  });

  it('halves and quarters the payload', () => {
    const float16 = quantize(vector, 'float16');
    const int8 = quantize(vector, 'int8');
    expect(float16.data.length).toBeGreaterThan(int8.data.length * 1.9);
    expect(float16.dimensions).toBe(vector.length);
    expect(int8.scale).toBeGreaterThan(0);
  });

  it('handles an all-zero vector', () => {
    const restored = dequantize(quantize(new Array<number>(16).fill(0), 'int8'));
    expect(Array.from(restored)).toEqual(new Array<number>(16).fill(0));
  });

  it('rejects a payload of the wrong length', () => {
    const corrupt = { ...quantize(vector, 'int8'), dimensions: 999 };
    expect(() => dequantize(corrupt)).toThrow(/expected 999/);
  });

  it('survives a JSON round trip', () => {
    const quantized = quantize(vector, 'float16');
    const restored = dequantize(JSON.parse(JSON.stringify(quantized)) as typeof quantized);
    expect(restored[0] as number).toBeCloseTo(vector[0] as number, 3);
  });

  it('converts float16 bit patterns exactly at known values', () => {
    expect(float16ToFloat32(float32ToFloat16(1))).toBe(1);
    expect(float16ToFloat32(float32ToFloat16(-2))).toBe(-2);
    expect(float16ToFloat32(float32ToFloat16(0))).toBe(0);
    expect(float16ToFloat32(float32ToFloat16(0.5))).toBe(0.5);
    expect(float16ToFloat32(float32ToFloat16(1e6))).toBe(Number.POSITIVE_INFINITY);
  });
});
