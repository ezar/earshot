import { describe, expect, it, vi } from 'vitest';

import { SAMPLE_RATE_HZ, WINDOW_SAMPLES } from '../src/constants.js';
import { createEngine } from '../src/engine.js';
import { createClassifier, maxScoreOf, mergeClassifications, scoreOf } from '../src/models/classifier.js';
import { averageEmbeddings, createEmbedder } from '../src/models/embedder.js';
import type { TasksAudioModule } from '../src/models/tasks-audio.js';
import type { EngineRequest, EngineResponse } from '../src/worker/protocol.js';
import { createFeatureExtractor } from '../src/dsp/features.js';
import { createFramer } from '../src/dsp/framing.js';
import { tone } from '../fixtures/synthetic/index.js';

/**
 * A Worker stand-in that runs the same framing and feature extraction the real
 * worker does, so the engine's protocol is exercised end to end without WASM.
 */
function createFakeWorker(): Worker & { requests: EngineRequest['type'][] } {
  const framer = createFramer();
  const extractor = createFeatureExtractor();
  const requests: EngineRequest['type'][] = [];
  const worker = {
    requests,
    onmessage: null as ((event: MessageEvent<EngineResponse>) => void) | null,
    onerror: null,
    terminate: vi.fn(),
    postMessage(request: EngineRequest): void {
      requests.push(request.type);
      const reply = (response: EngineResponse): void => {
        queueMicrotask(() => worker.onmessage?.({ data: response } as MessageEvent<EngineResponse>));
      };
      switch (request.type) {
        case 'init':
          reply({ type: 'ready', id: request.id, classifier: true, embedder: true });
          return;
        case 'push': {
          const windows = framer.push(request.samples).map((frame) => {
            const features = extractor.extract(frame.samples);
            return {
              t: frame.t,
              embedding: [features.rmsDbfs],
              classes: [{ label: 'Mechanical fan', score: 0.5 }],
              rmsDbfs: features.rmsDbfs,
              features,
            };
          });
          reply({ type: 'windows', id: request.id, windows });
          return;
        }
        default:
          reply({ type: 'done', id: request.id });
      }
    },
  };
  return worker as unknown as Worker & { requests: EngineRequest['type'][] };
}

const models = { wasmBaseUrl: 'https://example.invalid/wasm', classifierUrl: 'a', embedderUrl: 'b' };

describe('createEngine', () => {
  it('initializes and reports which models loaded', async () => {
    const worker = createFakeWorker();
    const engine = await createEngine({ workerUrl: 'ignored', models, createWorker: () => worker });
    expect(engine.hasClassifier).toBe(true);
    expect(engine.hasEmbedder).toBe(true);
    expect(worker.requests[0]).toBe('init');
  });

  it('turns pushed audio into windows on the documented grid', async () => {
    const engine = await createEngine({ workerUrl: 'x', models, createWorker: createFakeWorker });
    const windows = await engine.push(tone(440, { seconds: 3 }, 0.3));
    expect(windows.length).toBeGreaterThan(3);
    expect(windows[0]?.t).toBeCloseTo(0, 6);
    expect(windows[1]?.t).toBeCloseTo(0.4875, 6);
    expect(windows[0]?.features.logMel.length).toBe(64);
  });

  it('accumulates across pushes rather than dropping partial chunks', async () => {
    const engine = await createEngine({ workerUrl: 'x', models, createWorker: createFakeWorker });
    const audio = tone(440, { seconds: 3 }, 0.3);
    let total = 0;
    for (let offset = 0; offset < audio.length; offset += 1024) {
      total += (await engine.push(audio.slice(offset, offset + 1024))).length;
    }
    expect(total).toBe(Math.floor((audio.length - WINDOW_SAMPLES) / 7800) + 1);
  });

  it('notifies subscribers of every window', async () => {
    const engine = await createEngine({ workerUrl: 'x', models, createWorker: createFakeWorker });
    const seen: number[] = [];
    const unsubscribe = engine.onWindow((window) => seen.push(window.t));
    const windows = await engine.push(tone(440, { seconds: 3 }, 0.3));
    expect(seen.length).toBe(windows.length);
    unsubscribe();
    await engine.push(tone(440, { seconds: 2 }, 0.3));
    expect(seen.length).toBe(windows.length);
  });

  it('terminates the worker on close', async () => {
    const worker = createFakeWorker();
    const engine = await createEngine({ workerUrl: 'x', models, createWorker: () => worker });
    await engine.close();
    expect(worker.terminate).toHaveBeenCalled();
  });

  it('surfaces a worker error as a rejection', async () => {
    const failing = {
      onmessage: null as ((event: MessageEvent<EngineResponse>) => void) | null,
      onerror: null,
      terminate: vi.fn(),
      postMessage(request: EngineRequest): void {
        queueMicrotask(() =>
          failing.onmessage?.({
            data: { type: 'error', id: request.id, message: 'model not found' },
          } as MessageEvent<EngineResponse>),
        );
      },
    };
    await expect(
      createEngine({ workerUrl: 'x', models, createWorker: () => failing as unknown as Worker }),
    ).rejects.toThrow(/model not found/);
  });
});

describe('model wrappers', () => {
  const fakeTasks = (embedding: number[], categories: { categoryName: string; score: number }[]) =>
    ({
      FilesetResolver: { forAudioTasks: async () => ({}) },
      AudioClassifier: {
        createFromOptions: async () => ({
          classify: () => [{ classifications: [{ categories }] }],
          close: () => undefined,
        }),
      },
      AudioEmbedder: {
        createFromOptions: async () => ({
          embed: () => [{ embeddings: [{ floatEmbedding: embedding }] }],
          close: () => undefined,
        }),
      },
    }) as unknown as TasksAudioModule;

  it('classifies through the injected module', async () => {
    const tasks = fakeTasks([], [
      { categoryName: 'Cat', score: 0.8 },
      { categoryName: 'Speech', score: 0.2 },
    ]);
    const classifier = await createClassifier({
      wasmBaseUrl: 'w',
      classifierUrl: 'c',
      loadTasksAudio: async () => tasks,
    });
    const classes = classifier.classify(new Float32Array(SAMPLE_RATE_HZ));
    expect(classes[0]?.label).toBe('Cat');
    expect(scoreOf(classes, 'Speech')).toBeCloseTo(0.2, 6);
    expect(maxScoreOf(classes, ['Speech', 'Cat'])).toBeCloseTo(0.8, 6);
    classifier.close();
  });

  it('embeds through the injected module', async () => {
    const embedder = await createEmbedder({
      wasmBaseUrl: 'w',
      embedderUrl: 'e',
      loadTasksAudio: async () => fakeTasks([0.1, 0.2, 0.3], []),
    });
    expect(Array.from(embedder.embed(new Float32Array(SAMPLE_RATE_HZ)))).toEqual([
      expect.closeTo(0.1, 6),
      expect.closeTo(0.2, 6),
      expect.closeTo(0.3, 6),
    ]);
    embedder.close();
  });

  it('requires the app to supply model URLs', async () => {
    await expect(createClassifier({ wasmBaseUrl: 'w' })).rejects.toThrow(/classifierUrl/);
    await expect(createEmbedder({ wasmBaseUrl: 'w' })).rejects.toThrow(/embedderUrl/);
  });

  it('keeps the strongest score per class across patches', () => {
    const merged = mergeClassifications(
      [
        { classifications: [{ categories: [{ categoryName: 'Cat', score: 0.2 }] }] },
        { classifications: [{ categories: [{ categoryName: 'Cat', score: 0.9 }] }] },
      ],
      5,
    );
    expect(merged).toEqual([{ label: 'Cat', score: 0.9 }]);
  });

  it('averages embeddings across patches', () => {
    const averaged = averageEmbeddings([
      { embeddings: [{ floatEmbedding: [0, 2] }] },
      { embeddings: [{ floatEmbedding: [2, 4] }] },
    ]);
    expect(Array.from(averaged)).toEqual([1, 3]);
  });

  it('returns an empty embedding when the model gives nothing', () => {
    expect(averageEmbeddings([{ embeddings: [] }]).length).toBe(0);
  });
});
