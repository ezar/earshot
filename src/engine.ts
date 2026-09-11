/**
 * The engine: the main-thread handle on the Worker that turns audio chunks
 * into {@link WindowResult}s.
 *
 * The engine owns the framing, so callers may push chunks of any size — a
 * 2048-sample worklet chunk, a whole WAV file — and receive complete
 * 0.975 s windows on the 0.4875 s grid.
 */

import { SAMPLE_RATE_HZ } from './constants.js';
import type { FeatureOptions } from './dsp/features.js';
import type { ModelUrls } from './models/tasks-audio.js';
import type { WindowResult } from './util/types.js';
import type { EngineRequest, EngineResponse } from './worker/protocol.js';

/** Options for {@link createEngine}. */
export interface EngineOptions {
  /** URL of the `earshot/worker` entry point, loaded with `?worker&url`. */
  readonly workerUrl: string;
  /** Model and WASM locations; earshot never supplies defaults. */
  readonly models: Omit<ModelUrls, 'loadTasksAudio'>;
  /** Feature extraction overrides. */
  readonly features?: FeatureOptions;
  /** Sample rate of the audio that will be pushed, in Hz. Defaults to {@link SAMPLE_RATE_HZ}. */
  readonly sampleRateHz?: number;
  /**
   * Construct the Worker yourself, e.g. to control its name or credentials.
   *
   * The default builds a **classic** worker, which is what MediaPipe needs;
   * override this only if you are supplying models some other way.
   */
  readonly createWorker?: (url: string) => Worker;
}

/** A running engine. */
export interface Engine {
  /**
   * Push audio and receive every window that completed.
   *
   * @param samples - Mono samples at the engine's sample rate. The buffer is
   *   transferred to the worker and must not be reused by the caller.
   */
  push(samples: Float32Array): Promise<WindowResult[]>;
  /** Subscribe to windows produced by any `push`; returns an unsubscribe function. */
  onWindow(listener: (window: WindowResult) => void): () => void;
  /** Drop buffered audio that has not yet formed a window. */
  reset(): Promise<void>;
  /** Whether a classifier was loaded. */
  readonly hasClassifier: boolean;
  /** Whether an embedder was loaded. */
  readonly hasEmbedder: boolean;
  /** Terminate the worker and release the models. */
  close(): Promise<void>;
}

/**
 * Start the engine worker and load the models.
 *
 * @param options - Worker URL, model URLs and feature overrides.
 * @throws When the worker cannot start or a model fails to load.
 */
export async function createEngine(options: EngineOptions): Promise<Engine> {
  const worker = (options.createWorker ?? defaultCreateWorker)(options.workerUrl);
  const pending = new Map<number, { resolve: (response: EngineResponse) => void; reject: (error: Error) => void }>();
  const listeners = new Set<(window: WindowResult) => void>();
  let nextId = 1;

  worker.onmessage = (event: MessageEvent<EngineResponse>) => {
    const response = event.data;
    const entry = pending.get(response.id);
    if (response.type === 'windows') {
      for (const window of response.windows) {
        for (const listener of listeners) listener(window);
      }
    }
    if (entry === undefined) return;
    pending.delete(response.id);
    if (response.type === 'error') entry.reject(new Error(`earshot: ${response.message}`));
    else entry.resolve(response);
  };
  worker.onerror = (event: ErrorEvent) => {
    const error = new Error(`earshot: engine worker failed: ${event.message}`);
    for (const entry of pending.values()) entry.reject(error);
    pending.clear();
  };

  const send = (request: EngineRequest, transfer: Transferable[] = []): Promise<EngineResponse> =>
    new Promise<EngineResponse>((resolve, reject) => {
      pending.set(request.id, { resolve, reject });
      worker.postMessage(request, transfer);
    });

  const ready = await send({
    type: 'init',
    id: nextId++,
    models: options.models,
    ...(options.features === undefined ? {} : { features: options.features }),
    sampleRateHz: options.sampleRateHz ?? SAMPLE_RATE_HZ,
  });
  if (ready.type !== 'ready') {
    worker.terminate();
    throw new Error('earshot: the engine worker did not become ready');
  }

  return {
    async push(samples: Float32Array): Promise<WindowResult[]> {
      const response = await send({ type: 'push', id: nextId++, samples }, [samples.buffer]);
      return response.type === 'windows' ? response.windows.slice() : [];
    },
    onWindow(listener): () => void {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    async reset(): Promise<void> {
      await send({ type: 'reset', id: nextId++ });
    },
    hasClassifier: ready.classifier,
    hasEmbedder: ready.embedder,
    async close(): Promise<void> {
      try {
        await send({ type: 'close', id: nextId++ });
      } finally {
        listeners.clear();
        worker.terminate();
      }
    },
  };
}

/**
 * Create the engine worker as a **classic** worker.
 *
 * MediaPipe loads its WASM glue with `importScripts`, which an ES module worker
 * does not support, and falls back to injecting a `<script>` element, which
 * needs a `document` a worker does not have. A classic worker is the only
 * context where both the models and the Worker requirement can hold at once;
 * see `docs/decisions/0007-mediapipe-cannot-load-in-a-module-worker.md`.
 *
 * Consumers must therefore build workers as IIFE, not ESM:
 * `worker: { format: 'iife' }` in `vite.config.ts`.
 */
function defaultCreateWorker(url: string): Worker {
  return new Worker(url, { name: 'earshot-engine' });
}
