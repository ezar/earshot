/**
 * The `earshot/worker` entry point.
 *
 * Consumers load it as a URL and hand that URL to {@link createEngine}:
 *
 * ```ts
 * import workerUrl from 'earshot/worker?worker&url';
 * const engine = await createEngine({ workerUrl, models });
 * ```
 *
 * Everything expensive — WASM inference, the STFT, the feature extraction —
 * happens here, so the page's main thread stays free for rendering.
 */

/// <reference lib="webworker" />

import { SAMPLE_RATE_HZ } from '../constants.js';
import { createFeatureExtractor, type FeatureExtractor } from '../dsp/features.js';
import { createFramer, type Framer } from '../dsp/framing.js';
import { createGuards, type Guards } from '../guards/index.js';
import { createClassifier, type Classifier } from '../models/classifier.js';
import { createEmbedder, type Embedder } from '../models/embedder.js';
import type { WindowResult } from '../util/types.js';
import type { EngineRequest, EngineResponse } from './protocol.js';

interface EngineState {
  framer: Framer;
  extractor: FeatureExtractor;
  classifier: Classifier | null;
  embedder: Embedder | null;
  guards: Guards | null;
  embedRejectedWindows: boolean;
}

let state: EngineState | null = null;

const scope = globalThis as unknown as DedicatedWorkerGlobalScope;

const post = (message: EngineResponse): void => {
  scope.postMessage(message);
};

scope.onmessage = (event: MessageEvent<EngineRequest>): void => {
  void handle(event.data);
};

async function handle(request: EngineRequest): Promise<void> {
  try {
    switch (request.type) {
      case 'init': {
        const sampleRateHz = request.sampleRateHz || SAMPLE_RATE_HZ;
        const classifier =
          request.models.classifierUrl === undefined ? null : await createClassifier(request.models);
        const embedder = request.models.embedderUrl === undefined ? null : await createEmbedder(request.models);
        state = {
          framer: createFramer({ sampleRateHz }),
          extractor: createFeatureExtractor({ ...request.features, sampleRateHz }),
          classifier,
          embedder,
          guards: request.guards === undefined ? null : createGuards(request.guards),
          embedRejectedWindows: request.embedRejectedWindows ?? false,
        };
        post({
          type: 'ready',
          id: request.id,
          classifier: classifier !== null,
          embedder: embedder !== null,
        });
        return;
      }
      case 'push': {
        const current = requireState();
        const windows: WindowResult[] = [];
        for (const frame of current.framer.push(request.samples)) {
          windows.push(analyse(current, frame.t, frame.samples));
        }
        post({ type: 'windows', id: request.id, windows });
        return;
      }
      case 'flush': {
        requireState().framer.reset();
        post({ type: 'done', id: request.id });
        return;
      }
      case 'reset': {
        const current = requireState();
        current.framer.reset();
        post({ type: 'done', id: request.id });
        return;
      }
      case 'close': {
        state?.classifier?.close();
        state?.embedder?.close();
        state = null;
        post({ type: 'done', id: request.id });
        scope.close();
        return;
      }
    }
  } catch (error) {
    post({ type: 'error', id: request.id, message: describeError(error) });
  }
}

/**
 * Derive everything the engine reports for one window.
 *
 * The order matters for cost. Features and classification come first, because
 * guards need a level and a set of class scores to judge a window. The embedder
 * runs last and, when guards are configured, only on windows they accept: it is
 * roughly half the engine's per-window cost, and an embedding the host app is
 * going to discard is worth nothing.
 */
function analyse(current: EngineState, t: number, samples: Float32Array): WindowResult {
  const features = current.extractor.extract(samples);
  const classes = current.classifier === null ? [] : current.classifier.classify(samples);
  const base = { t, classes, rmsDbfs: features.rmsDbfs, features };

  if (current.guards === null) {
    const embedding = current.embedder === null ? [] : Array.from(current.embedder.embed(samples));
    return { ...base, embedding };
  }

  const guard = current.guards.check({ ...base, embedding: [] });
  const shouldEmbed = current.embedder !== null && (guard.accepted || current.embedRejectedWindows);
  const embedding = shouldEmbed ? Array.from((current.embedder as Embedder).embed(samples)) : [];
  return { ...base, embedding, guard };
}

function requireState(): EngineState {
  if (state === null) throw new Error('earshot: the engine worker received a message before init');
  return state;
}

function describeError(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}
