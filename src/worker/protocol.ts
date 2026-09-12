/**
 * The message protocol between {@link createEngine} on the main thread and the
 * `earshot/worker` entry point.
 *
 * Every message is structured-clone-safe. Audio travels as a transferable
 * `Float32Array`, so pushing a chunk costs no copy.
 */

import type { ModelUrls } from '../models/tasks-audio.js';
import type { FeatureOptions } from '../dsp/features.js';
import type { GuardConfig } from '../guards/index.js';
import type { WindowResult } from '../util/types.js';

/** Messages the main thread sends to the worker. */
export type EngineRequest =
  | {
      readonly type: 'init';
      readonly id: number;
      readonly models: Omit<ModelUrls, 'loadTasksAudio'>;
      readonly features?: FeatureOptions;
      readonly guards?: GuardConfig;
      readonly embedRejectedWindows?: boolean;
      readonly sampleRateHz: number;
    }
  | { readonly type: 'push'; readonly id: number; readonly samples: Float32Array }
  | { readonly type: 'flush'; readonly id: number }
  | { readonly type: 'reset'; readonly id: number }
  | { readonly type: 'close'; readonly id: number };

/** Messages the worker sends back. */
export type EngineResponse =
  | { readonly type: 'ready'; readonly id: number; readonly classifier: boolean; readonly embedder: boolean }
  | { readonly type: 'windows'; readonly id: number; readonly windows: readonly WindowResult[] }
  | { readonly type: 'done'; readonly id: number }
  | { readonly type: 'error'; readonly id: number; readonly message: string };
