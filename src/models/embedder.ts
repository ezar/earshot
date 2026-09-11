/** Thin wrapper over MediaPipe's YAMNet `AudioEmbedder`. */

import { SAMPLE_RATE_HZ } from '../constants.js';
import { defaultTasksAudioLoader, type ModelUrls, type TasksAudioEmbedder } from './tasks-audio.js';

/** Options for {@link createEmbedder}. */
export interface EmbedderOptions extends ModelUrls {
  /** L2-normalize embeddings before returning them. Defaults to true. */
  readonly l2Normalize?: boolean;
}

/** A loaded audio embedder. */
export interface Embedder {
  /**
   * Embed one analysis window.
   *
   * @param samples - Mono window at {@link SAMPLE_RATE_HZ}.
   * @returns The embedding vector; empty when the model returns nothing.
   */
  embed(samples: Float32Array): Float32Array;
  /** Release the underlying WASM instance. */
  close(): void;
}

/**
 * Load the YAMNet embedder from app-provided URLs.
 *
 * @throws When `embedderUrl` is missing or the task file cannot be loaded.
 */
export async function createEmbedder(options: EmbedderOptions): Promise<Embedder> {
  if (options.embedderUrl === undefined) {
    throw new Error('earshot: createEmbedder requires models.embedderUrl');
  }
  const tasks = await (options.loadTasksAudio ?? defaultTasksAudioLoader)();
  const fileset = await tasks.FilesetResolver.forAudioTasks(options.wasmBaseUrl);
  const embedder: TasksAudioEmbedder = await tasks.AudioEmbedder.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: options.embedderUrl },
    l2Normalize: options.l2Normalize ?? true,
    quantize: false,
  });

  return {
    embed(samples: Float32Array): Float32Array {
      const results = embedder.embed(samples, SAMPLE_RATE_HZ);
      return averageEmbeddings(results);
    },
    close(): void {
      embedder.close();
    },
  };
}

/**
 * Average the per-patch embeddings MediaPipe returns for one window.
 *
 * Averaging (rather than taking the first patch) keeps the vector stable when a
 * window straddles a transient.
 */
export function averageEmbeddings(
  results: readonly { readonly embeddings: readonly { readonly floatEmbedding?: readonly number[] }[] }[],
): Float32Array {
  let accumulator: Float32Array | null = null;
  let count = 0;
  for (const result of results) {
    for (const embedding of result.embeddings) {
      const values = embedding.floatEmbedding;
      if (values === undefined || values.length === 0) continue;
      if (accumulator === null) accumulator = new Float32Array(values.length);
      const length = Math.min(accumulator.length, values.length);
      for (let i = 0; i < length; i += 1) {
        accumulator[i] = (accumulator[i] as number) + (values[i] as number);
      }
      count += 1;
    }
  }
  if (accumulator === null || count === 0) return new Float32Array(0);
  for (let i = 0; i < accumulator.length; i += 1) {
    accumulator[i] = (accumulator[i] as number) / count;
  }
  return accumulator;
}
