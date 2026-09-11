/**
 * Optional CLAP embeddings, published as the `earshot/clap` entry point.
 *
 * CLAP embeddings are far richer than YAMNet's and support text queries, but
 * the model is tens of megabytes and `@huggingface/transformers` is a large
 * dependency. Keeping it behind its own entry point means the core bundle never
 * pulls either: an app that does not import `earshot/clap` pays nothing.
 *
 * ```ts
 * import { createClapEmbedder } from 'earshot/clap';
 * const clap = await createClapEmbedder({ modelId: 'Xenova/clap-htsat-unfused' });
 * const vector = await clap.embedAudio(pcm16k);
 * ```
 */

import { SAMPLE_RATE_HZ } from '../constants.js';
import { cosineSimilarity, l2Normalize } from '../util/math.js';

/** A minimal view of the transformers.js pipeline factory earshot uses. */
export interface TransformersModule {
  pipeline(task: string, model: string, options?: Record<string, unknown>): Promise<TransformersPipeline>;
}

/** A transformers.js feature-extraction pipeline. */
export type TransformersPipeline = (
  input: unknown,
  options?: Record<string, unknown>,
) => Promise<{ data: ArrayLike<number>; dims?: readonly number[] }>;

/** Options for {@link createClapEmbedder}. */
export interface ClapOptions {
  /** Hugging Face model id, e.g. `'Xenova/clap-htsat-unfused'`. */
  readonly modelId: string;
  /** Run in WebGPU when available. Defaults to false (WASM). */
  readonly webgpu?: boolean;
  /** Override the module loader, e.g. to bundle transformers.js statically. */
  readonly loadTransformers?: () => Promise<TransformersModule>;
  /** L2-normalize the returned vectors. Defaults to true. */
  readonly l2Normalize?: boolean;
}

/** A loaded CLAP embedder. */
export interface ClapEmbedder {
  /**
   * Embed audio into the shared audio-text space.
   *
   * @param pcm16k - Mono samples at {@link SAMPLE_RATE_HZ}.
   */
  embedAudio(pcm16k: Float32Array): Promise<Float32Array>;
  /**
   * Embed a natural-language description into the same space.
   *
   * @param text - e.g. `"a cat meowing"`.
   */
  embedText(text: string): Promise<Float32Array>;
  /** Release the underlying pipelines. */
  close(): Promise<void>;
}

const defaultLoadTransformers = async (): Promise<TransformersModule> => {
  const specifier = '@huggingface/transformers';
  const module: unknown = await import(/* @vite-ignore */ specifier);
  return module as TransformersModule;
};

/**
 * Load a CLAP embedder.
 *
 * The audio and text towers are loaded lazily and independently, so an app that
 * only ever calls {@link ClapEmbedder.embedAudio} never downloads the text tower.
 *
 * @param options - Model id and runtime preferences.
 */
export async function createClapEmbedder(options: ClapOptions): Promise<ClapEmbedder> {
  const transformers = await (options.loadTransformers ?? defaultLoadTransformers)();
  const normalize = options.l2Normalize ?? true;
  const pipelineOptions: Record<string, unknown> = options.webgpu === true ? { device: 'webgpu' } : {};

  let audioPipeline: TransformersPipeline | null = null;
  let textPipeline: TransformersPipeline | null = null;

  const finish = (data: ArrayLike<number>): Float32Array => {
    const vector = Float32Array.from(data as ArrayLike<number>);
    return normalize ? l2Normalize(vector) : vector;
  };

  return {
    async embedAudio(pcm16k: Float32Array): Promise<Float32Array> {
      audioPipeline ??= await transformers.pipeline('feature-extraction', options.modelId, pipelineOptions);
      const output = await audioPipeline(pcm16k, { sampling_rate: SAMPLE_RATE_HZ });
      return finish(output.data);
    },
    async embedText(text: string): Promise<Float32Array> {
      textPipeline ??= await transformers.pipeline('feature-extraction', options.modelId, pipelineOptions);
      const output = await textPipeline(text, { pooling: 'mean', normalize: false });
      return finish(output.data);
    },
    async close(): Promise<void> {
      audioPipeline = null;
      textPipeline = null;
    },
  };
}

/**
 * Rank text descriptions against an audio embedding.
 *
 * @param audioEmbedding - Vector from {@link ClapEmbedder.embedAudio}.
 * @param textEmbeddings - Vectors from {@link ClapEmbedder.embedText}, with their labels.
 * @returns Labels sorted by descending cosine similarity.
 */
export function rankByText(
  audioEmbedding: ArrayLike<number>,
  textEmbeddings: readonly { readonly label: string; readonly embedding: ArrayLike<number> }[],
): { label: string; similarity: number }[] {
  return textEmbeddings
    .map((entry) => ({ label: entry.label, similarity: cosineSimilarity(audioEmbedding, entry.embedding) }))
    .sort((a, b) => b.similarity - a.similarity);
}
