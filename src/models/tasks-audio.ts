/**
 * The narrow slice of MediaPipe Tasks Audio that earshot depends on.
 *
 * earshot never imports `@mediapipe/tasks-audio` statically and never embeds a
 * model or WASM location: apps self-host the task files and pass their URLs in.
 * The default loader resolves the package at runtime; apps that bundle with
 * Vite should pass their own `loadTasksAudio` so the dependency stays static
 * and analysable.
 */

/** A single category returned by the classifier. */
export interface TasksCategory {
  readonly categoryName?: string;
  readonly displayName?: string;
  readonly score: number;
}

/** One classification result, covering one internal patch of audio. */
export interface TasksClassificationResult {
  readonly classifications: readonly { readonly categories: readonly TasksCategory[] }[];
}

/** One embedding result. */
export interface TasksEmbeddingResult {
  readonly embeddings: readonly { readonly floatEmbedding?: readonly number[] }[];
}

/** The MediaPipe `AudioClassifier` surface earshot uses. */
export interface TasksAudioClassifier {
  classify(data: Float32Array, sampleRateHz: number): readonly TasksClassificationResult[];
  close(): void;
}

/** The MediaPipe `AudioEmbedder` surface earshot uses. */
export interface TasksAudioEmbedder {
  embed(data: Float32Array, sampleRateHz: number): readonly TasksEmbeddingResult[];
  close(): void;
}

/**
 * The MediaPipe module surface earshot uses.
 *
 * `AudioEmbedder` is optional because MediaPipe removed it: it ships in
 * `@mediapipe/tasks-audio` up to and including 0.10.21 and is absent from
 * 0.10.32 onwards, including 1.x. See {@link EMBEDDER_MAX_VERSION} and
 * `docs/decisions/0006-mediapipe-dropped-the-audio-embedder.md`.
 */
export interface TasksAudioModule {
  readonly FilesetResolver: {
    forAudioTasks(wasmBaseUrl: string): Promise<unknown>;
  };
  readonly AudioClassifier: {
    createFromOptions(fileset: unknown, options: unknown): Promise<TasksAudioClassifier>;
  };
  readonly AudioEmbedder?: {
    createFromOptions(fileset: unknown, options: unknown): Promise<TasksAudioEmbedder>;
  };
}

/**
 * Last `@mediapipe/tasks-audio` release that ships `AudioEmbedder`.
 *
 * The classifier is present in every supported release; the embedder is not.
 * An app that needs embeddings — a profile in the `'embedding'` feature space,
 * or Meowlogue's kNN identity — must pin at or below this version.
 */
export const EMBEDDER_MAX_VERSION = '0.10.21';

/** A function that resolves the MediaPipe Tasks Audio module. */
export type TasksAudioLoader = () => Promise<TasksAudioModule>;

/**
 * Default loader: a runtime import of `@mediapipe/tasks-audio`.
 *
 * The specifier is held in a variable so that neither TypeScript nor a bundler
 * tries to resolve it at build time; apps that want static bundling should pass
 * their own loader instead.
 */
export const defaultTasksAudioLoader: TasksAudioLoader = async () => {
  const module: unknown = await import('@mediapipe/tasks-audio');
  return module as TasksAudioModule;
};

/** URLs of the model and runtime files; always supplied by the host app. */
export interface ModelUrls {
  /** URL of the YAMNet classifier `.tflite` task file. */
  readonly classifierUrl?: string;
  /** URL of the YAMNet embedder `.tflite` task file. */
  readonly embedderUrl?: string;
  /** Base URL of the MediaPipe WASM assets directory. */
  readonly wasmBaseUrl: string;
  /** Override the module loader, e.g. to bundle MediaPipe statically. */
  readonly loadTasksAudio?: TasksAudioLoader;
}
