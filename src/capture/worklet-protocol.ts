/**
 * The contract between the AudioWorklet processor and the main thread.
 *
 * The processor itself is plain JavaScript (`capture-worklet.js`) because
 * `AudioWorkletGlobalScope.addModule` loads a URL that the browser executes
 * directly: a `?url` import of a TypeScript file hands the browser TypeScript,
 * which it cannot run. Keeping the names and message shapes here lets the main
 * thread stay fully typed without importing the processor's module.
 */

/** Name the processor registers itself under. */
export const CAPTURE_PROCESSOR_NAME = 'earshot-capture';

/**
 * Message posted from the worklet to the main thread on every full chunk.
 *
 * `samples` is transferred, and it belongs to the processor's pool: the main
 * thread must copy anything it wants to keep and then transfer the buffer back
 * with a {@link CaptureReleaseMessage}, or the processor runs out of buffers and
 * starts allocating on the audio thread.
 */
export interface CaptureChunkMessage {
  readonly type: 'chunk';
  /** Mono samples at the AudioContext's sample rate. */
  readonly samples: Float32Array;
  /** Number of samples the processor has seen since it started. */
  readonly totalSamples: number;
  /**
   * Buffers the processor has had to allocate because none were returned in
   * time. Zero on a healthy page; a rising value means the main thread is too
   * slow to give them back, and each one is a garbage collection on the audio
   * thread.
   */
  readonly starved: number;
}

/** Message posted once, when the processor is constructed. */
export interface CaptureReadyMessage {
  readonly type: 'ready';
  /** Sample rate of the audio graph the processor is running in, in Hz. */
  readonly sampleRate: number;
  /** Buffers in the processor's pool. */
  readonly poolSize: number;
}

/** Every message the processor can post. */
export type CaptureMessage = CaptureChunkMessage | CaptureReadyMessage;

/** Message the main thread posts to hand a chunk's buffer back to the pool. */
export interface CaptureReleaseMessage {
  readonly type: 'release';
  /** The exact buffer that arrived in a {@link CaptureChunkMessage}. */
  readonly samples: Float32Array;
}

/** Options accepted through `AudioWorkletNode`'s `processorOptions`. */
export interface CaptureProcessorOptions {
  /** Samples per posted chunk. Defaults to 2048. */
  readonly chunkSamples?: number;
  /**
   * Buffers the processor keeps in its pool. Defaults to 4, minimum 2. Raise it
   * only if `starved` climbs on a page that cannot be made more responsive.
   */
  readonly poolSize?: number;
}
