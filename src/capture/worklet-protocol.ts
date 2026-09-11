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

/** Message posted from the worklet to the main thread on every full chunk. */
export interface CaptureChunkMessage {
  readonly type: 'chunk';
  /** Mono samples at the AudioContext's sample rate. */
  readonly samples: Float32Array;
  /** Number of samples the processor has seen since it started. */
  readonly totalSamples: number;
}

/** Message posted once, when the processor is constructed. */
export interface CaptureReadyMessage {
  readonly type: 'ready';
  /** Sample rate of the audio graph the processor is running in, in Hz. */
  readonly sampleRate: number;
}

/** Every message the processor can post. */
export type CaptureMessage = CaptureChunkMessage | CaptureReadyMessage;

/** Options accepted through `AudioWorkletNode`'s `processorOptions`. */
export interface CaptureProcessorOptions {
  /** Samples per posted chunk. Defaults to 2048. */
  readonly chunkSamples?: number;
}
