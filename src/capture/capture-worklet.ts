/**
 * The AudioWorklet processor, published as the `earshot/capture-worklet`
 * entry point.
 *
 * Consumers load it as a URL and hand that URL to {@link createCapture}:
 *
 * ```ts
 * import workletUrl from 'earshot/capture-worklet?url';
 * const capture = await createCapture({ workletUrl });
 * ```
 *
 * `process` runs on the audio thread every 128 samples. It allocates nothing:
 * a ring buffer is sized once in the constructor and a single transferable
 * chunk is posted whenever it fills.
 */

/// <reference lib="dom" />

/** Message posted from the worklet to the main thread on every full chunk. */
export interface CaptureChunkMessage {
  readonly type: 'chunk';
  /** Mono samples at the AudioContext's sample rate. */
  readonly samples: Float32Array;
  /** Number of samples the processor has seen since it started. */
  readonly totalSamples: number;
}

/** Options accepted through `AudioWorkletNode`'s `processorOptions`. */
export interface CaptureProcessorOptions {
  /** Samples per posted chunk. Defaults to 2048. */
  readonly chunkSamples?: number;
}

declare const sampleRate: number;
declare const currentFrame: number;
declare function registerProcessor(name: string, processor: unknown): void;

/** Name the processor registers itself under. */
export const CAPTURE_PROCESSOR_NAME = 'earshot-capture';

// The worklet global scope is not the module scope of the host page, so the
// class is declared and registered here and the file is never imported for its
// runtime value on the main thread.
if (typeof registerProcessor === 'function') {
  class EarshotCaptureProcessor {
    private readonly chunk: Float32Array;
    private fill = 0;
    private total = 0;
    private readonly port: MessagePort;

    constructor(options?: { processorOptions?: CaptureProcessorOptions }) {
      const chunkSamples = options?.processorOptions?.chunkSamples ?? 2048;
      this.chunk = new Float32Array(chunkSamples);
      // `port` is provided by AudioWorkletProcessor; typed loosely to avoid
      // depending on the worklet ambient types in consuming projects.
      this.port = (this as unknown as { port: MessagePort }).port;
      this.port.postMessage({ type: 'ready', sampleRate, currentFrame });
    }

    process(inputs: Float32Array[][]): boolean {
      const input = inputs[0];
      const channel = input?.[0];
      if (channel === undefined) return true;
      const chunk = this.chunk;
      const capacity = chunk.length;
      for (let i = 0; i < channel.length; i += 1) {
        chunk[this.fill] = channel[i] as number;
        this.fill += 1;
        if (this.fill === capacity) {
          const copy = chunk.slice();
          this.total += capacity;
          const message: CaptureChunkMessage = {
            type: 'chunk',
            samples: copy,
            totalSamples: this.total,
          };
          this.port.postMessage(message, [copy.buffer]);
          this.fill = 0;
        }
      }
      return true;
    }
  }

  registerProcessor(CAPTURE_PROCESSOR_NAME, EarshotCaptureProcessor);
}
