/**
 * The AudioWorklet processor, published as the `earshot/capture-worklet`
 * entry point.
 *
 * Consumers load it as a URL and hand that URL to `createCapture`:
 *
 * ```ts
 * import workletUrl from 'earshot/capture-worklet?url';
 * const capture = await createCapture({ workletUrl });
 * ```
 *
 * This file is plain JavaScript, not TypeScript, and imports nothing. The
 * browser executes the URL passed to `audioWorklet.addModule` as-is, so a
 * TypeScript source would arrive at the audio thread untranspiled, and static
 * imports inside a worklet module are not reliably supported. Its types and the
 * processor name live in `worklet-protocol.ts`, which the main thread uses.
 *
 * `process` runs on the audio thread every 128 samples and **allocates
 * nothing**. A chunk has to be transferred to reach the main thread, and a
 * transferred buffer is detached and cannot be refilled, so the processor keeps
 * a pool of buffers and the main thread transfers each one back once it is
 * done with it. Allocating here instead would put a garbage collection on the
 * audio thread roughly twenty times a second, and a GC pause there is a gap in
 * the recording.
 */

/* global AudioWorkletProcessor, registerProcessor, sampleRate */

/** Must match `CAPTURE_PROCESSOR_NAME` in `worklet-protocol.ts`. */
const CAPTURE_PROCESSOR_NAME = 'earshot-capture';

/**
 * Buffers in the pool.
 *
 * One is being filled, one is usually in flight, and the rest absorb a slow
 * main thread. Four covers roughly 170 ms of round-trip latency at the default
 * chunk size, far more than a healthy page needs.
 */
const DEFAULT_POOL_SIZE = 4;

class EarshotCaptureProcessor extends AudioWorkletProcessor {
  /**
   * @param {{ processorOptions?: { chunkSamples?: number, poolSize?: number } }} [options]
   */
  constructor(options) {
    super();
    const chunkSamples = options?.processorOptions?.chunkSamples ?? 2048;
    const poolSize = Math.max(2, options?.processorOptions?.poolSize ?? DEFAULT_POOL_SIZE);

    /** Buffers not currently in flight. `process` fills the last one. */
    this.free = [];
    for (let i = 0; i < poolSize; i += 1) this.free.push(new Float32Array(chunkSamples));
    this.chunkSamples = chunkSamples;
    /** The buffer being filled; null when the pool is momentarily empty. */
    this.chunk = this.free.pop();
    this.fill = 0;
    this.total = 0;
    /**
     * Buffers allocated beyond the pool because every one was still in flight.
     * Zero on a healthy page; a rising count means the main thread is not
     * returning them, and is reported so that it is observable rather than
     * silently costing a GC.
     */
    this.starved = 0;

    this.port.onmessage = (event) => {
      const message = event.data;
      // The main thread returns a buffer once it has copied what it needs.
      if (message?.type === 'release' && message.samples instanceof Float32Array) {
        if (message.samples.length === this.chunkSamples) this.free.push(message.samples);
      }
    };

    this.port.postMessage({ type: 'ready', sampleRate, poolSize });
  }

  /**
   * @param {Float32Array[][]} inputs - Per-input, per-channel blocks of 128 samples.
   * @returns {boolean} Always true: the node lives until the graph disconnects it.
   */
  process(inputs) {
    const channel = inputs[0]?.[0];
    if (channel === undefined) return true;

    for (let i = 0; i < channel.length; i += 1) {
      if (this.chunk === null) {
        // Every buffer is in flight. Dropping audio would be worse than the one
        // allocation, so take the allocation and make it visible.
        this.chunk = this.free.pop() ?? null;
        if (this.chunk === null) {
          this.chunk = new Float32Array(this.chunkSamples);
          this.starved += 1;
        }
        this.fill = 0;
      }
      this.chunk[this.fill] = channel[i];
      this.fill += 1;
      if (this.fill === this.chunk.length) {
        const full = this.chunk;
        this.total += full.length;
        this.port.postMessage(
          { type: 'chunk', samples: full, totalSamples: this.total, starved: this.starved },
          [full.buffer],
        );
        this.chunk = this.free.pop() ?? null;
        this.fill = 0;
      }
    }
    return true;
  }
}

registerProcessor(CAPTURE_PROCESSOR_NAME, EarshotCaptureProcessor);
