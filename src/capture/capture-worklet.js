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
 * `process` runs on the audio thread every 128 samples and allocates nothing
 * beyond the one chunk it transfers away: the ring buffer is sized once in the
 * constructor and reused for the lifetime of the node.
 */

/* global AudioWorkletProcessor, registerProcessor, sampleRate */

/** Must match `CAPTURE_PROCESSOR_NAME` in `worklet-protocol.ts`. */
const CAPTURE_PROCESSOR_NAME = 'earshot-capture';

class EarshotCaptureProcessor extends AudioWorkletProcessor {
  /**
   * @param {{ processorOptions?: { chunkSamples?: number } }} [options]
   */
  constructor(options) {
    super();
    const chunkSamples = options?.processorOptions?.chunkSamples ?? 2048;
    this.chunk = new Float32Array(chunkSamples);
    this.fill = 0;
    this.total = 0;
    this.port.postMessage({ type: 'ready', sampleRate });
  }

  /**
   * @param {Float32Array[][]} inputs - Per-input, per-channel blocks of 128 samples.
   * @returns {boolean} Always true: the node lives until the graph disconnects it.
   */
  process(inputs) {
    const channel = inputs[0]?.[0];
    if (channel === undefined) return true;
    const chunk = this.chunk;
    const capacity = chunk.length;
    for (let i = 0; i < channel.length; i += 1) {
      chunk[this.fill] = channel[i];
      this.fill += 1;
      if (this.fill === capacity) {
        // The copy is transferred, so the caller may retain it and the ring
        // buffer stays available for the next block without reallocating.
        const copy = chunk.slice();
        this.total += capacity;
        this.port.postMessage({ type: 'chunk', samples: copy, totalSamples: this.total }, [copy.buffer]);
        this.fill = 0;
      }
    }
    return true;
  }
}

registerProcessor(CAPTURE_PROCESSOR_NAME, EarshotCaptureProcessor);
