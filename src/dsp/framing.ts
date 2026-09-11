/**
 * Turning a continuous stream of PCM into overlapping analysis windows.
 */

import { HOP_SAMPLES, SAMPLE_RATE_HZ, WINDOW_SAMPLES } from '../constants.js';

/** One analysis window together with its position in the stream. */
export interface Frame {
  /** Window start time, in seconds since the framer was created. */
  readonly t: number;
  /** Exactly {@link WINDOW_SAMPLES} samples at {@link SAMPLE_RATE_HZ}. */
  readonly samples: Float32Array;
}

/** Options for {@link createFramer}. */
export interface FramerOptions {
  /** Window length, in samples. Defaults to {@link WINDOW_SAMPLES}. */
  readonly windowSamples?: number;
  /** Hop between windows, in samples. Defaults to {@link HOP_SAMPLES}. */
  readonly hopSamples?: number;
  /** Sample rate of the incoming audio, in Hz. Defaults to {@link SAMPLE_RATE_HZ}. */
  readonly sampleRateHz?: number;
}

/** Stateful framer: push arbitrary chunks, pull fixed-size overlapping windows. */
export interface Framer {
  /**
   * Append samples and return every window that became complete.
   *
   * @param chunk - Mono samples at the framer's sample rate.
   */
  push(chunk: Float32Array): Frame[];
  /** Number of buffered samples that have not yet produced a window. */
  readonly pending: number;
  /** Drop all buffered audio and reset the clock to zero. */
  reset(): void;
}

/**
 * Create a framer producing overlapping windows.
 *
 * The framer keeps a ring of at most `windowSamples + hopSamples` samples and
 * copies each emitted window out, so callers may retain the frames they get.
 */
export function createFramer(options: FramerOptions = {}): Framer {
  const windowSamples = options.windowSamples ?? WINDOW_SAMPLES;
  const hopSamples = options.hopSamples ?? HOP_SAMPLES;
  const sampleRateHz = options.sampleRateHz ?? SAMPLE_RATE_HZ;
  if (hopSamples <= 0 || hopSamples > windowSamples) {
    throw new Error(`hopSamples must be in (0, windowSamples], received ${hopSamples}`);
  }

  let buffer = new Float32Array(0);
  let consumedSamples = 0;

  return {
    push(chunk: Float32Array): Frame[] {
      if (chunk.length > 0) {
        const merged = new Float32Array(buffer.length + chunk.length);
        merged.set(buffer, 0);
        merged.set(chunk, buffer.length);
        buffer = merged;
      }
      const frames: Frame[] = [];
      let offset = 0;
      while (buffer.length - offset >= windowSamples) {
        frames.push({
          t: (consumedSamples + offset) / sampleRateHz,
          samples: buffer.slice(offset, offset + windowSamples),
        });
        offset += hopSamples;
      }
      if (offset > 0) {
        buffer = buffer.slice(offset);
        consumedSamples += offset;
      }
      return frames;
    },
    get pending(): number {
      return buffer.length;
    },
    reset(): void {
      buffer = new Float32Array(0);
      consumedSamples = 0;
    },
  };
}

/**
 * Split a finite buffer into overlapping windows in one call.
 *
 * Unlike {@link createFramer} this keeps no state and drops the tail that does
 * not fill a whole window.
 */
export function frameBuffer(samples: Float32Array, options: FramerOptions = {}): Frame[] {
  const framer = createFramer(options);
  return framer.push(samples);
}
