/**
 * Sample-rate conversion down to {@link SAMPLE_RATE_HZ}.
 *
 * Browsers hand us whatever rate the device runs at (commonly 44.1 or 48 kHz),
 * so every buffer is low-pass filtered and linearly interpolated to 16 kHz
 * before anything else touches it.
 */

import { SAMPLE_RATE_HZ } from '../constants.js';

/**
 * Resample a mono buffer to `targetRateHz`.
 *
 * Downsampling first applies a windowed-sinc low-pass at the target Nyquist to
 * keep aliasing out of the mel bands; upsampling and pass-through skip it.
 *
 * @param input - Mono samples in `[-1, 1]`.
 * @param sourceRateHz - Sample rate of `input`, in Hz.
 * @param targetRateHz - Desired sample rate, in Hz. Defaults to {@link SAMPLE_RATE_HZ}.
 * @returns A new buffer at `targetRateHz`.
 */
export function resample(
  input: Float32Array,
  sourceRateHz: number,
  targetRateHz: number = SAMPLE_RATE_HZ,
): Float32Array {
  if (sourceRateHz <= 0) throw new Error(`sourceRateHz must be positive, received ${sourceRateHz}`);
  if (sourceRateHz === targetRateHz) return input.slice();
  if (input.length === 0) return new Float32Array(0);

  const filtered =
    targetRateHz < sourceRateHz
      ? lowPass(input, (0.45 * targetRateHz) / sourceRateHz)
      : input;

  const ratio = sourceRateHz / targetRateHz;
  const outLength = Math.max(1, Math.floor(input.length / ratio));
  const out = new Float32Array(outLength);
  for (let i = 0; i < outLength; i += 1) {
    const position = i * ratio;
    const index = Math.floor(position);
    const fraction = position - index;
    const a = filtered[index] as number;
    const b = (filtered[index + 1] ?? a) as number;
    out[i] = a + (b - a) * fraction;
  }
  return out;
}

/**
 * Linear-phase FIR low-pass with a Blackman window.
 *
 * @param input - Mono samples.
 * @param cutoffNormalized - Cutoff as a fraction of the source sample rate, in `(0, 0.5)`.
 */
function lowPass(input: Float32Array, cutoffNormalized: number): Float32Array {
  const taps = 63;
  const half = (taps - 1) / 2;
  const kernel = new Float32Array(taps);
  let sum = 0;
  for (let i = 0; i < taps; i += 1) {
    const n = i - half;
    const sinc = n === 0 ? 2 * cutoffNormalized : Math.sin(2 * Math.PI * cutoffNormalized * n) / (Math.PI * n);
    const window =
      0.42 - 0.5 * Math.cos((2 * Math.PI * i) / (taps - 1)) + 0.08 * Math.cos((4 * Math.PI * i) / (taps - 1));
    const value = sinc * window;
    kernel[i] = value;
    sum += value;
  }
  for (let i = 0; i < taps; i += 1) kernel[i] = (kernel[i] as number) / sum;

  const out = new Float32Array(input.length);
  for (let i = 0; i < input.length; i += 1) {
    let acc = 0;
    for (let k = 0; k < taps; k += 1) {
      const j = i + k - half;
      if (j >= 0 && j < input.length) acc += (input[j] as number) * (kernel[k] as number);
    }
    out[i] = acc;
  }
  return out;
}

/**
 * Mix an interleaved multi-channel buffer down to mono by averaging channels.
 *
 * @param interleaved - Interleaved samples.
 * @param channels - Channel count; 1 returns a copy.
 */
export function toMono(interleaved: Float32Array, channels: number): Float32Array {
  if (channels <= 1) return interleaved.slice();
  const frames = Math.floor(interleaved.length / channels);
  const out = new Float32Array(frames);
  for (let i = 0; i < frames; i += 1) {
    let sum = 0;
    for (let c = 0; c < channels; c += 1) sum += interleaved[i * channels + c] as number;
    out[i] = sum / channels;
  }
  return out;
}
