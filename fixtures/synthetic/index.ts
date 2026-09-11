/**
 * Synthetic audio generators.
 *
 * Every earshot algorithm is tested against signals whose ground truth is known
 * exactly — a 440 Hz tone really is at 440 Hz — before any recorded dataset or
 * app is involved. Generators are deterministic: the same seed always produces
 * the same buffer, so a failing test is reproducible.
 */

import { SAMPLE_RATE_HZ } from '../../src/constants.js';
import { createRandom } from '../../src/util/math.js';

/** Options shared by every generator. */
export interface SignalOptions {
  /** Duration, in seconds. */
  readonly seconds: number;
  /** Sample rate, in Hz. Defaults to {@link SAMPLE_RATE_HZ}. */
  readonly sampleRateHz?: number;
  /** Seed for any randomness. Defaults to 1. */
  readonly seed?: number;
}

/** Digital silence. */
export function silence(options: SignalOptions): Float32Array {
  return new Float32Array(sampleCount(options));
}

/**
 * A pure sine tone.
 *
 * @param frequencyHz - Tone frequency, in Hz.
 * @param amplitude - Linear peak amplitude in `[0, 1]`. Defaults to 0.3.
 */
export function tone(frequencyHz: number, options: SignalOptions, amplitude = 0.3): Float32Array {
  const rate = options.sampleRateHz ?? SAMPLE_RATE_HZ;
  const out = new Float32Array(sampleCount(options));
  for (let i = 0; i < out.length; i += 1) {
    out[i] = amplitude * Math.sin((2 * Math.PI * frequencyHz * i) / rate);
  }
  return out;
}

/** White noise, uniform in `[-amplitude, amplitude]`. */
export function whiteNoise(options: SignalOptions, amplitude = 0.1): Float32Array {
  const random = createRandom(options.seed ?? 1);
  const out = new Float32Array(sampleCount(options));
  for (let i = 0; i < out.length; i += 1) out[i] = (random() * 2 - 1) * amplitude;
  return out;
}

/**
 * Pink noise (1/f) via the Voss-McCartney algorithm.
 *
 * Pink noise stands in for the broadband hum of a real machine far better than
 * white noise does, because its energy falls with frequency the way mechanical
 * noise does.
 */
export function pinkNoise(options: SignalOptions, amplitude = 0.1): Float32Array {
  const random = createRandom(options.seed ?? 1);
  const out = new Float32Array(sampleCount(options));
  const rows = 16;
  const values = new Float64Array(rows);
  let running = 0;
  for (let i = 0; i < rows; i += 1) {
    values[i] = random() * 2 - 1;
    running += values[i] as number;
  }
  for (let i = 0; i < out.length; i += 1) {
    // Each row updates at half the rate of the one above it.
    const index = i === 0 ? 0 : countTrailingZeros(i) % rows;
    running -= values[index] as number;
    values[index] = random() * 2 - 1;
    running += values[index] as number;
    out[i] = (running / rows) * amplitude * 3;
  }
  return out;
}

/** Add a narrowband tone on top of an existing buffer, simulating a bearing whine. */
export function injectWhine(
  base: Float32Array,
  frequencyHz: number,
  amplitude = 0.05,
  sampleRateHz = SAMPLE_RATE_HZ,
): Float32Array {
  const out = base.slice();
  for (let i = 0; i < out.length; i += 1) {
    out[i] = (out[i] as number) + amplitude * Math.sin((2 * Math.PI * frequencyHz * i) / sampleRateHz);
  }
  return out;
}

/**
 * Evenly spaced knocks.
 *
 * A knock is modelled as an impulse exciting a few damped resonances, which is
 * what a struck panel actually does: one sharp spectral onset followed by a
 * decaying ring. A burst of fresh random noise would instead contain many
 * independent spectral changes and would make an onset detector look worse than
 * it is.
 *
 * @param intervalSeconds - Time between knocks, in seconds.
 * @param decaySeconds - Exponential decay time of each knock, in seconds.
 * @param amplitude - Peak amplitude of a knock in `[0, 1]`.
 */
export function knocks(
  options: SignalOptions,
  intervalSeconds = 0.5,
  decaySeconds = 0.03,
  amplitude = 0.6,
): Float32Array {
  const rate = options.sampleRateHz ?? SAMPLE_RATE_HZ;
  const out = new Float32Array(sampleCount(options));
  const period = Math.round(intervalSeconds * rate);
  const decay = decaySeconds * rate;
  const modesHz = [180, 460, 1130];
  // Ring out until the envelope is inaudible rather than truncating it: a hard
  // cut is a step discontinuity, and a step is a broadband click that any
  // honest onset detector would rightly report as a second event.
  const tail = Math.min(period, Math.ceil(decay * 14));
  for (let start = 0; start < out.length; start += period) {
    for (let i = 0; i < tail && start + i < out.length; i += 1) {
      const envelope = Math.exp(-i / decay);
      let value = 0;
      for (let m = 0; m < modesHz.length; m += 1) {
        value += Math.sin((2 * Math.PI * (modesHz[m] as number) * i) / rate) / (m + 1);
      }
      out[start + i] = (out[start + i] as number) + amplitude * envelope * value * 0.6;
    }
  }
  return out;
}

/** Multiply a buffer by a constant gain, in dB. */
export function applyGainDb(samples: Float32Array, gainDb: number): Float32Array {
  const gain = 10 ** (gainDb / 20);
  const out = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i += 1) out[i] = (samples[i] as number) * gain;
  return out;
}

/** Amplitude-modulate a buffer at `rateHz` with modulation depth `depth` in `[0, 1]`. */
export function amplitudeModulate(
  samples: Float32Array,
  rateHz: number,
  depth = 0.5,
  sampleRateHz = SAMPLE_RATE_HZ,
): Float32Array {
  const out = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i += 1) {
    const modulator = 1 - depth + depth * (0.5 + 0.5 * Math.sin((2 * Math.PI * rateHz * i) / sampleRateHz));
    out[i] = (samples[i] as number) * modulator;
  }
  return out;
}

/** Shape of a synthetic vocalization's pitch contour. */
export type ContourShape = 'flat' | 'rising' | 'falling' | 'arc';

/** Options for {@link harmonicCall}. */
export interface CallOptions extends SignalOptions {
  /** Fundamental frequency at the start of the call, in Hz. */
  readonly startHz: number;
  /** Fundamental frequency at the end of the call, in Hz. Defaults to `startHz`. */
  readonly endHz?: number;
  /** Contour shape. Defaults to `'flat'`. */
  readonly shape?: ContourShape;
  /** Harmonics above the fundamental. Defaults to 5. */
  readonly harmonics?: number;
  /** Peak amplitude in `[0, 1]`. Defaults to 0.4. */
  readonly amplitude?: number;
  /** Attack and release time of the amplitude envelope, in seconds. Defaults to 0.03. */
  readonly fadeSeconds?: number;
}

/**
 * A harmonic call with a pitch contour and a smooth amplitude envelope —
 * a stand-in for a cat vocalization.
 *
 * Harmonic amplitudes fall as `1 / n`, which is close enough to a real meow's
 * spectrum for the pitch tracker and the segmenter to be exercised honestly.
 */
export function harmonicCall(options: CallOptions): Float32Array {
  const rate = options.sampleRateHz ?? SAMPLE_RATE_HZ;
  const length = sampleCount(options);
  const out = new Float32Array(length);
  const startHz = options.startHz;
  const endHz = options.endHz ?? startHz;
  const shape = options.shape ?? 'flat';
  const harmonics = options.harmonics ?? 5;
  const amplitude = options.amplitude ?? 0.4;
  const fade = Math.max(1, Math.round((options.fadeSeconds ?? 0.03) * rate));

  let phase = 0;
  for (let i = 0; i < length; i += 1) {
    const position = length === 1 ? 0 : i / (length - 1);
    const f0 = contourFrequency(startHz, endHz, shape, position);
    phase += (2 * Math.PI * f0) / rate;
    let value = 0;
    for (let h = 1; h <= harmonics; h += 1) {
      if (f0 * h >= rate / 2) break;
      value += Math.sin(phase * h) / h;
    }
    const attack = Math.min(1, i / fade);
    const release = Math.min(1, (length - 1 - i) / fade);
    out[i] = amplitude * value * Math.min(attack, release);
  }
  return out;
}

function contourFrequency(startHz: number, endHz: number, shape: ContourShape, position: number): number {
  switch (shape) {
    case 'flat':
      return startHz;
    case 'rising':
      return startHz + (Math.max(endHz, startHz) - startHz) * position;
    case 'falling':
      return startHz - (startHz - Math.min(endHz, startHz)) * position;
    case 'arc':
      return startHz + (endHz - startHz) * Math.sin(Math.PI * position);
  }
}

/**
 * A human imitation of an animal call: the same contour, but with a lower
 * fundamental, formant-like emphasis and a noisier, breathier spectrum.
 */
export function humanImitation(options: CallOptions): Float32Array {
  const call = harmonicCall({ ...options, startHz: options.startHz * 0.55, harmonics: 12 });
  const breath = whiteNoise(options, (options.amplitude ?? 0.4) * 0.25);
  const out = new Float32Array(call.length);
  for (let i = 0; i < out.length; i += 1) {
    const envelope = Math.min(1, i / 200, (out.length - i) / 200);
    out[i] = (call[i] as number) + (breath[i] as number) * envelope;
  }
  return out;
}

/** Concatenate buffers, optionally separating them with silence. */
export function concat(parts: readonly Float32Array[], gapSamples = 0): Float32Array {
  const total = parts.reduce((sum, part) => sum + part.length + gapSamples, 0);
  const out = new Float32Array(Math.max(0, total - gapSamples));
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.length + gapSamples;
  }
  return out;
}

/** Add two buffers sample by sample, truncating to the shorter one. */
export function mix(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(Math.min(a.length, b.length));
  for (let i = 0; i < out.length; i += 1) out[i] = (a[i] as number) + (b[i] as number);
  return out;
}

function sampleCount(options: SignalOptions): number {
  return Math.max(0, Math.round(options.seconds * (options.sampleRateHz ?? SAMPLE_RATE_HZ)));
}

function countTrailingZeros(value: number): number {
  let count = 0;
  let n = value;
  while ((n & 1) === 0 && count < 31) {
    n >>>= 1;
    count += 1;
  }
  return count;
}
