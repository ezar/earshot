/**
 * Short-time spectral analysis: magnitude spectrogram, log-mel bands,
 * flatness, centroid, prominent peaks and octave band energies.
 */

import {
  FFT_SIZE,
  MEL_BANDS,
  MEL_MAX_HZ,
  MEL_MIN_HZ,
  SAMPLE_RATE_HZ,
  STFT_HOP_SAMPLES,
  STFT_WINDOW_SAMPLES,
} from '../constants.js';
import type { BandEnergy, SpectralPeak } from '../util/types.js';
import { powerToDb } from '../util/math.js';
import { Fft, binToHz, hannWindow, magnitudeBins } from './fft.js';

/** A magnitude spectrogram plus the geometry needed to interpret it. */
export interface Spectrogram {
  /** `frames * bins` linear magnitudes, row-major. */
  readonly data: Float32Array;
  /** Number of time frames. */
  readonly frames: number;
  /** Number of frequency bins per frame, `fftSize / 2 + 1`. */
  readonly bins: number;
  /** Spacing between frames, in seconds. */
  readonly hopSeconds: number;
  /** Sample rate the spectrogram was computed at, in Hz. */
  readonly sampleRateHz: number;
  /** Transform size used, in samples. */
  readonly fftSize: number;
}

/** Options for {@link computeSpectrogram}. */
export interface SpectrogramOptions {
  /** STFT window length, in samples. Defaults to {@link STFT_WINDOW_SAMPLES}. */
  readonly windowSamples?: number;
  /** STFT hop, in samples. Defaults to {@link STFT_HOP_SAMPLES}. */
  readonly hopSamples?: number;
  /** Transform size, in samples. Defaults to {@link FFT_SIZE}. */
  readonly fftSize?: number;
  /** Sample rate of the input, in Hz. Defaults to {@link SAMPLE_RATE_HZ}. */
  readonly sampleRateHz?: number;
}

/**
 * Compute the magnitude spectrogram of a buffer.
 *
 * Magnitudes are normalized by the window's coherent gain, so a full-scale
 * sine lands near 0 dBFS in its bin regardless of the window length.
 */
export function computeSpectrogram(samples: Float32Array, options: SpectrogramOptions = {}): Spectrogram {
  const windowSamples = options.windowSamples ?? STFT_WINDOW_SAMPLES;
  const hopSamples = options.hopSamples ?? STFT_HOP_SAMPLES;
  const fftSize = options.fftSize ?? FFT_SIZE;
  const sampleRateHz = options.sampleRateHz ?? SAMPLE_RATE_HZ;

  const bins = magnitudeBins(fftSize);
  const frames = samples.length < windowSamples ? 0 : Math.floor((samples.length - windowSamples) / hopSamples) + 1;
  const data = new Float32Array(frames * bins);
  if (frames === 0) {
    return { data, frames, bins, hopSeconds: hopSamples / sampleRateHz, sampleRateHz, fftSize };
  }

  const fft = new Fft(fftSize);
  const window = hannWindow(windowSamples);
  let coherentGain = 0;
  for (let i = 0; i < windowSamples; i += 1) coherentGain += window[i] as number;
  const scale = 2 / Math.max(coherentGain, 1e-9);

  const windowed = new Float32Array(windowSamples);
  const magnitude = new Float32Array(bins);
  for (let f = 0; f < frames; f += 1) {
    const offset = f * hopSamples;
    for (let i = 0; i < windowSamples; i += 1) {
      windowed[i] = (samples[offset + i] as number) * (window[i] as number);
    }
    fft.realMagnitude(windowed, magnitude);
    for (let b = 0; b < bins; b += 1) {
      data[f * bins + b] = (magnitude[b] as number) * scale;
    }
  }
  return { data, frames, bins, hopSeconds: hopSamples / sampleRateHz, sampleRateHz, fftSize };
}

/** Average the magnitude spectrogram over time into a single spectrum. */
export function averageSpectrum(spectrogram: Spectrogram): Float32Array {
  const { data, frames, bins } = spectrogram;
  const out = new Float32Array(bins);
  if (frames === 0) return out;
  for (let f = 0; f < frames; f += 1) {
    for (let b = 0; b < bins; b += 1) out[b] = (out[b] as number) + (data[f * bins + b] as number);
  }
  for (let b = 0; b < bins; b += 1) out[b] = (out[b] as number) / frames;
  return out;
}

/** Convert a frequency in Hz to the mel scale (HTK formula). */
export function hzToMel(hz: number): number {
  return 2595 * Math.log10(1 + hz / 700);
}

/** Convert a mel value back to Hz (HTK formula). */
export function melToHz(mel: number): number {
  return 700 * (10 ** (mel / 2595) - 1);
}

/**
 * Build a triangular mel filterbank matrix, `bands * bins`, row-major.
 *
 * Filters are area-normalized so that a flat spectrum yields equal band values
 * regardless of how many FFT bins fall inside each triangle.
 */
export function melFilterbank(
  bands: number = MEL_BANDS,
  fftSize: number = FFT_SIZE,
  sampleRateHz: number = SAMPLE_RATE_HZ,
  minHz: number = MEL_MIN_HZ,
  maxHz: number = MEL_MAX_HZ,
): Float32Array {
  const bins = magnitudeBins(fftSize);
  const matrix = new Float32Array(bands * bins);
  const lowMel = hzToMel(minHz);
  const highMel = hzToMel(maxHz);
  const edges = new Float32Array(bands + 2);
  for (let i = 0; i < bands + 2; i += 1) {
    edges[i] = melToHz(lowMel + ((highMel - lowMel) * i) / (bands + 1));
  }
  for (let m = 0; m < bands; m += 1) {
    const left = edges[m] as number;
    const centre = edges[m + 1] as number;
    const right = edges[m + 2] as number;
    let sum = 0;
    for (let b = 0; b < bins; b += 1) {
      const hz = binToHz(b, fftSize, sampleRateHz);
      let weight = 0;
      if (hz > left && hz <= centre) weight = (hz - left) / Math.max(centre - left, 1e-9);
      else if (hz > centre && hz < right) weight = (right - hz) / Math.max(right - centre, 1e-9);
      matrix[m * bins + b] = weight;
      sum += weight;
    }
    if (sum > 0) {
      for (let b = 0; b < bins; b += 1) {
        matrix[m * bins + b] = (matrix[m * bins + b] as number) / sum;
      }
    }
  }
  return matrix;
}

/**
 * Project a spectrum through a mel filterbank and take the log.
 *
 * @param spectrum - Linear magnitudes of length `fftSize / 2 + 1`.
 * @param filterbank - Matrix from {@link melFilterbank}.
 * @param bands - Number of mel bands in `filterbank`.
 * @returns Band levels in dB, floored at -120 dB.
 */
export function logMel(spectrum: Float32Array, filterbank: Float32Array, bands: number = MEL_BANDS): Float32Array {
  const bins = spectrum.length;
  const out = new Float32Array(bands);
  for (let m = 0; m < bands; m += 1) {
    let acc = 0;
    for (let b = 0; b < bins; b += 1) {
      const magnitude = spectrum[b] as number;
      acc += magnitude * magnitude * (filterbank[m * bins + b] as number);
    }
    out[m] = powerToDb(acc);
  }
  return out;
}

/**
 * Wiener entropy (spectral flatness) of a magnitude spectrum, in `[0, 1]`.
 * Tonal signals approach 0; white noise approaches 1.
 */
export function spectralFlatness(spectrum: Float32Array): number {
  let logSum = 0;
  let linearSum = 0;
  let count = 0;
  for (let b = 1; b < spectrum.length; b += 1) {
    const power = (spectrum[b] as number) ** 2 + 1e-20;
    logSum += Math.log(power);
    linearSum += power;
    count += 1;
  }
  if (count === 0 || linearSum <= 0) return 0;
  const geometric = Math.exp(logSum / count);
  const arithmetic = linearSum / count;
  return Math.min(1, geometric / arithmetic);
}

/** Spectral centroid of a magnitude spectrum, in Hz. */
export function spectralCentroidHz(
  spectrum: Float32Array,
  fftSize: number = FFT_SIZE,
  sampleRateHz: number = SAMPLE_RATE_HZ,
): number {
  let weighted = 0;
  let total = 0;
  for (let b = 1; b < spectrum.length; b += 1) {
    const magnitude = spectrum[b] as number;
    weighted += magnitude * binToHz(b, fftSize, sampleRateHz);
    total += magnitude;
  }
  return total > 0 ? weighted / total : 0;
}

/** Default octave band edges, in Hz, covering the 16 kHz working range. */
export const OCTAVE_BAND_EDGES_HZ: readonly number[] = [
  62.5, 125, 250, 500, 1000, 2000, 4000, 8000,
];

/**
 * Integrate a magnitude spectrum into octave bands.
 *
 * @param spectrum - Linear magnitudes.
 * @param edgesHz - Ascending band edges; `n` edges produce `n - 1` bands.
 */
export function bandEnergies(
  spectrum: Float32Array,
  edgesHz: readonly number[] = OCTAVE_BAND_EDGES_HZ,
  fftSize: number = FFT_SIZE,
  sampleRateHz: number = SAMPLE_RATE_HZ,
): BandEnergy[] {
  const out: BandEnergy[] = [];
  for (let i = 0; i + 1 < edgesHz.length; i += 1) {
    const lowHz = edgesHz[i] as number;
    const highHz = edgesHz[i + 1] as number;
    let power = 0;
    for (let b = 1; b < spectrum.length; b += 1) {
      const hz = binToHz(b, fftSize, sampleRateHz);
      if (hz >= lowHz && hz < highHz) power += (spectrum[b] as number) ** 2;
    }
    out.push({ lowHz, highHz, levelDbfs: powerToDb(power) });
  }
  return out;
}

/** Options for {@link findPeaks}. */
export interface PeakOptions {
  /** Maximum number of peaks to return. Defaults to 5. */
  readonly maxPeaks?: number;
  /** Minimum prominence above the local floor, in dB. Defaults to 6. */
  readonly minProminenceDb?: number;
  /** Half-width of the neighbourhood used as the local floor, in Hz. Defaults to 300. */
  readonly floorRadiusHz?: number;
}

/**
 * Find prominent spectral peaks.
 *
 * Prominence is measured against the median level of a neighbourhood of
 * `floorRadiusHz` on each side, which is robust to broadband level changes and
 * therefore usable as a machine-whine detector across different rooms.
 *
 * @returns Peaks sorted by descending prominence.
 */
export function findPeaks(
  spectrum: Float32Array,
  options: PeakOptions = {},
  fftSize: number = FFT_SIZE,
  sampleRateHz: number = SAMPLE_RATE_HZ,
): SpectralPeak[] {
  const maxPeaks = options.maxPeaks ?? 5;
  const minProminenceDb = options.minProminenceDb ?? 6;
  const floorRadiusHz = options.floorRadiusHz ?? 300;
  const binHz = sampleRateHz / fftSize;
  const radiusBins = Math.max(2, Math.round(floorRadiusHz / binHz));

  const db = new Float32Array(spectrum.length);
  for (let b = 0; b < spectrum.length; b += 1) {
    db[b] = powerToDb((spectrum[b] as number) ** 2);
  }

  const candidates: SpectralPeak[] = [];
  const neighbourhood: number[] = [];
  for (let b = 1; b + 1 < db.length; b += 1) {
    const here = db[b] as number;
    if (here <= (db[b - 1] as number) || here < (db[b + 1] as number)) continue;
    neighbourhood.length = 0;
    const from = Math.max(1, b - radiusBins);
    const to = Math.min(db.length - 1, b + radiusBins);
    for (let j = from; j < to; j += 1) {
      if (Math.abs(j - b) > 1) neighbourhood.push(db[j] as number);
    }
    if (neighbourhood.length === 0) continue;
    neighbourhood.sort((x, y) => x - y);
    const floor = neighbourhood[Math.floor(neighbourhood.length / 2)] as number;
    const prominenceDb = here - floor;
    if (prominenceDb < minProminenceDb) continue;
    candidates.push({
      frequencyHz: interpolatePeakHz(db, b, fftSize, sampleRateHz),
      levelDbfs: here,
      prominenceDb,
    });
  }
  candidates.sort((a, b) => b.prominenceDb - a.prominenceDb);
  return candidates.slice(0, maxPeaks);
}

/** Parabolic interpolation of a peak's true frequency from its dB neighbours. */
function interpolatePeakHz(db: Float32Array, index: number, fftSize: number, sampleRateHz: number): number {
  const left = db[index - 1] as number;
  const centre = db[index] as number;
  const right = db[index + 1] as number;
  const denominator = left - 2 * centre + right;
  const offset = denominator === 0 ? 0 : (0.5 * (left - right)) / denominator;
  return binToHz(index + Math.max(-0.5, Math.min(0.5, offset)), fftSize, sampleRateHz);
}
