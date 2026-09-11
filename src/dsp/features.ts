/**
 * The window feature extractor: everything interpretable earshot derives from
 * one 0.975 s window, computed once and shared by scoring, guards and events.
 */

import { FFT_SIZE, MEL_BANDS, SAMPLE_RATE_HZ } from '../constants.js';
import type { WindowFeatures } from '../util/types.js';
import { amplitudeEnvelope, rmsDbfs } from './level.js';
import { melFilterbank } from './spectrum.js';
import {
  averageSpectrum,
  bandEnergies,
  computeSpectrogram,
  findPeaks,
  logMel,
  spectralCentroidHz,
  spectralFlatness,
  OCTAVE_BAND_EDGES_HZ,
} from './spectrum.js';
import { detectOnsets, onsetPeriodicity, spectralFlux } from './onsets.js';
import { modulationRate } from './modulation.js';
import { mean } from '../util/math.js';

/** Options for {@link createFeatureExtractor}. */
export interface FeatureOptions {
  /** Sample rate of the windows to analyse, in Hz. Defaults to {@link SAMPLE_RATE_HZ}. */
  readonly sampleRateHz?: number;
  /** Number of mel bands. Defaults to {@link MEL_BANDS}. */
  readonly melBands?: number;
  /** Band edges for {@link bandEnergies}, in Hz. */
  readonly bandEdgesHz?: readonly number[];
  /** Maximum spectral peaks to report per window. Defaults to 5. */
  readonly maxPeaks?: number;
  /** Minimum peak prominence, in dB. Defaults to 6. */
  readonly minPeakProminenceDb?: number;
}

/** A reusable extractor holding the mel filterbank and FFT tables. */
export interface FeatureExtractor {
  /**
   * Extract features from one analysis window.
   *
   * @param samples - Mono window at the extractor's sample rate.
   */
  extract(samples: Float32Array): WindowFeatures;
}

/**
 * Create a feature extractor.
 *
 * The filterbank is built once; each {@link FeatureExtractor.extract} call runs
 * a single STFT over the window and derives every feature from it.
 */
export function createFeatureExtractor(options: FeatureOptions = {}): FeatureExtractor {
  const sampleRateHz = options.sampleRateHz ?? SAMPLE_RATE_HZ;
  const melBands = options.melBands ?? MEL_BANDS;
  const bandEdgesHz = options.bandEdgesHz ?? OCTAVE_BAND_EDGES_HZ;
  const filterbank = melFilterbank(melBands, FFT_SIZE, sampleRateHz);

  return {
    extract(samples: Float32Array): WindowFeatures {
      const spectrogram = computeSpectrogram(samples, { sampleRateHz });
      const spectrum = averageSpectrum(spectrogram);
      const flux = spectralFlux(spectrogram);
      const onsets = detectOnsets(flux);
      const periodicity = onsetPeriodicity(onsets);
      const envelope = amplitudeEnvelope(samples, 10, sampleRateHz);
      const modulation = modulationRate(envelope);

      const peakOptions =
        options.minPeakProminenceDb === undefined
          ? { maxPeaks: options.maxPeaks ?? 5 }
          : { maxPeaks: options.maxPeaks ?? 5, minProminenceDb: options.minPeakProminenceDb };

      return {
        rmsDbfs: rmsDbfs(samples),
        logMel: Array.from(logMel(spectrum, filterbank, melBands)),
        bands: bandEnergies(spectrum, bandEdgesHz, FFT_SIZE, sampleRateHz),
        peaks: findPeaks(spectrum, peakOptions, FFT_SIZE, sampleRateHz),
        spectralFlatness: spectralFlatness(spectrum),
        spectralCentroidHz: spectralCentroidHz(spectrum, FFT_SIZE, sampleRateHz),
        spectralFlux: mean(flux.strength),
        onsets,
        onsetPeriodicity: periodicity.strength,
        onsetPeriodSeconds: periodicity.periodSeconds,
        amplitudeModulationHz: modulation.rateHz,
        amplitudeModulationDepth: modulation.depth,
      };
    },
  };
}

/** Extract features from a single window without reusing an extractor. */
export function extractFeatures(samples: Float32Array, options: FeatureOptions = {}): WindowFeatures {
  return createFeatureExtractor(options).extract(samples);
}

/**
 * Flatten the numeric part of a feature set into a fixed-length vector.
 *
 * The spectral dimensions carry the *shape* of the spectrum, not its absolute
 * level: the mean is subtracted from the log-mel bands and from the octave
 * bands, and the overall level is kept as a single separate dimension. This
 * matters because the profile models these dimensions with a diagonal
 * covariance, which assumes they vary independently. They do not — turning the
 * gain up a decibel moves all 71 of them by exactly one decibel, and a diagonal
 * model reads that coherent shift as 71 independent surprises at once. Removing
 * the common mode leaves a gain change visible in exactly the dimension that
 * means "level", where it belongs.
 *
 * The order is stable across versions of earshot within a MINOR release and is
 * what {@link learnProfile} clusters on when `featureSpace` is `'features'`.
 */
export function featureVector(features: WindowFeatures): Float32Array {
  const out = new Float32Array(features.logMel.length + features.bands.length + 6);
  const melMean = mean(features.logMel);
  const bandLevels = features.bands.map((band) => band.levelDbfs);
  const bandMean = mean(bandLevels);
  let i = 0;
  for (const value of features.logMel) out[i++] = value - melMean;
  for (const value of bandLevels) out[i++] = value - bandMean;
  out[i++] = features.rmsDbfs;
  out[i++] = features.spectralFlatness;
  out[i++] = features.spectralCentroidHz;
  out[i++] = features.spectralFlux;
  out[i++] = features.amplitudeModulationHz;
  out[i++] = features.amplitudeModulationDepth;
  return out;
}

/** Human-readable names for each dimension of {@link featureVector}. */
export function featureVectorNames(features: WindowFeatures): string[] {
  const names: string[] = [];
  for (let i = 0; i < features.logMel.length; i += 1) names.push(`mel shape[${i}]`);
  for (const band of features.bands) names.push(`band balance ${band.lowHz}-${band.highHz} Hz`);
  names.push('level', 'spectral flatness', 'spectral centroid', 'spectral flux', 'AM rate', 'AM depth');
  return names;
}
