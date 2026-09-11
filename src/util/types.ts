/**
 * Serializable types crossing the earshot boundary.
 *
 * Every type in this file is plain JSON-compatible data (numbers, strings,
 * arrays and objects; never a typed array, a `Map` or a class instance) so
 * that consuming apps can persist it with Dexie and send it through
 * `postMessage` without a custom serializer.
 */

/** A class score emitted by the audio classifier. */
export interface ClassScore {
  /** Class name as reported by the model, e.g. `"Speech"`. */
  readonly label: string;
  /** Model confidence in `[0, 1]`. */
  readonly score: number;
}

/** Energy in one frequency band. */
export interface BandEnergy {
  /** Lower edge of the band, in Hz. */
  readonly lowHz: number;
  /** Upper edge of the band, in Hz. */
  readonly highHz: number;
  /** Band level, in dBFS. */
  readonly levelDbfs: number;
}

/** A prominent peak of the magnitude spectrum. */
export interface SpectralPeak {
  /** Peak frequency, in Hz (parabolically interpolated). */
  readonly frequencyHz: number;
  /** Peak level, in dBFS. */
  readonly levelDbfs: number;
  /** Height above the surrounding spectral floor, in dB. */
  readonly prominenceDb: number;
}

/** Interpretable features extracted from a single analysis window. */
export interface WindowFeatures {
  /** Broadband level, in dBFS. */
  readonly rmsDbfs: number;
  /** 64 log-mel band values, in dB, averaged over the window. */
  readonly logMel: readonly number[];
  /** Energy per octave band, in dBFS. */
  readonly bands: readonly BandEnergy[];
  /** Up to `maxPeaks` prominent spectral peaks, strongest first. */
  readonly peaks: readonly SpectralPeak[];
  /** Wiener entropy of the average spectrum in `[0, 1]`; 1 is white noise. */
  readonly spectralFlatness: number;
  /** Centre of mass of the spectrum, in Hz. */
  readonly spectralCentroidHz: number;
  /** Mean half-wave-rectified spectral flux across the window. */
  readonly spectralFlux: number;
  /** Onsets detected inside the window, in seconds relative to the window start. */
  readonly onsets: readonly number[];
  /**
   * Strength in `[0, 1]` of the dominant periodicity of the onset sequence;
   * 0 when the window holds too few onsets to judge.
   */
  readonly onsetPeriodicity: number;
  /** Period of that dominant onset periodicity, in seconds; 0 when undefined. */
  readonly onsetPeriodSeconds: number;
  /** Dominant amplitude-modulation rate, in Hz, searched in `[2, 20]` Hz. */
  readonly amplitudeModulationHz: number;
  /** Depth of that modulation in `[0, 1]`. */
  readonly amplitudeModulationDepth: number;
}

/** Everything the engine derives from one analysis window. */
export interface WindowResult {
  /** Window start time, in seconds since the capture started. */
  readonly t: number;
  /** YAMNet embedding; empty when the embedder is not configured. */
  readonly embedding: readonly number[];
  /** Top class scores, strongest first; empty when the classifier is not configured. */
  readonly classes: readonly ClassScore[];
  /** Broadband level of the window, in dBFS (mirrors `features.rmsDbfs`). */
  readonly rmsDbfs: number;
  /** Interpretable features of the window. */
  readonly features: WindowFeatures;
}

/** A voiced/unvoiced decision for one pitch frame. */
export interface PitchFrame {
  /** Frame centre, in seconds relative to the analysed buffer. */
  readonly t: number;
  /** Fundamental frequency, in Hz; 0 when the frame is unvoiced. */
  readonly f0Hz: number;
  /** Periodicity confidence in `[0, 1]`; 0 when the frame is unvoiced. */
  readonly confidence: number;
  /** Frame level, in dBFS. */
  readonly rmsDbfs: number;
}

/** The result of running the YIN tracker over a buffer. */
export interface PitchTrack {
  /** Per-frame pitch estimates at {@link PITCH_HOP_MS} resolution. */
  readonly frames: readonly PitchFrame[];
  /** Median f0 over voiced frames, in Hz; 0 when nothing is voiced. */
  readonly medianF0Hz: number;
  /** Fraction of frames that are voiced, in `[0, 1]`. */
  readonly voicedFraction: number;
  /** Lowest voiced f0, in Hz; 0 when nothing is voiced. */
  readonly minF0Hz: number;
  /** Highest voiced f0, in Hz; 0 when nothing is voiced. */
  readonly maxF0Hz: number;
  /**
   * Overall slope of the voiced f0 contour, in semitones per second.
   * Negative means the call falls; positive means it rises.
   */
  readonly contourSlopeSemitonesPerSecond: number;
}

/** A quantized embedding, ready to persist. */
export interface QuantizedEmbedding {
  /** Quantization scheme used to produce {@link data}. */
  readonly format: 'float16' | 'int8';
  /** Number of dimensions in the original vector. */
  readonly dimensions: number;
  /** Base64 of the packed little-endian payload. */
  readonly data: string;
  /** Dequantization scale; only meaningful for `int8`. */
  readonly scale: number;
}
