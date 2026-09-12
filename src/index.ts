/**
 * earshot — the shared in-browser audio engine behind SteadyHum and Meowlogue.
 *
 * The library has no UI, no persistence and makes no network calls: it takes
 * microphone audio in and hands back plain serializable objects that the host
 * app stores and renders.
 *
 * @see README.md for a runnable end-to-end example.
 */

// Constants.
export {
  DBFS_FLOOR,
  EMBEDDING_DIMENSIONS,
  ENVELOPE_HOP_MS,
  FFT_SIZE,
  HOP_SAMPLES,
  HOP_SECONDS,
  MEL_BANDS,
  MEL_MAX_HZ,
  MEL_MIN_HZ,
  PITCH_FRAME_MS,
  PITCH_HOP_MS,
  PITCH_MAX_HZ,
  PITCH_MIN_HZ,
  SAMPLE_RATE_HZ,
  SILENCE_EPSILON,
  STFT_HOP_MS,
  STFT_HOP_SAMPLES,
  STFT_WINDOW_MS,
  STFT_WINDOW_SAMPLES,
  WINDOW_SAMPLES,
  WINDOW_SECONDS,
} from './constants.js';

// Types.
export type {
  BandEnergy,
  ClassScore,
  PitchFrame,
  PitchTrack,
  QuantizedEmbedding,
  SpectralPeak,
  WindowFeatures,
  WindowResult,
} from './util/types.js';

// Capture.
export { createCapture } from './capture/index.js';
export type { AppliedConstraints, Capture, CaptureOptions } from './capture/index.js';
export { CAPTURE_PROCESSOR_NAME } from './capture/worklet-protocol.js';
export type {
  CaptureChunkMessage,
  CaptureMessage,
  CaptureProcessorOptions,
  CaptureReadyMessage,
} from './capture/worklet-protocol.js';

// Engine.
export { createEngine } from './engine.js';
export type { Engine, EngineOptions } from './engine.js';
export type { EngineRequest, EngineResponse } from './worker/protocol.js';

// DSP.
export { resample, toMono } from './dsp/resample.js';
export { createFramer, frameBuffer } from './dsp/framing.js';
export type { Frame, Framer, FramerOptions } from './dsp/framing.js';
export { amplitudeEnvelope, peakDbfs, rmsDbfs } from './dsp/level.js';
export type { Envelope } from './dsp/level.js';
export { Fft, binToHz, hannWindow, magnitudeBins } from './dsp/fft.js';
export {
  averageSpectrum,
  bandEnergies,
  computeSpectrogram,
  createSpectrogramAnalyzer,
  findPeaks,
  hzToMel,
  logMel,
  melFilterbank,
  melToHz,
  spectralCentroidHz,
  spectralFlatness,
  OCTAVE_BAND_EDGES_HZ,
} from './dsp/spectrum.js';
export type {
  PeakOptions,
  Spectrogram,
  SpectrogramAnalyzer,
  SpectrogramOptions,
} from './dsp/spectrum.js';
export { detectOnsets, onsetPeriodicity, spectralFlux } from './dsp/onsets.js';
export type { OnsetCurve, OnsetOptions, OnsetPeriodicity } from './dsp/onsets.js';
export { modulationRate } from './dsp/modulation.js';
export type { ModulationEstimate } from './dsp/modulation.js';
export { createFeatureExtractor, extractFeatures, featureVector, featureVectorNames } from './dsp/features.js';
export type { FeatureExtractor, FeatureOptions } from './dsp/features.js';
export { summarize as summarizePitch, trackPitch, yin } from './dsp/pitch.js';
export type { PitchOptions } from './dsp/pitch.js';

// Models.
export { createClassifier, maxScoreOf, mergeClassifications, scoreOf } from './models/classifier.js';
export type { Classifier, ClassifierOptions } from './models/classifier.js';
export { averageEmbeddings, createEmbedder } from './models/embedder.js';
export type { Embedder, EmbedderOptions } from './models/embedder.js';
export { defaultTasksAudioLoader, EMBEDDER_MAX_VERSION } from './models/tasks-audio.js';
export type { ModelUrls, TasksAudioLoader, TasksAudioModule } from './models/tasks-audio.js';

// Guards.
export {
  createGuards,
  humanVoiceScore,
  isPossiblyHuman,
  HUMAN_VOICE_CLASSES,
  INTERFERENCE_CLASSES,
} from './guards/index.js';
export type { GuardConfig, GuardReason, Guards, GuardVerdict } from './guards/index.js';

// Learning.
export { kmeans, selectClustering, silhouette, squaredDistance } from './learn/kmeans.js';
export type { Clustering, KmeansOptions, SelectClusteringOptions } from './learn/kmeans.js';
export {
  buildDistanceDistribution,
  distanceScore,
  fitDiagonalGaussian,
  mahalanobisDistance,
} from './learn/statistics.js';
export type { DiagonalGaussian, DistanceDistribution } from './learn/statistics.js';
export {
  calibrate,
  describableValues,
  learnProfile,
  toVector,
  DESCRIBABLE_FEATURES,
  PROFILE_SCHEMA_VERSION,
} from './learn/profile.js';
export type {
  CalibrationRecord,
  FeatureSpace,
  FeatureStat,
  LearnProfileOptions,
  Profile,
  ProfileState,
  ProfileThresholds,
  Verdict,
} from './learn/profile.js';

// Scoring.
export { findState, scoreCheck, scoreWindow, statusFor } from './score/score.js';
export type { CheckResult, ScoreOptions, Status, WindowScore } from './score/score.js';
export { describeDifference } from './score/describe.js';
export type { DescribeOptions, Descriptor } from './score/describe.js';
export { createStreamScorer } from './score/stream.js';
export type { DriftState, StreamScorer, StreamScorerOptions, StreamUpdate } from './score/stream.js';

// Events.
export { countSyllables, segmentBuffer, segmentEnvelope } from './events/segment.js';
export type { Segment, SegmentOptions } from './events/segment.js';
export { createEventDetector, eventHumanScore } from './events/detector.js';
export type { EventDetector, EventDetectorConfig, VocalEvent } from './events/detector.js';

// Classification.
export {
  buildPrototypes,
  createKnnClassifier,
  predictByPrototype,
  predictWith,
  KNN_SCHEMA_VERSION,
} from './classify/knn.js';
export type {
  Example,
  KnnClassifier,
  KnnOptions,
  KnnSnapshot,
  LabelScore,
  Neighbour,
  Prediction,
  Prototype,
} from './classify/knn.js';
export { crossValidate } from './classify/validate.js';
export type { LabelReport, ValidateOptions, ValidationReport } from './classify/validate.js';

// Utilities.
export {
  base64ToBytes,
  bytesToBase64,
  dequantize,
  float16ToFloat32,
  float32ToFloat16,
  quantize,
} from './util/quantize.js';
export type { QuantizationFormat } from './util/quantize.js';
export {
  amplitudeToDbfs,
  clamp,
  cosineDistance,
  cosineSimilarity,
  createRandom,
  dot,
  l2Normalize,
  mean,
  median,
  norm,
  percentile,
  percentileRank,
  powerToDb,
  semitones,
  standardDeviation,
} from './util/math.js';
