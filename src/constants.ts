/**
 * Framing and analysis constants shared by every earshot module.
 *
 * These values match the input contract of YAMNet as exposed by MediaPipe
 * Tasks Audio, so that the same frames feed the classifier, the embedder and
 * the hand-written feature extractors without resampling twice.
 */

/** Working sample rate of the whole engine, in Hz. */
export const SAMPLE_RATE_HZ = 16000;

/** Analysis window length, in seconds (YAMNet's patch length). */
export const WINDOW_SECONDS = 0.975;

/** Hop between consecutive analysis windows, in seconds (50 % overlap). */
export const HOP_SECONDS = 0.4875;

/** Analysis window length, in samples at {@link SAMPLE_RATE_HZ}. */
export const WINDOW_SAMPLES = Math.round(WINDOW_SECONDS * SAMPLE_RATE_HZ);

/** Hop between consecutive analysis windows, in samples. */
export const HOP_SAMPLES = Math.round(HOP_SECONDS * SAMPLE_RATE_HZ);

/** Short-time Fourier transform window length, in milliseconds. */
export const STFT_WINDOW_MS = 25;

/** Short-time Fourier transform hop, in milliseconds. */
export const STFT_HOP_MS = 10;

/** STFT window length, in samples. */
export const STFT_WINDOW_SAMPLES = Math.round((STFT_WINDOW_MS / 1000) * SAMPLE_RATE_HZ);

/** STFT hop, in samples. */
export const STFT_HOP_SAMPLES = Math.round((STFT_HOP_MS / 1000) * SAMPLE_RATE_HZ);

/** FFT size used for the STFT: the next power of two above {@link STFT_WINDOW_SAMPLES}. */
export const FFT_SIZE = 512;

/** Number of log-mel bands produced by {@link melSpectrogram}. */
export const MEL_BANDS = 64;

/** Lowest edge of the mel filterbank, in Hz. */
export const MEL_MIN_HZ = 125;

/** Highest edge of the mel filterbank, in Hz. */
export const MEL_MAX_HZ = 7500;

/** Lowest pitch the YIN tracker will report, in Hz. */
export const PITCH_MIN_HZ = 80;

/** Highest pitch the YIN tracker will report, in Hz. */
export const PITCH_MAX_HZ = 1200;

/** Pitch analysis frame length, in milliseconds. */
export const PITCH_FRAME_MS = 25;

/** Pitch analysis hop, in milliseconds. */
export const PITCH_HOP_MS = 10;

/** Resolution of the amplitude envelope used for event segmentation, in milliseconds. */
export const ENVELOPE_HOP_MS = 10;

/** Amplitude below which a sample is treated as digital silence (linear, full scale). */
export const SILENCE_EPSILON = 1e-7;

/** Level reported instead of `-Infinity` for digital silence, in dBFS. */
export const DBFS_FLOOR = -120;

/** Dimensionality of a YAMNet embedding vector. */
export const EMBEDDING_DIMENSIONS = 1024;
