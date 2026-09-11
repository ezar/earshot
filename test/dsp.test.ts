import { describe, expect, it } from 'vitest';

import { SAMPLE_RATE_HZ, WINDOW_SAMPLES, HOP_SAMPLES, FFT_SIZE } from '../src/constants.js';
import { Fft, binToHz, hannWindow } from '../src/dsp/fft.js';
import { resample, toMono } from '../src/dsp/resample.js';
import { createFramer, frameBuffer } from '../src/dsp/framing.js';
import { amplitudeEnvelope, peakDbfs, rmsDbfs } from '../src/dsp/level.js';
import {
  averageSpectrum,
  bandEnergies,
  computeSpectrogram,
  findPeaks,
  hzToMel,
  logMel,
  melFilterbank,
  melToHz,
  spectralCentroidHz,
  spectralFlatness,
} from '../src/dsp/spectrum.js';
import { detectOnsets, onsetPeriodicity, spectralFlux } from '../src/dsp/onsets.js';
import { modulationRate } from '../src/dsp/modulation.js';
import { extractFeatures, featureVector } from '../src/dsp/features.js';
import {
  amplitudeModulate,
  applyGainDb,
  knocks,
  pinkNoise,
  tone,
  whiteNoise,
} from '../fixtures/synthetic/index.js';

describe('fft', () => {
  it('puts a pure tone in the expected bin', () => {
    const fft = new Fft(512);
    const frequencyHz = SAMPLE_RATE_HZ / 512 * 40; // exactly bin 40
    const samples = tone(frequencyHz, { seconds: 512 / SAMPLE_RATE_HZ }, 1);
    const magnitude = new Float32Array(257);
    fft.realMagnitude(samples, magnitude);

    let peakBin = 0;
    for (let b = 1; b < magnitude.length; b += 1) {
      if ((magnitude[b] as number) > (magnitude[peakBin] as number)) peakBin = b;
    }
    expect(peakBin).toBe(40);
  });

  it('rejects non power-of-two sizes', () => {
    expect(() => new Fft(500)).toThrow(/power of two/);
  });

  it('produces a periodic Hann window that starts at zero', () => {
    const window = hannWindow(8);
    expect(window[0]).toBeCloseTo(0, 6);
    expect(window[4]).toBeCloseTo(1, 6);
  });

  it('maps bins to frequencies', () => {
    expect(binToHz(40, 512, SAMPLE_RATE_HZ)).toBeCloseTo(1250, 6);
  });
});

describe('resample', () => {
  it('halves the length when halving the rate', () => {
    const input = tone(440, { seconds: 1, sampleRateHz: 32000 });
    const output = resample(input, 32000, 16000);
    expect(output.length).toBe(16000);
  });

  it('preserves a tone below the new Nyquist', () => {
    const input = tone(440, { seconds: 0.5, sampleRateHz: 48000 }, 0.5);
    const output = resample(input, 48000, SAMPLE_RATE_HZ);
    const spectrum = averageSpectrum(computeSpectrogram(output));
    const peaks = findPeaks(spectrum, { maxPeaks: 1, minProminenceDb: 3 });
    expect(peaks[0]?.frequencyHz).toBeGreaterThan(400);
    expect(peaks[0]?.frequencyHz).toBeLessThan(480);
  });

  it('attenuates content above the new Nyquist instead of aliasing it down', () => {
    // 15 kHz at 48 kHz would alias to 1 kHz at 16 kHz without the low-pass.
    const input = tone(15000, { seconds: 0.5, sampleRateHz: 48000 }, 0.5);
    const output = resample(input, 48000, SAMPLE_RATE_HZ);
    expect(rmsDbfs(output)).toBeLessThan(rmsDbfs(input) - 20);
  });

  it('returns a copy when the rate already matches', () => {
    const input = tone(440, { seconds: 0.1 });
    const output = resample(input, SAMPLE_RATE_HZ, SAMPLE_RATE_HZ);
    expect(output).not.toBe(input);
    expect(Array.from(output)).toEqual(Array.from(input));
  });

  it('mixes interleaved channels down to mono', () => {
    const interleaved = Float32Array.from([1, -1, 2, 0]);
    expect(Array.from(toMono(interleaved, 2))).toEqual([0, 1]);
  });
});

describe('framing', () => {
  it('produces overlapping windows on the documented grid', () => {
    const samples = tone(440, { seconds: 3 });
    const frames = frameBuffer(samples);
    expect(frames[0]?.samples.length).toBe(WINDOW_SAMPLES);
    expect(frames[0]?.t).toBeCloseTo(0, 6);
    expect(frames[1]?.t).toBeCloseTo(HOP_SAMPLES / SAMPLE_RATE_HZ, 6);
    expect(frames.length).toBe(Math.floor((samples.length - WINDOW_SAMPLES) / HOP_SAMPLES) + 1);
  });

  it('is stream-equivalent to one-shot framing', () => {
    const samples = tone(440, { seconds: 3 });
    const framer = createFramer();
    const streamed: number[] = [];
    for (let offset = 0; offset < samples.length; offset += 1024) {
      for (const frame of framer.push(samples.slice(offset, offset + 1024))) streamed.push(frame.t);
    }
    expect(streamed).toEqual(frameBuffer(samples).map((frame) => frame.t));
  });

  it('rejects a hop larger than the window', () => {
    expect(() => createFramer({ windowSamples: 10, hopSamples: 20 })).toThrow(/hopSamples/);
  });
});

describe('level', () => {
  it('reports -3 dBFS for a full-scale sine', () => {
    expect(rmsDbfs(tone(440, { seconds: 0.5 }, 1))).toBeCloseTo(-3.01, 1);
    expect(peakDbfs(tone(440, { seconds: 0.5 }, 1))).toBeCloseTo(0, 1);
  });

  it('floors digital silence', () => {
    expect(rmsDbfs(new Float32Array(100))).toBe(-120);
  });

  it('tracks a level change in the envelope', () => {
    const quiet = tone(440, { seconds: 0.5 }, 0.01);
    const loud = tone(440, { seconds: 0.5 }, 0.5);
    const combined = new Float32Array(quiet.length + loud.length);
    combined.set(quiet, 0);
    combined.set(loud, quiet.length);
    const envelope = amplitudeEnvelope(combined);
    expect(envelope.hopSeconds).toBeCloseTo(0.01, 6);
    const first = envelope.levelsDbfs[5] as number;
    const last = envelope.levelsDbfs[envelope.levelsDbfs.length - 5] as number;
    expect(last - first).toBeGreaterThan(25);
  });
});

describe('spectrum', () => {
  it('round-trips the mel scale', () => {
    for (const hz of [125, 440, 1000, 7500]) {
      expect(melToHz(hzToMel(hz))).toBeCloseTo(hz, 3);
    }
  });

  it('builds an area-normalized filterbank', () => {
    const bank = melFilterbank(8, FFT_SIZE, SAMPLE_RATE_HZ);
    const bins = FFT_SIZE / 2 + 1;
    for (let m = 0; m < 8; m += 1) {
      let sum = 0;
      for (let b = 0; b < bins; b += 1) sum += bank[m * bins + b] as number;
      expect(sum).toBeCloseTo(1, 4);
    }
  });

  it('puts a tone in the mel band that contains it', () => {
    const spectrum = averageSpectrum(computeSpectrogram(tone(1000, { seconds: 1 }, 0.5)));
    const bank = melFilterbank(64, FFT_SIZE, SAMPLE_RATE_HZ);
    const bands = logMel(spectrum, bank, 64);
    let loudest = 0;
    for (let m = 1; m < bands.length; m += 1) {
      if ((bands[m] as number) > (bands[loudest] as number)) loudest = m;
    }
    // 1 kHz sits a little above the middle of a 125 Hz - 7.5 kHz mel bank.
    expect(loudest).toBeGreaterThan(15);
    expect(loudest).toBeLessThan(35);
  });

  it('separates tonal from noisy signals by flatness', () => {
    const tonal = spectralFlatness(averageSpectrum(computeSpectrogram(tone(1000, { seconds: 1 }, 0.5))));
    const noisy = spectralFlatness(averageSpectrum(computeSpectrogram(whiteNoise({ seconds: 1 }, 0.5))));
    expect(tonal).toBeLessThan(0.05);
    expect(noisy).toBeGreaterThan(0.3);
    expect(noisy).toBeGreaterThan(tonal * 5);
  });

  it('places the centroid near a single tone', () => {
    const spectrum = averageSpectrum(computeSpectrogram(tone(2000, { seconds: 1 }, 0.5)));
    expect(spectralCentroidHz(spectrum)).toBeGreaterThan(1600);
    expect(spectralCentroidHz(spectrum)).toBeLessThan(2600);
  });

  it('finds an injected whine with the right frequency', () => {
    const spectrum = averageSpectrum(computeSpectrogram(tone(3000, { seconds: 1 }, 0.3)));
    const peaks = findPeaks(spectrum, { maxPeaks: 3 });
    expect(peaks.length).toBeGreaterThan(0);
    expect(peaks[0]?.frequencyHz).toBeGreaterThan(2900);
    expect(peaks[0]?.frequencyHz).toBeLessThan(3100);
    expect(peaks[0]?.prominenceDb).toBeGreaterThan(20);
  });

  it('finds no prominent peaks in white noise', () => {
    const spectrum = averageSpectrum(computeSpectrogram(whiteNoise({ seconds: 1 }, 0.3)));
    expect(findPeaks(spectrum, { minProminenceDb: 20 }).length).toBe(0);
  });

  it('puts a low tone in the band that contains it', () => {
    const spectrum = averageSpectrum(computeSpectrogram(tone(300, { seconds: 1 }, 0.5)));
    const bands = bandEnergies(spectrum);
    let loudest = 0;
    for (let i = 1; i < bands.length; i += 1) {
      if ((bands[i]?.levelDbfs ?? -Infinity) > (bands[loudest]?.levelDbfs ?? -Infinity)) loudest = i;
    }
    expect(bands[loudest]?.lowHz).toBe(250);
    expect(bands[loudest]?.highHz).toBe(500);
  });
});

describe('onsets', () => {
  it('finds evenly spaced knocks and recovers their period', () => {
    const onsets = detectOnsets(spectralFlux(computeSpectrogram(knocks({ seconds: 4 }, 0.4))));
    expect(onsets.length).toBeGreaterThanOrEqual(6);
    const periodicity = onsetPeriodicity(onsets);
    expect(periodicity.periodSeconds).toBeCloseTo(0.4, 1);
    expect(periodicity.strength).toBeGreaterThan(0.9);
  });

  it('finds no onsets in a steady tone', () => {
    expect(detectOnsets(spectralFlux(computeSpectrogram(tone(440, { seconds: 4 }, 0.3)))).length).toBe(0);
  });

  it('does not call scattered noise onsets periodic', () => {
    const onsets = detectOnsets(spectralFlux(computeSpectrogram(pinkNoise({ seconds: 4 }, 0.2))));
    expect(onsetPeriodicity(onsets).strength).toBeLessThan(0.6);
  });

  it('is gain invariant', () => {
    const loud = knocks({ seconds: 4 }, 0.4, 0.03, 0.6);
    const quiet = applyGainDb(loud, -30);
    const onsetsOf = (s: Float32Array): number[] => detectOnsets(spectralFlux(computeSpectrogram(s)));
    expect(onsetsOf(quiet)).toEqual(onsetsOf(loud));
  });

  it('is undecided about fewer than three onsets', () => {
    expect(onsetPeriodicity([0.1, 0.5])).toEqual({ strength: 0, periodSeconds: 0 });
  });
});

describe('modulation', () => {
  it('recovers the modulation rate of an amplitude-modulated tone', () => {
    const carrier = tone(1000, { seconds: 4 }, 0.4);
    const modulated = amplitudeModulate(carrier, 7, 0.8);
    const estimate = modulationRate(amplitudeEnvelope(modulated));
    expect(estimate.rateHz).toBeGreaterThan(6);
    expect(estimate.rateHz).toBeLessThan(8);
    expect(estimate.depth).toBeGreaterThan(0.1);
  });

  it('reports nothing for an unmodulated tone', () => {
    const estimate = modulationRate(amplitudeEnvelope(tone(1000, { seconds: 4 }, 0.4)));
    expect(estimate.depth).toBeLessThan(0.3);
  });
});

describe('features', () => {
  it('extracts a complete feature set from one window', () => {
    const window = pinkNoise({ seconds: WINDOW_SAMPLES / SAMPLE_RATE_HZ }, 0.2);
    const features = extractFeatures(window);
    expect(features.logMel.length).toBe(64);
    expect(features.bands.length).toBe(7);
    expect(features.rmsDbfs).toBeLessThan(0);
    expect(features.rmsDbfs).toBeGreaterThan(-60);
    expect(Number.isFinite(features.spectralCentroidHz)).toBe(true);
  });

  it('flattens to a vector of stable length', () => {
    const a = featureVector(extractFeatures(pinkNoise({ seconds: 1 }, 0.2)));
    const b = featureVector(extractFeatures(tone(440, { seconds: 1 }, 0.2)));
    expect(a.length).toBe(64 + 7 + 6);
    expect(b.length).toBe(a.length);
    expect(a.every((value) => Number.isFinite(value))).toBe(true);
  });
});
