import { describe, expect, it } from 'vitest';

import { PITCH_MAX_HZ, PITCH_MIN_HZ, SAMPLE_RATE_HZ } from '../src/constants.js';
import { summarize, trackPitch, yin } from '../src/dsp/pitch.js';
import { applyGainDb, harmonicCall, silence, tone, whiteNoise } from '../fixtures/synthetic/index.js';

describe('yin', () => {
  it.each([100, 220, 440, 880, 1100])('recovers a %d Hz tone within 1 %%', (frequencyHz) => {
    const frame = tone(frequencyHz, { seconds: 0.1 }, 0.5);
    const estimate = yin(frame);
    expect(estimate.f0Hz).toBeGreaterThan(frequencyHz * 0.99);
    expect(estimate.f0Hz).toBeLessThan(frequencyHz * 1.01);
    expect(estimate.confidence).toBeGreaterThan(0.8);
  });

  it('recovers the fundamental of a harmonic stack, not a harmonic', () => {
    const frame = harmonicCall({ seconds: 0.1, startHz: 350, harmonics: 6, fadeSeconds: 0 });
    expect(yin(frame).f0Hz).toBeCloseTo(350, -1);
  });

  it('reports nothing voiced for white noise', () => {
    expect(yin(whiteNoise({ seconds: 0.1 }, 0.3)).f0Hz).toBe(0);
  });

  it('reports nothing for silence', () => {
    expect(yin(silence({ seconds: 0.1 })).f0Hz).toBe(0);
  });

  it('refuses frequencies outside the configured range', () => {
    expect(yin(tone(60, { seconds: 0.1 }, 0.5), { minHz: PITCH_MIN_HZ, maxHz: PITCH_MAX_HZ }).f0Hz).toBe(0);
    expect(yin(tone(2000, { seconds: 0.1 }, 0.5), { minHz: PITCH_MIN_HZ, maxHz: PITCH_MAX_HZ }).f0Hz).toBe(0);
  });

  it('is gain invariant', () => {
    const loud = tone(440, { seconds: 0.1 }, 0.8);
    expect(yin(applyGainDb(loud, -40)).f0Hz).toBeCloseTo(yin(loud).f0Hz, 3);
  });
});

describe('trackPitch', () => {
  it('tracks a steady tone across every frame', () => {
    const track = trackPitch(tone(440, { seconds: 1 }, 0.4));
    expect(track.voicedFraction).toBeGreaterThan(0.95);
    expect(track.medianF0Hz).toBeCloseTo(440, -1);
    expect(Math.abs(track.contourSlopeSemitonesPerSecond)).toBeLessThan(0.5);
  });

  it('reports a rising contour as a positive slope', () => {
    const track = trackPitch(
      harmonicCall({ seconds: 0.6, startHz: 300, endHz: 600, shape: 'rising', fadeSeconds: 0.02 }),
    );
    expect(track.contourSlopeSemitonesPerSecond).toBeGreaterThan(5);
    expect(track.maxF0Hz).toBeGreaterThan(track.minF0Hz * 1.5);
  });

  it('reports a falling contour as a negative slope', () => {
    const track = trackPitch(
      harmonicCall({ seconds: 0.6, startHz: 600, endHz: 300, shape: 'falling', fadeSeconds: 0.02 }),
    );
    expect(track.contourSlopeSemitonesPerSecond).toBeLessThan(-5);
  });

  it('marks silence as unvoiced', () => {
    const track = trackPitch(silence({ seconds: 0.5 }));
    expect(track.voicedFraction).toBe(0);
    expect(track.medianF0Hz).toBe(0);
    expect(track.frames.length).toBeGreaterThan(10);
  });

  it('produces frames on the documented 10 ms grid', () => {
    const track = trackPitch(tone(440, { seconds: 0.5 }, 0.4));
    const first = track.frames[0];
    const second = track.frames[1];
    expect(second !== undefined && first !== undefined).toBe(true);
    expect((second?.t ?? 0) - (first?.t ?? 0)).toBeCloseTo(0.01, 6);
  });

  it('honours a non-default sample rate', () => {
    const track = trackPitch(tone(440, { seconds: 0.5, sampleRateHz: 8000 }, 0.4), { sampleRateHz: 8000 });
    expect(track.medianF0Hz).toBeCloseTo(440, -1);
  });

  it('reports the working sample rate constant', () => {
    expect(SAMPLE_RATE_HZ).toBe(16000);
  });
});

describe('summarize', () => {
  it('derives the contour summaries from frames a caller built itself', () => {
    // Exported so an app that frames audio its own way can still get the
    // summaries without re-running the tracker.
    const frames = [
      { t: 0.0, f0Hz: 200, confidence: 0.9, rmsDbfs: -20 },
      { t: 0.1, f0Hz: 0, confidence: 0, rmsDbfs: -70 },
      { t: 0.2, f0Hz: 400, confidence: 0.9, rmsDbfs: -20 },
    ];
    const track = summarize(frames);
    expect(track.medianF0Hz).toBe(300);
    expect(track.minF0Hz).toBe(200);
    expect(track.maxF0Hz).toBe(400);
    expect(track.voicedFraction).toBeCloseTo(2 / 3, 6);
    // 200 Hz to 400 Hz is an octave, 12 semitones, over 0.2 s.
    expect(track.contourSlopeSemitonesPerSecond).toBeCloseTo(60, 0);
  });

  it('reports nothing voiced for an empty or silent track', () => {
    expect(summarize([]).medianF0Hz).toBe(0);
    expect(summarize([{ t: 0, f0Hz: 0, confidence: 0, rmsDbfs: -90 }]).voicedFraction).toBe(0);
  });
});
