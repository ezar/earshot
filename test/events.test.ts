import { describe, expect, it } from 'vitest';

import { SAMPLE_RATE_HZ } from '../src/constants.js';
import { countSyllables, segmentBuffer } from '../src/events/segment.js';
import { createEventDetector, eventHumanScore } from '../src/events/detector.js';
import { amplitudeEnvelope } from '../src/dsp/level.js';
import {
  concat,
  harmonicCall,
  humanImitation,
  pinkNoise,
  silence,
  whiteNoise,
} from '../fixtures/synthetic/index.js';
import { windowsFrom } from './helpers.js';

const gap = (seconds: number): Float32Array => silence({ seconds });

describe('segmentBuffer', () => {
  it('finds one call in a quiet buffer', () => {
    const audio = concat([gap(0.3), harmonicCall({ seconds: 0.4, startHz: 500 }), gap(0.3)]);
    const segments = segmentBuffer(audio);
    expect(segments.length).toBe(1);
    expect(segments[0]?.start).toBeGreaterThan(0.2);
    expect(segments[0]?.start).toBeLessThan(0.45);
    expect((segments[0]?.end ?? 0) - (segments[0]?.start ?? 0)).toBeGreaterThan(0.3);
  });

  it('finds three separate calls', () => {
    const call = (): Float32Array => harmonicCall({ seconds: 0.3, startHz: 600 });
    const audio = concat([gap(0.2), call(), gap(0.4), call(), gap(0.4), call(), gap(0.2)]);
    expect(segmentBuffer(audio).length).toBe(3);
  });

  it('bridges a very short gap into one call', () => {
    const call = (): Float32Array => harmonicCall({ seconds: 0.2, startHz: 600 });
    const audio = concat([gap(0.3), call(), gap(0.03), call(), gap(0.3)]);
    const segments = segmentBuffer(audio);
    expect(segments.length).toBe(1);
    expect(segments[0]?.syllables).toBeGreaterThanOrEqual(2);
  });

  it('finds nothing in silence', () => {
    expect(segmentBuffer(silence({ seconds: 2 })).length).toBe(0);
  });

  it('finds nothing in steady noise, which has no envelope structure', () => {
    expect(segmentBuffer(pinkNoise({ seconds: 2, seed: 4 }, 0.1)).length).toBe(0);
  });

  it('drops a segment shorter than the minimum duration', () => {
    const audio = concat([gap(0.3), harmonicCall({ seconds: 0.02, startHz: 600, fadeSeconds: 0.005 }), gap(0.3)]);
    expect(segmentBuffer(audio, { minDurationSeconds: 0.1 }).length).toBe(0);
  });

  it('truncates a segment longer than the maximum duration', () => {
    const audio = concat([gap(0.3), harmonicCall({ seconds: 3, startHz: 600 }), gap(0.3)]);
    const segments = segmentBuffer(audio, { maxDurationSeconds: 1 });
    expect((segments[0]?.end ?? 0) - (segments[0]?.start ?? 0)).toBeLessThanOrEqual(1.001);
  });
});

describe('countSyllables', () => {
  it('counts one lobe for a smooth call', () => {
    const envelope = amplitudeEnvelope(harmonicCall({ seconds: 0.4, startHz: 500 }));
    expect(countSyllables(envelope.levelsDbfs)).toBe(1);
  });

  it('counts two lobes for a double call', () => {
    const call = (): Float32Array => harmonicCall({ seconds: 0.2, startHz: 500 });
    const envelope = amplitudeEnvelope(concat([call(), gap(0.05), call()]));
    expect(countSyllables(envelope.levelsDbfs)).toBe(2);
  });

  it('counts nothing in an empty envelope', () => {
    expect(countSyllables(new Float32Array(0))).toBe(0);
  });
});

describe('createEventDetector', () => {
  const cat = [{ label: 'Cat', score: 0.7 }];
  const quiet = [{ label: 'Silence', score: 0.9 }];

  const detect = (
    audio: Float32Array,
    classes: readonly { label: string; score: number }[],
    config: Parameters<typeof createEventDetector>[0] = { triggerClasses: ['Cat'] },
  ) => {
    const detector = createEventDetector(config);
    const events = [];
    for (const window of windowsFrom(audio, undefined, classes)) {
      const samples = audio.slice(
        Math.round(window.t * SAMPLE_RATE_HZ),
        Math.round(window.t * SAMPLE_RATE_HZ) + Math.round(0.975 * SAMPLE_RATE_HZ),
      );
      events.push(...detector.push(window, samples));
    }
    events.push(...detector.flush());
    return events;
  };

  it('emits one event for one call', () => {
    const audio = concat([gap(0.5), harmonicCall({ seconds: 0.5, startHz: 600 }), gap(0.5)]);
    const events = detect(audio, cat);
    expect(events.length).toBe(1);
    expect(events[0]?.type).toBe('Cat');
    expect(events[0]?.confidence).toBeCloseTo(0.7, 5);
    expect(events[0]?.duration).toBeGreaterThan(0.3);
  });

  it('measures the pitch of the call it found', () => {
    const audio = concat([gap(0.5), harmonicCall({ seconds: 0.5, startHz: 600 }), gap(0.5)]);
    const event = detect(audio, cat)[0];
    expect(event?.pitch.medianF0Hz).toBeGreaterThan(550);
    expect(event?.pitch.medianF0Hz).toBeLessThan(650);
    expect(event?.pitch.voicedFraction).toBeGreaterThan(0.5);
  });

  it('reports a rising contour', () => {
    const audio = concat([
      gap(0.5),
      harmonicCall({ seconds: 0.5, startHz: 300, endHz: 700, shape: 'rising' }),
      gap(0.5),
    ]);
    const event = detect(audio, cat)[0];
    expect(event?.pitch.contourSlopeSemitonesPerSecond).toBeGreaterThan(3);
  });

  it('emits nothing when the trigger class is absent', () => {
    const audio = concat([gap(0.5), harmonicCall({ seconds: 0.5, startHz: 600 }), gap(0.5)]);
    expect(detect(audio, quiet).length).toBe(0);
  });

  it('emits nothing for silence even when the class fires', () => {
    expect(detect(silence({ seconds: 3 }), cat).length).toBe(0);
  });

  it('does not emit the same call twice from overlapping windows', () => {
    const audio = concat([gap(0.4), harmonicCall({ seconds: 0.6, startHz: 550 }), gap(0.4)]);
    expect(detect(audio, cat).length).toBe(1);
  });

  it('separates two calls that are far enough apart', () => {
    const call = (): Float32Array => harmonicCall({ seconds: 0.3, startHz: 600 });
    const audio = concat([gap(0.5), call(), gap(1.5), call(), gap(0.5)]);
    expect(detect(audio, cat).length).toBe(2);
  });

  it('flags a human imitation through the voice guard', () => {
    const audio = concat([gap(0.5), humanImitation({ seconds: 0.5, startHz: 600, seed: 2 }), gap(0.5)]);
    const events = detect(audio, [
      { label: 'Cat', score: 0.5 },
      { label: 'Speech', score: 0.6 },
    ]);
    expect(events[0]?.possibleHuman).toBe(true);
  });

  it('does not flag a plain call as human', () => {
    const audio = concat([gap(0.5), harmonicCall({ seconds: 0.5, startHz: 600 }), gap(0.5)]);
    expect(detect(audio, cat)[0]?.possibleHuman).toBe(false);
  });

  it('ignores a trigger below the score threshold', () => {
    const audio = concat([gap(0.5), harmonicCall({ seconds: 0.5, startHz: 600 }), gap(0.5)]);
    expect(detect(audio, [{ label: 'Cat', score: 0.05 }]).length).toBe(0);
  });

  it('requires at least one trigger class', () => {
    expect(() => createEventDetector({ triggerClasses: [] })).toThrow(/trigger class/);
  });

  it('resets without emitting', () => {
    const detector = createEventDetector({ triggerClasses: ['Cat'] });
    const audio = concat([gap(0.3), harmonicCall({ seconds: 0.5, startHz: 600 })]);
    for (const window of windowsFrom(audio, undefined, cat)) detector.push(window, audio.slice(0, 15600));
    detector.reset();
    expect(detector.flush().length).toBe(0);
  });

  it('carries features computed from the event, not the window', () => {
    const audio = concat([gap(0.5), harmonicCall({ seconds: 0.4, startHz: 800 }), gap(0.5)]);
    const event = detect(audio, cat)[0];
    expect(event?.features.peaks.length ?? 0).toBeGreaterThan(0);
    expect(event?.features.rmsDbfs ?? -120).toBeGreaterThan(-40);
    expect(event?.peakDbfs ?? -120).toBeGreaterThan(-30);
  });

  it('is not fooled into an event by broadband noise alone', () => {
    expect(detect(whiteNoise({ seconds: 3, seed: 9 }, 0.1), cat).length).toBe(0);
  });

  it('scores how human a detected event looks', () => {
    const audio = concat([gap(0.5), harmonicCall({ seconds: 0.5, startHz: 600 }), gap(0.5)]);
    const human = detect(audio, [
      { label: 'Cat', score: 0.5 },
      { label: 'Speech', score: 0.62 },
    ])[0];
    expect(human).toBeDefined();
    expect(eventHumanScore(human as NonNullable<typeof human>)).toBeCloseTo(0.62, 5);

    const plain = detect(audio, cat)[0];
    expect(eventHumanScore(plain as NonNullable<typeof plain>)).toBe(0);
  });
});
