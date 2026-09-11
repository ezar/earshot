import { describe, expect, it } from 'vitest';
import * as earshot from '../src/index.js';

describe('public API', () => {
  it('exports everything the README uses', () => {
    for (const name of [
      'createCapture', 'createEngine', 'createGuards', 'learnProfile', 'calibrate',
      'scoreCheck', 'scoreWindow', 'describeDifference', 'createStreamScorer',
      'createEventDetector', 'trackPitch', 'createKnnClassifier', 'crossValidate',
      'quantize', 'dequantize', 'SAMPLE_RATE_HZ', 'WINDOW_SECONDS', 'HOP_SECONDS', 'MEL_BANDS',
    ]) {
      expect(earshot, `missing export: ${name}`).toHaveProperty(name);
    }
  });

  it('pins the documented framing constants', () => {
    expect(earshot.SAMPLE_RATE_HZ).toBe(16000);
    expect(earshot.WINDOW_SECONDS).toBeCloseTo(0.975, 6);
    expect(earshot.HOP_SECONDS).toBeCloseTo(0.4875, 6);
    expect(earshot.WINDOW_SAMPLES).toBe(15600);
    expect(earshot.HOP_SAMPLES).toBe(7800);
    expect(earshot.MEL_BANDS).toBe(64);
    expect(earshot.PITCH_MIN_HZ).toBe(80);
    expect(earshot.PITCH_MAX_HZ).toBe(1200);
  });
});
