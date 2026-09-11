/**
 * The page Playwright drives during evaluation.
 *
 * It exposes `window.earshotEval` to the Node side. Everything runs through the
 * same engine, guards, profile and classifier a shipped app uses — the point of
 * evaluating in a real browser is that the WASM models, the Worker and the
 * AudioWorklet-shaped data path are the production ones, not a Node-side
 * reimplementation of them.
 */

import {
  createEngine,
  createGuards,
  createKnnClassifier,
  crossValidate,
  createEventDetector,
  learnProfile,
  scoreCheck,
  SAMPLE_RATE_HZ,
  type Engine,
  type WindowResult,
} from '../../src/index.js';
import workerUrl from '../../src/worker/engine-worker.ts?worker&url';

/** Model locations, injected by the Node side so nothing is hardcoded. */
export interface EvalModels {
  readonly wasmBaseUrl: string;
  readonly classifierUrl: string;
  readonly embedderUrl: string;
}

let engine: Engine | null = null;

const api = {
  /** Load the models once for the whole run. */
  async init(models: EvalModels): Promise<void> {
    engine = await createEngine({ workerUrl, models });
  },

  /** Run one buffer through the engine and return its windows. */
  async analyse(samples: number[]): Promise<WindowResult[]> {
    if (engine === null) throw new Error('eval: init() first');
    await engine.reset();
    return engine.push(Float32Array.from(samples));
  },

  /**
   * Anomaly detection: learn from the normal clips, then score every test clip.
   *
   * @returns One score per test clip, in input order.
   */
  async anomalyScores(normalClips: number[][], testClips: number[][]): Promise<number[]> {
    const guards = createGuards();
    const learning: WindowResult[] = [];
    for (const clip of normalClips) learning.push(...guards.filter(await api.analyse(clip)));
    const profile = learnProfile(learning);
    const scores: number[] = [];
    for (const clip of testClips) scores.push(scoreCheck(profile, await api.analyse(clip)).score);
    return scores;
  },

  /**
   * Vocalization detection: how many of the clips produce at least one event.
   */
  async detectionRate(clips: number[][], triggerClasses: string[]): Promise<number[]> {
    const found: number[] = [];
    for (const clip of clips) {
      const detector = createEventDetector({ triggerClasses, triggerScore: 0.1 });
      const samples = Float32Array.from(clip);
      const windows = await api.analyse(clip);
      let events = 0;
      for (const window of windows) {
        const offset = Math.round(window.t * SAMPLE_RATE_HZ);
        events += detector.push(window, samples.slice(offset, offset + Math.round(0.975 * SAMPLE_RATE_HZ))).length;
      }
      events += detector.flush().length;
      found.push(events);
    }
    return found;
  },

  /**
   * Identity: embed every labelled clip, then k-fold self-test the kNN.
   */
  async identityAccuracy(clips: { label: string; samples: number[] }[]): Promise<number> {
    const classifier = createKnnClassifier<string>();
    for (const clip of clips) {
      const windows = await api.analyse(clip.samples);
      const loudest = windows.reduce<WindowResult | null>(
        (best, window) => (best === null || window.rmsDbfs > best.rmsDbfs ? window : best),
        null,
      );
      if (loudest === null || loudest.embedding.length === 0) continue;
      classifier.add(clip.label, loudest.embedding);
    }
    return crossValidate(classifier.examples(), { folds: 5 }).accuracy;
  },
};

declare global {
  interface Window {
    earshotEval: typeof api;
  }
}

window.earshotEval = api;
document.getElementById('state')!.textContent = 'ready';
