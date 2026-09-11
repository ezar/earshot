/**
 * A minimal developer playground: open the microphone, run the engine and show
 * what comes out. It is not a product UI — it exists so that capture, model
 * loading and the worker protocol can be exercised in a real browser during
 * development.
 */

import { createCapture, createEngine, type Capture, type Engine } from '../src/index.js';
import workletUrl from '../src/capture/capture-worklet.js?url';
import workerUrl from '../src/worker/engine-worker.ts?worker&url';

const el = <T extends HTMLElement>(id: string): T => {
  const found = document.getElementById(id);
  if (found === null) throw new Error(`playground: missing #${id}`);
  return found as T;
};

const status = el<HTMLParagraphElement>('status');
const startButton = el<HTMLButtonElement>('start');
const stopButton = el<HTMLButtonElement>('stop');

let capture: Capture | null = null;
let engine: Engine | null = null;

startButton.addEventListener('click', () => {
  void start();
});
stopButton.addEventListener('click', () => {
  void stop();
});

async function start(): Promise<void> {
  startButton.disabled = true;
  status.textContent = 'Opening the microphone…';
  try {
    capture = await createCapture({ workletUrl });
    el<HTMLPreElement>('constraints').textContent = JSON.stringify(capture.appliedConstraints, null, 2);

    status.textContent = 'Loading the models…';
    engine = await createEngine({
      workerUrl,
      models: {
        wasmBaseUrl: el<HTMLInputElement>('wasm').value,
        classifierUrl: el<HTMLInputElement>('classifier').value,
        embedderUrl: el<HTMLInputElement>('embedder').value,
      },
    });

    engine.onWindow(render);
    capture.onChunk((samples) => {
      void engine?.push(samples);
    });

    status.textContent = `Running. Classifier: ${engine.hasClassifier}. Embedder: ${engine.hasEmbedder}.`;
    stopButton.disabled = false;
  } catch (error) {
    status.textContent = `Failed: ${error instanceof Error ? error.message : String(error)}`;
    startButton.disabled = false;
    await stop();
  }
}

async function stop(): Promise<void> {
  await capture?.stop();
  await engine?.close();
  capture = null;
  engine = null;
  startButton.disabled = false;
  stopButton.disabled = true;
  status.textContent = 'Stopped.';
}

function render(window: Parameters<Parameters<Engine['onWindow']>[0]>[0]): void {
  // -70 dBFS is the bottom of the meter; 0 dBFS is full.
  const fraction = Math.max(0, Math.min(1, (window.rmsDbfs + 70) / 70));
  el<HTMLDivElement>('level').style.width = `${(fraction * 100).toFixed(1)}%`;
  el<HTMLParagraphElement>('levelText').textContent = `${window.rmsDbfs.toFixed(1)} dBFS`;

  el<HTMLTableSectionElement>('classes').innerHTML = window.classes
    .slice(0, 6)
    .map((entry) => `<tr><td>${escapeHtml(entry.label)}</td><td class="n">${entry.score.toFixed(3)}</td></tr>`)
    .join('');

  const peak = window.features.peaks[0];
  const rows: [string, string][] = [
    ['spectral flatness', window.features.spectralFlatness.toFixed(3)],
    ['spectral centroid', `${window.features.spectralCentroidHz.toFixed(0)} Hz`],
    ['spectral flux', window.features.spectralFlux.toFixed(3)],
    ['onsets in window', String(window.features.onsets.length)],
    ['onset periodicity', window.features.onsetPeriodicity.toFixed(3)],
    ['AM rate', `${window.features.amplitudeModulationHz.toFixed(1)} Hz`],
    ['AM depth', window.features.amplitudeModulationDepth.toFixed(3)],
    ['strongest peak', peak === undefined ? '—' : `${peak.frequencyHz.toFixed(0)} Hz (+${peak.prominenceDb.toFixed(1)} dB)`],
    ['embedding dimensions', String(window.embedding.length)],
  ];
  el<HTMLTableSectionElement>('features').innerHTML = rows
    .map(([name, value]) => `<tr><td>${name}</td><td class="n">${escapeHtml(value)}</td></tr>`)
    .join('');
}

function escapeHtml(text: string): string {
  return text.replace(/[&<>"]/g, (character) => `&#${character.charCodeAt(0)};`);
}
