/**
 * The device benchmark.
 *
 * The spec's performance target names a 2022 mid-range Android phone, and no
 * amount of measuring on a desktop container settles whether it is met. This
 * page runs the real engine, with the real models, on whatever device opens it.
 *
 * It uses no microphone. Audio is generated in the page, so a run is
 * reproducible and needs no permission prompt — which also means the page works
 * over plain HTTP on a local network, with no hosting or certificate needed.
 */

import {
  createEngine,
  HOP_SECONDS,
  SAMPLE_RATE_HZ,
  WINDOW_SECONDS,
  type Engine,
  type GuardConfig,
} from '../../src/index.js';
import workerUrl from '../../src/worker/engine-worker.ts?worker&url';

/** Seconds of audio per pass. Long enough that fixed costs do not distort the mean. */
const SECONDS_PER_PASS = 30;

/** Passes per scenario. The first is discarded as warm-up. */
const PASSES = 4;

/** The target the spec sets, in milliseconds per window. */
const TARGET_MS = 30;

const MODELS = {
  wasmBaseUrl: './models/wasm',
  classifierUrl: './models/yamnet_classifier.tflite',
  embedderUrl: './models/yamnet_embedder.tflite',
} as const;

interface ScenarioResult {
  readonly name: string;
  readonly note: string;
  readonly medianMs: number;
  readonly passesMs: readonly number[];
  readonly windows: number;
  readonly embedded: number;
}

const el = <T extends HTMLElement>(id: string): T => {
  const found = document.getElementById(id);
  if (found === null) throw new Error(`bench: missing #${id}`);
  return found as T;
};

const status = el<HTMLParagraphElement>('status');
const progress = el<HTMLDivElement>('progress');
const runButton = el<HTMLButtonElement>('run');
const copyButton = el<HTMLButtonElement>('copy');

let steps = 0;
const TOTAL_STEPS = 8;
const step = (message: string): void => {
  steps += 1;
  progress.style.width = `${Math.min(100, (steps / TOTAL_STEPS) * 100)}%`;
  status.textContent = message;
};

/** Deterministic noise, so every device measures the same signal. */
function noise(seconds: number): Float32Array {
  const out = new Float32Array(Math.round(seconds * SAMPLE_RATE_HZ));
  let state = 1;
  for (let i = 0; i < out.length; i += 1) {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    out[i] = ((state / 0x7fffffff) * 2 - 1) * 0.2;
  }
  return out;
}

function silence(seconds: number): Float32Array {
  return new Float32Array(Math.round(seconds * SAMPLE_RATE_HZ));
}

const median = (values: readonly number[]): number => {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)] ?? 0;
};

/**
 * Run one scenario: build an engine, push the same audio several times, and
 * report the median per-window cost of the passes after the warm-up.
 */
async function scenario(
  name: string,
  note: string,
  audio: Float32Array,
  options: { readonly embedder: boolean; readonly guards?: GuardConfig },
): Promise<ScenarioResult> {
  const models = options.embedder
    ? MODELS
    : { wasmBaseUrl: MODELS.wasmBaseUrl, classifierUrl: MODELS.classifierUrl };
  const engine: Engine = await createEngine({
    workerUrl,
    models,
    ...(options.guards === undefined ? {} : { guards: options.guards }),
  });

  const passesMs: number[] = [];
  let windows = 0;
  let embedded = 0;
  try {
    for (let pass = 0; pass < PASSES; pass += 1) {
      step(`${name}: pass ${pass + 1} of ${PASSES}…`);
      await engine.reset();
      // push transfers the buffer, so each pass needs its own copy.
      const started = performance.now();
      const results = await engine.push(audio.slice());
      const elapsed = performance.now() - started;
      windows = results.length;
      embedded = results.filter((w) => w.embedding.length > 0).length;
      passesMs.push(elapsed / Math.max(1, results.length));
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  } finally {
    await engine.close();
  }

  const measured = passesMs.slice(1);
  return { name, note, medianMs: median(measured), passesMs, windows, embedded };
}

function describeDevice(): Record<string, string | number> {
  const nav = navigator as Navigator & { deviceMemory?: number };
  return {
    userAgent: navigator.userAgent,
    cores: navigator.hardwareConcurrency ?? 0,
    memoryGb: nav.deviceMemory ?? 0,
    screen: `${window.screen.width}x${window.screen.height}@${window.devicePixelRatio}`,
    secureContext: String(window.isSecureContext),
    origin: window.location.origin,
  };
}

function render(results: readonly ScenarioResult[], loadMs: number): void {
  const full = results.find((r) => r.name === 'Classifier + embedder');
  const headline = full?.medianMs ?? 0;
  const verdict = el<HTMLDivElement>('verdict');
  const ratio = headline / TARGET_MS;
  verdict.textContent =
    headline <= TARGET_MS
      ? `${headline.toFixed(1)} ms per window — within the ${TARGET_MS} ms budget (${(ratio * 100).toFixed(0)} % of it).`
      : `${headline.toFixed(1)} ms per window — over the ${TARGET_MS} ms budget by ${((ratio - 1) * 100).toFixed(0)} %.`;

  const rows: string[] = [
    `<tr><th>Scenario</th><th class="n">ms/window</th><th class="n">vs 30 ms</th></tr>`,
  ];
  for (const r of results) {
    rows.push(
      `<tr><td>${r.name}<br><span class="muted">${r.note}</span></td>` +
        `<td class="n">${r.medianMs.toFixed(1)}</td>` +
        `<td class="n">${((r.medianMs / TARGET_MS) * 100).toFixed(0)} %</td></tr>`,
    );
  }
  rows.push(`<tr><td>Model load</td><td class="n">${loadMs.toFixed(0)} ms</td><td class="n">—</td></tr>`);
  const table = el<HTMLTableElement>('results');
  (table.querySelector('tbody') as HTMLTableSectionElement).innerHTML = rows.join('');
  table.hidden = false;

  const payload = {
    earshotBenchmark: 1,
    at: new Date().toISOString(),
    targetMsPerWindow: TARGET_MS,
    windowSeconds: WINDOW_SECONDS,
    hopSeconds: HOP_SECONDS,
    sampleRateHz: SAMPLE_RATE_HZ,
    secondsPerPass: SECONDS_PER_PASS,
    passes: PASSES,
    modelLoadMs: Math.round(loadMs),
    device: describeDevice(),
    results: results.map((r) => ({
      name: r.name,
      medianMs: Number(r.medianMs.toFixed(2)),
      passesMs: r.passesMs.map((v) => Number(v.toFixed(2))),
      windows: r.windows,
      embedded: r.embedded,
    })),
  };
  const json = JSON.stringify(payload, null, 2);
  el<HTMLPreElement>('json').textContent = json;
  el<HTMLDetailsElement>('detail').hidden = false;
  copyButton.hidden = false;
  copyButton.onclick = () => {
    void navigator.clipboard.writeText(json).then(
      () => { copyButton.textContent = 'Copied'; },
      () => { copyButton.textContent = 'Copy failed — select the text below'; },
    );
  };
}

runButton.addEventListener('click', () => {
  void (async () => {
    runButton.disabled = true;
    steps = 0;
    try {
      step('Loading the models…');
      const loadStarted = performance.now();
      const warm = await createEngine({ workerUrl, models: MODELS });
      const loadMs = performance.now() - loadStarted;
      await warm.close();

      const results: ScenarioResult[] = [];
      results.push(
        await scenario('Classifier + embedder', 'the full engine, every window embedded',
          noise(SECONDS_PER_PASS), { embedder: true }),
      );
      results.push(
        await scenario('Classifier only', 'no embedder configured',
          noise(SECONDS_PER_PASS), { embedder: false }),
      );
      results.push(
        await scenario('Guards on, silence', 'embedder skipped for rejected windows',
          silence(SECONDS_PER_PASS), { embedder: true, guards: {} }),
      );

      progress.style.width = '100%';
      status.textContent = 'Done.';
      render(results, loadMs);
    } catch (error) {
      status.textContent = `Failed: ${error instanceof Error ? error.message : String(error)}`;
    } finally {
      runButton.disabled = false;
      runButton.textContent = 'Run again';
    }
  })();
});
