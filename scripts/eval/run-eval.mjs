#!/usr/bin/env node
/**
 * Run the evaluation suite.
 *
 * Playwright launches headless Chromium on a Vite dev server serving
 * `eval-page.ts`, which loads earshot and the real YAMNet task files. Dataset
 * WAV files are decoded in Node and streamed into the same pipeline a shipped
 * app uses, and the results are written to `.eval-out/` and summarised for
 * `docs/eval-results.md`.
 *
 *   pnpm eval:fetch --all      # download datasets (see docs/datasets.md)
 *   pnpm eval:run              # run every suite that has data
 *   pnpm eval:run -- catmeows  # run one suite
 *
 * Missing datasets are skipped with a message rather than failing the run, so
 * the harness is useful before every licence has been accepted.
 */

import { existsSync } from 'node:fs';
import { mkdir, readdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { decodeWavToMono16k } from './wav.mjs';

const root = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const dataDir = join(root, '.eval-data');
const outDir = join(root, '.eval-out');

const MODELS = {
  wasmBaseUrl: process.env['EARSHOT_WASM_BASE_URL'] ?? '/models/wasm',
  classifierUrl: process.env['EARSHOT_CLASSIFIER_URL'] ?? '/models/yamnet_classifier.tflite',
  embedderUrl: process.env['EARSHOT_EMBEDDER_URL'] ?? '/models/yamnet_embedder.tflite',
};

const requested = process.argv.slice(2).filter((argument) => !argument.startsWith('-'));
const suites = ['mimii', 'toyadmos', 'catmeows'].filter(
  (name) => requested.length === 0 || requested.includes(name),
);

const available = suites.filter((name) => existsSync(join(dataDir, name)));
const missing = suites.filter((name) => !available.includes(name));
for (const name of missing) {
  console.log(`skipping ${name}: no data in .eval-data/${name} (run pnpm eval:fetch ${name})`);
}
if (available.length === 0) {
  console.log('\nNothing to evaluate. See docs/datasets.md.');
  process.exit(0);
}

const { chromium } = await importOrExit('playwright');
const { createServer } = await importOrExit('vite');

const server = await createServer({
  root,
  server: { port: 5175 },
  worker: { format: 'es' },
  publicDir: process.env['EARSHOT_MODELS_DIR'] ?? join(root, 'scripts', 'eval', 'public'),
});
await server.listen();

const browser = await chromium.launch();
const page = await browser.newPage();
page.on('console', (message) => console.log(`  [page] ${message.text()}`));
await page.goto('http://localhost:5175/scripts/eval/eval-page.html');
await page.waitForFunction(() => document.getElementById('state')?.textContent === 'ready');
await page.evaluate((models) => window.earshotEval.init(models), MODELS);

const results = {};
for (const suite of available) {
  console.log(`\nrunning ${suite}`);
  results[suite] = await runSuite(suite, page);
}

await browser.close();
await server.close();

await mkdir(outDir, { recursive: true });
const stamp = new Date().toISOString().replace(/[:.]/g, '-');
const reportPath = join(outDir, `eval-${stamp}.json`);
await writeFile(reportPath, `${JSON.stringify(results, null, 2)}\n`);
console.log(`\nwrote ${reportPath}`);
console.log(renderMarkdown(results));

async function runSuite(suite, page) {
  if (suite === 'catmeows') return runCatMeows(page);
  return runAnomaly(suite, page);
}

/**
 * Anomaly detection: for each machine, learn from `normal` and score
 * `normal` + `abnormal`, then report the ROC AUC.
 */
async function runAnomaly(suite, page) {
  const machines = await readdir(join(dataDir, suite), { withFileTypes: true });
  const out = {};
  for (const machine of machines.filter((entry) => entry.isDirectory())) {
    const base = join(dataDir, suite, machine.name);
    const normal = await loadClips(join(base, 'normal'), 60);
    const abnormal = await loadClips(join(base, 'abnormal'), 60);
    if (normal.length < 20 || abnormal.length < 5) {
      console.log(`  ${machine.name}: not enough clips (${normal.length} normal, ${abnormal.length} abnormal)`);
      continue;
    }
    const split = Math.floor(normal.length / 2);
    const learn = normal.slice(0, split);
    const test = [...normal.slice(split), ...abnormal];
    const labels = [...normal.slice(split).map(() => 0), ...abnormal.map(() => 1)];

    const scores = await page.evaluate(
      ([a, b]) => window.earshotEval.anomalyScores(a, b),
      [learn.map(Array.from), test.map(Array.from)],
    );
    const auc = rocAuc(labels, scores);
    out[machine.name] = { auc, learnClips: learn.length, testClips: test.length };
    console.log(`  ${machine.name}: AUC ${auc.toFixed(3)} over ${test.length} clips`);
  }
  return out;
}

/** Detection recall, plus pair-identity accuracy over cats with enough clips. */
async function runCatMeows(page) {
  const clips = await loadLabelledClips(join(dataDir, 'catmeows'));
  if (clips.length === 0) {
    console.log('  no clips found');
    return {};
  }
  const counts = await page.evaluate(
    ([samples, triggers]) => window.earshotEval.detectionRate(samples, triggers),
    [clips.map((clip) => Array.from(clip.samples)), ['Cat', 'Meow', 'Caterwaul', 'Domestic animals, pets']],
  );
  const recall = counts.filter((count) => count > 0).length / counts.length;
  console.log(`  detection recall: ${(recall * 100).toFixed(1)} % over ${clips.length} clips`);

  const byCat = new Map();
  for (const clip of clips) byCat.set(clip.label, (byCat.get(clip.label) ?? 0) + 1);
  const eligible = [...byCat.entries()].filter(([, count]) => count >= 10).map(([label]) => label);

  const pairAccuracies = [];
  for (let i = 0; i < eligible.length; i += 1) {
    for (let j = i + 1; j < eligible.length; j += 1) {
      const pair = clips
        .filter((clip) => clip.label === eligible[i] || clip.label === eligible[j])
        .slice(0, 20)
        .map((clip) => ({ label: clip.label, samples: Array.from(clip.samples) }));
      pairAccuracies.push(await page.evaluate((p) => window.earshotEval.identityAccuracy(p), pair));
    }
  }
  const identity = pairAccuracies.length === 0 ? 0 : pairAccuracies.reduce((a, b) => a + b, 0) / pairAccuracies.length;
  console.log(`  pair identity: ${(identity * 100).toFixed(1)} % over ${pairAccuracies.length} pairs`);
  return { detectionRecall: recall, pairIdentity: identity, clips: clips.length, pairs: pairAccuracies.length };
}

async function loadClips(directory, limit) {
  if (!existsSync(directory)) return [];
  const names = (await readdir(directory)).filter((name) => name.toLowerCase().endsWith('.wav')).slice(0, limit);
  const clips = [];
  for (const name of names) clips.push(decodeWavToMono16k(await readFile(join(directory, name))));
  return clips;
}

/** CatMeows encodes the cat id in the filename: `<context>_<id>_<breed>...wav`. */
async function loadLabelledClips(directory) {
  if (!existsSync(directory)) return [];
  const clips = [];
  for (const entry of await readdir(directory, { withFileTypes: true, recursive: true })) {
    if (!entry.isFile() || !entry.name.toLowerCase().endsWith('.wav')) continue;
    const parts = entry.name.split('_');
    if (parts.length < 2) continue;
    const path = join(entry.parentPath ?? directory, entry.name);
    clips.push({ label: parts[1], samples: decodeWavToMono16k(await readFile(path)) });
  }
  return clips;
}

/** Area under the ROC curve, computed from rank statistics. */
function rocAuc(labels, scores) {
  const positives = labels.filter((label) => label === 1).length;
  const negatives = labels.length - positives;
  if (positives === 0 || negatives === 0) return 0;
  const order = scores.map((score, i) => ({ score, label: labels[i] })).sort((a, b) => a.score - b.score);
  let rankSum = 0;
  for (let i = 0; i < order.length; i += 1) {
    if (order[i].label === 1) rankSum += i + 1;
  }
  return (rankSum - (positives * (positives + 1)) / 2) / (positives * negatives);
}

function renderMarkdown(results) {
  const lines = ['', '| Suite | Metric | Value |', '| --- | --- | --- |'];
  for (const [suite, value] of Object.entries(results)) {
    for (const [key, entry] of Object.entries(value)) {
      if (typeof entry === 'number') lines.push(`| ${suite} | ${key} | ${entry.toFixed(3)} |`);
      else if (entry && typeof entry.auc === 'number') lines.push(`| ${suite} | ${key} AUC | ${entry.auc.toFixed(3)} |`);
    }
  }
  return lines.join('\n');
}

async function importOrExit(name) {
  try {
    return await import(name);
  } catch {
    console.error(`\n${name} is not installed. Run: pnpm add -D ${name}`);
    process.exit(1);
  }
}
