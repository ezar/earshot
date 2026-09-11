#!/usr/bin/env node
/**
 * Browser smoke test for the AudioWorklet processor.
 *
 * The unit tests cannot reach this code: `process` runs on the audio thread,
 * inside a global scope that only a real browser provides. This script loads
 * `src/capture/capture-worklet.js` into headless Chromium both ways a bundler
 * can deliver it — as a served asset and as an inlined `data:` URL, which is
 * what Vite produces for a file under its `assetsInlineLimit` — drives it with
 * a live oscillator, and checks that chunks arrive at the requested size.
 *
 *   pnpm add -D playwright && npx playwright install chromium
 *   pnpm smoke
 *
 * Set PLAYWRIGHT_CHROMIUM_PATH to use a Chromium that Playwright did not install.
 */

import { readFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const workletSource = readFileSync(join(root, 'src', 'capture', 'capture-worklet.js'), 'utf8');
const dataUrl = `data:text/javascript;base64,${Buffer.from(workletSource).toString('base64')}`;
const PORT = 8099;
const CHUNK_SAMPLES = 512;

let chromium;
try {
  ({ chromium } = await import('playwright'));
} catch {
  console.error('playwright is not installed. Run: pnpm add -D playwright && npx playwright install chromium');
  process.exit(1);
}

const server = createServer((request, response) => {
  if (request.url === '/capture-worklet.js') {
    response.writeHead(200, { 'content-type': 'text/javascript' });
    response.end(workletSource);
    return;
  }
  response.writeHead(200, { 'content-type': 'text/html' });
  response.end('<!doctype html><meta charset="utf-8"><title>earshot smoke</title>');
});
await new Promise((resolve) => server.listen(PORT, resolve));

const launchOptions = { args: ['--autoplay-policy=no-user-gesture-required'] };
const executablePath = process.env['PLAYWRIGHT_CHROMIUM_PATH'];
if (executablePath !== undefined) launchOptions.executablePath = executablePath;

const browser = await chromium.launch(launchOptions);
const page = await browser.newPage();
page.on('pageerror', (error) => console.error(`  page error: ${error.message}`));
await page.goto(`http://localhost:${PORT}/`);

const results = await page.evaluate(
  async ([inlineUrl, chunkSamples]) => {
    const out = {};
    for (const [name, url] of [['served asset', '/capture-worklet.js'], ['inlined data URL', inlineUrl]]) {
      const context = new AudioContext({ sampleRate: 48000 });
      try {
        await context.audioWorklet.addModule(url);
        const node = new AudioWorkletNode(context, 'earshot-capture', {
          numberOfInputs: 1,
          numberOfOutputs: 0,
          channelCount: 1,
          channelCountMode: 'explicit',
          processorOptions: { chunkSamples },
        });
        const chunks = await new Promise((resolve) => {
          const sizes = [];
          let ready = false;
          node.port.onmessage = (event) => {
            if (event.data.type === 'ready') ready = true;
            if (event.data.type !== 'chunk') return;
            sizes.push({ length: event.data.samples.length, total: event.data.totalSamples });
            if (sizes.length >= 3) resolve({ ready, sizes });
          };
          const oscillator = context.createOscillator();
          oscillator.frequency.value = 440;
          oscillator.connect(node);
          oscillator.start();
          setTimeout(() => resolve({ ready, sizes }), 3000);
        });
        out[name] = { ok: true, ...chunks };
      } catch (error) {
        out[name] = { ok: false, error: String(error) };
      }
      await context.close();
    }
    return out;
  },
  [dataUrl, CHUNK_SAMPLES],
);

await browser.close();
server.close();

let failed = false;
for (const [name, result] of Object.entries(results)) {
  if (!result.ok) {
    console.error(`FAIL ${name}: ${result.error}`);
    failed = true;
    continue;
  }
  const wrongSize = result.sizes.filter((chunk) => chunk.length !== CHUNK_SAMPLES);
  const monotonic = result.sizes.every(
    (chunk, i) => chunk.total === CHUNK_SAMPLES * (i + 1),
  );
  if (!result.ready || result.sizes.length < 3 || wrongSize.length > 0 || !monotonic) {
    console.error(`FAIL ${name}: ${JSON.stringify(result)}`);
    failed = true;
    continue;
  }
  console.log(`ok   ${name}: ${result.sizes.length} chunks of ${CHUNK_SAMPLES} samples, counter monotonic`);
}

process.exit(failed ? 1 : 0);
