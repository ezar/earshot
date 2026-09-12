import { fileURLToPath } from 'node:url';

import { defineConfig } from 'vite';

/**
 * The device benchmark.
 *
 * `worker.format: 'iife'` for the same reason every consumer needs it: the
 * engine worker is a classic worker because MediaPipe's WASM loader calls
 * `importScripts`. And because Vite serves workers as ES modules in dev
 * whatever this says, the benchmark is only ever built and previewed, never
 * served from the dev server.
 *
 * `assetsInlineLimit: 0` keeps the capture worklet a real file, matching what a
 * cautious consumer ships.
 */
export default defineConfig({
  root: fileURLToPath(new URL('.', import.meta.url)),
  // The models are staged here by `pnpm eval:fetch --models`.
  publicDir: fileURLToPath(new URL('../eval/public', import.meta.url)),
  // Relative, so the built folder works from any path: a LAN address, a
  // subdirectory, GitHub Pages, a phone opening a file over a tunnel.
  base: './',
  worker: { format: 'iife' },
  build: {
    outDir: fileURLToPath(new URL('../../.bench-out', import.meta.url)),
    emptyOutDir: true,
    assetsInlineLimit: 0,
    target: 'es2022',
  },
  preview: { port: 4173, host: true },
});
