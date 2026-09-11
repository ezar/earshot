import { fileURLToPath } from 'node:url';

import { defineConfig } from 'vite';

/**
 * The playground loads earshot straight from `src/`, exactly as a consuming app
 * does through the `github:` dependency, so anything that breaks in a real Vite
 * build breaks here first.
 *
 * `pnpm playground` builds and previews rather than running the dev server: Vite
 * serves workers as ES modules in dev whatever `worker.format` says, and
 * MediaPipe cannot load in a module worker (see
 * `docs/decisions/0007-mediapipe-cannot-load-in-a-module-worker.md`). Use
 * `pnpm playground:dsp` for the dev server when you only need capture, levels
 * and features and no model.
 */
export default defineConfig({
  // Relative to this file, so the config works from any working directory.
  root: fileURLToPath(new URL('.', import.meta.url)),
  server: { port: 5174, open: true },
  preview: { port: 5174, open: true },
  // Classic workers, not ESM: MediaPipe's WASM loader calls `importScripts`.
  worker: { format: 'iife' },
  build: {
    // Emit the capture worklet as a real file rather than inlining it.
    // At ~2.4 kB it sits under Vite's default 4 kB threshold, so a build would
    // otherwise hand `audioWorklet.addModule` a `data:` URL. Chromium accepts
    // one — `pnpm smoke` checks both paths — but support is not universal, and
    // the playground exists to catch what breaks in a real browser first. This
    // is the same setting the README recommends to consumers who are unsure of
    // their target.
    assetsInlineLimit: 0,
  },
});
