import { fileURLToPath } from 'node:url';

import { defineConfig } from 'vite';

/**
 * The playground loads earshot straight from `src/`, exactly as a consuming app
 * does through the `github:` dependency, so anything that breaks in a real Vite
 * build breaks here first.
 */
export default defineConfig({
  // Relative to this file, so the config works from any working directory.
  root: fileURLToPath(new URL('.', import.meta.url)),
  server: { port: 5174, open: true },
  worker: { format: 'es' },
});
