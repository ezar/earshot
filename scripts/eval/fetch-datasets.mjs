#!/usr/bin/env node
/**
 * Download the evaluation datasets into the git-ignored `.eval-data/`.
 *
 * Datasets are large and separately licensed, so nothing is downloaded without
 * an explicit run of this script, and each dataset must be named on the command
 * line or selected with `--all`.
 *
 *   node scripts/eval/fetch-datasets.mjs --all
 *   node scripts/eval/fetch-datasets.mjs catmeows
 *
 * See docs/datasets.md for the licences. You are responsible for accepting them.
 */

import { createWriteStream } from 'node:fs';
import { mkdir, stat } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';

const root = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const dataDir = join(root, '.eval-data');

/** Each entry names the archives to fetch and where they land. */
const DATASETS = {
  mimii: {
    licence: 'CC BY-SA 4.0',
    files: [
      ['mimii/fan.zip', 'https://zenodo.org/records/3384388/files/-6_dB_fan.zip'],
      ['mimii/pump.zip', 'https://zenodo.org/records/3384388/files/-6_dB_pump.zip'],
      ['mimii/valve.zip', 'https://zenodo.org/records/3384388/files/-6_dB_valve.zip'],
    ],
  },
  toyadmos: {
    licence: 'CC BY-NC-SA 4.0 (non-commercial)',
    files: [
      ['toyadmos/dev_toycar.zip', 'https://zenodo.org/records/3678171/files/dev_data_ToyCar.zip'],
      ['toyadmos/dev_toyconveyor.zip', 'https://zenodo.org/records/3678171/files/dev_data_ToyConveyor.zip'],
    ],
  },
  catmeows: {
    licence: 'CC BY 4.0',
    files: [['catmeows/dataset.zip', 'https://zenodo.org/records/4008297/files/dataset.zip']],
  },
};

const requested = process.argv.slice(2);
const names = requested.includes('--all') ? Object.keys(DATASETS) : requested.filter((name) => name in DATASETS);

if (names.length === 0) {
  console.error('Usage: node scripts/eval/fetch-datasets.mjs [--all | mimii | toyadmos | catmeows]');
  console.error('Licences are listed in docs/datasets.md; read them before downloading.');
  process.exit(1);
}

for (const name of names) {
  const dataset = DATASETS[name];
  console.log(`\n${name} — licence: ${dataset.licence}`);
  for (const [relativePath, url] of dataset.files) {
    const target = join(dataDir, relativePath);
    if (await exists(target)) {
      console.log(`  already present: ${relativePath}`);
      continue;
    }
    await mkdir(dirname(target), { recursive: true });
    console.log(`  downloading ${url}`);
    const response = await fetch(url);
    if (!response.ok || response.body === null) {
      console.error(`  failed (${response.status}); fetch it manually into ${target}`);
      continue;
    }
    await pipeline(Readable.fromWeb(response.body), createWriteStream(target));
    console.log(`  saved ${relativePath}`);
  }
}

console.log(`\nUnpack the archives under ${dataDir} and run: pnpm eval:run`);

async function exists(path) {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}
