# Evaluation results

Results are produced by `pnpm eval:run` (see `scripts/eval/` and
`docs/datasets.md`) and refreshed for every release tag.

Every number below comes from headless Chromium running the real YAMNet task
files through the same engine, guards, profile and classifier a shipped app
uses. Nothing here is measured on the synthetic fixtures — those are for unit
tests, where ground truth is exact and no dataset licence is involved.

## Status

**Datasets: not run. Models and pipeline: run, with findings.**

The harness was executed for the first time on 2026-09-11 against the real
YAMNet classifier and embedder in headless Chromium. The dataset suites could
not be run — see [Blocked](#blocked) — but loading and driving the real models
turned up two defects that no unit test could have caught, both recorded in
`docs/decisions/0007-mediapipe-cannot-load-in-a-module-worker.md`:

1. `defaultTasksAudioLoader` used a variable import specifier, so **no browser
   could resolve MediaPipe** and `createEngine` could never load a model. Fixed.
2. MediaPipe's WASM loader calls `importScripts`, which throws inside the ES
   **module** worker earshot creates. The same code loads fine on the main
   thread and in a classic worker. **Open** — the fix changes what consumers
   configure, so it is a maintainer decision.

## Blocked

`zenodo.org` is denied by this environment's network policy — every request,
including the site root and the API, is refused at CONNECT with HTTP 403. MIMII,
ToyADMOS and CatMeows are all hosted there, so none could be downloaded. This is
an environment limitation, not a licence or harness problem: run
`pnpm eval:fetch --all` from a machine that can reach Zenodo and the suites
below will produce numbers.

## Model loading and inference (measured)

Real `yamnet_classifier.tflite` (4.1 MB) and `yamnet_embedder.tflite` (12.9 MB),
MediaPipe `@mediapipe/tasks-audio@0.10.21`, headless Chromium in a desktop
container.

| Metric | Value | Notes |
| --- | --- | --- |
| Model load, main thread | 1094 ms | classifier + embedder |
| Model load, classic worker | 1199 ms | end to end through `createEngine` |
| Model load, module worker | **fails** | see decision `0007` |
| Embedding dimensions | 1024 | matches `EMBEDDING_DIMENSIONS` |
| classify + embed, one window | 17.4 ms | models only |
| Full engine, one window | 43.6 ms | embed + classify + features |

The full-engine figure is **above the 30 ms target**, on a desktop container
that is considerably faster than the 2022 mid-range Android phone the target
names. The gap between 17.4 ms and 43.6 ms is the feature extractor, not the
models, which is where any optimisation should start.

## Pipeline sanity against real audio (measured)

Synthetic probes with known content, classified by the real model. This is not
an accuracy benchmark — it checks that the pipeline is wired correctly and that
the label strings the guards and the event detector match are the ones YAMNet
actually emits.

| Probe | Top classes |
| --- | --- |
| 1 kHz sine | `Beep, bleep` 0.74, `Sine wave` 0.15, `Chirp tone` 0.04 |
| White noise | `Spray` 0.59, `Liquid` 0.41, `Hiss` 0.20 |
| Digital silence | `Silence` 0.80 |
| Harmonic call, rising-falling f0 | `Siren` 0.20, `Alarm` 0.15, `Screaming` 0.15 |
| 120 Hz tone + noise (motor-like) | `Sine wave` 0.33, `White noise` 0.26, `Hum` 0.15 |

Labels arrive as title-case English strings via `categoryName`, which is the
form `INTERFERENCE_CLASSES` and `HUMAN_VOICE_CLASSES` assume; `Silence` and
`Screaming` appear verbatim. The trigger classes for real cat audio
(`Cat`, `Meow`, `Caterwaul`) remain unverified — a synthetic harmonic sweep is
not a meow, and CatMeows is what would settle it.

Features computed on white noise by the full engine: spectral flatness 0.987
(white noise should approach 1), spectral centroid 4031 Hz (a flat spectrum to
8 kHz has its centroid near 4 kHz), RMS −18.7 dBFS. All three are physically
correct on real model-fed audio, not just on fixtures.

## Anomaly detection (MIMII, -6 dB SNR)

Half of each machine's normal clips build the profile; the other half plus every
abnormal clip are scored, and the ROC AUC is computed over those scores.

| Machine | AUC | Target | Result |
| --- | --- | --- | --- |
| fan | — | > 0.80 | blocked: dataset unreachable |
| pump | — | > 0.80 | blocked: dataset unreachable |
| valve | — | > 0.80 | blocked: dataset unreachable |

## Anomaly detection (ToyADMOS, DCASE 2020 Task 2 development set)

| Machine | AUC | Target | Result |
| --- | --- | --- | --- |
| ToyCar | — | — | blocked: dataset unreachable |
| ToyConveyor | — | — | blocked: dataset unreachable |

## Vocalization detection and identity (CatMeows)

Detection recall is the fraction of clips producing at least one event. Pair
identity is the mean 5-fold accuracy of the cosine kNN over every pair of cats
with at least ten clips, capped at ten examples per cat.

| Metric | Value | Target | Result |
| --- | --- | --- | --- |
| Meow detection recall | — | > 90 % | blocked: dataset unreachable |
| Pair identity accuracy | — | > 80 % | blocked: dataset unreachable |

## Performance

Per-window cost (embedding plus features) must stay under 30 ms on a 2022
mid-range Android phone.

| Device | Per window | Target | Result |
| --- | --- | --- | --- |
| Desktop container, headless Chromium | 43.6 ms | < 30 ms | **over budget** |
| 2022 mid-range Android | — | < 30 ms | not measured |

## Reproducing

```bash
pnpm add -D playwright && npx playwright install chromium
pnpm eval:fetch --models          # YAMNet task files + MediaPipe WASM assets
pnpm eval:fetch --all             # datasets; accept the licences in docs/datasets.md
# unpack the archives under .eval-data/
pnpm eval:run
```

`--models` is enough to reproduce every measured number on this page; the
dataset suites need the rest.
