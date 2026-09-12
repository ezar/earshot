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
   **module** worker earshot created. Fixed: the engine worker is now a classic
   worker, and consumers set `worker: { format: 'iife' }`. Verified end to end
   with the shipped defaults — `createEngine` loads the real models and produces
   1024-dimension embeddings.

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

**Methodology.** 30 s of audio (59 windows) per pass, four passes, the first
discarded as warm-up, median of the rest. An earlier revision of this page
reported 35.5-43.6 ms and called the budget missed; that was measurement error —
seven windows with a cold WASM runtime and no warm-up discarded. Those figures
are withdrawn.

| Metric | Value | Notes |
| --- | --- | --- |
| Model load, main thread | 1094 ms | classifier + embedder |
| Model load, classic worker | 833-1408 ms | end to end through `createEngine` |
| Model load, module worker | **fails** | see decision `0007` |
| Embedding dimensions | 1024 | matches `EMBEDDING_DIMENSIONS` |
| Full engine, one window | **17.5 ms** | classifier + embedder + features |
| Full engine, classifier only | 7.7 ms | no embedder configured |
| Full engine, window the guards reject | **7.3 ms** | embedder skipped; see below |

The embedder costs about 9.8 ms per window, over half the total. Model inference
dominates: the feature extractor is around 2 ms of the 17.5 ms, so even removing
it entirely would not change the picture much — which is why the guards option
below, which skips the embedder outright, matters more than any DSP work.

### Effect of the feature-extractor optimisations

Same methodology, each stage measured against the commit before it:

| Configuration | Original | Buffer reuse + flux logs | + real-input FFT |
| --- | --- | --- | --- |
| classifier + embedder | 19.5 ms | 17.3 ms | **17.5 ms** |
| classifier only | 9.1 ms | 8.7 ms | **7.7 ms** |

The classifier-only column is the honest measure of the DSP work, since it is
not swamped by the embedder: **9.1 ms to 7.7 ms, -15 %**. With the embedder
running, the DSP is a small enough share that the change sits inside the
run-to-run spread.

In a Node microbenchmark of `extract` alone (300 repetitions) the picture is
much sharper, **3.68 ms to 2.01 ms, -45 %**:

| Stage | `extract` | One 512-point transform |
| --- | --- | --- |
| Original | 3.68 ms | 0.018 ms |
| Reused STFT tables and buffers, flux logarithms carried forward | 3.12 ms | 0.018 ms |
| Real-input FFT | **2.01 ms** | **0.008 ms** |

Three changes account for it:

1. The STFT reuses its transform tables, analysis window and buffers instead of
   rebuilding them per window.
2. Spectral flux carries each magnitude's logarithm forward rather than
   recomputing it as the next frame's "previous", halving 49 152 logarithms per
   window to 24 576.
3. The FFT transforms a real signal with a complex transform of half the length
   plus a recombination pass, rather than a full-length complex transform with a
   zeroed imaginary part. Verified against a naive DFT over 42 signal and size
   combinations; worst relative error 4.4e-8.

### Effect of running the guards inside the worker

`createEngine({ guards })` evaluates the guards in the worker and skips the
embedder for rejected windows. The embedder is about half the per-window cost,
and an embedding the host app discards is worth nothing.

| Audio | Without `guards` | With `guards` | Windows embedded |
| --- | --- | --- | --- |
| Silence | 17.0 ms | **7.3 ms** | 0 of 60 |
| Loud noise, accepted | 17.4 ms | 17.2 ms | 60 of 60 |

It roughly halves the cost of windows that do not matter and costs nothing on
the ones that do.

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
| Desktop container, headless Chromium | 17.5 ms | < 30 ms | within budget |
| Same, window rejected by the guards | 7.3 ms | < 30 ms | within budget |
| 2022 mid-range Android | — | < 30 ms | **not measured** |

The target names a phone, and this container is considerably faster than one, so
being within budget here does not establish that the target is met. A mid-range
phone two to three times slower would sit at 35-50 ms. Measuring on real
hardware is the only way to settle it, and the embedder — about half the cost —
is the first thing to look at if it turns out to be missed.

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
