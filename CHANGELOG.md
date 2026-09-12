# Changelog

All notable changes to earshot are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
semantic versioning with the 0.x convention that **breaking changes bump MINOR**
and come with a migration note.

## [Unreleased]

### Added

- **`createEngine({ guards })` runs the guards inside the worker** and attaches
  the verdict to every window as `guard` — and skips the embedder for windows it
  rejects. The embedder is over half the per-window cost, and an embedding the
  host app discards is worth nothing: on silence the cost falls from 17.0 ms to
  **7.3 ms** per window, and on audio the guards accept there is no penalty
  (17.4 ms to 17.2 ms). `embedRejectedWindows: true` opts back in.
- `WindowResult.guard`, present only when the engine was given `guards`.
  `WindowGuard` and `WindowGuardReason` are exported.
- `createSpectrogramAnalyzer`, the reusable form of `computeSpectrogram`. The
  one-shot function is unchanged and still allocates per call; the analyzer
  lends its buffer and is only valid until the next call.
- `spectralFlux` takes an optional scratch buffer.

### Changed

- **The feature extractor is 45 % faster.** `extract` goes from 3.68 ms to
  2.01 ms in a Node microbenchmark over 300 repetitions, and the engine's
  classifier-only path from 9.1 ms to 7.7 ms per window in a browser. Three
  changes: the STFT reuses its transform tables, analysis window and buffers
  instead of rebuilding them per window; spectral flux carries each magnitude's
  logarithm forward rather than recomputing it as the next frame's "previous",
  halving 49 152 logarithms per window to 24 576; and the FFT transforms a real
  signal with a complex transform of half the length plus a recombination pass,
  rather than a full-length complex transform with a zeroed imaginary part.
  A 512-point transform goes from 0.018 ms to 0.008 ms.
- `Fft` now requires a size of at least 4, since the real-input path needs a
  half-length complex transform. Sizes below that were never usable for audio.

### Fixed

- **The per-window figures published in 0.4.0 were wrong.** `docs/eval-results.md`
  and the 0.4.0 changelog entry reported 35.5-43.6 ms and called the 30 ms budget
  missed. That was measurement error: seven windows with a cold WASM runtime and
  no warm-up discarded. Measured properly — 59 windows, four passes, first
  discarded, median — the figure is **17.3 ms, within budget** on this hardware.
  The target names a 2022 mid-range Android phone, which this container is not,
  so the target itself remains unverified.

## [0.4.0] — 2026-09-11

A MINOR bump rather than a PATCH: consumers must change their Vite config, and
this project's convention is that breaking changes bump MINOR while in 0.x.

**0.3.0 should not be used.** `createEngine` cannot load a model in any browser
on that tag — both defects below were present — so anything needing a class
score or an embedding fails. There is no workaround within 0.3.0.

### Migration from 0.3.0

1. Change `worker.format` in your Vite config:

   ```diff
    export default defineConfig({
      optimizeDeps: { exclude: ['earshot'] },
   -  worker: { format: 'es' },
   +  worker: { format: 'iife' },
    });
   ```

   The engine worker is now a classic worker. `worker.format` is global, so if
   your app has module workers of its own, they must become IIFE too.

2. Move any development of the model path off `vite dev`. Vite serves workers as
   ES modules in development whatever `worker.format` says, so the engine cannot
   load MediaPipe there. Use `vite build --watch` with `vite preview`. Capture,
   levels, features, pitch and segmentation never touch a model and are
   unaffected.

3. Nothing else changes. No API signature, type or return shape moved.

### Fixed

- `defaultTasksAudioLoader` held the MediaPipe specifier in a variable behind
  `@vite-ignore`, so no browser could resolve it and `createEngine` could never
  load a model. The specifier is now static, so the consumer's bundler resolves
  it and bundles it into the worker chunk.
- `CHANGELOG.md` and `docs/eval-results.md` reported 169 unit tests; the count
  is 173.

### Changed

- **The engine worker is a classic worker.** MediaPipe loads its WASM glue with
  `importScripts`, which an ES module worker does not support, and its fallback
  needs a `document` a worker does not have. See the migration note above and
  `docs/decisions/0007-mediapipe-cannot-load-in-a-module-worker.md`.
- Toolchain moved to the current releases: TypeScript 7.0.2 (from 5.7), Vite
  8.3.0 (from 6.0), Vitest 5.0.0 (from 2.1), `@types/node` 22.20.2, Playwright
  1.63.0. Node stays on 22, which every one of them supports.
- `TasksAudioModule.AudioEmbedder` is now optional, and `createEmbedder` throws
  a message naming the version constraint when the loaded MediaPipe build does
  not provide it. `EMBEDDER_MAX_VERSION` is exported.
- `pnpm playground` builds and previews instead of running the dev server, since
  the model path needs a build. `pnpm playground:dsp` keeps the dev server for
  work that touches no model.

### Added

- CI job `typescript-compatibility`: the sources are type-checked against
  TypeScript 5, 6 and 7. Consumers compile these sources with their own
  compiler, so the pinned version alone proves nothing; all three pass.
- `pnpm eval:fetch --models` downloads the YAMNet task files and stages the
  MediaPipe WASM assets. The harness previously assumed they were already in
  place and could not run a single window without them.
- `docs/eval-results.md` carries measured numbers for model loading, inference
  cost and pipeline sanity against real YAMNet, and says plainly which suites
  are blocked and why.

### Notes

**The evaluation harness ran against real models for the first time**, and that
is what found both defects above. Unit tests could not have: they use injected
stand-ins, so nothing had ever asked a browser to load MediaPipe.

**Per-window cost.** Reported here as 35.5-43.6 ms and over budget; that
measurement was wrong and is corrected in the Unreleased entry above. The
models, not the feature extractor, dominate the cost.

**The dataset suites are still unrun.** `zenodo.org` was unreachable from the
environment this work was done in, so MIMII, ToyADMOS and CatMeows could not be
downloaded and every target in `docs/eval-results.md` remains unmeasured.

**MediaPipe pinning is unchanged** from 0.3.0: apps needing embeddings must pin
`"@mediapipe/tasks-audio": "<=0.10.21"`.

## [0.3.0] — 2026-09-11

First release covering milestones E0 through E3. The whole library lands at once
rather than as three tags, because the modules were developed and tested
together; subsequent releases will be incremental.

### Added

**Capture (E0)**

- `earshot/capture-worklet`: an AudioWorklet processor that allocates nothing
  inside `process` and posts transferable chunks.
- `createCapture` requests echo cancellation, noise suppression and automatic
  gain control **off**, and reports what the browser actually applied through
  `appliedConstraints`, including an `unhonoured` list.

**DSP (E0, E1, E2)**

- `resample` with an anti-alias low-pass, `toMono`, `createFramer` and
  `frameBuffer` on the 0.975 s / 0.4875 s grid.
- Radix-2 `Fft`, `computeSpectrogram`, `melFilterbank`, `logMel`,
  `bandEnergies`, `findPeaks` with prominence, `spectralFlatness`,
  `spectralCentroidHz`.
- `spectralFlux`, `detectOnsets`, `onsetPeriodicity`, `modulationRate`.
- `trackPitch` and `yin`: a YIN tracker over 80–1200 Hz at 25 ms / 10 ms.
- `createFeatureExtractor`, `extractFeatures`, `featureVector`.

**Models (E0, E3)**

- `createClassifier` and `createEmbedder`: thin MediaPipe Tasks Audio wrappers.
  Model and WASM URLs are always injected; the MediaPipe module itself can be
  injected too, for apps that want it statically bundled.
- `earshot/clap`: optional CLAP embeddings over `@huggingface/transformers`.

**Engine (E0)**

- `createEngine` and the `earshot/worker` entry point: framing, inference and
  feature extraction off the main thread.

**Guards (E1)**

- `createGuards` with silence, level, clipping and interference rules, plus
  `humanVoiceScore` and `isPossiblyHuman` for the vocalization voice guard.

**Learning and scoring (E1, E3)**

- `kmeans`, `selectClustering`, `silhouette`.
- `fitDiagonalGaussian` with shrinkage, `mahalanobisDistance`,
  `buildDistanceDistribution`, `distanceScore`.
- `learnProfile`, `calibrate`, `scoreCheck`, `scoreWindow`, `statusFor`.
- `describeDifference` producing human-readable descriptors.
- `createStreamScorer` with smoothing, status hold and drift detection.

**Events (E2)**

- `segmentBuffer` and `segmentEnvelope` at 10 ms with hysteresis and gap
  bridging, `countSyllables`.
- `createEventDetector` fusing class triggers with envelope segmentation, with
  merging across overlapping windows and debounce.

**Classification (E2, E3)**

- `createKnnClassifier` (cosine, distance-weighted votes), `buildPrototypes`,
  `predictByPrototype`, `crossValidate`.
- `earshot/mlp`: optional TensorFlow.js head over `@tensorflow/tfjs`.

**Utilities**

- `quantize` and `dequantize` for float16 and int8 embeddings.

**Repository**

- 173 unit tests on synthetic fixtures (`fixtures/synthetic/`).
- `playground/`: a Vite page for live capture in a real browser.
- `scripts/smoke/`: a headless-Chromium check that the AudioWorklet processor
  loads and runs, both as a served asset and as an inlined `data:` URL. This
  covers the one code path unit tests cannot reach.
- `scripts/eval/`: a Playwright harness that streams dataset WAVs through the
  production pipeline in headless Chromium. Datasets and licences are documented
  in `docs/datasets.md`; results in `docs/eval-results.md`.
- Decision records in `docs/decisions/` for every place the implementation
  departs from the obvious reading of the app specs.

### Notes on algorithm choices

Five choices differ from what the app specs imply, each because a test on a
signal with exact ground truth showed the obvious approach failing. They are
recorded in full in `docs/decisions/`:

- `0001` — the feature vector carries spectral *shape* with the common mode
  removed. A diagonal covariance treats the mel bands as independent, so a
  uniform +1 dB gain change scored as the maximum anomaly.
- `0002` — status thresholds are derived from the learning session by scoring
  leave-one-block-out pseudo-checks. Fixed thresholds flagged held-out normal
  audio about half the time. `distanceScore` was also rescaled so the learned
  range occupies `[0, 0.5]` and novelty has room above it.
- `0003` — spectral flux floors magnitudes at -120 dBFS and onset picking takes
  an absolute, gain-invariant strength floor; periodicity uses phase coherence
  rather than the coefficient of variation of intervals.
- `0004` — YIN rejects frames whose fundamental is above `maxHz` instead of
  reporting a subharmonic.
- `0005` — K selection takes an absolute silhouette floor, because the
  coefficient is undefined at K = 1 and will otherwise split structureless data.

### Known gaps

- **No dataset evaluation has been run.** The harness is written and ready, but
  MIMII, ToyADMOS and CatMeows have not been downloaded, so every target in
  `docs/eval-results.md` is unmeasured.
- **Per-window cost has not been measured on a phone.** The 30 ms target on a
  2022 mid-range Android device is unverified.
- `earshot/clap` and `earshot/mlp` are type-checked and unit-tested only through
  injected module stand-ins; neither has been run against the real
  `@huggingface/transformers` or `@tensorflow/tfjs`.

[0.4.0]: https://github.com/ezar/earshot/releases/tag/v0.4.0
[0.3.0]: https://github.com/ezar/earshot/releases/tag/v0.3.0
