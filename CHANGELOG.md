# Changelog

All notable changes to earshot are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
semantic versioning with the 0.x convention that **breaking changes bump MINOR**
and come with a migration note.

## [Unreleased]

### Fixed

- `defaultTasksAudioLoader` held the MediaPipe specifier in a variable behind
  `@vite-ignore`, so no browser could resolve it and `createEngine` could never
  load a model. The specifier is now static, so the consumer's bundler resolves
  it. Found by running the evaluation harness against the real models for the
  first time; see `docs/decisions/0007-*.md`.

### Added

- `pnpm eval:fetch --models` downloads the YAMNet task files and stages the
  MediaPipe WASM assets. The harness previously assumed they were already in
  place and could not run a single window without them.
- `docs/eval-results.md` now carries measured numbers for model loading,
  inference cost and pipeline sanity against real YAMNet, and says plainly which
  suites are blocked and why.

### Notes

**`createEngine` cannot load models in a module Worker.** MediaPipe's WASM
loader calls `importScripts`, which throws inside an ES module worker; the same
code loads fine on the main thread and in a classic worker. The fix changes what
consumers configure (`worker: { format: 'iife' }`), so it is recorded as a
proposed decision in `docs/decisions/0007-*.md` rather than applied.

**Per-window cost is over budget.** 43.6 ms for embed + classify + features on a
desktop container, against a 30 ms target on a 2022 mid-range Android. The
models account for 17.4 ms of that; the rest is the feature extractor.

### Changed

- Toolchain moved to the current releases: TypeScript 7.0.2 (from 5.7), Vite
  8.3.0 (from 6.0), Vitest 5.0.0 (from 2.1), `@types/node` 22.20.2, Playwright
  1.63.0. Node stays on 22, which every one of them supports.
- `TasksAudioModule.AudioEmbedder` is now optional, and `createEmbedder` throws
  a message naming the version constraint when the loaded MediaPipe build does
  not provide it. `EMBEDDER_MAX_VERSION` is exported.

### Added

- CI job `typescript-compatibility`: the sources are type-checked against
  TypeScript 5, 6 and 7. Consumers compile these sources with their own
  compiler, so the pinned version alone proves nothing; all three pass.

### Fixed

- `CHANGELOG.md` and `docs/eval-results.md` reported 169 unit tests; the count
  is 173.

### Notes

**MediaPipe removed `AudioEmbedder`.** It ships in `@mediapipe/tasks-audio` up
to 0.10.21 and is absent from 0.10.32 onward, including 1.x, from both the types
and the runtime bundle — while the package README still documents it. Apps that
need embeddings (the `'embedding'` feature space, Meowlogue's kNN identity) must
pin `"@mediapipe/tasks-audio": "<=0.10.21"`. The classifier is unaffected on
every release, so the peer range stays permissive and classifier-only apps can
use the latest. Full analysis in
`docs/decisions/0006-mediapipe-dropped-the-audio-embedder.md`.

The other peer ranges already admit their latest releases and are unchanged:
`@huggingface/transformers` (latest 4.2.0, range `>=3`) and `@tensorflow/tfjs`
(latest 4.22.0, range `>=4`).

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

[0.3.0]: https://github.com/ezar/earshot/releases/tag/v0.3.0
