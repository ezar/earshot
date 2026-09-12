# earshot

The shared in-browser audio engine behind SteadyHum and Meowlogue.

earshot is a framework-agnostic TypeScript library that captures microphone
audio in the browser, frames it, runs small audio models locally (MediaPipe
YAMNet classifier and embedder), extracts interpretable features, and provides
the learning and scoring primitives both apps need: machine "normal" profiles
and anomaly scores for SteadyHum, and vocalization events, pitch features and
few-shot classifiers for Meowlogue.

It has **no UI**, **no persistence** and makes **no network calls**. Everything
it returns is plain serializable data that the host app stores and renders.

MIT licensed. Distributed from GitHub release tags, not npm.

## Install

```jsonc
// package.json
{
  "dependencies": {
    "earshot": "github:ezar/earshot#v0.5.0"
  }
}
```

> **Do not use `v0.3.0`.** `createEngine` cannot load a model in any browser on
> that tag, so nothing needing a class score or an embedding works. `v0.4.0` is
> the first usable release, and `v0.5.0` is the one to take: it is the same API,
> measurably faster, and it stops the capture worklet allocating on the audio
> thread. See the migration note in `CHANGELOG.md` if you are coming from
> `v0.3.0`; from `v0.4.0` there is nothing to migrate.

Always depend on a tag, never on a branch. There is no build step: the `exports`
map points at the TypeScript sources, and there are no install or prepare
scripts, so this works under pnpm 10, which blocks dependency lifecycle scripts
by default.

Because your project type-checks earshot's sources, they compile cleanly under
`strict`, `noUncheckedIndexedAccess` and `exactOptionalPropertyTypes`.

### Peer dependencies

| Package | Needed for | Optional |
| --- | --- | --- |
| `@mediapipe/tasks-audio` | the classifier and embedder | yes, if you only use the DSP |
| `@huggingface/transformers` | `earshot/clap` | yes |
| `@tensorflow/tfjs` | `earshot/mlp` | yes |

> **If you need embeddings, pin `"@mediapipe/tasks-audio": "<=0.10.21"`.**
> MediaPipe removed `AudioEmbedder` from 0.10.32 onward, including 1.x — it is
> gone from both the types and the runtime bundle, though the package README
> still documents it. The classifier is unaffected and works on every release.

Which version you need depends on what you use earshot for:

| You use | MediaPipe version | Why |
| --- | --- | --- |
| The DSP only (features, pitch, segmentation) | not needed | no model is loaded |
| Classification, and profiles in the `'features'` space | any, including 1.x | only `AudioClassifier` is touched |
| Profiles in the `'embedding'` space, or the kNN identity classifier | **`<=0.10.21`** | needs `AudioEmbedder` |

Without an embedder, `createEngine` runs classifier-only and `learnProfile`
falls back to the `'features'` space on its own, so a profile still works.
`createEmbedder` throws with this constraint spelled out rather than failing on
an undefined reference, and `EMBEDDER_MAX_VERSION` is exported so you can assert
on it instead of hardcoding the number:

```ts
import { EMBEDDER_MAX_VERSION } from 'earshot'; // '0.10.21'
```

Full analysis in `docs/decisions/0006-mediapipe-dropped-the-audio-embedder.md`.

### Vite configuration

```ts
// vite.config.ts
import { defineConfig } from 'vite';

export default defineConfig({
  optimizeDeps: { exclude: ['earshot'] },
  worker: { format: 'iife' },
});
```

Both settings are required.

`optimizeDeps.exclude` because earshot ships TypeScript, and the dependency
pre-bundler cannot process it.

`worker.format: 'iife'` because the engine worker is a **classic** worker, not a
module worker: MediaPipe loads its WASM glue with `importScripts`, which module
workers do not support. Note that `worker.format` is global, so an app that has
its own module workers cannot have both.

The AudioWorklet processor (`earshot/capture-worklet`) is plain JavaScript, not
TypeScript, because `audioWorklet.addModule` hands its URL straight to the
browser — a `?url` import of a `.ts` file would deliver TypeScript to the audio
thread. Vite may either emit it as a separate asset or inline it as a `data:`
URL, depending on `assetsInlineLimit`; `pnpm smoke` verifies both paths load and
run in Chromium. If you target a browser you are unsure about, set
`build.assetsInlineLimit: 0` so the file is always emitted as a real asset.

## Entry points

| Specifier | Contents |
| --- | --- |
| `earshot` | the core API |
| `earshot/worker` | the engine Worker |
| `earshot/capture-worklet` | the AudioWorklet processor |
| `earshot/clap` | optional CLAP embeddings (pulls `@huggingface/transformers`) |
| `earshot/mlp` | optional TensorFlow.js head (pulls `@tensorflow/tfjs`) |

The two optional entry points exist so the core bundle never pulls their
dependencies: an app that does not import them pays nothing for them.

## Self-hosting the models

earshot never hardcodes a model or WASM location. Copy the YAMNet task files and
the MediaPipe WASM assets into your app's `public/models/` and pass their URLs
in. Nothing is fetched from a CDN at runtime.

```
public/models/
  wasm/                       # from @mediapipe/tasks-audio's wasm directory
  yamnet_classifier.tflite
  yamnet_embedder.tflite
```

## Quick start

```ts
import { createCapture, createEngine, createGuards } from 'earshot';
import workletUrl from 'earshot/capture-worklet?url';
import workerUrl from 'earshot/worker?worker&url';

const models = {
  wasmBaseUrl: '/models/wasm',
  classifierUrl: '/models/yamnet_classifier.tflite',
  embedderUrl: '/models/yamnet_embedder.tflite',
};

const capture = await createCapture({ workletUrl });
const engine = await createEngine({ workerUrl, models });
const guards = createGuards();

engine.onWindow((window) => {
  if (!guards.check(window).accepted) return;
  console.log(window.t, window.rmsDbfs, window.classes[0]?.label);
});

capture.onChunk((samples) => void engine.push(samples));

// later
await capture.stop();
await engine.close();
```

### Running the guards inside the worker

Pass `guards` to `createEngine` and the worker evaluates them itself, attaching
the verdict to every window as `guard` — and **skipping the embedder for windows
it rejects**:

```ts
const engine = await createEngine({
  workerUrl,
  models,
  guards: {},                 // same options as createGuards
  // embedRejectedWindows: true,   // opt back in if you want every embedding
});

engine.onWindow((window) => {
  if (window.guard?.accepted !== true) return;   // no embedding was computed
  …
});
```

This is the single biggest performance lever the engine has. The embedder is
roughly half the per-window cost, and an embedding the app is going to discard
is worth nothing. Measured with the real models:

| Audio | Without `guards` | With `guards` |
| --- | --- | --- |
| Silence | 17.0 ms/window | **7.3 ms/window** |
| Loud noise, accepted | 17.4 ms/window | 17.2 ms/window |

So it roughly halves the cost of the windows that do not matter, and costs
nothing on the ones that do.

> **The model path does not run under `vite dev`.** Vite serves workers as ES
> modules in development whatever `worker.format` says, and MediaPipe cannot
> load in a module worker. Exercise anything that needs a class score or an
> embedding against a build — `vite build --watch` alongside `vite preview`.
> Capture, levels, features, pitch and segmentation never touch a model and work
> in dev as normal. See `docs/decisions/0007-*.md`.

The worklet allocates nothing on the audio thread. A chunk has to be transferred
to reach the main thread and a transferred buffer cannot be refilled, so the
processor keeps a pool and `createCapture` hands each buffer straight back.
`capture.starvedBuffers` counts any the processor had to allocate because the
main thread was too slow to return them — it should stay at zero, and each one
is a garbage collection on the audio thread, which is a gap in the recording.

`createCapture` asks the browser for echo cancellation, noise suppression and
automatic gain control to be **off** — all three rewrite the signal in ways that
destroy the stationarity a machine profile depends on, and AGC makes dBFS levels
incomparable between checks. Browsers do not always comply, so check what you
got:

```ts
const { sampleRateHz, unhonoured, deviceLabel } = capture.appliedConstraints;
if (unhonoured.length > 0) {
  console.warn(`levels may not be comparable; browser kept ${unhonoured.join(', ')} on`);
}
```

## Learning and scoring a machine (SteadyHum)

```ts
import { calibrate, describeDifference, learnProfile, scoreCheck } from 'earshot';

// 1. Collect windows from one or more learning sessions, guard-filtered.
const profile = learnProfile(learningWindows);
// store `profile` with Dexie; it is plain JSON

// 2. Score a later check against it.
const result = scoreCheck(profile, checkWindows);
result.score;   // 0..1
result.status;  // 'normal' | 'watch' | 'anomalous'

// 3. Explain the verdict in words.
for (const descriptor of describeDifference(profile, result.dominantStateId, checkWindows)) {
  console.log(descriptor.text); // "Level is louder than usual, by 14.2 dBFS."
}

// 4. Fold the user's judgement back in.
const updated = calibrate(profile, 'normal', checkWindows);
```

A few things worth knowing about how this behaves:

- **States.** A machine rarely has one steady sound — a fridge cycles, a pump
  alternates between load states. `learnProfile` clusters the learning windows
  and gives the profile one *state* per distinguishable mode, choosing K by
  silhouette subject to a minimum cluster size. Scoring compares a check against
  the nearest state rather than against a single blurred average.
- **Thresholds are derived, not fixed.** A check score is a high percentile of
  its window scores, so its normal value depends on the machine and on how long
  a check is. `learnProfile` measures that by scoring leave-one-block-out
  pseudo-checks from the learning session itself and places the thresholds above
  what it saw. A hardcoded pair flags ordinary audio roughly half the time.
- **The score's scale is meaningful.** Everything the profile actually heard
  while learning maps into `[0, 0.5]`; the top half of the scale is reserved for
  distances it never saw, approaching 1 as the check gets stranger. So a
  slightly unusual check and a wildly unusual one stay far apart.
- **Level sensitivity is graded.** A decibel of drift is ordinary room variation
  and will not raise an alarm; a big level change will.

### Watch mode

```ts
import { createStreamScorer } from 'earshot';

const scorer = createStreamScorer(profile);
engine.onWindow((window) => {
  const update = scorer.push(window);
  if (update.statusChanged) notify(update.status);
  if (update.drift.drifting) notify(`baseline rising by ${update.drift.delta.toFixed(2)}`);
});
```

The stream scorer smooths window scores so a single lorry driving past does not
raise an alarm, holds a new status for several windows before reporting it, and
separately watches for *drift*: a slow rise in the baseline that no single
window would ever flag.

## Detecting and identifying vocalizations (Meowlogue)

> **This section needs `"@mediapipe/tasks-audio": "<=0.10.21"`.**
>
> ```jsonc
> // package.json
> {
>   "dependencies": {
>     "earshot": "github:ezar/earshot#v0.5.0",
>     "@mediapipe/tasks-audio": "<=0.10.21"
>   }
> }
> ```
>
> Identity rests on cosine distance between YAMNet embeddings, and MediaPipe
> dropped `AudioEmbedder` after 0.10.21 (see [Peer dependencies](#peer-dependencies)).
> There is no fallback for this one: kNN over the interpretable features is not a
> substitute for kNN over a learned embedding. Without the pin, `createEngine`
> rejects at startup, because loading the embedder is part of initialising the
> worker. Dropping `embedderUrl` gets you past that, but then every
> `window.embedding` is empty and `createKnnClassifier` has nothing to separate
> one cat from another.
>
> Detection, segmentation, syllable counting and pitch need none of this — they
> run off the classifier and the DSP, so they work on any MediaPipe version.

```ts
import {
  createEventDetector,
  createKnnClassifier,
  crossValidate,
  quantize,
  SAMPLE_RATE_HZ,
} from 'earshot';

const detector = createEventDetector({ triggerClasses: ['Cat', 'Meow', 'Caterwaul'] });
const classifier = createKnnClassifier<string>();

// The detector needs the window's own audio as well as the engine's result:
// YAMNet says *what* is in a window, the envelope says exactly *when*.
for (const event of detector.push(window, windowSamples)) {
  event.duration;      // seconds
  event.syllables;     // amplitude lobes
  event.pitch.medianF0Hz;
  event.pitch.contourSlopeSemitonesPerSecond; // negative falls, positive rises
  event.possibleHuman; // a human-voice class was confident in the trigger window

  classifier.add('luna', window.embedding);
}

// Tell the user how well their examples actually separate.
const report = crossValidate(classifier.examples(), { folds: 5 });
report.accuracy;          // 0..1
report.labels[0]?.recall; // per cat

// Store embeddings compactly: float16 halves the size, int8 quarters it.
const stored = quantize(window.embedding, 'float16');
```

`trackPitch` is also available directly, and reports only estimates that are
periodic *inside* the search range — a 2 kHz kettle comes back unvoiced rather
than as a confident 1 kHz call.

```ts
import { trackPitch } from 'earshot';
const track = trackPitch(pcm16k); // 80-1200 Hz, 25 ms frames, 10 ms hop
```

## Framing constants

Named exports, so nothing has to guess the grid:

```ts
import { HOP_SECONDS, SAMPLE_RATE_HZ, WINDOW_SECONDS, MEL_BANDS } from 'earshot';
// 16000 Hz, 0.975 s windows, 0.4875 s hop, 64 mel bands
```

Every public function's JSDoc carries units on its numeric parameters (Hz,
dBFS, seconds, ms).

## Persistence

earshot stores nothing. `Profile`, `KnnSnapshot`, `QuantizedEmbedding`,
`WindowResult` and everything else crossing the boundary is JSON-compatible:
plain numbers, strings, arrays and objects — never a typed array, `Map` or class
instance. Persist them with Dexie and hand them back later.

`Profile` and `KnnSnapshot` carry a `schemaVersion`; check it when you load.

### Storing many profiles

A profile learned in the `'embedding'` space carries a 1024-dimension mean and
variance per state, and as JSON text that runs to roughly 135 kB for three
states — almost all of it digits. `compactProfile` quantizes those to float16
and `expandProfile` restores them:

```ts
import { compactProfile, expandProfile } from 'earshot';

await db.profiles.put(compactProfile(profile));
const restored = expandProfile(await db.profiles.get(id));
```

A round trip moves a check score by well under a thousandth. Only the embedding
space is quantized: `'features'`-space profiles pass through untouched, because
their dimensions run from negative decibels to thousands of hertz — where
float16's three significant digits are a visible error — and there are only
about seventy of them per state, so there is nothing to save.

## Developing

```bash
pnpm install
pnpm test          # vitest, unit tests on synthetic fixtures
pnpm typecheck     # tsc --noEmit under the consumers' strictness
pnpm playground    # a real browser: builds, then previews (models need a build)
pnpm playground:dsp # dev server, for the DSP-only path
pnpm smoke         # headless Chromium: loads the worklet and checks it runs
pnpm bench         # build a benchmark page and serve it to a phone on your LAN
```

`pnpm bench` answers the one question CI cannot: what the engine costs on real
hardware. It needs no HTTPS and no hosting — the page uses no microphone, so a
plain LAN address is enough. See `docs/benchmarking-on-a-phone.md`.

`pnpm smoke` needs Playwright and a browser
(`pnpm add -D playwright && npx playwright install chromium`). It covers the one
part of earshot unit tests cannot reach: `process` runs on the audio thread, in
a global scope only a real browser provides.

To develop against an app, `pnpm link ../earshot` in the app. Never commit that
link.

Evaluation against public datasets is separate, needs explicit downloads and
licence acceptance, and is documented in `docs/datasets.md`. Results live in
`docs/eval-results.md`.

## Releases

Releases are git tags `vMAJOR.MINOR.PATCH`, each with a `CHANGELOG.md` entry and
an updated `docs/eval-results.md`. While in 0.x, breaking changes bump MINOR and
come with a migration note in the changelog. Each app updates its tag
deliberately.
