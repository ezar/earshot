# 0007 — MediaPipe cannot load in a module Worker

**Status:** accepted

## Context

The engine had never been run against the real models: the unit tests use
injected stand-ins, and no dataset evaluation had been executed. The first
attempt to drive the evaluation page in headless Chromium with the real YAMNet
task files hit two failures in a row, both fatal to every consuming app.

### 1. The default loader could not resolve MediaPipe

`defaultTasksAudioLoader` held the specifier in a variable with `@vite-ignore`,
so that neither TypeScript nor a bundler would try to resolve it:

```ts
const specifier = '@mediapipe/tasks-audio';
await import(/* @vite-ignore */ specifier);
```

That works in Node and fails in every browser: a bare specifier has no meaning
at runtime without an import map, so the call raised

```
Failed to resolve module specifier '@mediapipe/tasks-audio'
```

`createEngine` therefore could never load a model, and the README's quick start
could not work as written. Injecting `loadTasksAudio` does not help either,
because the import happens inside the Worker and a function cannot cross
`postMessage`.

**Fixed** by making the specifier static, so the consumer's bundler resolves it
and bundles it into the worker chunk. The engine worker chunk grows from 10.5 kB
to 62 kB in a production build, which is MediaPipe being included.

### 2. MediaPipe's WASM loader needs `importScripts`

With the specifier resolving, the next failure was:

```
Failed to execute 'importScripts' on 'WorkerGlobalScope':
Module scripts don't support importScripts().
```

`@mediapipe/tasks-audio@0.10.21` loads its WASM glue like this:

```js
if ("function" != typeof importScripts) {
  // fall back to injecting a <script> element via document
}
importScripts(url)
```

Inside an ES **module** worker, `importScripts` exists but throws, and
`document` does not exist — so neither branch can succeed. earshot creates its
worker with `{ type: 'module' }`.

Isolating the context confirms the diagnosis: the identical code loads fine on
the **main thread** (1094 ms, 1024-dimension embeddings), and fine inside a
**classic** worker (1199 ms), and only fails in a module worker.

`forAudioTasks(basePath, useModule)` gained a `useModule` flag in 1.x that looks
made for this, but 1.x is exactly the line that dropped `AudioEmbedder` (see
`0006`), so it is not available to anyone who needs embeddings.

## Decision

**The engine worker is a classic worker.** `createEngine` constructs it without
`{ type: 'module' }`, and consumers set `worker: { format: 'iife' }` in their
Vite config instead of `'es'`.

Measured end to end with the real models in that configuration:

| | |
| --- | --- |
| Models loaded in a classic worker | 1199-1408 ms |
| Embedding dimensions | 1024, matching `EMBEDDING_DIMENSIONS` |
| One window, embed + classify + features | 43.6 ms (desktop container) |

Alternatives, both rejected:

- **Run the models on the main thread.** Contradicts the spec's "the engine runs
  in a Worker" and gives up the reason the Worker exists.
- **Require MediaPipe 1.x with `useModule: true`.** Keeps module workers, but
  costs the embedder, and with it the `'embedding'` feature space and the kNN
  identity classifier (see `0006`).

## Consequences

**`worker.format` is global in a Vite config**, so an app cannot have earshot's
classic worker and its own module workers at the same time. Apps with other
workers must build those as IIFE too.

**The model path does not run under `vite dev`.** Vite serves workers as ES
modules in development whatever `worker.format` says, so the engine hits the
`importScripts` failure there. Verified across both worker formats and both the
`?worker&url` and `?worker` import forms — all four combinations fail in dev and
the build succeeds:

| Mode | Worker Vite produces | Models load |
| --- | --- | --- |
| `vite dev`, any setting | module | no |
| `vite build`, `format: 'iife'` | classic | **yes** |
| `vite build`, `format: 'es'` | module | no |

So anything that needs a class score or an embedding must be exercised against a
build: `vite build --watch` alongside `vite preview`, which is what
`pnpm playground` now does. `pnpm playground:dsp` keeps the dev server for the
capture, level and feature work that never touches a model.

This is a Vite limitation rather than an earshot defect, but it is the kind of
thing that costs an afternoon if it is not written down.

Everything that does not touch MediaPipe — the DSP, guards, learning, scoring,
events and classification — is unaffected in every mode.
