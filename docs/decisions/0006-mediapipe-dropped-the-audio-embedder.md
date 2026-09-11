# 0006 — MediaPipe dropped `AudioEmbedder`, so the embedder pins an older release

**Status:** accepted

## Context

The spec calls for "thin wrappers over MediaPipe Tasks Audio (AudioClassifier
and AudioEmbedder with YAMNet)", and earshot's peer range was written as
`@mediapipe/tasks-audio: >=0.10`.

That range is wrong. Checking every published release:

| Release | `AudioClassifier` | `AudioEmbedder` |
| --- | --- | --- |
| 0.10.0 – 0.10.21 | present | present |
| 0.10.32 – 1.0.1 (latest) | present | **absent** |

`AudioEmbedder` is gone from both `audio.d.ts` and the runtime bundle from
0.10.32 onward — 0.10.22 through 0.10.31 were never published as stable. The
package README still documents `AudioEmbedder.createFromModelPath`, so the
removal is easy to miss until an app crashes.

This is not cosmetic. Embeddings drive the `'embedding'` feature space that
`learnProfile` prefers whenever they are available, and they are the entire
basis of Meowlogue's kNN identity. On MediaPipe 1.x the embedding path dies with
"cannot read properties of undefined", inside a Worker, the first time a user
tries to learn a profile.

Where it still exists, the API is exactly what earshot's wrapper assumes:
`embed(audioData: Float32Array, sampleRate?: number): AudioEmbedderResult[]`,
results carrying `floatEmbedding?: number[]`, and `l2Normalize` / `quantize`
options. The wrapper is correct; only the version range was wrong.

## Decision

1. `TasksAudioModule.AudioEmbedder` is optional, because that is the truth about
   current MediaPipe builds.
2. `createEmbedder` checks for it and throws a message naming the constraint and
   both ways out, rather than letting an app hit an `undefined` dereference.
3. `EMBEDDER_MAX_VERSION` (`'0.10.21'`) is exported, so apps can assert against
   it rather than hardcoding the number.
4. The peer range stays permissive (`>=0.10`). Narrowing it to `<0.10.32` would
   also block the classifier, which works fine on every release — and an app
   that only classifies, or that learns profiles in the `'features'` space, has
   no reason to be held back.

## Consequences

- An app that needs embeddings must pin `"@mediapipe/tasks-audio": "<=0.10.21"`.
  This is stated in the README next to the peer dependency table.
- An app that does not can use the latest MediaPipe. `createEngine` already
  treats a missing `embedderUrl` as "no embedder", and `learnProfile` falls back
  to the `'features'` space automatically, so SteadyHum degrades rather than
  breaks.
- Meowlogue's identity feature has no such fallback: cosine kNN over
  interpretable features is not a substitute for kNN over a learned embedding.
  Meowlogue must pin the older MediaPipe.

## Alternatives considered

- **Pinning the peer range to `<0.10.32`.** Rejected: it punishes
  classifier-only consumers for a limitation that does not affect them.
- **Replacing the embedder.** A TensorFlow.js YAMNet, or the CLAP wrapper behind
  `earshot/clap`, could supply embeddings instead. Both are real options and
  both are larger changes than this decision record covers; if MediaPipe never
  restores `AudioEmbedder`, one of them becomes the path forward. Raised for the
  maintainer rather than decided here.
