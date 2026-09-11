# Evaluation results

Results are produced by `pnpm eval:run` (see `scripts/eval/` and
`docs/datasets.md`) and refreshed for every release tag.

Every number below comes from headless Chromium running the real YAMNet task
files through the same engine, guards, profile and classifier a shipped app
uses. Nothing here is measured on the synthetic fixtures — those are for unit
tests, where ground truth is exact and no dataset licence is involved.

## Status

**Not yet run.** The library's algorithms are covered by 173 unit tests on
synthetic fixtures, and the harness below is written and ready, but no dataset
has been downloaded or evaluated. The tables are the shape the results will
take; the targets are inherited from the app specs.

To produce them:

```bash
pnpm add -D playwright && npx playwright install chromium
pnpm eval:fetch --all            # accept the licences in docs/datasets.md first
# unpack the archives under .eval-data/, then place the YAMNet task files and
# MediaPipe WASM assets where EARSHOT_MODELS_DIR points
pnpm eval:run
```

## Anomaly detection (MIMII, -6 dB SNR)

Half of each machine's normal clips build the profile; the other half plus every
abnormal clip are scored, and the ROC AUC is computed over those scores.

| Machine | AUC | Target | Result |
| --- | --- | --- | --- |
| fan | — | > 0.80 | not run |
| pump | — | > 0.80 | not run |
| valve | — | > 0.80 | not run |

## Anomaly detection (ToyADMOS, DCASE 2020 Task 2 development set)

| Machine | AUC | Target | Result |
| --- | --- | --- | --- |
| ToyCar | — | — | not run |
| ToyConveyor | — | — | not run |

## Vocalization detection and identity (CatMeows)

Detection recall is the fraction of clips producing at least one event. Pair
identity is the mean 5-fold accuracy of the cosine kNN over every pair of cats
with at least ten clips, capped at ten examples per cat.

| Metric | Value | Target | Result |
| --- | --- | --- | --- |
| Meow detection recall | — | > 90 % | not run |
| Pair identity accuracy | — | > 80 % | not run |

## Performance

Per-window cost (embedding plus features) must stay under 30 ms on a 2022
mid-range Android phone.

| Device | Per window | Target | Result |
| --- | --- | --- | --- |
| — | — | < 30 ms | not measured |
