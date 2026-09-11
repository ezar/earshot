# 0002 — Status thresholds are derived from the learning session

**Status:** accepted

## Context

A check score is an aggregate of its window scores — the 90th percentile, so
that a machine sounding wrong for a couple of seconds registers while one stray
window does not.

Window scores are positions within the learned distance distribution. That makes
the aggregate's *normal* value a property of the aggregation, not of the machine:
the 90th percentile of scores drawn from the learning distribution sits near 0.9
by construction. Fixed thresholds of `watch = 0.9` / `anomalous = 0.97` therefore
flagged held-out normal audio roughly half the time. Measured: held-out normal
checks scored 0.71, 0.74, 0.92 and 0.98 against a `watch` threshold of 0.9.

The normal value also depends on how many windows a check holds and on how
tightly the machine's own sound clusters, so no single constant can work across
machines.

## Decision

`learnProfile` derives the thresholds. The learning windows are cut into
contiguous pseudo-checks; each one is scored against per-state models refitted
*without* it — the same leave-one-block-out idea as cross-validation — which
yields the distribution of check scores this machine produces when nothing is
wrong. `watch` is placed above the worst of them and `anomalous` a clear step
beyond, both clamped to a sane band.

Blocks are contiguous rather than random because consecutive windows overlap by
50 %: a random split would leak half of every held-out window back into the
training side and produce thresholds far too tight.

`learnProfile` still accepts explicit `watchThreshold` and `anomalousThreshold`
overrides, and `calibrate` moves the derived values in response to user verdicts.

Separately, `distanceScore` maps everything inside the learned range onto
`[0, 0.5]` and reserves the top half for distances never seen, approaching 1
exponentially. The earlier mapping compressed all novelty into the top 1 % of
the scale, which left no room for thresholds or for ranking one anomaly above
another.

## Consequences

- Held-out normal audio scores 0.37 to 0.51 against a derived `watch` of 0.72;
  every injected fault scores 1.0.
- A profile learned from too little audio (fewer than three blocks) falls back to
  a deliberately permissive pair, so a thin learning session produces few false
  alarms rather than constant ones.
- `Profile.thresholds` now varies between machines. Apps must not assume a
  particular numeric value, and must re-derive rather than carry thresholds
  across profiles.
