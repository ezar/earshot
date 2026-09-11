# 0004 — YIN rejects out-of-range fundamentals instead of reporting subharmonics

**Status:** accepted

## Context

The pitch tracker searches lags corresponding to 80–1200 Hz. A signal whose true
period is *shorter* than the shortest searched lag is also periodic at every
multiple of that period, so a lag-limited search finds a confident minimum at a
subharmonic and reports it.

Measured: a 2 kHz tone came back as a confident 1001 Hz estimate. For Meowlogue
that means a kettle or a smoke alarm is reported as a vocalization in the middle
of a cat's range.

## Decision

`yin` computes the cumulative mean normalized difference from lag 2, not from
`minLag`, and rejects the frame when any out-of-range lag explains it at least as
well as the best in-range one. The search itself still starts at `minLag`.

## Consequences

- A 2 kHz tone is reported unvoiced. A 350 Hz harmonic stack, whose upper
  harmonics fall in the out-of-range region but which is not *periodic* there, is
  unaffected.
- The cost is a slightly longer difference-function loop, negligible against the
  loop already running to `maxLag`.
- Callers wanting the higher range should widen `maxHz` rather than reading a
  subharmonic.
