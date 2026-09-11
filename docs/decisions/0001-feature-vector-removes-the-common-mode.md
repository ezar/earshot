# 0001 — The feature vector carries spectral shape, not absolute band levels

**Status:** accepted

## Context

The app specs describe a profile built from log-mel bands, octave band energies
and a set of scalar features, modelled with a **diagonal** covariance and scored
by Mahalanobis distance. The obvious flattening puts each band's absolute level
into its own dimension.

That combination has a failure mode. A diagonal covariance assumes its
dimensions vary independently. The mel and octave bands do not: turning the gain
up by one decibel moves all 71 of them by exactly one decibel. The model reads
that coherent shift as 71 independent surprises happening at once, and the
distance grows as the square root of the dimension count.

Measured on a profile learned from 40 s of stationary noise, a uniform **+1 dB**
gain change scored 1.0 — the maximum — indistinguishable from a bearing failure.
A decibel of drift is ordinary variation in mic position and room conditions,
so the profile would have been useless in practice.

## Decision

`featureVector` subtracts the mean from the log-mel bands and from the octave
bands, keeping their *shape*, and carries the overall level as one separate
dimension.

`describableValues` does the same for descriptors: band values are reported
relative to the window's overall level.

## Consequences

- A gain change now moves one dimension out of 71 instead of all of them, and
  sensitivity to level is graded: +1 dB and +2 dB score as normal, +4 dB reaches
  `watch`, +12 dB is firmly `anomalous`.
- A new tonal component, a spectral tilt, or a change in noise character still
  register strongly, because those change the *shape*.
- Descriptors no longer restate a level change seven times over. A pure gain
  change produces exactly one descriptor: "level".
- The profile remains level-sensitive, which is the point of capturing with AGC
  off — just no longer absurdly so.

## Alternatives considered

- **Full covariance.** Correct in principle, but needs far more data than one
  learning session provides, and the app specs call for diagonal with shrinkage.
- **Dropping level entirely.** Rejected: a machine getting louder is one of the
  signals SteadyHum exists to catch.
