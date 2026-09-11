# 0005 — K selection needs an absolute silhouette floor

**Status:** accepted

## Context

The silhouette coefficient is undefined for a single cluster. Any implementation
that compares K = 1 against K > 1 has to invent a value for K = 1, and whatever
it invents, silhouette will happily split a structureless cloud: a uniform
two-dimensional blob scores 0.41 at K = 2 and 0.47 at K = 4.

With K = 1 treated as scoring 0 and only a relative margin to beat, a machine
with one steady sound was given four states. Every extra state dilutes the
evidence behind that state's distance distribution, which is what scoring
depends on.

## Decision

`selectClustering` takes `minSilhouette`, default 0.6: a candidate `K > 1` must
clear that absolute floor before the relative margin is even considered.

The floor is calibrated on the gap that actually exists in the data. Measured
across synthetic sets: a structureless cloud peaks at 0.47 across every K, while
two well-separated groups score 0.97 and three score 0.96.

## Consequences

- A single-mode machine gets one state; a machine with two genuinely distinct
  operating modes gets two.
- Modes that overlap heavily are merged into one state rather than split. That is
  the conservative direction: a merged state has a wider distance distribution
  and raises fewer false alarms than two thin ones.
- `minClusterFraction` (default 8 %) still applies independently, so a handful of
  outliers cannot become a state either.
