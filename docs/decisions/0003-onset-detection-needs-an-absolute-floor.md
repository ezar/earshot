# 0003 — Onset detection needs an absolute floor, and periodicity needs phase coherence

**Status:** accepted

## Context

Two problems surfaced while testing spectral flux onsets against synthetic
signals whose ground truth is exact.

**A steady tone produced onsets.** Spectral flux differences log-compressed
magnitudes. With an epsilon of `1e-9`, bins holding nothing but numerical noise
swung across orders of magnitude in the log domain, and the adaptive median
threshold — which is purely relative — happily called those swings onsets. A
perfectly steady 440 Hz tone yielded 52 onsets over four seconds.

**Scattered noise looked rhythmic.** Periodicity was `1 - CV` over inter-onset
intervals. The detector's own minimum spacing makes random intervals look more
even than they are, so pink noise scored 0.65 — close to what genuinely even
knocks should score.

## Decision

1. The log compression floors magnitudes at `1e-6`, which is -120 dBFS on the
   normalized scale `computeSpectrogram` produces. Bins below that all compress
   to the same value instead of fluctuating.
2. `detectOnsets` takes a `minStrength` floor, default 0.05 — roughly 1 dB of
   mean per-bin change. Because the curve is a *log* difference, this floor is
   invariant to the recording's gain, so it does not quietly become a level gate.
3. `onsetPeriodicity` reports the resultant length of the onset times wrapped
   onto a candidate period — the standard circular-statistics measure of phase
   coherence — searching ±40 % around the median interval.

## Consequences

- A steady tone now yields zero onsets; evenly spaced knocks yield one per knock
  and score 1.0 with the correct period; pink noise scores 0.37.
- Onset detection is verifiably gain invariant: the same onsets are found in a
  signal and in a copy 30 dB quieter.
- A sequence that misses the occasional onset still resolves to the underlying
  period rather than to twice it.
