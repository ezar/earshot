# Evaluation datasets

earshot's evaluation targets come from the app specs, and are measured on public
datasets rather than on the synthetic fixtures the unit tests use. The datasets
are **not** vendored: `pnpm eval:fetch` downloads them into `.eval-data/`, which
is git-ignored.

Read and accept each licence before downloading. earshot ships no dataset audio
and no derived model weights.

## MIMII — Malfunctioning Industrial Machine Investigation and Inspection

- Normal and anomalous recordings of fans, pumps, valves and slide rails at
  several signal-to-noise ratios.
- Used for: anomaly detection AUC on `fan`, `pump` and `valve`.
- Licence: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).
- Source: <https://zenodo.org/record/3384388>
- Citation: Purohit et al., *MIMII Dataset: Sound Dataset for Malfunctioning
  Industrial Machine Investigation and Inspection*, DCASE 2019.

## ToyADMOS — DCASE 2020 Task 2 development set

- Miniature machine recordings (ToyCar, ToyConveyor) with normal and anomalous
  conditions, used alongside MIMII in the DCASE 2020 Task 2 benchmark.
- Used for: anomaly detection AUC, as a second family of machines.
- Licence: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0).
  **Non-commercial**: use it for evaluating earshot, not inside a shipped app.
- Source: <https://zenodo.org/record/3678171>
- Citation: Koizumi et al., *ToyADMOS: A Dataset of Miniature-Machine Operating
  Sounds for Anomalous Sound Detection*, WASPAA 2019.

## CatMeows

- 440 meows from 21 cats in three contexts (brushing, isolation, waiting for
  food), each labelled with the cat's identity, breed, sex and context.
- Used for: meow detection recall, pitch sanity, and pair identity accuracy.
- Licence: Creative Commons Attribution 4.0 International (CC BY 4.0).
- Source: <https://zenodo.org/record/4008297>
- Citation: Ludovico et al., *CatMeows: A Publicly-Available Dataset of Cat
  Vocalizations*, MMM 2021.

## Targets

Inherited from the app specs and recorded in `docs/eval-results.md`:

| Metric | Dataset | Target |
| --- | --- | --- |
| Anomaly AUC, fan | MIMII | > 0.80 |
| Anomaly AUC, pump | MIMII | > 0.80 |
| Anomaly AUC, valve | MIMII | > 0.80 |
| Meow detection recall | CatMeows | > 90 % |
| Pair identity accuracy, 10 examples per cat | CatMeows | > 80 % |
