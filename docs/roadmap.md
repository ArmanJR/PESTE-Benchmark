# Version roadmap

## Version 1 — completed

- [x] Pin the FLEURS Persian dataset revision and seal the 4,341-row manifest.
- [x] Publish the 871-recording `fa-v1` accuracy benchmark and uncertainty analysis.
- [x] Preserve the release and retired results at git tag `1.0.0`.

## Version 2 — current release

- [x] Rebase the hardware contract on one NVIDIA RTX 6000 Ada Generation 48 GB GPU.
- [x] Replace singleton adapter calls with deterministic per-model native batching.
- [x] Replace resource ranking with steady-state throughput and RTF.
- [x] Add batch calibration, timing journals, and resumed-speed invalidation.
- [x] Add the Vast.ai VM reference acquisition/build lifecycle.
- [x] Move persisted specifications, manifests, requests, predictions, bundles, and generated JSON
  to schema 2.
- [x] Preserve `fleurs-fa-ir-v1`, `fa-v1`, checkpoint revisions, accuracy metrics, uncertainty, and
  paired comparisons.
- [ ] Calibrate all eight model batch sizes on an official host.
- [ ] Produce one fresh uninterrupted schema-2 bundle for every model.
- [ ] Publish the complete RTX accuracy and speed boards.

## Version 3 — `fa-v2` normalization milestone

### Implemented groundwork

- [x] Add and register an experimental `fa-v2` normalizer.
- [x] Preserve the `fa-v1` Unicode, letter, diacritic, punctuation, and whitespace rules.
- [x] Spell integers and comma-grouped integers below 10^18 in Persian.
- [x] Normalize decimal numbers.
- [x] Normalize clock-like `HH:MM` expressions and remaining colon-separated ratios.
- [x] Normalize slash fractions with denominators from 2 through 10.
- [x] Normalize the vulgar fractions `¼`, `½`, and `¾`.
- [x] Retain digit sequences at or above 10^18 as one ASCII-canonical token and emit a structured
  warning.
- [x] Make the runner score with the normalization version declared by its suite.
- [x] Add a `fleurs-fa-ir-v2` suite declaration that reuses the pinned v1 manifest.
- [x] Add regression tests for normalization, metric equivalence, manifest reuse, and runner
  propagation.

### Remaining work

- [ ] Review Persian number spellings with appropriate language expertise.
- [ ] Decide how dotted forms distinguish decimals, grouped thousands, and clock notation.
- [ ] Decide how colon forms distinguish times from ratios outside the covered patterns.
- [ ] Decide how hyphenated scores, ranges, and year spans should be normalized.
- [ ] Review the oversized-number fallback and supported scale limit.
- [ ] Confirm that manifest reuse is the intended authoritative v2 suite contract.
- [ ] Define an auditable migration or rescoring policy for existing raw predictions.
- [ ] Run or rescore every supported model under the finalized v2 contract.
- [ ] Publish independent v2 result bundles and generated leaderboard artifacts.
- [ ] Update the active README, benchmark contract, maintainer guide, and reproduction commands.

## Notes

- Published suite, normalization, model, hardware-profile, and result contracts are immutable.
- Scores from different normalization or hardware-profile versions must not share a leaderboard.
- Hyphen-separated numbers are currently spelled independently without an inferred connector.
- The experimental `fleurs-fa-ir-v2` declaration changes normalization only and is not the active
  release suite.
