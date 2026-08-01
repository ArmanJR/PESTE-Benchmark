# Version roadmap

## Version 1

- [x] Pin the `google/fleurs` Persian `fa_ir` dataset revision.
- [x] Seal the 4,341-row manifest and 871-recording evaluation split.
- [x] Implement and register the immutable `fa-v1` normalization policy.
- [x] Run the supported models on the official Jetson hardware profile.
- [x] Publish CER, WER, uncertainty intervals, resource measurements, predictions, and logs.
- [x] Rank accuracy primarily by CER to reduce Persian spacing sensitivity.
- [x] Document ZWNJ-policy sensitivity.
- [x] Document the known digit-versus-spelled-number formatting bias.

## Version 2

### Implemented groundwork

- [x] Add and register an experimental `fa-v2` normalizer.
- [x] Preserve the `fa-v1` Unicode, letter, diacritic, punctuation, and whitespace rules.
- [x] Spell integers and comma-grouped integers below 10^18 in Persian.
- [x] Normalize decimal numbers.
- [x] Normalize clock-like `HH:MM` expressions and remaining colon-separated ratios.
- [x] Normalize slash fractions with denominators from 2 through 10.
- [x] Normalize the vulgar fractions `¼`, `½`, and `¾`.
- [x] Retain digit sequences at or above 10^18 as one ASCII-canonical token and emit a
  structured warning.
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

- Published suite, normalization, and result contracts are immutable.
- Scores from different normalization versions must not be mixed in one leaderboard.
- Hyphen-separated numbers are currently spelled independently without an inferred connector.
- The existing v2 suite declaration points to the exact v1 manifest and changes only the
  normalization version.
