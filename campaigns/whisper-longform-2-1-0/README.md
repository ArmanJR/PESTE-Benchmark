# Whisper long-form 2.1.0 campaign

This campaign recalibrates and reruns all 29 `transformers-whisper` models after correcting the
30-second input truncation in PESTE 2.1.0. Candidate repository revisions are unchanged from the
published 2.0 specifications.

Qualification must use the provisional batch-size-one model specifications and a digest-pinned
2.1.0 calibration carrier. After reviewing and applying every profile, rebuild the carrier from
the exact code-plus-profiles commit and use a fresh doctor-approved host for official runs.
Qualification evidence must include exact-dimension, monotonic singleton conformance; evidence
from the earlier permissive guard must not be applied.

All 29 candidates qualified on the official RTX 6000 Ada profile using source revision
`6e653b817820c981fc2e1183f547d5dc726e3019` and calibration image digest
`sha256:313f66838173fe4b95c301831e7cec08dfbe2caab0d13b816144d1cc3df08172`.
The reviewed profiles select batch size 1 for 24 models and batch size 2 for five models. No model
selects a candidate after a smaller batch diverges from singleton output.

Old Whisper bundles remain immutable audit evidence but become stale through their model digests.
The 13 unaffected non-Whisper 2.0 bundles remain current.
