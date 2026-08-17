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

Official runs completed successfully for all 29 candidates on 2026-08-17. Every run processed all
871 test recordings in one uninterrupted attempt, published valid speed measurements, and used
source revision `e7833ac24e5e2db15a66ae267066bff576d872c5` with image digest
`sha256:7c81292cfd47fe2ed956bf5b81f8cef5d78d1e9558299650fd9d883973bd14ef` on GPU
`GPU-384be5d9-429b-1084-226b-7ba0453860f4`. The test split contains 13 recordings longer than
30 seconds, up to 39.24 seconds; every official bundle includes all 13 and records automatic
Whisper long-form activation.

The regenerated publication boards contain exactly 42 models: the 29 refreshed Whisper bundles
and the 13 unchanged non-Whisper 2.0 bundles. Generated JSON and CSV rows expose the model digest,
runtime image tag and digest, and source revision needed to audit this mixed provenance.
