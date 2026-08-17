# Whisper long-form 2.1.0 campaign

This campaign recalibrates and reruns all 29 `transformers-whisper` models after correcting the
30-second input truncation in PESTE 2.1.0. Candidate repository revisions are unchanged from the
published 2.0 specifications.

Qualification must use the provisional batch-size-one model specifications and a digest-pinned
2.1.0 calibration carrier. After reviewing and applying every profile, rebuild the carrier from
the exact code-plus-profiles commit and use a fresh doctor-approved host for official runs.
Qualification evidence must include exact-dimension, monotonic singleton conformance; evidence
from the earlier permissive guard must not be applied.

Old Whisper bundles remain immutable audit evidence but become stale through their model digests.
The 13 unaffected non-Whisper 2.0 bundles remain current.
