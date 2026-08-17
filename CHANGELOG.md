# Changelog

## 2.1.0

- Corrected the Transformers Whisper adapter so recordings longer than 30 seconds are no longer
  truncated during feature extraction.
- Added automatic native sequential long-form decoding: ordinary short-form decoding remains in
  use through the model's segment limit, while longer batches use internal timestamp tokens and
  still publish text-only transcriptions.
- Strengthened legacy Whisper generation metadata, long-recording smoke coverage, semantic model
  policy, and generated row-level runtime provenance.
- Hardened batch calibration so large candidates exercise their exact batch dimension and cannot
  be selected after a smaller candidate demonstrates batch-sensitive output.
- Added the exact 29-model `whisper-longform-2-1-0` recalibration and result-refresh campaign.
- Qualified all 29 Whisper candidates on the official RTX 6000 Ada profile; exact-dimension,
  monotonic calibration selected batch size 1 for 24 models and batch size 2 for five models.
- Preserved schema 2, `fleurs-fa-ir-v1`, `fa-v1`, metrics, checkpoint revisions, dependency pins,
  and the 13 unaffected non-Whisper 2.0 result identities.

## 2.0.0

Breaking major release.

- Rebased the official hardware contract on one NVIDIA RTX 6000 Ada Generation 48 GB GPU with
  driver major 580 or newer, 300 W power/board limits, and ECC disabled.
- Replaced batch-size-one adapter inference with deterministic, real padded/native batching and a
  committed per-model speed profile.
- Replaced the memory-efficiency board with steady-state end-to-end audio throughput and RTF.
- Added deterministic batch calibration, two excluded warmups, exact CUDA timer boundaries,
  append-only batch timing journals, and invalid speed for resumed runs.
- Added ordinary Vast.ai container provisioning, lifecycle, doctor retry/destruction, direct SSH
  execution, and a public digest-pinned GHCR carrier image as the reference acquisition path. The
  doctor, not the marketplace, defines host compatibility.
- Moved all persisted contracts to schema 2. Schema-1 JSON is rejected with an explicit
  unsupported-version error.
- Changed only the `schema_version` field in all 4,341 manifest rows. Audio references, hashes,
  transcripts, durations, upstream identity, and splits are byte-for-byte unchanged; the manifest
  and suite digests necessarily changed.
- Preserved `fleurs-fa-ir-v1`, `fa-v1`, checkpoint revisions, native precision, decoding policy,
  determinism, CER/WER, bootstrap intervals, paired CER comparisons, offline inference, and
  read-only caches.
- Retired every v1 result bundle instead of migrating it. All v2 result bundles must be generated
  fresh on the new contract.
- Qualified 37 additional model-only candidates, rejected three padding-unsafe CTC processors,
  published 33 new complete official bundles, and expanded the accuracy, speed, and Pareto boards
  from 8 to 41 models. The Shenava Rizeh full-run failure remains auditable but unranked.

## 1.0.0

The historical release used a Jetson AGX Orin 32 GB host, batch-size-one inference, and a
memory-efficiency leaderboard. Its results are retired, remain preserved at git tag `1.0.0`, and
are not comparable with v2.
