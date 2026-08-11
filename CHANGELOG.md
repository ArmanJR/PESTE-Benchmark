# Changelog

## 2.0.0

Breaking major release.

- Rebased the official hardware contract on one NVIDIA RTX 6000 Ada Generation 48 GB GPU with
  driver `580.142`, 300 W power/board limits, and ECC disabled.
- Replaced batch-size-one adapter inference with deterministic, real padded/native batching and a
  committed per-model speed profile.
- Replaced the memory-efficiency board with steady-state end-to-end audio throughput and RTF.
- Added deterministic batch calibration, two excluded warmups, exact CUDA timer boundaries,
  append-only batch timing journals, and invalid speed for resumed runs.
- Added Vast.ai VM provisioning, lifecycle, bootstrap, doctor retry/destruction, and remote Docker
  image builds as the reference acquisition path. The doctor, not the marketplace, defines host
  compatibility.
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

## 1.0.0

The historical release used a Jetson AGX Orin 32 GB host, batch-size-one inference, and a
memory-efficiency leaderboard. Its results are retired, remain preserved at git tag `1.0.0`, and
are not comparable with v2.
