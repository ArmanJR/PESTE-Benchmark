# Benchmark contract

This document defines PESTE 2.0.0 inputs, execution policy, metrics, artifacts, and ranking rules.
A result is comparable only when its suite, normalization, model, speed profile, runtime, and
hardware contracts match.

## Dataset and normalization

The active suite is [`fleurs-fa-ir-v1`](../suites/fleurs-fa-ir-v1/suite.json), derived from the
Persian `fa_ir` configuration of `google/fleurs` at revision
`70bb2e84b976b7e960aa89f1c648e09c59f894dd` under CC BY 4.0.

| Split | Rows | Use |
|---|---:|---|
| train | 3,101 | Materialized and verified |
| validation | 369 | Materialized and verified |
| test | 871 | Official evaluation |

The schema-2 manifest has SHA-256
`031a6871898bfb6b994164da49739ca6d50b79af91c7af59e6883ae820520488`. Its only change from the
historical manifest is the `schema_version` field; audio references, hashes, transcripts, source
identity, durations, and splits are unchanged.

Audio is canonical 16-kHz mono PCM-16 WAV. Preparation verifies every materialized recording
against the immutable manifest. The `fa-v1` normalizer remains unchanged: NFKC, Arabic/Persian
letter folding, tatweel and diacritic removal, digit-glyph folding, ZWNJ/punctuation/symbol
replacement with spaces, then whitespace collapse. Empty normalized references are invalid; empty
predictions are scored.

CER removes normalized whitespace before character edits. WER retains word boundaries. The known
digit-versus-spoken-number and Persian spacing sensitivities remain part of `fa-v1`.

## Model and batching contract

Each model JSON pins repository revision, adapter, native dtype, language, generation policy,
runtime image, and `speed_profile`. A speed profile contains:

- `hardware_profile_id = rtx-6000-ada-v1`;
- one positive, calibrated `batch_size`.

The suffix in `rtx-6000-ada-v1` is the revision of the hardware profile, independent of the PESTE
major version. Both speed fields contribute to the model digest. Official runs use the committed
batch size exactly and never retune, reduce it after OOM, or substitute singleton loops.

Every adapter accepts multiple canonical audio paths and returns exactly one transcription per
input in the same order. Whisper and CTC use padded processor batches, Qwen uses its native batched
transcription request, and NeMo calls `model.transcribe(paths, batch_size=n)`. Output cardinality
violations are hard failures.

Checkpoint-native precision, language, decoder, token handling, and generation limits remain
fixed. Quantization, offload, compilation, external language models, and fallback behavior are
prohibited.

## Hardware profile

The authoritative profile is
[`hardware/rtx-6000-ada-v1.json`](../hardware/rtx-6000-ada-v1.json):

- exactly one full physical `NVIDIA RTX 6000 Ada Generation` with 48 GB VRAM;
- x86-64 host, at least 8 vCPUs and 64 GiB RAM;
- at least 100 GiB free local storage;
- NVIDIA driver major 580 or newer;
- ECC exactly `Disabled`;
- power limit and board maximum both exactly 300 W;
- no active performance-limiting clock event (NVML's `0x1` GPU-idle bit is expected and allowed
  while the doctor is probing an otherwise idle device);
- no competing process on the assigned GPU; and
- exactly one matching numbered NVIDIA GPU device visible in the carrier container.

CPU model and GPU UUID are recorded but not pinned. The exact driver, container OS, host kernel,
image digest, and source revision are recorded. Timed execution occurs in the public PESTE carrier
image selected by immutable GHCR digest and built from the pinned x86-64 NGC PyTorch base.

`peste doctor` is authoritative. Marketplace filters only preselect plausible hosts. Vast.ai is
the reference acquisition path, but any compatible SSH-accessible carrier container passing the
same doctor is acceptable. The reference search requires one ordinary container GPU, verified
hosting, high reliability, driver `>=580.0.0`, sufficient CPU/RAM/disk, and 300 W capability. It
requests 200 GB so the doctor can still require 100 GiB free after the carrier image is present.

## Determinism and isolation

Official inference uses one CUDA device, fixed Python/NumPy/PyTorch/CUDA seeds, deterministic
PyTorch algorithms, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, TF32 disabled, cuDNN autotuning disabled,
and model evaluation/inference mode. Preparation and prefetch run as root; their cache trees are
then root-owned and timed work runs as an unprivileged user that cannot modify them. Network access
is denied for smoke, calibration, and inference by the digest-pinned native socket guard, in
addition to the framework offline variables. The doctor proves both controls before acceptance.

Framework families remain isolated in separate virtual environments within the same digest-pinned
carrier image so all models can run on one accepted host:

| Runtime | Adapters |
|---|---|
| `modern` | Transformers Whisper, Qwen, and greedy CTC |
| `nemo` | NeMo default RNNT |

## Batch calibration

`peste model profile-speed` selects 128 deterministic duration quantiles, including shortest and
longest evaluation recordings. It evaluates candidates `1, 2, 4, 8, 16, 32, 64, 128` with two
warmups and three measured passes each.

A candidate is rejected when it OOMs, exceeds 85% of total VRAM on the longest-duration stress
batch, violates output cardinality/order, or differs from singleton normalized output on a fixed
16-recording conformance set. Among safe candidates, the smallest reaching at least 95% of the
best safe throughput is selected. Memory telemetry exists only during calibration and is not a
published benchmark metric.

## Official timing protocol

1. Validate every evaluation audio path and read the audio into the OS cache.
2. Load the model outside the timed region.
3. Sort evaluation work by `(duration_seconds, original_sequence)`.
4. Run two excluded warmup batches at representative median duration.
5. Make one complete measured pass over all 871 test recordings.
6. Call `torch.cuda.synchronize()` immediately before starting and immediately after returning
   from every timed adapter call.
7. Restore original manifest order in `predictions.jsonl`.

Timed work includes audio loading, preprocessing, transfers, inference/generation, decoding, and
postprocessing. Model load, audio priming, warmup, scoring, journal writes, logging, and
orchestration are excluded.

```text
audio_throughput_x = total_audio_seconds / processing_seconds
RTF                = processing_seconds / total_audio_seconds
```

Both values and their source times are stored with a reciprocity invariant. The speed board sorts
throughput descending, then CER, WER, and model ID. The accuracy board sorts CER, WER, and model
ID. The derived Pareto board includes speed-valid models and minimizes CER while maximizing
throughput. Model A point-dominates model B when A has equal-or-lower CER and equal-or-higher
throughput, with at least one strict advantage. A dominance edge is statistically supported only
when the paired 95% CER-difference interval for A minus B remains below zero. CER confidence bars
measure test-set sampling uncertainty; the single-run speed coordinate has no confidence interval.
Pareto efficiency is a classification, not a total ranking, and no composite score is defined.
The Pareto SVG inverts the logarithmic CER axis so improving directions are visually up and right;
its tick labels remain the untransformed CER values. Confidence bars are hidden by default and can
be toggled in interactive SVG rendering. The displayed axis ends at CER 1; any worse point is
labeled at the lower boundary as off-scale without changing its stored or tabulated value.

## Resume and artifacts

Each completed measured batch is appended atomically to `timing.jsonl`, including original
sequences, sample IDs, audio duration, processing duration, and prediction records. Recovery
requires a contiguous prefix matching the deterministic batch plan. `predictions.jsonl` is
materialized from the journal in original manifest order.

A resumed successful run retains predictions and accuracy, but `speed.valid` is false with an
explicit reason. It is excluded only from the speed board. Canonical speed publication requires a
fresh uninterrupted run.

An official directory contains:

| Artifact | Purpose |
|---|---|
| `request.json` | Immutable schema-2 request and spec digests |
| `run.json` | Status, environment, `speed`, `model_facts`, aggregates, references |
| `timing.jsonl` | Append-only measured-batch journal and timing artifact |
| `predictions.jsonl` | Manifest-ordered text and edit counts |
| `runner.jsonl` | Structured runner log |
| `container.jsonl` | Captured container log |
| `diagnostics.json` | External failure details, when needed |

`model_facts` contains native dtype, parameter count, and checkpoint bytes. Environment data
records the GPU product, driver, ECC, power limit, CPU model, GPU UUID, dependency/runtime
versions, source revision, and optional cloud instance/host provenance.

## Accuracy uncertainty

CER and WER retain deterministic 95% utterance-level percentile-bootstrap intervals with 10,000
replicates and seed `20250731`. Adjacent accuracy rows retain paired CER-difference intervals.
These intervals measure test-set sampling uncertainty only; they do not cover training variation,
speaker clustering, data overlap, domain shift, or multiplicity.

## Ranking eligibility

Accuracy ranking requires a tracked, successful, complete schema-2 bundle with matching suite and
model digests and a committed source revision. Speed ranking additionally requires `speed.valid =
true`. Multiple successful official bundles for one model ID are rejected. Stale, incomplete,
failed, killed, and OOM bundles remain auditable but unranked.

Markdown, SVG, JSON, CSV, and the README marker block are deterministic derived artifacts. They
are never authoritative metric sources.
