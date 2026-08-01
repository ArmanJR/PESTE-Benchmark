# Benchmark contract

This document defines the inputs, execution policy, metrics, artifacts, and publication rules for
the current PESTE benchmark. A result is comparable only when it follows the same suite, model,
normalization, inference, and hardware contracts.

## Dataset contract

The current suite is [`fleurs-fa-ir-v1`](../suites/fleurs-fa-ir-v1/suite.json), derived from the
Persian `fa_ir` configuration of `google/fleurs`.

| Property | Value |
|---|---|
| Repository | `google/fleurs` |
| Revision | `70bb2e84b976b7e960aa89f1c648e09c59f894dd` |
| License | CC BY 4.0 |
| Evaluation split | `test` |
| Manifest | [`manifest.jsonl`](../suites/fleurs-fa-ir-v1/manifest.jsonl) |
| Manifest SHA-256 | `76e87e96769cd63ce5d5abbc7827563644e55c3ba3fdc4041f4359a67435c061` |

| Split | Rows | Benchmark use |
|---|---:|---|
| train | 3,101 | Materialized and verified, not scored |
| validation | 369 | Materialized and verified, not scored |
| test | 871 | Official ranking |

The manifest records the upstream ID and row index, exact transcript, duration, canonical audio
hash and path, source revision, and license. Dataset preparation decodes every recording once into
16-kHz mono PCM-16 WAV and verifies the materialized data against the committed manifest.

The upstream `transcription` field is the scoring reference. Empty normalized references are not
valid suite data. Empty model predictions are valid outputs and are scored as errors.

Published manifests are immutable. A source revision, transcript, audio, split, or normalization
change requires a new suite ID and independent leaderboard.

## Text normalization

The `fa-v1` normalizer is applied identically to references and predictions. It performs:

1. Unicode NFKC normalization;
2. Arabic/Persian letter-variant unification;
3. tatweel and diacritic removal;
4. Persian and Arabic-Indic digit conversion to ASCII;
5. replacement of ZWNJ, non-breaking spaces, punctuation, and symbols with spaces; and
6. whitespace collapse and trimming.

Normalization intentionally removes punctuation from the accuracy measurement. Punctuation
quality is not a separate metric in the current suite. Replacing ZWNJ with a space also makes WER
sensitive to Persian word-segmentation conventions: a model that emits joined compounds can
receive word substitutions and deletions even when its letters match the reference.

### Number-format sensitivity

`fa-v1` canonicalizes digit glyphs but does not equate digits with spoken Persian number words.
The `fleurs-fa-ir-v1` test split has digit glyphs in 153 of 871 references. Consequently, a model
that emits `۱۹۶۷` matches a reference containing `۱۹۶۷` after glyph folding, while a model that
emits the semantically equivalent `هزار و نهصد و شصت و هفت` does not. In the corresponding
FLEURS reference sentence, that representation difference alone produces one substitution and six
insertions under WER. Published v1 scores retain this known formatting bias because their suite and
normalization contracts are immutable.

## Model contract

Every checkpoint has an immutable JSON specification under [`models/`](../models) containing its
Hugging Face repository and revision, adapter, native dtype, declared license, language policy,
generation policy, and runtime image.

The model digest is stored in every run request and result bundle. A result whose model digest no
longer matches the corresponding specification is excluded from generated rankings.

Supported model proposals must follow an existing adapter contract. See
[Adding a model](adding-a-model.md). New inference interfaces require maintainer-owned adapter and
runtime work; see the [maintainer guide](maintainer-guide.md).

## Execution contract

Official runs use:

- one CUDA device and batch size 1;
- checkpoint-native benchmark precision;
- canonical manifest order;
- fixed Python, NumPy, PyTorch, and CUDA seeds;
- strict PyTorch deterministic algorithms;
- a fixed CUDA BLAS workspace configuration;
- disabled TF32 and cuDNN autotuning;
- inference/evaluation mode; and
- no fallback decoder, precision, quantization, offload, or compilation.

Dataset and checkpoint acquisition run in network-enabled preparation containers. Official
inference containers have networking disabled and mount dataset/model caches read-only.

Framework families run in separate images with frozen dependencies:

| Runtime | Use |
|---|---|
| `modern` | Transformers 5.14.1 Whisper, Qwen, and standard greedy CTC adapters |
| `nemo` | NeMo ASR and the default RNNT adapter |

Runtime Dockerfiles and lockfiles under [`runtimes/`](../runtimes) are authoritative.

The `modern` runtime's `transformers-ctc` path uses the standard `AutoProcessor` and
`AutoModelForCTC` interfaces with fixed batch-size-one greedy decoding. It groups repeated CTC
tokens, preserves special tokens including `<unk>`, and does not use beam search or an external
language model. The reference specification is
[`wav2vec2-large-xlsr-53-persian.json`](../models/wav2vec2-large-xlsr-53-persian.json).

## Hardware contract

The official profile is:

- Jetson AGX Orin 32GB;
- Ubuntu 22.04;
- JetPack 6.2 / L4T R36.4.7;
- host CUDA 12.6;
- NVIDIA container runtime;
- MAXN power mode;
- at least 60 GiB free cache storage; and
- no competing NVIDIA containers or detected host GPU processes.

`peste doctor` enforces this profile before smoke tests and full runs. The captured profile, runtime
image digest, Python/PyTorch/CUDA versions, installed dependencies, seed, and source revision are
stored in the result bundle.

Jetson CUDA memory is unified system/GPU memory. Peak CUDA values therefore should not be compared
directly with process VRAM measurements from discrete GPUs.

## Metrics

WER and CER are corpus-level Levenshtein error rates:

```text
WER = total word substitutions + deletions + insertions
      -------------------------------------------------
                   total reference words

CER = total character substitutions + deletions + insertions
      ------------------------------------------------------
                 total reference characters
```

CER removes normalized whitespace before counting characters. Word accuracy and memory efficiency
are derived values:

```text
word_accuracy_pct = 100 × max(0, 1 − WER)
memory_efficiency = word_accuracy_pct / peak_cuda_reserved_gib
```

The accuracy board sorts by CER ascending, WER ascending, then stable model ID. CER is primary
because it ignores normalized whitespace and is therefore robust to Persian ZWNJ/word-segmentation
variation. WER remains a complementary measure of word-level transcription and orthographic
segmentation. The memory board sorts by memory efficiency descending, WER ascending, then model ID;
because memory efficiency derives from word accuracy, that secondary board remains
segmentation-sensitive.

### Resource measurement

Peak CUDA reserved and allocated memory are reset before checkpoint loading and measured through
the complete run. Peak process RSS is also retained. Resumed runs carry forward the highest
measurements from earlier segments.

### Statistical uncertainty

Published CER and WER include deterministic 95% percentile-bootstrap intervals. Each replicate
resamples the 871 test utterances with replacement and recomputes the corpus error rate from the
resampled edit-count and reference-unit totals. The generator uses 10,000 replicates and seed
`20250731`.

The numeric leaderboard order is based on point estimates; it is not a claim that every adjacent
model differs significantly. To evaluate those gaps, the generator also publishes paired
utterance-bootstrap intervals for the CER difference between each adjacent pair in point-estimate
order. A difference is marked resolved at 95% only when its interval excludes zero. These are
unadjusted pointwise intervals, not simultaneous family-wise guarantees. They quantify sampling
uncertainty over this test set and do not capture model-training variation, speaker clustering,
training-data overlap, dataset bias, or deployment-domain shift.

### ZWNJ-policy sensitivity

In the current test manifest, 622 of 871 references (71.4%) contain ZWNJ. As a sensitivity check,
removing ZWNJ instead of converting it to a space changes corpus WER materially and changes the
model order. For example, `whisper-large-v3` changes from 0.1980 to 0.2882 WER, while
`whisper-large-persian-steja` changes from 0.2648 to 0.1917 and moves from fourth to first under
that alternative. These alternative values are diagnostic only; official `fleurs-fa-ir-v1`
scores retain the immutable `fa-v1` policy.

## Result bundles

An official result directory contains:

| Artifact | Purpose |
|---|---|
| `request.json` | Immutable run request and digests |
| `run.json` | Status, aggregates, environment, memory, and artifact references |
| `predictions.jsonl` | Append-only per-sample references, predictions, normalized text, and edit counts |
| `runner.jsonl` | Structured runner log |
| `container.jsonl` | Captured container output |
| `diagnostics.json` | External failure diagnostics when a container cannot produce a bundle |

Resume validates the original request, append-only prediction count, and sample sequence. It is
available only for failed or killed runs, not OOM runs.

## Ranking eligibility

A result enters a generated leaderboard only when:

- `run.json` is tracked as an intended official result;
- the suite ID and digest match the current suite specification;
- the model digest matches the current model specification;
- status is `success` and aggregate metrics are present;
- the prediction count equals the complete evaluation split; and
- the recorded source revision is not `uncommitted`.

The generator rejects multiple successful official runs for the same model ID. Failed, killed,
incomplete, stale, and OOM bundles remain available for audit but are not ranked.

Maintainers regenerate Markdown, SVG, JSON, CSV, and the README leaderboard block from eligible
bundles. Generated artifacts are never the source of benchmark metrics.
