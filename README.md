# PSST: Persian Speech-to-Script Test Benchmark

PSST v1 is a reproducible **FLEURS Persian leaderboard** for eight pinned automatic
speech-recognition checkpoints. It evaluates the 871-row `test` split of the Persian `fa_ir`
configuration with a fixed text normalizer and corpus-level WER/CER. It is not a definitive
measure of Persian ASR: FLEURS is public read speech, may overlap model training data, and does
not represent conversational, noisy, accented, domain-specific, or long-form Persian.

Version 1 deliberately excludes speed, timestamps, diarization, streaming, punctuation quality,
robustness subsets, confidence intervals, parameter tuning, prompts, hotwords, quantization,
offload, compilation, and alternative decoding searches. It publishes static Markdown, JSON,
and CSV—not a web application.

<!-- LEADERBOARD:START -->

## Normalized accuracy — FLEURS Persian test split

| Rank | Model | WER | CER | Word accuracy |
|---:|---|---:|---:|---:|
| 1 | `whisper-large-v3` | 0.1980 | 0.0599 | 80.20% |
| 2 | `whisper-large-v3-turbo` | 0.2041 | 0.0650 | 79.59% |
| 3 | `qwen3-asr-1-7b` | 0.2417 | 0.0892 | 75.83% |
| 4 | `whisper-large-persian-steja` | 0.2648 | 0.0589 | 73.52% |
| 5 | `vibevoice-asr` | 0.2704 | 0.1378 | 72.96% |
| 6 | `qwen3-asr-0-6b` | 0.4803 | 0.2086 | 51.97% |
| 7 | `whisper-persian-paulwalker` | 0.9430 | 1.4341 | 5.70% |

## Accuracy per peak CUDA memory — Jetson AGX Orin 32GB

| Rank | Model | Accuracy / reserved GiB | WER | Peak CUDA reserved GiB |
|---:|---|---:|---:|---:|
| 1 | `whisper-large-v3-turbo` | 45.3785 | 0.2041 | 1.754 |
| 2 | `whisper-large-v3` | 23.2527 | 0.1980 | 3.449 |
| 3 | `whisper-large-persian-steja` | 22.8268 | 0.2648 | 3.221 |
| 4 | `qwen3-asr-0-6b` | 18.6197 | 0.4803 | 2.791 |
| 5 | `qwen3-asr-1-7b` | 14.8179 | 0.2417 | 5.117 |
| 6 | `whisper-persian-paulwalker` | 14.8123 | 0.9430 | 0.385 |
| 7 | `vibevoice-asr` | 3.8305 | 0.2704 | 19.047 |

Peak CUDA memory is unified system/GPU memory and is not directly comparable with process VRAM reported on discrete GPUs.

<!-- LEADERBOARD:END -->

`nvidia-fastconformer-fa` is unranked: its official native-FP32 run exhausted CUDA
memory during the default RNNT decoder CUDA-graph warmup before the first prediction. The
benchmark contract forbids fallback decoding or precision changes, and the OOM bundle retains
the full diagnostics.

The accuracy board sorts normalized WER ascending, then normalized CER ascending, then stable
model ID. CER removes whitespace. The efficiency board uses:

```text
word_accuracy_pct = 100 × max(0, 1 − WER)
memory_efficiency = word_accuracy_pct / peak_cuda_reserved_gib
```

On Jetson, CUDA uses unified system/GPU memory. `torch.cuda.max_memory_reserved()` is measured
from before checkpoint loading through the full batch-size-one test run. The result is not
directly comparable with process VRAM measurements from discrete GPUs.

## Reproduce

Prerequisites are Python 3.12, [`uv`](https://docs.astral.sh/uv/), Docker configured to reach the
Jetson over SSH, and the three prebuilt runtime images described below. The official host alias is
`jetson`; Docker communication uses `ssh://jetson`. Commands are non-interactive.

```bash
uv sync --frozen --all-groups
uv run --frozen psst doctor --host ssh://jetson
uv run --frozen psst dataset prepare --suite fleurs-fa-ir-v1 --host ssh://jetson
uv run --frozen psst model validate --model whisper-large-v3
uv run --frozen psst model validate --model whisper-large-v3 --host ssh://jetson
uv run --frozen psst run --suite fleurs-fa-ir-v1 --model whisper-large-v3 --host ssh://jetson
uv run --frozen psst run --suite fleurs-fa-ir-v1 --model whisper-large-v3 --host ssh://jetson --resume
uv run --frozen psst run-all --suite fleurs-fa-ir-v1 --host ssh://jetson
uv run --frozen psst leaderboard --suite fleurs-fa-ir-v1
```

Set `HF_TOKEN` in the invoking environment for gated/private Hugging Face access and higher Hub
rate limits. Never place tokens in model specifications, commands, logs, or committed files. The
orchestrator passes the token only to network-enabled prefetch containers. Official inference runs
with container networking disabled.

The NVIDIA base image is subject to the [NGC terms of use](https://www.nvidia.com/en-us/data-center/products/ngc/).
It must be available on the Jetson before `doctor` can inspect CUDA. PSST never redistributes model
weights or FLEURS audio.

## Immutable dataset contract

[`suite.json`](suites/fleurs-fa-ir-v1/suite.json) pins `google/fleurs` to revision
`70bb2e84b976b7e960aa89f1c648e09c59f894dd`. The checked
[`manifest.jsonl`](suites/fleurs-fa-ir-v1/manifest.jsonl) has SHA-256
`76e87e96769cd63ce5d5abbc7827563644e55c3ba3fdc4041f4359a67435c061` and records, in canonical
split/index order, each upstream ID, transcript, duration, canonical audio hash, source revision,
and license. It contains all 4,341 `fa_ir` rows:

| Split | Rows | Ranking use |
|---|---:|---|
| train | 3,101 | downloaded, not scored |
| validation | 369 | downloaded, not scored |
| test | 871 | official rankings |

Audio is decoded once into ignored persistent 16-kHz mono PCM-16 WAV storage. Preparation always
checks the materialized data against the committed manifest. The reference is FLEURS’ exact
`transcription` field. A transcript correction, source revision, normalization change, or audio
change creates a new suite such as `fleurs-fa-ir-v2`; published v1 manifests and results are never
edited. Future PSST-owned recordings likewise receive immutable suite IDs and separate
leaderboards.

The `fa-v1` normalizer applies Unicode NFKC, unifies Arabic/Persian letter variants, removes
tatweel and diacritics, converts Persian and Arabic-Indic digits to ASCII, converts ZWNJ,
non-breaking spaces, punctuation, and symbols to spaces, then collapses whitespace. It is applied
identically to references and predictions. Empty normalized references invalidate a suite; empty
predictions are valid errors.

FLEURS content is licensed under CC BY 4.0. PSST’s code and benchmark definitions are Apache-2.0.
See [`NOTICE`](NOTICE) for attribution.

## Pinned models

| Stable ID | Hugging Face checkpoint | Revision | Adapter | Native dtype | License |
|---|---|---|---|---|---|
| `whisper-large-v3` | `openai/whisper-large-v3` | `06f233fe…` | Transformers Whisper | FP16 | Apache-2.0 |
| `whisper-large-v3-turbo` | `openai/whisper-large-v3-turbo` | `41f01f3f…` | Transformers Whisper | FP16 | MIT |
| `vibevoice-asr` | `microsoft/VibeVoice-ASR` | `d0c9efdb…` | VibeVoice | BF16 | MIT |
| `qwen3-asr-0-6b` | `Qwen/Qwen3-ASR-0.6B-hf` | `7f1569a4…` | Transformers Qwen | BF16 | Apache-2.0 |
| `qwen3-asr-1-7b` | `Qwen/Qwen3-ASR-1.7B-hf` | `bcd2b5b7…` | Transformers Qwen | BF16 | Apache-2.0 |
| `nvidia-fastconformer-fa` | `nvidia/stt_fa_fastconformer_hybrid_large` | `249cf5bf…` | NeMo RNNT | FP32 | CC-BY-4.0 |
| `whisper-persian-paulwalker` | `Paulwalker4884/whisper-persian` | `80f96e52…` | Transformers Whisper | FP32 | Apache-2.0 |
| `whisper-large-persian-steja` | `steja/whisper-large-persian` | `4c8e5a01…` | Transformers Whisper | FP16 | Apache-2.0 |

The full immutable revisions and generation policies live in [`models/`](models). Whisper uses
Persian transcription with 444 generated tokens and no timestamp scoring. Whisper's four forced
Persian/transcription start tokens plus those 444 tokens exactly fit its 448-position decoder;
requesting 448 new tokens is rejected by Transformers 5.14.1. Qwen uses its recommended
`apply_transcription_request`, language `fa`, and 256 generated tokens. VibeVoice uses automatic
language detection, BF16/SDPA, deterministic one-beam generation, and 512 generated tokens; only
the concatenated segment text is scored while raw and structured output are retained. NVIDIA uses
the checkpoint’s default RNNT behavior without an external language model.

## Isolated Jetson runtimes

All images extend the JetPack-6.2-compatible NVIDIA PyTorch 25.06 iGPU ARM64 image pinned to
`sha256:90f3c17838fde28d5c7ae2d5bfbc8a4c587d3797767ea96cdd48fe82e3613f3b`. Each Dockerfile copies
`uv` 0.8.14 and installs from its own frozen lock:

- `modern`: Transformers 5.14.1 and PEFT 0.20.0 for Whisper/Qwen.
- `vibevoice`: Microsoft code commit `94da20d…`, Transformers 4.57.6, and the Qwen tokenizer at
  `d1497293…`.
- `nemo`: `nemo_toolkit[asr]` 2.7.3.

Every run uses one CUDA device, batch size 1, checkpoint-native precision, deterministic seeds and
strict deterministic algorithms with a fixed CUDA BLAS workspace, disabled TF32/autotuning,
inference/evaluation mode, manifest order, and no fallback. An OOM or
cgroup kill retains diagnostics and stays unranked. Persistent named volumes cache pinned Hub
snapshots and canonical audio. Network-enabled preparation/prefetch is separated from offline
official inference.

The NVIDIA Jetson PyTorch wheel omits distributed-training support. Narrow, inference-only runtime
compatibility modules make optional FSDP/DDP availability checks return false and bypass eager
training imports; they do not emulate collectives or alter single-device model computation.

`doctor` enforces the official profile: Jetson AGX Orin 32GB, Ubuntu 22.04, L4T R36.4.7 / JetPack
6.2, CUDA 12.6, NVIDIA runtime, MAXN, adequate storage, and no competing NVIDIA containers. The
profile is captured in every run bundle.

## Results and contributions

Each official result directory contains immutable `run.json`, append-only `predictions.jsonl`, and
JSONL logs. Bundles record suite/model digests, PSST revision, image digest, dependencies,
CUDA/PyTorch, hardware state, seed, measured parameters, checkpoint bytes, peak CUDA reserved and
allocated memory, and peak process RSS. Resume validates the original request and append-only
sample sequence and keeps the maximum memory observed across segments.

Only successful, complete, committed 871-sample bundles enter generated leaderboards. Maintainers
must rerun contributed model specs/adapters on `ssh://jetson`; contributor-provided scores are not
official. A new model using an existing adapter normally needs only a declarative JSON spec. New
adapters must include mocked contract tests and a real one-sample Jetson validation before a full
evaluation.

Before proposing a new immutable suite, document provenance and license, pin every source revision,
materialize canonical audio, reject empty normalized references, generate a never-overwritten
manifest, record its digest in the suite spec, and add an independent leaderboard. The maintainer
sealer [`tools/freeze_suite.py`](tools/freeze_suite.py) refuses to overwrite an existing manifest.

Standard JSON logging includes ISO timestamps, levels, and run/model/sample progress context. Full
tracebacks are retained for failures. Credentials and private environment values must never be
logged. See the CI workflow for the exact formatting, lint, type, test, spec, and generated-output
checks required for contributions.

## Citation

```bibtex
@software{jafarnezhad_psst_persian_speech_to_script_test_benchmark,
  author  = {Jafarnezhad, Arman},
  title   = {PSST: Persian Speech-to-Script Test Benchmark},
  year    = {2026},
  url     = {https://github.com/ArmanJR/PSST-Benchmark},
  version = {2026.07.31}
}
```
