<p align="center">
  <img src="docs/peste-logo.png" alt="PESTE logo" width="520">
</p>

# PESTE: Persian Speech to Text benchmark

PESTE (**PE**rsian **S**peech to **TE**xt) is a reproducible benchmark and leaderboard for Persian automatic speech recognition (ASR).

## At a glance

- **Release:** `v1`
- **Suite:** [`fleurs-fa-ir-v1`](suites/fleurs-fa-ir-v1/suite.json)
- **Dataset:** [`google/fleurs`](https://huggingface.co/datasets/google/fleurs), Persian `fa_ir`
  configuration
- **Evaluation set:** `test` split, 871 recordings
- **Accuracy metrics:** Corpus-level CER (primary) and WER
- **Efficiency metric:** Word accuracy per peak CUDA reserved GiB
- **Official hardware:** Jetson AGX Orin 32GB, JetPack 6.2 / L4T R36.4.7, host CUDA 12.6, MAXN
- **Inference policy:** One CUDA device, batch size 1, checkpoint-native precision, deterministic
  decoding

## Leaderboard

<!-- LEADERBOARD:START -->

### Normalized accuracy

![Normalized accuracy leaderboard](generated/leaderboard-accuracy.svg)

| Order | Model | CER | WER | Word accuracy |
|---:|---|---:|---:|---:|
| 1 | [`whisper-large-persian-steja`](https://huggingface.co/steja/whisper-large-persian) | 0.0589<br><sub>95% CI: 0.0542–0.0640</sub> | 0.2648<br><sub>95% CI: 0.2560–0.2737</sub> | 73.52% |
| 2 | [`whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3) | 0.0599<br><sub>95% CI: 0.0552–0.0652</sub> | 0.1980<br><sub>95% CI: 0.1897–0.2064</sub> | 80.20% |
| 3 | [`whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo) | 0.0650<br><sub>95% CI: 0.0576–0.0740</sub> | 0.2041<br><sub>95% CI: 0.1949–0.2135</sub> | 79.59% |
| 4 | [`qwen3-asr-1-7b`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) | 0.0892<br><sub>95% CI: 0.0844–0.0942</sub> | 0.2417<br><sub>95% CI: 0.2332–0.2505</sub> | 75.83% |
| 5 | [`vibevoice-asr`](https://huggingface.co/microsoft/VibeVoice-ASR) | 0.1378<br><sub>95% CI: 0.1266–0.1503</sub> | 0.2704<br><sub>95% CI: 0.2588–0.2823</sub> | 72.96% |
| 6 | [`qwen3-asr-0-6b`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) | 0.2086<br><sub>95% CI: 0.2013–0.2162</sub> | 0.4803<br><sub>95% CI: 0.4696–0.4907</sub> | 51.97% |
| 7 | [`whisper-persian-paulwalker`](https://huggingface.co/Paulwalker4884/whisper-persian) | 1.4341<br><sub>95% CI: 1.3055–1.5698</sub> | 0.9430<br><sub>95% CI: 0.8961–1.0015</sub> | 5.70% |

CER is the primary ranking metric because Persian WER is orthography-sensitive: fa-v1 converts ZWNJ to spaces, while CER ignores normalized whitespace. WER and derived word accuracy remain complementary, segmentation-sensitive measurements.

Point-estimate order does not establish statistical significance. Intervals use a deterministic 10,000-replicate utterance-level percentile bootstrap at 95% confidence with seed 20250731. Paired intervals containing zero are reported as no clear difference; these intervals measure test-set sampling uncertainty only.

#### Paired adjacent CER comparisons

| Adjacent models | ΔCER | Paired 95% range | Evidence |
|---|---:|---:|---|
| [`whisper-large-persian-steja`](https://huggingface.co/steja/whisper-large-persian) − [`whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3) | −0.11 pp | −0.53 to 0.29 pp | No clear difference |
| [`whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3) − [`whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo) | −0.50 pp | −1.20 to 0.02 pp | No clear difference |
| [`whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo) − [`qwen3-asr-1-7b`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) | −2.42 pp | −3.12 to −1.58 pp | First model has lower CER |
| [`qwen3-asr-1-7b`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) − [`vibevoice-asr`](https://huggingface.co/microsoft/VibeVoice-ASR) | −4.86 pp | −6.01 to −3.82 pp | First model has lower CER |
| [`vibevoice-asr`](https://huggingface.co/microsoft/VibeVoice-ASR) − [`qwen3-asr-0-6b`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) | −7.08 pp | −8.17 to −5.84 pp | First model has lower CER |
| [`qwen3-asr-0-6b`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) − [`whisper-persian-paulwalker`](https://huggingface.co/Paulwalker4884/whisper-persian) | −122.55 pp | −136.06 to −109.68 pp | First model has lower CER |

### Accuracy per peak CUDA memory

![Accuracy per peak CUDA memory leaderboard](generated/leaderboard-memory.svg)

| Rank | Model | Accuracy / reserved GiB | WER | Peak CUDA reserved GiB |
|---:|---|---:|---:|---:|
| 1 | [`whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo) | 45.3785 | 0.2041 | 1.754 |
| 2 | [`whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3) | 23.2527 | 0.1980 | 3.449 |
| 3 | [`whisper-large-persian-steja`](https://huggingface.co/steja/whisper-large-persian) | 22.8268 | 0.2648 | 3.221 |
| 4 | [`qwen3-asr-0-6b`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) | 18.6197 | 0.4803 | 2.791 |
| 5 | [`qwen3-asr-1-7b`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) | 14.8179 | 0.2417 | 5.117 |
| 6 | [`whisper-persian-paulwalker`](https://huggingface.co/Paulwalker4884/whisper-persian) | 14.8123 | 0.9430 | 0.385 |
| 7 | [`vibevoice-asr`](https://huggingface.co/microsoft/VibeVoice-ASR) | 3.8305 | 0.2704 | 19.047 |

Peak CUDA memory is unified system/GPU memory and is not directly comparable with process VRAM reported on discrete GPUs.

<!-- LEADERBOARD:END -->

### Reading the results

- **CER** is corpus-level character error rate after whitespace removal; lower is better and it is
  the primary accuracy ranking metric.
- **WER** is corpus-level word error rate; lower is better. It is sensitive to Persian spacing and
  ZWNJ conventions.
- **Word accuracy** is `100 × max(0, 1 − WER)`.
- **Accuracy / reserved GiB** is word accuracy divided by peak CUDA reserved memory; higher is
  better.

#### Normalization and scoring example

For an illustrative sample, the official normalizer replaces ZWNJ with a space and removes
punctuation. CER then removes normalized whitespace before scoring, while WER preserves word
boundaries.

| Stage | Reference | Prediction |
|---|---|---|
| Raw text | `می‌روم خانه.` | `میروم خانه` |
| After `fa-v1` | `می روم خانه` | `میروم خانه` |

- Sample CER is `0 / 9 = 0.0000`: the character sequences match after whitespace removal.
- Sample WER is `(1 substitution + 1 deletion) / 3 = 0.6667`: the token sequences differ.

Official scores aggregate edit counts over all 871 recordings rather than averaging per-sample
error rates.

The accuracy board sorts by CER, WER, then stable model ID. The efficiency board sorts by memory
efficiency, WER, then model ID. Only complete official result bundles whose suite and model
digests match the current specifications are ranked. Failed and out-of-memory runs remain
auditable but unranked.

### ZWNJ sensitivity

The official `fa-v1` normalizer converts ZWNJ to a space. This affects WER because joined and split
Persian compounds become different word-token sequences; CER removes normalized whitespace and is
not affected by that segmentation choice. ZWNJ occurs in 622 of the 871 test references (71.4%).

Recomputing the published raw predictions with ZWNJ removed instead of replaced by a space gives
the following sensitivity analysis. These are diagnostic values, not alternative official scores.

| Model | Official WER, ZWNJ → space | Diagnostic WER, ZWNJ → join | Diagnostic rank |
|---|---:|---:|---:|
| `whisper-large-persian-steja` | 0.2648 | 0.1917 | 1 |
| `qwen3-asr-1-7b` | 0.2417 | 0.2662 | 2 |
| `whisper-large-v3` | 0.1980 | 0.2882 | 3 |
| `whisper-large-v3-turbo` | 0.2041 | 0.2944 | 4 |
| `vibevoice-asr` | 0.2704 | 0.3274 | 5 |
| `qwen3-asr-0-6b` | 0.4803 | 0.5049 | 6 |
| `whisper-persian-paulwalker` | 0.9430 | 0.9663 | 7 |

### Number-format sensitivity

The published `fleurs-fa-ir-v1` scores are sensitive to whether a model writes a spoken number as
digits or Persian words. Its immutable `fa-v1` policy converts Persian and Arabic-Indic digit
glyphs to ASCII, but does not perform inverse text normalization. Digits occur in 153 of the 871
test references. For example, `۱۹۶۷` becomes `1967`, while the equivalent
`هزار و نهصد و شصت و هفت` remains words and receives seven word-level edit operations in its
reference sentence. This is a formatting bias in the published v1 results, not evidence of seven
recognition errors.

## Documentation

- [Propose a compatible model](docs/adding-a-model.md)
- [Contribute source code](docs/contributing.md)
- [Benchmark contract](docs/benchmark-contract.md)
- [Maintainer guide](docs/maintainer-guide.md)
- [Project roadmap](docs/roadmap.md)
- [Accuracy plot](generated/leaderboard-accuracy.svg),
  [memory-efficiency plot](generated/leaderboard-memory.svg),
  [JSON results](generated/leaderboard.json), and [CSV results](generated/leaderboard.csv)

## Scope and limitations

FLEURS is a public read-speech corpus and may overlap model training data. It does not represent
conversational, noisy, accented, domain-specific, or long-form Persian. These results do not
establish production suitability or general robustness.

This release measures normalized transcription accuracy, test-set sampling uncertainty, and peak
memory. It does not measure speed, latency, timestamps, diarization, streaming, punctuation
quality, model-training uncertainty, dataset bias, or robustness subsets. It excludes prompts,
hotwords, quantization, offload, compilation, external language models, and alternative decoding
searches.

`nvidia-fastconformer-fa` is unranked because its official native-FP32 run exhausted CUDA memory
during RNNT decoder CUDA-graph warmup. The benchmark does not change precision or decoding policy
after a failure.

See the [benchmark contract](docs/benchmark-contract.md) for the pinned dataset revision,
manifest, normalization rules, deterministic inference controls, result-bundle contents, and
ranking eligibility.

## How the benchmark works

1. Dataset audio is materialized into canonical 16-kHz mono PCM-16 WAV files and verified against
   an immutable manifest.
2. Checkpoints and any required auxiliary artifacts are downloaded at pinned revisions.
3. Official inference runs offline in framework-specific containers against read-only caches.
4. Predictions are normalized and scored in manifest order using corpus-level WER and CER.
5. Complete result bundles generate deterministic Markdown, SVG, JSON, and CSV leaderboards.

Each successful bundle publishes one JSONL record per evaluation sample. This abbreviated
[published record](results/fleurs-fa-ir-v1/fleurs-fa-ir-v1-whisper-large-v3-20260801T090117Z/predictions.jsonl)
shows the raw and normalized text together with its edit counts:

```json
{
  "sample_id": "test-000000",
  "reference": "این سند بر اساس …",
  "prediction": "این صند بر اساس …",
  "normalized_reference": "این سند بر اساس …",
  "normalized_prediction": "این صند بر اساس …",
  "word_substitutions": 3,
  "word_deletions": 1,
  "word_insertions": 0,
  "word_reference_units": 30,
  "character_substitutions": 4,
  "character_deletions": 5,
  "character_insertions": 2,
  "character_reference_units": 107
}
```

## Reproduce the benchmark

Official reproduction requires Python 3.12, [`uv`](https://docs.astral.sh/uv/), non-interactive
Docker-over-SSH access to a Jetson matching the official profile, the NVIDIA container runtime,
and at least 60 GiB of persistent cache storage. The commands below assume an SSH host alias named
`jetson`.

Install the host environment and build the isolated runtime images:

```bash
uv sync --frozen --all-groups
docker --host ssh://jetson pull nvcr.io/nvidia/pytorch@sha256:90f3c17838fde28d5c7ae2d5bfbc8a4c587d3797767ea96cdd48fe82e3613f3b
docker --host ssh://jetson build --file runtimes/modern/Dockerfile --tag peste-modern:1.0.0 .
docker --host ssh://jetson build --file runtimes/vibevoice/Dockerfile --tag peste-vibevoice:1.0.0 .
docker --host ssh://jetson build --file runtimes/nemo/Dockerfile --tag peste-nemo:1.0.0 .
```

Validate the host, prepare the dataset, validate a model, and run it:

```bash
uv run --frozen peste doctor --host ssh://jetson
uv run --frozen peste dataset prepare --suite fleurs-fa-ir-v1 --host ssh://jetson
uv run --frozen peste model validate --model whisper-large-v3
uv run --frozen peste model validate --model whisper-large-v3 --host ssh://jetson
uv run --frozen peste run --suite fleurs-fa-ir-v1 --model whisper-large-v3 --host ssh://jetson
uv run --frozen peste leaderboard --suite fleurs-fa-ir-v1
```

Store `HF_TOKEN` in the ignored `.env` file when gated access or higher Hub rate limits are
required, then add `--env-file .env` to `uv run`. Tokens are passed only to network-enabled
preparation containers; official inference runs with networking disabled.

Use `--resume` only for the latest failed or killed run. OOM runs are not resumable. `run-all` is
intended for a result set without existing successful bundles.

## Contributing

PESTE accepts two categories of contribution:

1. **Propose a model for benchmarking.** Add a pinned Hugging Face checkpoint that is compatible
   with an existing adapter. Contributors provide the model specification and open a pull request;
   maintainers perform the official Jetson evaluation and publish the score. Follow
   [Adding a model](docs/adding-a-model.md).
2. **Improve the benchmark source.** Changes to orchestration, scoring, adapters, runtimes,
   validation, generated outputs, tests, or documentation follow the
   [source contribution guide](docs/contributing.md).

Models that do not satisfy an existing adapter contract are not model-proposal PRs. Support for a
new architecture or inference API is benchmark-maintainer work and is documented separately in
the [maintainer guide](docs/maintainer-guide.md).

## License and attribution

PESTE code and benchmark definitions are licensed under Apache-2.0. FLEURS content is licensed
under CC BY 4.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) for terms and attribution.

## Citation

```bibtex
@software{jafarnezhad_peste_persian_speech_to_text_benchmark,
  author  = {Jafarnezhad, Arman},
  title   = {PESTE: Persian Speech to Text benchmark},
  year    = {2026},
  url     = {https://github.com/ArmanJR/PESTE-Benchmark},
  version = {2026.07.31}
}
```
