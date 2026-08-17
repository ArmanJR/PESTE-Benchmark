<p align="center">
  <img src="docs/peste-logo.png" alt="PESTE logo" width="520">
</p>

# PESTE: Persian Speech to Text benchmark

PESTE (**PE**rsian **S**peech to **TE**xt) is a benchmark and leaderboard for
Persian automatic speech recognition.

## At a glance

- **Release:** `v2.1.0`
- **Suite:** [`fleurs-fa-ir-v1`](suites/fleurs-fa-ir-v1/suite.json), 871 test recordings
- **Normalization:** immutable `fa-v1`
- **Accuracy:** corpus CER (primary), WER, deterministic bootstrap uncertainty, and paired CER
  comparisons
- **Speed:** steady-state end-to-end audio throughput and real-time factor (RTF)
- **Official profile:** one NVIDIA RTX 6000 Ada Generation 48 GB GPU (`rtx-6000-ada-v1`)
- **Inference:** deterministic per-model batching, checkpoint-native precision, automatic native
  Whisper long-form decoding, offline execution, and read-only dataset/checkpoint caches
- **Runtime:** digest-pinned public GHCR image with isolated modern and NeMo environments

## Leaderboard

<!-- LEADERBOARD:START -->

The tables show the top 10 models. See the [full leaderboard](docs/full-leaderboard.md) for complete standings.

Result provenance: 13 results from `ghcr.io/armanjr/peste-benchmark:2.0.0`, 29 results from `ghcr.io/armanjr/peste-benchmark:2.1.0`. The generated JSON and CSV include each row's model digest, image digest, and source revision.

### Normalized accuracy

![Normalized accuracy leaderboard](generated/leaderboard-accuracy.svg)

| Order | Model | CER | WER | Word accuracy |
|---:|---|---:|---:|---:|
| 1 | [whisper-persian-v4-nezamisafa](https://huggingface.co/nezamisafa/whisper-persian-v4) | 0.0497<br><sub>95% CI: 0.0447–0.0553</sub> | 0.1318<br><sub>95% CI: 0.1238–0.1398</sub> | 86.82% |
| 2 | [visualears-fastconformer-fa-full-ab](https://huggingface.co/Reza2kn/visualears-fastconformer-fa-full-ab) | 0.0518<br><sub>95% CI: 0.0475–0.0565</sub> | 0.1552<br><sub>95% CI: 0.1481–0.1624</sub> | 84.48% |
| 3 | [whisper-large-fa-v1-vhdm](https://huggingface.co/vhdm/whisper-large-fa-v1) | 0.0550<br><sub>95% CI: 0.0463–0.0689</sub> | 0.1457<br><sub>95% CI: 0.1358–0.1577</sub> | 85.43% |
| 4 | [whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | 0.0597<br><sub>95% CI: 0.0558–0.0640</sub> | 0.2037<br><sub>95% CI: 0.1957–0.2119</sub> | 79.63% |
| 5 | [whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | 0.0610<br><sub>95% CI: 0.0564–0.0663</sub> | 0.2027<br><sub>95% CI: 0.1945–0.2111</sub> | 79.73% |
| 6 | [shenava-rizeh-v1-0](https://huggingface.co/Reza2kn/Shenava-Rizeh-v1.0) | 0.0640<br><sub>95% CI: 0.0589–0.0694</sub> | 0.1555<br><sub>95% CI: 0.1470–0.1638</sub> | 84.45% |
| 7 | [xls-r-1b-fa-cv8-ghofrani](https://huggingface.co/ghofrani/xls-r-1b-fa-cv8) | 0.0704<br><sub>95% CI: 0.0655–0.0757</sub> | 0.3042<br><sub>95% CI: 0.2953–0.3133</sub> | 69.58% |
| 8 | [whisper-small-fa-7-taesiri](https://huggingface.co/taesiri/whisper-small-fa-7) | 0.0727<br><sub>95% CI: 0.0661–0.0798</sub> | 0.2805<br><sub>95% CI: 0.2710–0.2901</sub> | 71.95% |
| 9 | [persian-speech-transcription-wav2vec2-v1-seyedali](https://huggingface.co/SeyedAli/Persian-Speech-Transcription-Wav2Vec2-V1) | 0.0740<br><sub>95% CI: 0.0690–0.0794</sub> | 0.2972<br><sub>95% CI: 0.2876–0.3066</sub> | 70.28% |
| 10 | [wav2vec2-large-xlsr-53-persian](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-persian) | 0.0741<br><sub>95% CI: 0.0690–0.0793</sub> | 0.3269<br><sub>95% CI: 0.3179–0.3357</sub> | 67.31% |

CER is primary because Persian WER is sensitive to word segmentation. The 95% intervals use a deterministic 10,000-replicate utterance bootstrap with seed 20250731; point-estimate order alone does not establish a significant difference.

### Steady-state speed

![Steady-state speed leaderboard](generated/leaderboard-speed.svg)

| Rank | Model | Batch | Throughput | RTF | Processing s | Audio s |
|---:|---|---:|---:|---:|---:|---:|
| 1 | [nvidia-fastconformer-fa](https://huggingface.co/nvidia/stt_fa_fastconformer_hybrid_large) | 32 | 1199.914× | 0.00083 | 11.102 | 13321.860 |
| 2 | [shenava-rizeh-v1-0](https://huggingface.co/Reza2kn/Shenava-Rizeh-v1.0) | 16 | 675.031× | 0.00148 | 19.735 | 13321.860 |
| 3 | [wav2vec2-large-xlsr-53-persian](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-persian) | 1 | 470.167× | 0.00213 | 28.334 | 13321.860 |
| 4 | [visualears-fastconformer-fa-full-ab](https://huggingface.co/Reza2kn/visualears-fastconformer-fa-full-ab) | 32 | 378.349× | 0.00264 | 35.211 | 13321.860 |
| 5 | [wav2vec2-large-xlsr-persian-v3-masoumehb](https://huggingface.co/masoumehb/wav2vec2-large-xlsr-persian-v3) | 1 | 296.335× | 0.00337 | 44.955 | 13321.860 |
| 6 | [wav2vec2-xls-r-300m-fa-alifarokh](https://huggingface.co/alifarokh/wav2vec2-xls-r-300m-fa) | 1 | 291.422× | 0.00343 | 45.713 | 13321.860 |
| 7 | [xls-r-1b-fa-cv8-ghofrani](https://huggingface.co/ghofrani/xls-r-1b-fa-cv8) | 1 | 289.030× | 0.00346 | 46.092 | 13321.860 |
| 8 | [wav2vec2-large-xlsr-persian-v2-m3hrdadfi](https://huggingface.co/m3hrdadfi/wav2vec2-large-xlsr-persian-v2) | 1 | 288.740× | 0.00346 | 46.138 | 13321.860 |
| 9 | [persian-speech-transcription-wav2vec2-v1-seyedali](https://huggingface.co/SeyedAli/Persian-Speech-Transcription-Wav2Vec2-V1) | 1 | 286.634× | 0.00349 | 46.477 | 13321.860 |
| 10 | [wav2vec2-large-xlsr-persian-shemo-m3hrdadfi](https://huggingface.co/m3hrdadfi/wav2vec2-large-xlsr-persian-shemo) | 1 | 285.490× | 0.00350 | 46.663 | 13321.860 |

Throughput is total audio seconds divided by measured processing seconds; RTF is its reciprocal. Resumed runs retain accuracy but are excluded here.

### Accuracy-speed Pareto efficiency

![Accuracy-speed Pareto efficiency](generated/leaderboard-pareto.svg)

| Model | CER (95% CI) | Throughput (× real time) |
|---|---:|---:|
| [nvidia-fastconformer-fa](https://huggingface.co/nvidia/stt_fa_fastconformer_hybrid_large) | 0.2646 (0.2552–0.2741) | 1199.914× |
| [shenava-rizeh-v1-0](https://huggingface.co/Reza2kn/Shenava-Rizeh-v1.0) | 0.0640 (0.0589–0.0694) | 675.031× |
| [visualears-fastconformer-fa-full-ab](https://huggingface.co/Reza2kn/visualears-fastconformer-fa-full-ab) | 0.0518 (0.0475–0.0565) | 378.349× |
| [whisper-persian-v4-nezamisafa](https://huggingface.co/nezamisafa/whisper-persian-v4) | 0.0497 (0.0447–0.0553) | 6.352× |

A model is Pareto-efficient when no other speed-valid model has equal-or-lower CER and equal-or-higher throughput with at least one strict advantage. This is a trade-off classification, not a composite score; JSON and CSV artifacts retain the full dominance analysis.

<!-- LEADERBOARD:END -->

## Reproduce

Install the locked environment, configure Vast.ai, and use the immutable image reference from a
successful `Runtime image` workflow artifact:

```bash
uv sync --frozen --all-groups
uv run vastai set api-key <key>
image_ref=$(jq -r .image_reference runtime-image.json)
uv run peste cloud up --image "$image_ref" --max-dph <maximum-hourly-price>
uv run peste dataset prepare --suite fleurs-fa-ir-v1 --host <ssh-url>
uv run peste model validate --model <model-id> --host <ssh-url>
uv run peste run --suite fleurs-fa-ir-v1 --model <model-id> --host <ssh-url>
uv run peste leaderboard --suite fleurs-fa-ir-v1
uv run peste cloud down
```

Always destroy the instance after copying results. See the [maintainer guide](docs/maintainer-guide.md)
for image builds, batch calibration, campaigns, recovery, and publication checks.

## Dataset and scoring

The suite verifies pinned FLEURS audio against its committed manifest and applies immutable `fa-v1`
normalization to references and predictions. CER ignores normalized whitespace; WER preserves word
boundaries. See the [benchmark contract](docs/benchmark-contract.md) for the complete rules.

## Documentation

- [Benchmark contract](docs/benchmark-contract.md)
- [Adding a model](docs/adding-a-model.md)
- [Maintainer guide](docs/maintainer-guide.md)
- [Contributing source changes](docs/contributing.md)
- [Version roadmap](docs/roadmap.md)
- [Changelog](CHANGELOG.md)

## Scope and limitations

FLEURS is public read speech and may overlap model training data. It does not represent all
conversational, noisy, accented, domain-specific, streaming, or long-form Persian. The benchmark
does not evaluate timestamps, diarization, punctuation quality, training uncertainty, deployment
robustness, or production suitability. Prompts, hotwords, quantization, offload, compilation,
external language models, and policy-changing fallbacks are excluded.

Accuracy intervals describe test-utterance sampling uncertainty. Speed is one uninterrupted pass
per model and has no repeated-run confidence interval. PESTE 2.1 regenerates the 29 affected
Whisper results while retaining 13 unchanged non-Whisper 2.0 bundles; the generated JSON/CSV and
individual result bundles carry the exact model, image, source, and hardware provenance.

The Shenava collection includes FLEURS-fa evaluation artifacts, and the
[`visualears-fastconformer-fa-full-ab`](https://huggingface.co/Reza2kn/visualears-fastconformer-fa-full-ab)
card reports external-language-model calibration on a FLEURS-256 slice. PESTE used the fixed
default RNNT decoder without that language model, but the published VisualEars score should not be
treated as clean held-out evidence.

## License and citation

PESTE code and definitions are Apache-2.0. FLEURS content is CC BY 4.0; see [`NOTICE`](NOTICE).

```bibtex
@software{jafarnezhad_peste_persian_speech_to_text_benchmark,
  author  = {Jafarnezhad, Arman},
  title   = {PESTE: Persian Speech to Text benchmark},
  year    = {2026},
  url     = {https://github.com/ArmanJR/PESTE-Benchmark},
  version = {2.1.0}
}
```
