# PESTE leaderboard — `fleurs-fa-ir-v1`

## Normalized accuracy

![Normalized accuracy leaderboard](leaderboard-accuracy.svg)

| Order | Model | CER | WER | Word accuracy |
|---:|---|---:|---:|---:|
| 1 | [whisper-large-persian-steja](https://huggingface.co/steja/whisper-large-persian) | 0.0584<br><sub>95% CI: 0.0538–0.0636</sub> | 0.2643<br><sub>95% CI: 0.2554–0.2732</sub> | 73.57% |
| 2 | [whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | 0.0605<br><sub>95% CI: 0.0560–0.0655</sub> | 0.2044<br><sub>95% CI: 0.1961–0.2129</sub> | 79.56% |
| 3 | [whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | 0.0629<br><sub>95% CI: 0.0570–0.0699</sub> | 0.2042<br><sub>95% CI: 0.1951–0.2135</sub> | 79.58% |
| 4 | [wav2vec2-large-xlsr-53-persian](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-persian) | 0.0741<br><sub>95% CI: 0.0690–0.0793</sub> | 0.3269<br><sub>95% CI: 0.3179–0.3357</sub> | 67.31% |
| 5 | [qwen3-asr-1-7b](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) | 0.0892<br><sub>95% CI: 0.0845–0.0944</sub> | 0.2410<br><sub>95% CI: 0.2324–0.2501</sub> | 75.90% |
| 6 | [qwen3-asr-0-6b](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) | 0.2091<br><sub>95% CI: 0.2022–0.2164</sub> | 0.4811<br><sub>95% CI: 0.4704–0.4915</sub> | 51.89% |
| 7 | [nvidia-fastconformer-fa](https://huggingface.co/nvidia/stt_fa_fastconformer_hybrid_large) | 0.2646<br><sub>95% CI: 0.2552–0.2741</sub> | 0.4461<br><sub>95% CI: 0.4360–0.4565</sub> | 55.39% |
| 8 | [whisper-persian-paulwalker](https://huggingface.co/Paulwalker4884/whisper-persian) | 1.4278<br><sub>95% CI: 1.2999–1.5608</sub> | 0.9320<br><sub>95% CI: 0.8926–0.9774</sub> | 6.80% |

CER is the primary ranking metric because Persian WER is orthography-sensitive: fa-v1 converts ZWNJ to spaces, while CER ignores normalized whitespace. WER and derived word accuracy remain complementary, segmentation-sensitive measurements.

Point-estimate order does not establish statistical significance. Intervals use a deterministic 10,000-replicate utterance-level percentile bootstrap at 95% confidence with seed 20250731. Paired intervals containing zero are reported as no clear difference; these intervals measure test-set sampling uncertainty only.

### Paired adjacent CER comparisons

| Adjacent models | ΔCER | Paired 95% range | Evidence |
|---|---:|---:|---|
| [whisper-large-persian-steja](https://huggingface.co/steja/whisper-large-persian) − [whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | −0.21 pp | −0.61 to 0.17 pp | No clear difference |
| [whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) − [whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | −0.24 pp | −0.59 to 0.05 pp | No clear difference |
| [whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) − [wav2vec2-large-xlsr-53-persian](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-persian) | −1.12 pp | −1.71 to −0.40 pp | First model has lower CER |
| [wav2vec2-large-xlsr-53-persian](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-persian) − [qwen3-asr-1-7b](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) | −1.52 pp | −1.99 to −1.05 pp | First model has lower CER |
| [qwen3-asr-1-7b](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) − [qwen3-asr-0-6b](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) | −11.99 pp | −12.56 to −11.42 pp | First model has lower CER |
| [qwen3-asr-0-6b](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) − [nvidia-fastconformer-fa](https://huggingface.co/nvidia/stt_fa_fastconformer_hybrid_large) | −5.55 pp | −6.52 to −4.56 pp | First model has lower CER |
| [nvidia-fastconformer-fa](https://huggingface.co/nvidia/stt_fa_fastconformer_hybrid_large) − [whisper-persian-paulwalker](https://huggingface.co/Paulwalker4884/whisper-persian) | −116.32 pp | −129.61 to −103.50 pp | First model has lower CER |

## Steady-state speed

![Steady-state speed leaderboard](leaderboard-speed.svg)

| Rank | Model | Batch | Throughput | RTF | Processing s | Audio s |
|---:|---|---:|---:|---:|---:|---:|
| 1 | [nvidia-fastconformer-fa](https://huggingface.co/nvidia/stt_fa_fastconformer_hybrid_large) | 32 | 1199.914× | 0.00083 | 11.102 | 13321.860 |
| 2 | [wav2vec2-large-xlsr-53-persian](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-persian) | 1 | 470.167× | 0.00213 | 28.334 | 13321.860 |
| 3 | [whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | 1 | 50.196× | 0.01992 | 265.399 | 13321.860 |
| 4 | [whisper-persian-paulwalker](https://huggingface.co/Paulwalker4884/whisper-persian) | 1 | 13.183× | 0.07586 | 1010.571 | 13321.860 |
| 5 | [qwen3-asr-0-6b](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) | 1 | 11.213× | 0.08919 | 1188.118 | 13321.860 |
| 6 | [qwen3-asr-1-7b](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) | 1 | 10.755× | 0.09298 | 1238.672 | 13321.860 |
| 7 | [whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | 1 | 10.183× | 0.09820 | 1308.217 | 13321.860 |
| 8 | [whisper-large-persian-steja](https://huggingface.co/steja/whisper-large-persian) | 1 | 8.201× | 0.12194 | 1624.458 | 13321.860 |

Throughput is total audio seconds divided by measured processing seconds; RTF is its reciprocal. Resumed runs retain accuracy but are excluded here.

## Accuracy-speed Pareto efficiency

![Accuracy-speed Pareto efficiency](leaderboard-pareto.svg)

| Order | Model | CER | Throughput | Point frontier | Supported frontier | Point dominators | Supported dominators |
|---:|---|---:|---:|---|---|---|---|
| 1 | [nvidia-fastconformer-fa](https://huggingface.co/nvidia/stt_fa_fastconformer_hybrid_large) | 0.2646<br><sub>95% CI: 0.2552–0.2741</sub> | 1199.914× | Yes | Yes | — | — |
| 2 | [wav2vec2-large-xlsr-53-persian](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-persian) | 0.0741<br><sub>95% CI: 0.0690–0.0793</sub> | 470.167× | Yes | Yes | — | — |
| 3 | [whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | 0.0629<br><sub>95% CI: 0.0570–0.0699</sub> | 50.196× | Yes | Yes | — | — |
| 4 | [whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | 0.0605<br><sub>95% CI: 0.0560–0.0655</sub> | 10.183× | Yes | Yes | — | — |
| 5 | [whisper-large-persian-steja](https://huggingface.co/steja/whisper-large-persian) | 0.0584<br><sub>95% CI: 0.0538–0.0636</sub> | 8.201× | Yes | Yes | — | — |
| 6 | [whisper-persian-paulwalker](https://huggingface.co/Paulwalker4884/whisper-persian) | 1.4278<br><sub>95% CI: 1.2999–1.5608</sub> | 13.183× | No | No | [nvidia-fastconformer-fa](https://huggingface.co/nvidia/stt_fa_fastconformer_hybrid_large)<br>[wav2vec2-large-xlsr-53-persian](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-persian)<br>[whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | [nvidia-fastconformer-fa](https://huggingface.co/nvidia/stt_fa_fastconformer_hybrid_large)<br>[wav2vec2-large-xlsr-53-persian](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-persian)<br>[whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) |
| 7 | [qwen3-asr-0-6b](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) | 0.2091<br><sub>95% CI: 0.2022–0.2164</sub> | 11.213× | No | No | [wav2vec2-large-xlsr-53-persian](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-persian)<br>[whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | [wav2vec2-large-xlsr-53-persian](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-persian)<br>[whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) |
| 8 | [qwen3-asr-1-7b](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) | 0.0892<br><sub>95% CI: 0.0845–0.0944</sub> | 10.755× | No | No | [wav2vec2-large-xlsr-53-persian](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-persian)<br>[whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | [wav2vec2-large-xlsr-53-persian](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-persian)<br>[whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) |

A speed-valid model is Pareto-efficient when no other model has both equal-or-lower CER and equal-or-higher throughput, with at least one strict advantage. Point dominators use the published estimates. Supported dominators additionally require the paired 95% CER-difference interval to remain below zero. CER intervals measure test-set sampling uncertainty; speed is a single deterministic run without a confidence interval. The plot inverts its logarithmic CER axis so visually better directions are up and right while tick labels remain raw CER. Pareto status is a trade-off classification, not a composite score.
