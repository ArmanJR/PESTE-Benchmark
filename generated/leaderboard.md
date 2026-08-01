# PESTE leaderboard — `fleurs-fa-ir-v1`

## Normalized accuracy

![Normalized accuracy leaderboard](leaderboard-accuracy.svg)

| Order | Model | CER | WER | Word accuracy |
|---:|---|---:|---:|---:|
| 1 | [`whisper-large-persian-steja`](https://huggingface.co/steja/whisper-large-persian) | 0.0589<br><sub>95% CI: 0.0542–0.0640</sub> | 0.2648<br><sub>95% CI: 0.2560–0.2737</sub> | 73.52% |
| 2 | [`whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3) | 0.0599<br><sub>95% CI: 0.0552–0.0652</sub> | 0.1980<br><sub>95% CI: 0.1897–0.2064</sub> | 80.20% |
| 3 | [`whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo) | 0.0650<br><sub>95% CI: 0.0576–0.0740</sub> | 0.2041<br><sub>95% CI: 0.1949–0.2135</sub> | 79.59% |
| 4 | [`qwen3-asr-1-7b`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) | 0.0892<br><sub>95% CI: 0.0844–0.0942</sub> | 0.2417<br><sub>95% CI: 0.2332–0.2505</sub> | 75.83% |
| 5 | [`qwen3-asr-0-6b`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) | 0.2086<br><sub>95% CI: 0.2013–0.2162</sub> | 0.4803<br><sub>95% CI: 0.4696–0.4907</sub> | 51.97% |
| 6 | [`whisper-persian-paulwalker`](https://huggingface.co/Paulwalker4884/whisper-persian) | 1.4341<br><sub>95% CI: 1.3055–1.5698</sub> | 0.9430<br><sub>95% CI: 0.8961–1.0015</sub> | 5.70% |

CER is the primary ranking metric because Persian WER is orthography-sensitive: fa-v1 converts ZWNJ to spaces, while CER ignores normalized whitespace. WER and derived word accuracy remain complementary, segmentation-sensitive measurements.

Point-estimate order does not establish statistical significance. Intervals use a deterministic 10,000-replicate utterance-level percentile bootstrap at 95% confidence with seed 20250731. Paired intervals containing zero are reported as no clear difference; these intervals measure test-set sampling uncertainty only.

### Paired adjacent CER comparisons

| Adjacent models | ΔCER | Paired 95% range | Evidence |
|---|---:|---:|---|
| [`whisper-large-persian-steja`](https://huggingface.co/steja/whisper-large-persian) − [`whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3) | −0.11 pp | −0.53 to 0.29 pp | No clear difference |
| [`whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3) − [`whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo) | −0.50 pp | −1.20 to 0.02 pp | No clear difference |
| [`whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo) − [`qwen3-asr-1-7b`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) | −2.42 pp | −3.12 to −1.58 pp | First model has lower CER |
| [`qwen3-asr-1-7b`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) − [`qwen3-asr-0-6b`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) | −11.94 pp | −12.57 to −11.34 pp | First model has lower CER |
| [`qwen3-asr-0-6b`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) − [`whisper-persian-paulwalker`](https://huggingface.co/Paulwalker4884/whisper-persian) | −122.55 pp | −136.06 to −109.68 pp | First model has lower CER |

## Accuracy per peak CUDA memory

![Accuracy per peak CUDA memory leaderboard](leaderboard-memory.svg)

| Rank | Model | Accuracy / reserved GiB | WER | Peak CUDA reserved GiB |
|---:|---|---:|---:|---:|
| 1 | [`whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo) | 45.3785 | 0.2041 | 1.754 |
| 2 | [`whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3) | 23.2527 | 0.1980 | 3.449 |
| 3 | [`whisper-large-persian-steja`](https://huggingface.co/steja/whisper-large-persian) | 22.8268 | 0.2648 | 3.221 |
| 4 | [`qwen3-asr-0-6b`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) | 18.6197 | 0.4803 | 2.791 |
| 5 | [`qwen3-asr-1-7b`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) | 14.8179 | 0.2417 | 5.117 |
| 6 | [`whisper-persian-paulwalker`](https://huggingface.co/Paulwalker4884/whisper-persian) | 14.8123 | 0.9430 | 0.385 |

Peak CUDA memory is unified system/GPU memory and is not directly comparable with process VRAM reported on discrete GPUs.
