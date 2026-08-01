# PESTE leaderboard — `fleurs-fa-ir-v1`

## Normalized accuracy

![Normalized accuracy leaderboard](leaderboard-accuracy.svg)

| Rank | Model | WER | CER | Word accuracy |
|---:|---|---:|---:|---:|
| 1 | [`whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3) | 0.1980 | 0.0599 | 80.20% |
| 2 | [`whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo) | 0.2041 | 0.0650 | 79.59% |
| 3 | [`qwen3-asr-1-7b`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) | 0.2417 | 0.0892 | 75.83% |
| 4 | [`whisper-large-persian-steja`](https://huggingface.co/steja/whisper-large-persian) | 0.2648 | 0.0589 | 73.52% |
| 5 | [`vibevoice-asr`](https://huggingface.co/microsoft/VibeVoice-ASR) | 0.2704 | 0.1378 | 72.96% |
| 6 | [`qwen3-asr-0-6b`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) | 0.4803 | 0.2086 | 51.97% |
| 7 | [`whisper-persian-paulwalker`](https://huggingface.co/Paulwalker4884/whisper-persian) | 0.9430 | 1.4341 | 5.70% |

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
| 7 | [`vibevoice-asr`](https://huggingface.co/microsoft/VibeVoice-ASR) | 3.8305 | 0.2704 | 19.047 |

Peak CUDA memory is unified system/GPU memory and is not directly comparable with process VRAM reported on discrete GPUs.
