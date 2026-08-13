# Compatible 37 campaign

This campaign covers exactly the 37 additional model-only candidates classified as compatible in
[`asr-model-compatibility.md`](../../asr-model-compatibility.md): 25 Transformers Whisper, 10
Transformers CTC, and 2 NeMo RNNT checkpoints. The eight already-published models and every
conditional or incompatible repository are excluded.

The pinned Hugging Face metadata was rechecked on 2026-08-12. All 37 revisions remained public,
ungated, immutable, and licensed as recorded. Generic snapshot prefetch would retain
127,079,029,213 bytes (118.35 GiB); `steja/whisper-small-persian` alone accounts for
29,984,186,828 bytes because its repository contains many files. Calibration and official
sessions therefore request 400 GB rather than the normal 200 GB allocation so the doctor can
continue to enforce its 100 GiB free-space reserve.

Qualification completed on 2026-08-12 from the digest-pinned calibration carrier. Thirty-four
models passed pinned prefetch, real offline smoke, multi-item singleton equivalence,
cardinality/order checks, 85% VRAM headroom, and deterministic 95%-knee selection. Their selected
batches are 27 at batch 1, four at batch 2, one at batch 4, one at batch 32, and one at batch 128.
The three Zoha checkpoints failed because their processors omitted the attention mask required for
padding-safe CTC decoding; their provisional specifications were removed.

The tracked `qualification-summary.json` preserves compact outcomes and evidence hashes. Verbose
smoke, calibration, and failure logs remain under ignored `campaign-evidence/`. Calibration
throughput is evidence, not an unofficial benchmark score.

The Shenava collection includes FLEURS-fa evaluation artifacts, and the Full A+B card says an
external-language-model setting was calibrated on a FLEURS-256 slice. PESTE used its fixed default
RNNT decoder without an external language model, but the resulting VisualEars score should not be
treated as clean held-out evidence. No alternate decoder, precision fallback, or model-specific
repair was permitted.

## Official evaluation

The final carrier image was built by GitHub Actions run
[`31594241706`](https://github.com/ArmanJR/PESTE-Benchmark/actions/runs/31594241706) from source
revision `78fdc159e15e0e91ef3b9c2c10cfeb98417b27b0` and was executed by immutable digest:

```text
ghcr.io/armanjr/peste-benchmark@sha256:c5206ece402769a0540c7cdde4876b6ce8797645315df340588cab21dcd01fc7
```

All 34 qualifiers received one official attempt on one doctor-approved RTX 6000 Ada Generation
48 GB GPU (`GPU-edc964ee-2a79-5059-949e-8a2dda9d2861`). Thirty-three completed the full 871-sample
evaluation with two excluded warmups and valid uninterrupted speed measurements. Shenava Rizeh
completed 768 samples in six batch-128 measurements, then failed on the final 103 longest samples
when NeMo's Conformer `conv2d` exceeded PyTorch's 32-bit tensor-indexing limit. Its partial bundle
is retained as failed and unranked; changing the committed batch after seeing the result would
violate the campaign contract. The three qualification failures were not run officially.

| Model | Outcome | Batch | CER | WER | Throughput |
|---|---|---:|---:|---:|---:|
| `whisper-small-persian-steja` | Success | 1 | 9.38% | 34.91% | 6.369× |
| `neuraspeech-whisperbase-neurai` | Success | 1 | 6.70% | 21.63% | 10.425× |
| `whisper-tiny-fa-aictsharif` | Success | 1 | 13.51% | 45.33% | 14.008× |
| `whisper-base-fa-aictsharif` | Success | 1 | 11.39% | 40.15% | 10.702× |
| `whisper-small-fa-aictsharif` | Success | 1 | 7.92% | 29.85% | 6.137× |
| `whisper-medium-fa-aictsharif` | Success | 2 | 6.58% | 26.04% | 5.944× |
| `whisper-large-v2-fa-aictsharif` | Success | 2 | 5.10% | 23.26% | 4.597× |
| `whisper-fa-small-v1-seyedali` | Success | 1 | 9.27% | 33.53% | 6.339× |
| `persian-whisper-large-v3-10-percent-17-0-one-epoch-mohammadreza-halakoo` | Success | 1 | 6.23% | 27.43% | 2.619× |
| `whisper-large-fa-v1-vhdm` | Success | 1 | 5.35% | 14.48% | 12.676× |
| `whisper-persian-v4-nezamisafa` | Success | 1 | 4.93% | 13.12% | 2.445× |
| `whisper-v3-turbo-persian-v1-0-nezamisafa` | Success | 1 | 7.58% | 26.06% | 13.922× |
| `whisper-small-persian-v1-aliyzd95` | Success | 1 | 10.34% | 30.72% | 6.152× |
| `whisper-small-fa-mohammadgholizadeh` | Success | 1 | 12.43% | 43.65% | 6.446× |
| `whisper-fa-tinyyy-hackergeek98` | Success | 1 | 18.85% | 45.43% | 13.462× |
| `whisper-small-fa-7-taesiri` | Success | 1 | 7.43% | 28.12% | 6.182× |
| `whisper-small-persian-amirmohseni` | Success | 2 | 7.59% | 22.05% | 11.112× |
| `whisper-small-ctejarat-makhataei` | Success | 1 | 20.89% | 48.26% | 6.543× |
| `whisper-small-persian-alism98` | Success | 4 | 8.02% | 31.57% | 21.438× |
| `whisper-small-fa-mjavadf` | Success | 1 | 10.62% | 38.45% | 7.060× |
| `whisper-tiny` | Success | 1 | 97.21% | 165.05% | 8.443× |
| `whisper-base` | Success | 1 | 43.32% | 100.89% | 9.808× |
| `whisper-small` | Success | 2 | 24.38% | 60.40% | 9.829× |
| `whisper-medium` | Success | 1 | 14.19% | 38.80% | 3.292× |
| `whisper-large-v2` | Success | 1 | 8.19% | 25.30% | 2.616× |
| `shenava-rizeh-v1-0` | Official run failed | 128 | — | — | Invalid |
| `visualears-fastconformer-fa-full-ab` | Success | 32 | 5.18% | 15.52% | 378.349× |
| `wav2vec2-large-xlsr-persian-m3hrdadfi` | Success | 1 | 8.52% | 33.80% | 281.193× |
| `wav2vec2-large-xlsr-persian-v2-m3hrdadfi` | Success | 1 | 8.35% | 34.41% | 288.740× |
| `wav2vec2-large-xlsr-persian-shemo-m3hrdadfi` | Success | 1 | 10.11% | 30.66% | 285.490× |
| `xls-r-1b-fa-cv8-ghofrani` | Success | 1 | 7.04% | 30.42% | 289.030× |
| `wav2vec2-xls-r-300m-fa-alifarokh` | Success | 1 | 9.31% | 36.36% | 291.422× |
| `persian-speech-transcription-wav2vec2-v1-seyedali` | Success | 1 | 7.40% | 29.72% | 286.634× |
| `wav2vec2-base-common-voice-persian-colab-zoha` | Qualification failed | — | — | — | — |
| `wav2vec2-base-common-voice-40p-persian-colab-zoha` | Qualification failed | — | — | — | — |
| `wav2vec2-xlsr-persian-50p-zoha` | Qualification failed | — | — | — | — |
| `wav2vec2-large-xlsr-persian-v3-masoumehb` | Success | 1 | 91.83% | 100.00% | 296.335× |

The authoritative uncertainty intervals, paired comparisons, ranking, and Pareto analysis are in
the generated 41-model leaderboard. The tracked `qualification-summary.json` links every
qualified candidate to its official run ID and preserves the exact official environment.
