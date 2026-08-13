<p align="center">
  <img src="docs/peste-logo.png" alt="PESTE logo" width="520">
</p>

# PESTE: Persian Speech to Text benchmark

PESTE (**PE**rsian **S**peech to **TE**xt) is a benchmark and leaderboard for
Persian automatic speech recognition.

## At a glance

- **Release:** `v2`
- **Suite:** [`fleurs-fa-ir-v1`](suites/fleurs-fa-ir-v1/suite.json), 871 test recordings
- **Normalization:** immutable `fa-v1`
- **Accuracy:** corpus CER (primary), WER, deterministic bootstrap uncertainty, and paired CER
  comparisons
- **Speed:** steady-state end-to-end audio throughput and real-time factor (RTF)
- **Official profile:** one NVIDIA RTX 6000 Ada Generation 48 GB GPU, driver major 580 or newer, 300 W,
  ECC disabled, at least 8 vCPUs, 64 GiB RAM, and 100 GiB free local storage
- **Inference:** deterministic per-model batching, checkpoint-native precision, offline execution,
  and read-only dataset/checkpoint caches
- **Carrier image:** a digest-pinned public GHCR image with isolated modern and NeMo environments,
  built from x86-64 NGC PyTorch 25.06,
  `nvcr.io/nvidia/pytorch@sha256:3cb18e2c438db8af2d3a659ca27fac5da328640261c38c48a34edcd223c38af9`

The hardware-profile identifier is `rtx-6000-ada-v1`. Its `-v1` suffix versions the hardware
contract itself and is independent of the PESTE release version.

## Leaderboard

<!-- LEADERBOARD:START -->

The tables show the top 10 models. See the [full leaderboard and paired CER comparisons](docs/full-leaderboard.md) for complete standings.

### Normalized accuracy

![Normalized accuracy leaderboard](generated/leaderboard-accuracy.svg)

| Order | Model | CER | WER | Word accuracy |
|---:|---|---:|---:|---:|
| 1 | [whisper-persian-v4-nezamisafa](https://huggingface.co/nezamisafa/whisper-persian-v4) | 0.0493<br><sub>95% CI: 0.0443–0.0550</sub> | 0.1312<br><sub>95% CI: 0.1233–0.1393</sub> | 86.88% |
| 2 | [whisper-large-v2-fa-aictsharif](https://huggingface.co/aictsharif/whisper-large-v2-fa) | 0.0510<br><sub>95% CI: 0.0445–0.0589</sub> | 0.2326<br><sub>95% CI: 0.2233–0.2424</sub> | 76.74% |
| 3 | [visualears-fastconformer-fa-full-ab](https://huggingface.co/Reza2kn/visualears-fastconformer-fa-full-ab) | 0.0518<br><sub>95% CI: 0.0475–0.0565</sub> | 0.1552<br><sub>95% CI: 0.1481–0.1624</sub> | 84.48% |
| 4 | [whisper-large-fa-v1-vhdm](https://huggingface.co/vhdm/whisper-large-fa-v1) | 0.0535<br><sub>95% CI: 0.0473–0.0611</sub> | 0.1448<br><sub>95% CI: 0.1361–0.1542</sub> | 85.52% |
| 5 | [whisper-large-persian-steja](https://huggingface.co/steja/whisper-large-persian) | 0.0584<br><sub>95% CI: 0.0538–0.0636</sub> | 0.2643<br><sub>95% CI: 0.2554–0.2732</sub> | 73.57% |
| 6 | [whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | 0.0605<br><sub>95% CI: 0.0560–0.0655</sub> | 0.2044<br><sub>95% CI: 0.1961–0.2129</sub> | 79.56% |
| 7 | [persian-whisper-large-v3-10-percent-17-0-one-epoch-mohammadreza-halakoo](https://huggingface.co/MohammadReza-Halakoo/persian-whisper-large-v3-10-percent-17-0-one-epoch) | 0.0623<br><sub>95% CI: 0.0574–0.0675</sub> | 0.2743<br><sub>95% CI: 0.2654–0.2831</sub> | 72.57% |
| 8 | [whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | 0.0629<br><sub>95% CI: 0.0570–0.0699</sub> | 0.2042<br><sub>95% CI: 0.1951–0.2135</sub> | 79.58% |
| 9 | [shenava-rizeh-v1-0](https://huggingface.co/Reza2kn/Shenava-Rizeh-v1.0) | 0.0640<br><sub>95% CI: 0.0589–0.0694</sub> | 0.1555<br><sub>95% CI: 0.1470–0.1638</sub> | 84.45% |
| 10 | [whisper-medium-fa-aictsharif](https://huggingface.co/aictsharif/whisper-medium-fa) | 0.0658<br><sub>95% CI: 0.0517–0.0836</sub> | 0.2604<br><sub>95% CI: 0.2434–0.2804</sub> | 73.96% |

CER is the primary ranking metric because Persian WER is orthography-sensitive: fa-v1 converts ZWNJ to spaces, while CER ignores normalized whitespace. WER and derived word accuracy remain complementary, segmentation-sensitive measurements.

Point-estimate order does not establish statistical significance. Intervals use a deterministic 10,000-replicate utterance-level percentile bootstrap at 95% confidence with seed 20250731. Paired intervals containing zero are reported as no clear difference; these intervals measure test-set sampling uncertainty only.

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
| [whisper-large-v2-fa-aictsharif](https://huggingface.co/aictsharif/whisper-large-v2-fa) | 0.0510 (0.0445–0.0589) | 4.597× |
| [whisper-persian-v4-nezamisafa](https://huggingface.co/nezamisafa/whisper-persian-v4) | 0.0493 (0.0443–0.0550) | 2.445× |

A speed-valid model is Pareto-efficient when no other model has both equal-or-lower CER and equal-or-higher throughput, with at least one strict advantage. The table lists only that point-estimate frontier; the machine-readable artifacts retain classifications and dominators for every speed-valid model. Supported dominance additionally requires the paired 95% CER-difference interval to remain below zero. CER intervals measure test-set sampling uncertainty; speed is a single deterministic run without a confidence interval. The plot inverts its logarithmic CER axis so visually better directions are up and right while tick labels remain raw CER. CER confidence bars are hidden by default and can be toggled when the SVG is rendered interactively. The displayed CER axis ends at 1; worse values remain labeled at the lower boundary as off-scale points. Pareto status is a trade-off classification, not a composite score.

<!-- LEADERBOARD:END -->

Accuracy ties resolve by WER and model ID. Speed ties resolve by CER, WER, and model ID. The Pareto
table contains only the point-estimate frontier; the generated JSON and CSV retain point-estimate
and paired-CER-supported dominance for every speed-valid model.

## What is timed

The model is loaded and all evaluation audio is validated and primed into the OS cache before
timing. Evaluation rows are ordered deterministically by duration and original sequence. Two
representative median-duration batches warm the model without contributing to the measurement.
The timed region around each subsequent adapter call includes audio loading, preprocessing,
device transfer, inference/generation, decoding, and postprocessing. CUDA is synchronized
immediately before starting and immediately after completing each timed call.

One complete pass over all 871 test recordings is measured. Predictions are then restored to
manifest order. A resumed run keeps recovered predictions and accuracy, but its speed is marked
invalid; publication on the speed board requires a fresh uninterrupted run.

## Model batch calibration

Each model specification carries a `speed_profile` with the hardware-profile ID and a fixed batch
size. Maintainers determine it with:

```bash
uv run peste model profile-speed --model <model-id> --host <ssh-url>
```

Calibration uses 128 duration-quantile recordings, candidates from 1 through 128, two warmups and
three measured passes. It rejects OOM, more than 85% VRAM use on the longest-duration stress batch,
output cardinality/order failures, and normalized divergence from a fixed singleton conformance
set. The smallest safe candidate reaching at least 95% of the best throughput is selected. Full
runs never retune or silently fall back.

## Reproduce on Vast.ai

Ordinary Vast.ai container instances are the reference acquisition path, not part of the
comparison contract. Any SSH-accessible carrier container accepted by `peste doctor` is eligible.
Timed execution occurs directly inside the digest-pinned PESTE carrier image.

Install the locked environment and configure the Vast.ai API key once outside the repository:

```bash
uv sync --frozen --all-groups
uv run vastai set api-key <key>
```

`VAST_API_KEY` can override the stored key. Keys are redacted from subprocess logging and must not
be committed.

Build the selected commit with the manually triggered `Runtime image` GitHub Actions workflow.
The workflow publishes `ghcr.io/armanjr/peste-benchmark`, verifies both isolated runtime
environments and the native offline guard, and uploads `runtime-image.json` containing the
immutable image reference. The GHCR package must be public so Vast can pull it without receiving a
registry credential.

```bash
gh workflow run runtime-image.yml --ref <40-character-commit>
# Wait for the run, then download its runtime-image-<commit> artifact.
```

Provision and validate an ordinary container using that exact digest:

```bash
image_ref=$(jq -r .image_reference runtime-image.json)
uv run peste cloud up --image "$image_ref" --max-dph <maximum-hourly-price>
uv run peste cloud status
```

The normal allocation is 200 GB. Multi-model campaigns must total the pinned Hub snapshots first
and request sufficient storage with `cloud up --disk-gb <size>` while retaining the doctor's
required 100 GiB free-space reserve.

`cloud up` preselects one verified RTX 6000 Ada offer with driver 580 or newer, allocates 200 GB,
allows up to one hour for the large carrier to become SSH-ready, destroys every rejected attempt,
and prints the instance ID, accepted price, image digest, and a direct
`ssh://root@<ip>:<port>` URL. Use that URL unchanged with the existing commands:

```bash
uv run peste doctor --host <ssh-url>
uv run peste dataset prepare --suite fleurs-fa-ir-v1 --host <ssh-url>
uv run peste model validate --model <model-id> --host <ssh-url>
uv run peste model profile-speed --model <model-id> --host <ssh-url>
```

Repeat validation and profiling for every model, write each reported
`selected_batch_size` into its model JSON, review the candidate evidence, and commit all reviewed
profiles together. Because model specifications are copied into the carrier image and contribute
to request digests, destroy the calibration instance, build the updated commit, and provision a
fresh container from the new workflow digest before any full run:

```bash
uv run peste validate-specs
uv run peste cloud down
# Trigger the Runtime image workflow for the commit containing calibrated profiles.
image_ref=$(jq -r .image_reference runtime-image.json)
uv run peste cloud up --image "$image_ref" --max-dph <maximum-hourly-price>
uv run peste dataset prepare --suite fleurs-fa-ir-v1 --host <ssh-url>
uv run peste run-all --suite fleurs-fa-ir-v1 --host <ssh-url>
uv run peste leaderboard --suite fleurs-fa-ir-v1
uv run peste check-generated
uv run peste cloud down
```

The two Python dependency stacks remain isolated in `/opt/venvs/modern` and `/opt/venvs/nemo`
inside one image, so every model can use the same physical host. Fresh container storage is
ephemeral: audio and checkpoints are recreated for every session. Result bundles are copied back
to this checkout after each run. `cloud down` destroys labeled instances because stopped instances
continue billing storage.

If an instance dies during a run, resume can recover predictions and accuracy, but the speed result
is invalid. Repeat that model from a fresh run on a new doctor-approved host for publication.

## Dataset and scoring

Dataset preparation materializes the pinned FLEURS Persian audio as 16-kHz mono PCM-16 WAV and
verifies every file against the committed 4,341-row manifest. The `fa-v1` normalizer applies NFKC,
letter folding, diacritic and punctuation removal, digit-glyph folding, ZWNJ-to-space conversion,
and whitespace collapse to both reference and prediction.

CER removes normalized whitespace before scoring. WER preserves normalized word boundaries. Both
are corpus rates formed by summing edit and reference counts across all recordings. Empty
predictions remain scoreable; empty normalized references are invalid suite data.

## Documentation

- [Benchmark contract](docs/benchmark-contract.md)
- [Adding a model](docs/adding-a-model.md)
- [Maintainer guide](docs/maintainer-guide.md)
- [Contributing source changes](docs/contributing.md)
- [Version roadmap](docs/roadmap.md)
- [Changelog](CHANGELOG.md)

## Version history

PESTE 1.0.0 used a Jetson AGX Orin host, batch-size-one inference, and a memory-efficiency board.
Those results are retired and not comparable with v2; they remain preserved at git tag `1.0.0`.
Version 2.0.0 rebases all official results on the RTX profile and schema 2.

## Scope and limitations

FLEURS is public read speech and may overlap model training data. It does not represent all
conversational, noisy, accented, domain-specific, streaming, or long-form Persian. The benchmark
does not evaluate timestamps, diarization, punctuation quality, training uncertainty, deployment
robustness, or production suitability. Prompts, hotwords, quantization, offload, compilation,
external language models, and policy-changing fallbacks are excluded.

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
  version = {2.0.0}
}
```
