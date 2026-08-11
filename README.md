<p align="center">
  <img src="docs/peste-logo.png" alt="PESTE logo" width="520">
</p>

# PESTE: Persian Speech to Text benchmark

PESTE (**PE**rsian **S**peech to **TE**xt) is a reproducible benchmark and leaderboard for
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

### Normalized accuracy

![Normalized accuracy leaderboard](generated/leaderboard-accuracy.svg)

| Order | Model | CER | WER | Word accuracy |
|---:|---|---:|---:|---:|
| — | No complete official results yet | — | — | — |

CER is the primary ranking metric because Persian WER is orthography-sensitive: fa-v1 converts ZWNJ to spaces, while CER ignores normalized whitespace. WER and derived word accuracy remain complementary, segmentation-sensitive measurements.

Point-estimate order does not establish statistical significance. Intervals use a deterministic 10,000-replicate utterance-level percentile bootstrap at 95% confidence with seed 20250731. Paired intervals containing zero are reported as no clear difference; these intervals measure test-set sampling uncertainty only.

### Steady-state speed

![Steady-state speed leaderboard](generated/leaderboard-speed.svg)

| Rank | Model | Batch | Throughput | RTF | Processing s | Audio s |
|---:|---|---:|---:|---:|---:|---:|
| — | No complete official results yet | — | — | — | — | — |

Throughput is total audio seconds divided by measured processing seconds; RTF is its reciprocal. Resumed runs retain accuracy but are excluded here.

<!-- LEADERBOARD:END -->

> **Pre-publication state:** the committed `batch_size: 1` values are safe bootstrap placeholders,
> not calibration claims. The empty boards are intentional until all eight profiles are measured
> on an accepted host and followed by fresh uninterrupted runs.

The accuracy board sorts by CER, WER, then model ID. The speed board sorts valid uninterrupted runs
by throughput descending, CER, WER, then model ID. There is no composite accuracy/speed score.

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
`selected_batch_size` into its model JSON, review the candidate evidence, and commit all eight
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
