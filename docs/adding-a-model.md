# Adding a model

This guide is for a Hugging Face Persian ASR checkpoint already compatible with one supported
batched adapter. Contributors propose an immutable model specification; maintainers calibrate and
run it on the official RTX hardware profile.

## Supported adapters

| Adapter | Contract | Reference |
|---|---|---|
| `transformers-whisper` | `AutoProcessor` + `AutoModelForSpeechSeq2Seq`, non-truncating padded audio, automatic internal long-form timestamps, Persian text output | [`whisper-large-v3.json`](../models/whisper-large-v3.json) |
| `transformers-qwen` | `AutoProcessor.apply_transcription_request` + `AutoModelForMultimodalLM`, native padded multi-audio request | [`qwen3-asr-1-7b.json`](../models/qwen3-asr-1-7b.json) |
| `transformers-ctc` | `AutoProcessor` + `AutoModelForCTC`, padded greedy CTC batch with fixed token policy | [`wav2vec2-large-xlsr-53-persian.json`](../models/wav2vec2-large-xlsr-53-persian.json) |
| `nemo-rnnt` | one `.nemo` checkpoint, default RNNT, native `transcribe(paths, batch_size=n)` | [`nvidia-fastconformer-fa.json`](../models/nvidia-fastconformer-fa.json) |
| `nemo-ctc` | one hybrid `.nemo` checkpoint, auxiliary greedy CTC decoder, native `transcribe(paths, batch_size=n)` | [`shenava-rizeh-v1-0.json`](../models/shenava-rizeh-v1-0.json) |

The adapter must return exactly one ordered transcription per input and preserve singleton-
normalized output under batching. A checkpoint is incompatible with a model-only proposal when it
requires custom remote code, a different prompt or decoder, new dependencies, another precision,
quantization, offload, compilation, or model-specific postprocessing.

## Model specification

Use a full 40-character checkpoint revision and the repository-declared license. A Whisper
example is:

```json
{
  "schema_version": 2,
  "model_id": "example-persian-whisper",
  "repository": "organization/example-persian-whisper",
  "revision": "0123456789abcdef0123456789abcdef01234567",
  "adapter": "transformers-whisper",
  "native_dtype": "float16",
  "license": "Apache-2.0",
  "language": "fa",
  "generation": {
    "task": "transcribe",
    "max_new_tokens": 444,
    "return_timestamps": "auto"
  },
  "runtime": {
    "name": "modern",
    "image": "ghcr.io/armanjr/peste-benchmark:2.1.0",
    "dockerfile": "runtimes/Dockerfile"
  },
  "speed_profile": {
    "hardware_profile_id": "rtx-6000-ada-v1",
    "batch_size": 8
  }
}
```

| Field | Requirement |
|---|---|
| `schema_version` | Must remain `2` |
| `model_id` | Stable lowercase letters, digits, and hyphens; matches filename |
| `repository` | Hugging Face `owner/name` |
| `revision` | Full immutable lowercase commit hash |
| `adapter` | One supported key above |
| `native_dtype` | `float16`, `bfloat16`, or `float32`; no fallback |
| `language`, `generation`, `runtime` | Match the selected adapter reference policy |
| `speed_profile.hardware_profile_id` | Exactly `rtx-6000-ada-v1` |
| `speed_profile.batch_size` | Positive integer selected by official calibration |

The `-v1` hardware-profile revision is independent of the PESTE release. Both speed-profile fields
affect the model digest and force results to regenerate when changed.

For Whisper, `return_timestamps = "auto"` means timestamp tokens are internal control tokens only:
the adapter enables them when the processed batch exceeds one native 30-second segment, then
decodes text with special tokens removed. Contributors must not replace this with manual
truncation, ad-hoc chunks, or published timestamp output.

For a tracked multi-model campaign, batch size 1 may appear in the calibration-image commit as an
explicitly provisional value because schema 2 requires a positive integer. It is not an official
profile. `peste campaign run` refuses to run a candidate until reviewed calibration evidence has
been applied and its selected batch size matches the campaign state.

## Contributor validation

```bash
uv sync --frozen --all-groups
uv run peste model validate --model example-persian-whisper
uv run peste validate-specs
uv run pytest
```

Do not submit weights, audio, predictions, unofficial scores, generated boards, or credentials.
Include repository URL, revision, license, adapter evidence, native dtype, and validation commands
in the proposal.

## Maintainer qualification

Maintainers publish the carrier image for the candidate commit, provision a doctor-approved
container, prefetch the checkpoint, run the real offline smoke test twice, and exercise multi-item
padding, masks, dtype/device movement, ordered decoding, cardinality, and singleton equivalence.
They then run `peste model profile-speed` and commit the selected batch size. The carrier image
must be rebuilt after that commit because it contains the model specification; only then may one
fresh uninterrupted full evaluation run.

A model remains unranked if it OOMs, diverges, violates adapter cardinality/order, or cannot finish
under the fixed contract. Policy is never changed after observing a failure merely to obtain a
score.
