# Adding a model

This guide is for proposing a Hugging Face ASR checkpoint that is already compatible with one of
PESTE's supported adapters. The contribution is a model specification, not a benchmark score.
Maintainers validate the checkpoint on the official Jetson, run the complete suite, and publish
accepted results.

If the checkpoint requires a new inference API, dependency stack, decoding policy, or adapter,
open an issue describing the model instead of implementing support in a model-proposal pull
request. New adapter support is maintained under the [maintainer guide](maintainer-guide.md).

## 1. Confirm adapter compatibility

Choose an adapter based on the checkpoint's actual loading and transcription interface, not only
its architecture name.

| Adapter | Compatible checkpoint contract | Reference specification |
|---|---|---|
| `transformers-whisper` | Loads with `AutoProcessor` and `AutoModelForSpeechSeq2Seq`; supports Persian, transcription, and no-timestamp decoder tokens; does not require custom remote code | [`whisper-large-v3.json`](../models/whisper-large-v3.json) |
| `transformers-qwen` | Loads with `AutoProcessor` and `AutoModelForMultimodalLM`; processor implements `apply_transcription_request`; decoded output supports `transcription_only` | [`qwen3-asr-1-7b.json`](../models/qwen3-asr-1-7b.json) |
| `transformers-ctc` | Loads a standard CTC checkpoint with `AutoProcessor` and `AutoModelForCTC`; supports single-sample greedy decoding without an external language model or custom remote code | [`wav2vec2-large-xlsr-53-persian.json`](../models/wav2vec2-large-xlsr-53-persian.json) |
| `vibevoice` | Uses the supported VibeVoice ASR model and processor classes, pinned auxiliary tokenizer, BF16/SDPA, and fixed deterministic generation policy | [`vibevoice-asr.json`](../models/vibevoice-asr.json) |
| `nemo-rnnt` | Hugging Face snapshot contains exactly one `.nemo` checkpoint loadable by `ASRModel.restore_from`; supports the default RNNT decoder at batch size 1 | [`nvidia-fastconformer-fa.json`](../models/nvidia-fastconformer-fa.json) |

A checkpoint is not compatible when it requires any of the following:

- another model or processor class;
- custom remote code not already supported by the adapter;
- a different prompt, language, tokenizer, decoder, or generation policy;
- an additional runtime dependency or system package;
- quantization, offload, compilation, or a precision override; or
- preprocessing or postprocessing that the adapter does not implement.

The `transformers-ctc` contract is limited to batch-size-one greedy decoding with token grouping
enabled. It retains special tokens such as `<unk>` so recognition failures remain scoreable, and
does not support beam search, external language models, MMS language adapters, custom remote code,
or model-specific postprocessing.

Do not change adapter code, runtime dependencies, or benchmark policy to make a checkpoint fit a
model-proposal pull request.

## 2. Check the checkpoint

Before creating a specification, confirm that:

- the checkpoint performs Persian speech recognition;
- its Hugging Face repository is accessible to benchmark maintainers;
- the repository identifies a license that permits evaluation;
- the model card describes the architecture and expected inference interface;
- an immutable 40-character commit revision is available; and
- the checkpoint is not already represented under [`models/`](../models).

Use the full commit revision, not `main`, a tag, a branch, or an abbreviated hash. Record the
license declared for the checkpoint; do not infer a license from its base model.

## 3. Create the model specification

Copy the reference JSON file for the selected adapter to `models/<model-id>.json`. A Whisper
proposal, for example, has this structure:

```json
{
  "schema_version": 1,
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
    "return_timestamps": false
  },
  "runtime": {
    "name": "modern",
    "image": "peste-modern:1.0.0",
    "dockerfile": "runtimes/modern/Dockerfile"
  }
}
```

### Field requirements

| Field | Requirement |
|---|---|
| `schema_version` | Must remain `1` for the current schema |
| `model_id` | Stable lowercase identifier using letters, digits, and hyphens; must match the filename |
| `repository` | Hugging Face repository in `owner/name` form |
| `revision` | Full immutable 40-character lowercase commit hash |
| `adapter` | One of the supported adapter keys listed above |
| `native_dtype` | Checkpoint-native benchmark precision: `float16`, `bfloat16`, or `float32` |
| `license` | License identifier declared by the checkpoint repository |
| `language` | Copy from the selected adapter's reference specification |
| `generation` | Copy exactly from the selected adapter's reference specification |
| `runtime` | Copy exactly from the selected adapter's reference specification |

The benchmark intentionally holds language, generation, and runtime policy constant within each
adapter. A proposal that needs different values requires maintainer review as an adapter or
benchmark-contract change.

## 4. Validate the proposal

Install the project environment and validate the specification:

```bash
uv sync --frozen --all-groups
uv run --frozen peste model validate --model example-persian-whisper
uv run --frozen peste validate-specs
uv run --frozen pytest
```

Local validation checks the schema, immutable policy, and referenced runtime files. It does not
download or execute the checkpoint; passing it is necessary but does not prove compatibility.

Do not add predictions, result bundles, generated leaderboards, or claimed benchmark scores to
the proposal. Contributor-provided runs are not official results.

## 5. Open the pull request

Keep the pull request limited to the new model specification unless a documentation correction is
directly required. Include the following in the description:

- model ID and Hugging Face URL;
- immutable revision;
- declared license;
- selected adapter and evidence that the checkpoint uses its supported interface;
- native precision;
- whether Hugging Face access approval or authentication is required; and
- the local validation commands that passed.

Before submission, confirm:

- [ ] The filename and `model_id` match.
- [ ] The repository, revision, and license are exact.
- [ ] Generation and runtime fields match the adapter reference specification.
- [ ] No existing model specification or published result was edited.
- [ ] No credentials, downloaded weights, audio, predictions, or unofficial scores are included.
- [ ] Tests and specification validation pass.

## Maintainer evaluation

After the proposal is accepted for evaluation, maintainers rebuild the relevant runtime image,
prefetch the pinned checkpoint, run the deterministic one-sample smoke test, and execute the
complete suite on the official hardware. A compatible model may still remain unranked if it fails,
runs out of memory, produces nondeterministic output, or cannot complete the suite under the fixed
contract. Failure diagnostics are retained rather than changing the policy for that model.

Accepted official results are published by maintainers in a separate result update. The model,
its Hugging Face link, and its metrics then appear automatically in the generated tables and plot.
