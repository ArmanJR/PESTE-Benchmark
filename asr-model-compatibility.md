# Persian ASR model compatibility with PESTE 2.1

Benchmark-state review: **2026-08-16**

Campaign qualification update: **2026-08-12**. All 37 interface-compatible candidates were
tested on a doctor-approved RTX 6000 Ada host from the immutable calibration image. Thirty-four
passed real offline smoke, multi-item singleton equivalence, cardinality/order checks, VRAM
limits, and deterministic batch calibration. Three Zoha checkpoints failed the attention-mask
requirement for padding-safe CTC decoding and their provisional specifications were removed.

Official campaign update: **2026-08-13**. All 34 qualifiers received one fresh, uninterrupted
attempt on a new doctor-approved RTX 6000 Ada host using the final digest-pinned image. Thirty-three
completed all 871 evaluation recordings with valid speed measurements. Shenava Rizeh completed
768 recordings in six batch-128 measurements, then failed on the final 103 longest recordings at
NeMo's Conformer `conv2d` with PyTorch's 32-bit tensor-indexing limit. Its partial bundle is
retained as failed and unranked; it was not retried with a post-qualification batch change.

Corrective follow-up: **2026-08-13**. Shenava Rizeh was admitted through the dedicated
`nemo-ctc` adapter, conservatively recalibrated to batch 16 with longest-recording stress, and
completed a fresh uninterrupted 871-recording run. The original failed bundle remains auditable.

PESTE 2.1 refresh: **2026-08-16**. The 29 Whisper specifications now use non-truncating input and
automatic native long-form decoding. Their 2.0 bundles are historical evidence but are stale under
the new model digests until the tracked `whisper-longform-2-1-0` campaign finishes. The 13
unchanged non-Whisper bundles remain current.

Base inventory: [`asr-models-on-hf.md`](asr-models-on-hf.md) (**82 repositories**, Hub audit
performed **2026-08-01**). Four subsequently requested independent ASR checkpoints are also
counted below, for **86 models total**.

## Answer

Of the 86 counted models:

| Status | Models | Meaning |
|---|---:|---|
| Already evaluated successfully | 42 | A pinned schema-2 model specification and successful official result bundle exist; 29 Whisper bundles require the 2.1 refresh before ranking again. |
| Qualified; official evaluation failed | 0 | Shenava Rizeh's original failure was superseded by its successful dedicated-CTC follow-up. |
| Qualified; official evaluation pending | 0 | Every successful qualifier received an official full-run attempt. |
| Compatible candidate not yet qualified | 0 | Every admitted candidate in the campaign has reached a qualification outcome. |
| Conditional candidate | 9 | The inference interface fits, but access or a repository-declared license blocks admission. |
| Interface-compatible, but failed v2 qualification | 3 | The three Zoha processors did not emit the attention mask required for padding-safe batched CTC decoding. |
| Not compatible with the current adapters/backend | 32 | A new adapter/runtime, different decoding policy, or repository repair is required. |
| **Total** | **86** | The 82-model base inventory and four subsequent additions are each classified once below. |

The current evidence supports **42 models that use the existing adapters and have complete
official results**. The original interface/packaging audit found 37 additional candidates,
but qualification correctly rejected three; interface compatibility alone is not a benchmark
admission guarantee, and qualification does not guarantee completion on every evaluation length.

PESTE 1.0.0 used schema 1, batch-size-one inference, and a Jetson AGX Orin. Those result bundles
are retired and are not evidence of current evaluation status. The classifications below use the
schema 2, the `rtx-6000-ada-v1` hardware profile, calibrated native batching, and the five current
adapters: `transformers-whisper`, `transformers-qwen`, `transformers-ctc`, `nemo-rnnt`, and
`nemo-ctc`.

## Successful official evaluations (42)

These are the strongest evidence: each repository has a pinned specification under `models/` and
a successful result bundle under `results/fleurs-fa-ir-v1/`. The eight entries below predate the
37-model campaign; the other 34 successful bundles, including Shenava Rizeh's corrective follow-up,
are documented in the outcomes that follow. Whisper's 2.0 bundles remain historical until their
2.1 replacements complete.

| Hugging Face model card | Current adapter | Pinned revision | Evidence |
|---|---|---|---|
| [`steja/whisper-large-persian`](https://huggingface.co/steja/whisper-large-persian) | `transformers-whisper` | `4c8e5a01a1a684aa59b36a70b2ce408ad780af11` | Successful complete schema-2 run; batch 1 |
| [`Paulwalker4884/whisper-persian`](https://huggingface.co/Paulwalker4884/whisper-persian) | `transformers-whisper` | `80f96e52051b23f239db9b9798c41ed04d5aa568` | Successful complete schema-2 run; batch 1; the repository's LoRA packaging loads through the supported auto classes |
| [`openai/whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3) | `transformers-whisper` | `06f233fe06e710322aca913c1bc4249a0d71fce1` | Successful complete schema-2 run; batch 1 |
| [`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo) | `transformers-whisper` | `41f01f3fe87f28c78e2fbf8b568835947dd65ed9` | Successful complete schema-2 run; batch 1 |
| [`Qwen/Qwen3-ASR-0.6B-hf`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) | `transformers-qwen` | `7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c` | Successful complete schema-2 run; batch 1 |
| [`Qwen/Qwen3-ASR-1.7B-hf`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) | `transformers-qwen` | `bcd2b5b7f32b480ab5790554cfa8347f246a14f3` | Successful complete schema-2 run; batch 1 |
| [`jonatasgrosman/wav2vec2-large-xlsr-53-persian`](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-persian) | `transformers-ctc` | `234714078a1398a9db88194c5a40fefe6f376dc1` | Successful complete schema-2 run; batch 1 |
| [`nvidia/stt_fa_fastconformer_hybrid_large`](https://huggingface.co/nvidia/stt_fa_fastconformer_hybrid_large) | `nemo-rnnt` | `249cf5bf70dda7220a60ddeeecff2f6aad8e1784` | Successful complete schema-2 run; calibrated batch 32 |

All eight bundles report `status: success`, cover all 871 evaluation recordings, and have valid,
uninterrupted speed measurements. The NVIDIA result supersedes its retired v1 CUDA-OOM attempt;
the v2 adapter disables RNNT CUDA graphs for variable-length batched inference and the v2 run
completed under the fixed current policy.

## Qualification and official campaign outcomes (37)

These repositories matched an existing adapter and declared a usable license. The campaign fixed
their exact identities and tested all 37 without changing precision, decoding policy, or model
artifacts. Thirty-four received calibrated schema-2 specifications; 33 subsequently completed the
official evaluation and one produced a failed, unranked full-run attempt. The three failed
provisional specifications were removed. Except where explicitly identified as a subsequent
audit, “Observed revision” is the immutable Hub commit recorded on 2026-08-01; metadata and access
were rechecked on 2026-08-12.

### Persian Whisper fine-tunes (20)

All 20 exposed a standard root-level Whisper checkpoint and processor. Under the modern runtime's
pinned Transformers 5.14.1 stack, `AutoConfig` and `AutoProcessor` loaded without remote code, and
their tokenizers contained `<|fa|>`, `<|transcribe|>`, and `<|notimestamps|>`. They fit the loading
and decoding policy of `transformers-whisper`; checkpoints lacking modern generation metadata are
covered by its legacy-generation shim. All 20 passed real v2 qualification: 16 selected batch 1,
three selected batch 2, and one selected batch 4. All 20 then completed successful official runs.

| Hugging Face model card | Observed revision | Declared license |
|---|---|---|
| [`steja/whisper-small-persian`](https://huggingface.co/steja/whisper-small-persian) | `8c600b6b69f552730abab77a6a46311d673ea584` | Apache-2.0 |
| [`Neurai/NeuraSpeech_WhisperBase`](https://huggingface.co/Neurai/NeuraSpeech_WhisperBase) | `d0f95e6dcbfffe41a17d9a5d276a4a92ac19faa6` | Apache-2.0 |
| [`aictsharif/whisper-tiny-fa`](https://huggingface.co/aictsharif/whisper-tiny-fa) | `5cabeea3f4de08c66bfd7f6fc5449ca6b47f5115` | Apache-2.0 |
| [`aictsharif/whisper-base-fa`](https://huggingface.co/aictsharif/whisper-base-fa) | `9d6937cf8feac93c893e158a8176a051ecb3cc2a` | Apache-2.0 |
| [`aictsharif/whisper-small-fa`](https://huggingface.co/aictsharif/whisper-small-fa) | `21a6bdfb4955cc0dfd8d13cc8a3dff183b760631` | Apache-2.0 |
| [`aictsharif/whisper-medium-fa`](https://huggingface.co/aictsharif/whisper-medium-fa) | `d8deaeed83a869c78be19b6e5f44cd2a2472173d` | Apache-2.0 |
| [`aictsharif/whisper-large-v2-fa`](https://huggingface.co/aictsharif/whisper-large-v2-fa) | `ba330761086f7b5086c83d129f576c20d2426f49` | Apache-2.0 |
| [`SeyedAli/whisper-fa-small-v1`](https://huggingface.co/SeyedAli/whisper-fa-small-v1) | `06f1b8b524d488dc68bd89beee34861390eab732` | MIT |
| [`MohammadReza-Halakoo/persian-whisper-large-v3-10-percent-17-0-one-epoch`](https://huggingface.co/MohammadReza-Halakoo/persian-whisper-large-v3-10-percent-17-0-one-epoch) | `3b32d4db7767a534bc315967e16a5fd6dd9d9cbb` | Apache-2.0 |
| [`nezamisafa/whisper-persian-v4`](https://huggingface.co/nezamisafa/whisper-persian-v4) | `b84fc89f5d8c6a08acbd0930c74010f8bb555253` | Apache-2.0 |
| [`nezamisafa/whisper-v3-turbo-persian-v1.0`](https://huggingface.co/nezamisafa/whisper-v3-turbo-persian-v1.0) | `d9dfe321e0bcbd0bfe2ae1f5d72e1f8b580518cf` | MIT |
| [`aliyzd95/whisper-small-persian-v1`](https://huggingface.co/aliyzd95/whisper-small-persian-v1) | `6f667ff324f2a4918e7df54d52f638a1c8d8263c` | Apache-2.0 |
| [`MohammadGholizadeh/whisper-small-fa`](https://huggingface.co/MohammadGholizadeh/whisper-small-fa) | `d13d3798d2319e3fdfdf2f38f0eb3e7f18ace4f1` | MIT |
| [`hackergeek98/whisper-fa-tinyyy`](https://huggingface.co/hackergeek98/whisper-fa-tinyyy) | `e3eba5cdb1a64de5689ad231ee9aeff365b16094` | MIT |
| [`taesiri/whisper-small-fa-7`](https://huggingface.co/taesiri/whisper-small-fa-7) | `e3f1fa1f55ce8e24bcecce1bccb5940077df92a6` | Apache-2.0 |
| [`AmirMohseni/whisper-small-persian`](https://huggingface.co/AmirMohseni/whisper-small-persian) | `349c3edab4a19cc9c59147a69effb718fdc6cd38` | Apache-2.0 |
| [`makhataei/Whisper-Small-Ctejarat`](https://huggingface.co/makhataei/Whisper-Small-Ctejarat) | `16332d3dbba6e0b5372c2ee42d0b9d4b1a20973e` | Apache-2.0 |
| [`alism98/whisper-small-persian`](https://huggingface.co/alism98/whisper-small-persian) | `4b7c6eb71d6495d16e9991fb95197d3611b972ff` | CreativeML Open RAIL-M |
| [`mjavadf/whisper-small-fa`](https://huggingface.co/mjavadf/whisper-small-fa) | `ff66e51dc5b4c7763d28803775efc4ab88546277` | Apache-2.0 |

### Multilingual Whisper baselines (5)

The cards use the same `AutoProcessor` plus `AutoModelForSpeechSeq2Seq` interface as the current
adapter, and these are the multilingual checkpoints (not the English-only `.en` variants). All
five passed v2 qualification and completed successful official runs; `whisper-small` selected
batch 2 and the other four selected batch 1.

| Hugging Face model card | Observed revision | Declared license |
|---|---|---|
| [`openai/whisper-tiny`](https://huggingface.co/openai/whisper-tiny) | `169d4a4341b33bc18d8881c4b69c2e104e1cc0af` | Apache-2.0 |
| [`openai/whisper-base`](https://huggingface.co/openai/whisper-base) | `e37978b90ca9030d5170a5c07aadb050351a65bb` | Apache-2.0 |
| [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) | `973afd24965f72e36ca33b3055d56a652f456b4d` | Apache-2.0 |
| [`openai/whisper-medium`](https://huggingface.co/openai/whisper-medium) | `abdf7c39ab9d0397620ccaea8974cc764cd0953e` | Apache-2.0 |
| [`openai/whisper-large-v2`](https://huggingface.co/openai/whisper-large-v2) | `ae4642769ce2ad8fc292556ccea8e901f1530655` | Apache-2.0 |

### Qualified NeMo models (2)

| Hugging Face model card | Observed revision | Declared license | Compatibility evidence |
|---|---|---|---|
| [`Reza2kn/Shenava-Rizeh-v1.0`](https://huggingface.co/Reza2kn/Shenava-Rizeh-v1.0) | `74c96b7c23d8611dd4d0c775744f43bc4fb9c2ec` | Apache-2.0 | Successful dedicated `nemo-ctc` follow-up; batch 16; all 871 recordings; the earlier batch-128 RNNT-policy failure remains diagnostic |
| [`Reza2kn/visualears-fastconformer-fa-full-ab`](https://huggingface.co/Reza2kn/visualears-fastconformer-fa-full-ab) | `7f43a9d41d06328605257f0f28542c2f2332ed55` | Apache-2.0 | Successful complete official run; batch 32; CER 5.18%; WER 15.52%; 378.349× throughput |

### Transformers CTC candidates (10)

The `transformers-ctc` adapter supports standard root-level `AutoProcessor` plus
`AutoModelForCTC` checkpoints using deterministic padded batches and greedy decoding without an
external language model. Seven candidates passed real v2 smoke, attention-mask-safe decoding,
singleton equivalence, and RTX calibration at batch 1, then completed successful official runs.
The three Zoha checkpoints loaded but failed smoke because their processors did not emit an
attention mask; the adapter refuses to publish padding-unsafe batched output.

| Hugging Face model card | Observed revision | Declared license | Compatibility evidence |
|---|---|---|---|
| [`m3hrdadfi/wav2vec2-large-xlsr-persian`](https://huggingface.co/m3hrdadfi/wav2vec2-large-xlsr-persian) | `a6fc7cdc898c6ec218e7f337a4835c3cd1ab8fab` | Apache-2.0 | Qualified v2; batch 1 |
| [`m3hrdadfi/wav2vec2-large-xlsr-persian-v2`](https://huggingface.co/m3hrdadfi/wav2vec2-large-xlsr-persian-v2) | `599d7361d87b6ea3ca5d64a993e8ad8c942c48eb` | Apache-2.0 | Qualified v2 with 315,479,720 parameters; batch 1 |
| [`m3hrdadfi/wav2vec2-large-xlsr-persian-shemo`](https://huggingface.co/m3hrdadfi/wav2vec2-large-xlsr-persian-shemo) | `f9aa526bb0408f48543d0359dca089555adefc05` | Apache-2.0 | Qualified v2; batch 1 |
| [`ghofrani/xls-r-1b-fa-cv8`](https://huggingface.co/ghofrani/xls-r-1b-fa-cv8) | `c38ce46e838cade8ecadc7ff5ad5fb58fd7cda95` | Apache-2.0 | Qualified v2 with 315,567,870 parameters despite the repository name; batch 1 |
| [`alifarokh/wav2vec2-xls-r-300m-fa`](https://huggingface.co/alifarokh/wav2vec2-xls-r-300m-fa) | `79d44772d3bfc1f9000748c8478781662a5fbc64` | MIT | Qualified v2; batch 1 |
| [`SeyedAli/Persian-Speech-Transcription-Wav2Vec2-V1`](https://huggingface.co/SeyedAli/Persian-Speech-Transcription-Wav2Vec2-V1) | `21623b1ffbdcb4c79bf7bd74737ab30237db4b66` | MIT | Qualified v2; batch 1 |
| [`zoha/wav2vec2-base-common-voice-persian-colab`](https://huggingface.co/zoha/wav2vec2-base-common-voice-persian-colab) | `6267762ef6345f5e673123a26c873a4e340c08e2` | Apache-2.0 | Failed v2 smoke: processor omitted the required attention mask |
| [`zoha/wav2vec2-base-common-voice-40p-persian-colab`](https://huggingface.co/zoha/wav2vec2-base-common-voice-40p-persian-colab) | `fca861e14529491fba97caa766e52394bd8616c4` | Apache-2.0 | Failed v2 smoke: processor omitted the required attention mask |
| [`zoha/wav2vec2-xlsr-persian-50p`](https://huggingface.co/zoha/wav2vec2-xlsr-persian-50p) | `e780f1e94a4b181fc88fef263281996c491cd60f` | Apache-2.0 | Failed v2 smoke: processor omitted the required attention mask |
| [`masoumehb/wav2vec2-large-xlsr-persian-v3`](https://huggingface.co/masoumehb/wav2vec2-large-xlsr-persian-v3) | `918f655ca45ef4b729b496288139114a3fdf2b1a` | Apache-2.0 | Qualified v2; batch 1 |

The tracked summary records compact qualification status, selected batches, evidence hashes, and
official run outcomes. Verbose smoke, calibration, and failure logs remain outside git. All 33
initially complete campaign bundles plus Shenava Rizeh's corrective follow-up produced the
42-model 2.0 leaderboard; the failed Shenava bundle remains diagnostic and unranked. Qualification
throughput is not a benchmark score.

## Conditional candidates (9)

These do not appear to require a new inference architecture, but they were **not ready for
benchmark admission at the 2026-08-01 Hub audit**.

| Hugging Face model card | Observed revision | Existing adapter | Current blocker |
|---|---|---|---|
| [`C1Tech/whisper_small_persian`](https://huggingface.co/C1Tech/whisper_small_persian) | `685a77c4fbf88e8281a939290e63369797f8cdad` | `transformers-whisper` | Manually gated. Public metadata shows a complete standard Whisper checkpoint and Apache-2.0, but the unauthenticated processor probe received HTTP 401. It becomes a candidate only after maintainers obtain access and verify it with `HF_TOKEN`. |
| [`Yasaman/whisper_fa`](https://huggingface.co/Yasaman/whisper_fa) | `d55dd6374fbabfe314734a6cff6dbaa6c892646d` | `transformers-whisper` | Processor/token probe passed, but the repository declares no license. The current model specification requires the repository-declared license. |
| [`makhataei/Whisper-Small-Common-Voice`](https://huggingface.co/makhataei/Whisper-Small-Common-Voice) | `784438568649d229933e24d9af0fa7afc325e230` | `transformers-whisper` | Complete root checkpoint and processor/token probe passed, but the model card declares no license. |
| [`alisharifi/whisper-farsi`](https://huggingface.co/alisharifi/whisper-farsi) | `b7e73920b692fcb7c8eaf1f1494c84e73712d352` | `transformers-whisper` | Complete root checkpoint and processor/token probe passed, but the model card declares no license. |
| [`m3hrdadfi/wav2vec2-large-xlsr-persian-v3`](https://huggingface.co/m3hrdadfi/wav2vec2-large-xlsr-persian-v3) | `f3ceecb54fc81bb796f1565429bcf5599cd0e24d` | `transformers-ctc` | Complete standard CTC packaging, but no license is declared. |
| [`lnxdx/Wav2Vec2-Large-XLSR-Persian-ShEMO`](https://huggingface.co/lnxdx/Wav2Vec2-Large-XLSR-Persian-ShEMO) | `d18a45886779905dea3826db0cacd07841c48be6` | `transformers-ctc` | Complete standard CTC packaging, but no license is declared. |
| [`zoha/wav2vec2-base-common-voice-90p-persian-colab`](https://huggingface.co/zoha/wav2vec2-base-common-voice-90p-persian-colab) | `879a28d30e1db43b6da7d43aef8a2fb69f6f33d3` | `transformers-ctc` | Complete standard CTC packaging, but the repository-declared license required for a model specification is absent. |
| [`zoha/wav2vec2-xlsr-persian`](https://huggingface.co/zoha/wav2vec2-xlsr-persian) | `a05ebf85f6cd760096447ccd3760cc31d0b40474` | `transformers-ctc` | Complete standard CTC packaging, but the repository-declared license required for a model specification is absent. |
| [`rtler/wav2vec2-large-xls-r-300m-persian-12-colab`](https://huggingface.co/rtler/wav2vec2-large-xls-r-300m-persian-12-colab) | `e7875f0b245c144c0fcb6d415480b760e39d7132` | `transformers-ctc` | Complete standard CTC packaging, but the repository-declared license required for a model specification is absent. |

## Not compatible with the current adapters/backend (32)

### Whisper-named repositories that do not satisfy the Whisper adapter (7)

| Hugging Face model card | Why it cannot run now |
|---|---|
| [`MahdinourabadiAI/whisper-medium-fa`](https://huggingface.co/MahdinourabadiAI/whisper-medium-fa) | The repository was not accessible in the Hub audit: the API returned HTTP 401 with no model metadata/card, so it could not be inspected, pinned, or prefetched. |
| [`speechbrain/asr-whisper-large-v2-commonvoice-fa`](https://huggingface.co/speechbrain/asr-whisper-large-v2-commonvoice-fa) | The card requires SpeechBrain `WhisperASR.from_hparams`; the repository is packaged as a SpeechBrain pipeline, not a root `AutoProcessor`/`AutoModelForSpeechSeq2Seq` checkpoint. SpeechBrain is absent from the current runtime. |
| [`speechbrain/asr-whisper-medium-commonvoice-fa`](https://huggingface.co/speechbrain/asr-whisper-medium-commonvoice-fa) | Same SpeechBrain-specific `WhisperASR.from_hparams` interface and packaging; it needs a new adapter/runtime. |
| [`hezarai/whisper-small-fa`](https://huggingface.co/hezarai/whisper-small-fa) | The card requires `hezar.models.Model.load`; the Hub root has no Transformers config/processor/weights. The current runtime has no `hezar` dependency or adapter. |
| [`SanayAI/whisper-large-v3-persian-common-voice-17`](https://huggingface.co/SanayAI/whisper-large-v3-persian-common-voice-17) | The repository currently contains only `.gitattributes`: no card, config, processor, or checkpoint. |
| [`rezaFarsh/whisper-small-persian`](https://huggingface.co/rezaFarsh/whisper-small-persian) | The root has model weights/config and a feature extractor but no tokenizer assets. `AutoProcessor` constructed an incomplete tokenizer, and the required Persian/transcribe/no-timestamp token check failed; the current adapter would reject it. |
| [`mohammadMira/persian-whisper`](https://huggingface.co/mohammadMira/persian-whisper) | Only `.gitattributes` and an effectively empty README are present; there is no checkpoint, config, or processor. |

### CTC repositories that still fail the contract (11)

The standard Transformers CTC path is now supported, but the fixed contract intentionally has no
KenLM/`pyctcdecode` integration, MMS language-adapter selection, custom preprocessing repair, or
fallback behavior. These repositories still require a different decoding policy, additional
runtime dependencies, or repository repair. Standard CTC checkpoints blocked only by missing
license metadata are classified as conditional candidates above.

| Hugging Face model card | Card/files finding |
|---|---|
| [`SLPL/Sharif-wav2vec2`](https://huggingface.co/SLPL/Sharif-wav2vec2) | A historical pinned v1 Jetson smoke failed at `AutoProcessor` loading because `Wav2Vec2ProcessorWithLM` requires the absent `pyctcdecode` dependency; the card also requires KenLM |
| [`rsd16/wav2vec2-large-xlsr-53-fine-tuned-farsi`](https://huggingface.co/rsd16/wav2vec2-large-xlsr-53-fine-tuned-farsi) | CTC checkpoint with LM assets; the required external-LM path is unsupported, and no card/license is present |
| [`rsd16/wav2vec2-large-xlsr-53-fine-tuned-farsi-v2`](https://huggingface.co/rsd16/wav2vec2-large-xlsr-53-fine-tuned-farsi-v2) | No card or complete root config/processor |
| [`zoha/wav2vec2-base-common-voice-80p-persian-colab`](https://huggingface.co/zoha/wav2vec2-base-common-voice-80p-persian-colab) | Incomplete root repository with no card/config/processor |
| [`zoha/wav2vec2-base-common-voice-50p-persian-colab`](https://huggingface.co/zoha/wav2vec2-base-common-voice-50p-persian-colab) | CTC checkpoint packaged for external language-model decoding, which the fixed greedy contract does not support |
| [`Rasooli/wav2vec2-large-xls-r-300m-Farsi-colab`](https://huggingface.co/Rasooli/wav2vec2-large-xls-r-300m-Farsi-colab) | Incomplete root repository without model config/processor/weights; no card/license |
| [`Rasooli3003/wav2vec2-large-xls-r-300m-Farsi-colab`](https://huggingface.co/Rasooli3003/wav2vec2-large-xls-r-300m-Farsi-colab) | Incomplete root repository without model config/processor/weights; no card/license |
| [`pourzare/wav2vec2-large-xls-r-300m-persian-colab`](https://huggingface.co/pourzare/wav2vec2-large-xls-r-300m-persian-colab) | Incomplete root repository without card, model config, processor, or weights |
| [`manifoldix/xlsr-fa-lm`](https://huggingface.co/manifoldix/xlsr-fa-lm) | `Wav2Vec2ForCTC` plus required external-LM decoding; the fixed greedy contract does not support that processor/decoder path |
| [`tntchack/wav2vec_lm_farsi`](https://huggingface.co/tntchack/wav2vec_lm_farsi) | LM-oriented CTC repository requiring an unsupported decoder path; no card/license is present |
| [`AmirrezaV1/ASR-persian`](https://huggingface.co/AmirrezaV1/ASR-persian) | The root processor packaging is incomplete, so the standard CTC adapter cannot load it reliably |

### NeMo repositories that fail the current `nemo-rnnt` contract (7)

The adapter requires exactly one root-level `.nemo` archive, restores it with `ASRModel`, forces
the RNNT decoder, disables decoder CUDA graphs for variable-length batches, and calls native
`transcribe` with the configured batch size.

| Hugging Face model card | Why it cannot run now |
|---|---|
| [`Neurai/NeuraSpeech_900h`](https://huggingface.co/Neurai/NeuraSpeech_900h) | The card identifies FastConformer-TDT, while the adapter forces RNNT; the root also contains two `.nemo` archives, so the loader fails its exact-one check before restore. |
| [`MohammadGholizadeh/parakeet-ctc-1.1b-persian.nemo`](https://huggingface.co/MohammadGholizadeh/parakeet-ctc-1.1b-persian.nemo) | The card and archive are Parakeet CTC. There is one archive, but forcing `decoder_type="rnnt"` is incompatible. |
| [`alifarokh/nemo-conformer-medium-fa`](https://huggingface.co/alifarokh/nemo-conformer-medium-fa) | The card is sparse, but the single archive's embedded config targets `EncDecCTCModel` with greedy CTC decoding, not RNNT. |
| [`Reza2kn/Shenava-Koochik-v1.0`](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0) | At revision `45b4912965d37a60aa4e78e481cb4f8aa14029c7`, the root contains two `.nemo` archives. The current adapter requires exactly one and refuses to choose between them. |
| [`Reza2kn/Shenava-Koochik-0.9`](https://huggingface.co/Reza2kn/Shenava-Koochik-0.9) | Hybrid RNNT/CTC architecture, but nine root-level `.nemo` archives are present. The exact-one loader contract fails. |
| [`Reza2kn/Shenava-Rizeh-Pizeh-v1.0`](https://huggingface.co/Reza2kn/Shenava-Rizeh-Pizeh-v1.0) | Revision `bc3eeef5dbc38e8175ad133db5de95f148c320b1` has exactly one root archive, but its embedded target is `EncDecCTCModelBPE` with greedy CTC decoding; it has no RNNT decoder for this adapter. |
| [`Reza2kn/visualears-fastconformer-fa-69m-256x40-warmstart`](https://huggingface.co/Reza2kn/visualears-fastconformer-fa-69m-256x40-warmstart) | Added in the 2026-08-11 supplementary audit. Revision `9416a62bf215edcb85b6945332f462604e423e63` stores the final `.nemo` under `final/`, so the exact-one-root loader sees zero checkpoints. Selecting a nested artifact requires a loader contract change and decoder qualification. |

### Other multilingual/audio-language models requiring another interface (5)

| Hugging Face model card | Why it cannot run now |
|---|---|
| [`facebook/seamless-m4t-v2-large`](https://huggingface.co/facebook/seamless-m4t-v2-large) | Uses `SeamlessM4Tv2Model` and task/language-specific generation. No SeamlessM4T adapter or policy exists. |
| [`facebook/mms-1b-all`](https://huggingface.co/facebook/mms-1b-all) | The base CTC checkpoint loads and is deterministic, but a historical pinned v1 Jetson diagnostic confirmed that the generic adapter leaves `target_lang="eng"` and produced zero Arabic-script characters. Persian requires `set_target_lang("fas")` plus `load_adapter("fas")`, which remain unsupported. Generic prefetch would also acquire all 3,606 repository files (about 29.2 GB), including every language adapter. |
| [`Qwen/Qwen3-ASR-0.6B`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | The card packages this for the `qwen-asr` wrapper. It is the legacy layout (`thinker.*` checkpoint keys), not the native Transformers layout (`model.*`) consumed by the current adapter; use the already-supported `-hf` repository instead. |
| [`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | Same `qwen-asr`/legacy `thinker.*` layout; use the already-supported `Qwen3-ASR-1.7B-hf` repository instead. |
| [`microsoft/VibeVoice-ASR`](https://huggingface.co/microsoft/VibeVoice-ASR) | Its dedicated adapter, runtime, model specification, and result bundle were removed before v2. None of the four current adapters implements the VibeVoice processor/model interface, and its retired v1 result is not a current evaluation. Supporting it again would require a maintained adapter/runtime plus a v2 batching and calibration policy. |

### Vosk Farsi models requiring a new backend (2)

Source: [official Vosk model catalog](https://alphacephei.com/vosk/models)

| Model | Published facts | Why it cannot run now |
|---|---|---|
| `vosk-model-fa-0.42` | Apache-2.0; 1.6 GB; official catalog reports 16.7 WER on CV17 and 11.1 on its FLEURS evaluation | PESTE has no Vosk/Kaldi adapter or dependency, and its Hugging Face revision/prefetch contract cannot pin the external Vosk ZIP. |
| `vosk-model-small-fa-0.42` | Apache-2.0; 53 MB; official catalog reports 23.4 WER on CV17 and 14.0 on its FLEURS evaluation | Same unsupported Vosk interface, runtime, and artifact-source contract. |

Both are legitimate future-adapter candidates, but neither can be expressed as a current model-only
proposal. A new backend must define immutable artifact digests, deterministic full-utterance result
assembly, real multi-item execution, output order/cardinality, singleton equivalence, CPU/GPU
policy, and RTX calibration. The catalog's FLEURS numbers are not PESTE scores because their exact
split, transcript source, normalization, decoder settings, and timing contract are not established
as equal to PESTE v2.

## Supplementary requested audits

These entries were requested after the 82-repository inventory was frozen. The headline counts now
include four independent additions: Full A+B, the 69M warm-start, and the two Vosk Farsi models.
The Shenava collection contains 28 model repositories, but most are deployment exports or training
artifacts rather than independent raw-audio ASR checkpoints. The classification therefore
separates canonical checkpoints from alternate formats.

### Shenava 1.0 collection

Collection: [`Reza2kn/Shenava 1.0`](https://huggingface.co/collections/Reza2kn/shenava-10-open-streaming-persian-asr-and-captioning)

| Repository or group | Current status | PESTE v2 assessment |
|---|---|---|
| [`Reza2kn/Shenava-Rizeh-v1.0`](https://huggingface.co/Reza2kn/Shenava-Rizeh-v1.0) | **Official evaluation successful** | Already counted above. A dedicated `nemo-ctc` follow-up recalibrated revision `74c96b7c23d8611dd4d0c775744f43bc4fb9c2ec` to batch 16 and completed all 871 recordings with valid speed; the earlier failed bundle is retained. |
| [`Reza2kn/visualears-fastconformer-fa-full-ab`](https://huggingface.co/Reza2kn/visualears-fastconformer-fa-full-ab) | **Official evaluation successful** | Counted above as a subsequent addition. Revision `7f43a9d41d06328605257f0f28542c2f2332ed55` completed all 871 recordings at batch 32 with valid speed. |
| [`Reza2kn/Shenava-Koochik-v1.0`](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0) | **Not compatible with the current adapter** | Already counted above. Its current revision has two root `.nemo` files, while `nemo-rnnt` deliberately requires exactly one. Selecting one by filename would be a loader/policy change, not a model-only proposal. |
| [`Reza2kn/Shenava-Rizeh-Pizeh-v1.0`](https://huggingface.co/Reza2kn/Shenava-Rizeh-Pizeh-v1.0) | **Not compatible with the current adapter** | Already counted above. Its sole archive embeds `EncDecCTCModelBPE`; the current NeMo adapter forces RNNT and PESTE has no NeMo-CTC adapter. |
| [`Reza2kn/visualears-fastconformer-fa-69m-256x40-warmstart`](https://huggingface.co/Reza2kn/visualears-fastconformer-fa-69m-256x40-warmstart) | **Not compatible with the current adapter** | Counted above as a subsequent addition. Revision `9416a62bf215edcb85b6945332f462604e423e63` stores the final `.nemo` under `final/`, so the exact-one-root loader sees zero checkpoints. Supporting a selected nested artifact requires a loader contract change and subsequent decoder qualification. |
| [`Reza2kn/visualears-asr-full-ab-checkpoints`](https://huggingface.co/Reza2kn/visualears-asr-full-ab-checkpoints) | **Not an admissible model repository** | This is a training/checkpoint archive with many nested `.ckpt` and `.nemo` artifacts and no ASR pipeline tag or one canonical root checkpoint. It is provenance material, not a model-only candidate. |
| ONNX, CoreML, LiteRT, sherpa-onnx, and tract repositories | **Not compatible with current adapters** | PESTE has no adapter/runtime for these formats. Several exports are fixed-frame acoustic CTC cores that take precomputed features rather than complete raw-audio-to-text pipelines, so adding them would also require a new preprocessing and decoding policy. They are alternate deployments of parent models and should not be counted as independent checkpoints. |
| [`Reza2kn/ShenavaSanj-v1.0`](https://huggingface.co/Reza2kn/ShenavaSanj-v1.0) | **Out of scope** | It is a token-classification word-importance scorer for Semantic WER, not an ASR model. |

The actionable result is **two qualified Shenava NeMo models** for the existing backend: Rizeh
v1.0 and `visualears-fastconformer-fa-full-ab`. Both have complete official results; Rizeh's
earlier failed diagnostic remains part of the audit trail.

Compatibility does not validate the collection's published scores. The cards report a different
ITN/digit-normalization and attention-context convention, and the collection includes FLEURS-fa
evaluation artifacts. The Koochik Hub page also lists `fleurs-fa-benchmark` under “Datasets used
to train,” while the Full A+B card says an external-LM setting was calibrated on a FLEURS-256
slice. Before publication, maintainers should document any FLEURS training, model-selection, or
decoder-tuning overlap. PESTE's fixed default-RNNT/no-external-LM policy remains unchanged.

## Audit basis and limits

The PESTE-side classification was reconciled on 2026-08-11 against:

- the current adapter registry and implementations in `src/peste/adapters/transformers.py` and
  `src/peste/adapters/nemo.py`;
- the schema-2 model and ranking contracts in `docs/adding-a-model.md` and
  `docs/benchmark-contract.md`;
- the current pinned specifications under `models/`, 42 successful complete schema-2 bundles, and
  the earlier failed Shenava Rizeh diagnostic under `results/fleurs-fa-ir-v1/`;
- the modern runtime pins, including Transformers 5.14.1 and PEFT 0.20.0, plus the isolated NeMo
  2.7.3 runtime; and
- the current RTX batching, calibration, offline execution, and result-eligibility rules.

Candidate and rejection evidence is carried forward from the 2026-08-01 Hub audit:

- model cards, Hub metadata, file trees, and immutable observed revisions for all 82 repositories;
- processor-only probes for plausible Whisper candidates under the pinned modern runtime;
- complete-root packaging checks for standard Transformers CTC candidates;
- historical pinned, offline v1 Jetson singleton smoke tests for four CTC checkpoints;
- historical negative probes confirming the `pyctcdecode` failure for the LM-packaged Sharif
  model and the default-English MMS path without its Persian adapter; and
- embedded `model_config.yaml` inspection for ambiguous NeMo archives.

The supplementary audit additionally used current Hugging Face API metadata and file trees,
partial archive inspection sufficient to extract the NeMo `model_config.yaml` before the weight
payload, the official Vosk model catalog, and the Vosk API's published recognizer interfaces.

Hub repositories and access rules can change. An observed revision is immutable, but its recorded
metadata does not establish current access or v2 compatibility. Before admission, recheck the
repository-declared license and access, validate the final schema-2 specification and native
dtype, then perform pinned prefetch, two offline smoke runs, multi-item conformance, official batch
calibration, and a fresh full evaluation. Model-card claims, file audits, processor probes, and v1
singleton smokes do not prove that a v2 run will complete.
