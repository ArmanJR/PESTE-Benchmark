# Persian and Farsi Automatic Speech Recognition Models on Hugging Face

## Research scope and validation

This review identified **82 Hugging Face model repositories relevant to Persian/Farsi ASR** as of **August 1, 2026**. The inventory contains **68 Persian-specialized repositories**—fine-tuned models, Persian CTC checkpoints, SpeechBrain pipelines, NeMo/FastConformer models, and associated Persian checkpoints—and **14 multilingual foundation ASR or audio-language models with Persian coverage**.

The search combined Hugging Face model pages, organization and user inventories, Persian-model collections, task and language filters, and the Open Persian ASR Leaderboard. Hugging Face’s Persian collections confirm several established Wav2Vec2/XLS-R and Whisper repositories, while the Persian leaderboard independently references models such as `jonatasgrosman/wav2vec2-large-xlsr-53-persian`, `ghofrani/xls-r-1b-fa-cv8`, and multilingual Whisper baselines. citeturn18view0turn18view1turn18view2

| Category | Repositories |
|---|---:|
| Persian Whisper and SpeechBrain | 33 |
| Persian Wav2Vec2, XLS-R, MMS, and CTC | 27 |
| Persian NeMo, FastConformer, and Parakeet | 8 |
| Multilingual ASR and audio-language models with Persian coverage | 14 |
| **Total** | **82** |

The symbol **★** marks a model supplied in the original request. The supplied `steja/whisper-large-persian-steja` URL did not resolve as a public model during verification; the active matching Hugging Face repository is `steja/whisper-large-persian`, which is included below. Hugging Face currently indexes that corrected repository as Persian automatic speech recognition. citeturn6view0turn14search0turn18view1

## Whisper and SpeechBrain repositories

This group covers Persian-specific Whisper fine-tunes, SpeechBrain pipelines built around Whisper, LoRA-style adaptations, and a small number of older or sparsely documented Whisper repositories. The five `aictsharif` variants are separate fine-tunes covering Whisper Tiny through Large-v2. The two SpeechBrain repositories package Persian Common Voice Whisper models for use through SpeechBrain rather than only the Transformers pipeline. citeturn10search12turn10search0turn10search4turn2search0turn2search19

| Name | Architecture or status | Main Hugging Face URL |
|---|---|---|
| ★ `steja/whisper-large-persian` | Whisper Large; corrected form of supplied URL | <https://huggingface.co/steja/whisper-large-persian> |
| `steja/whisper-small-persian` | Whisper Small Persian fine-tune | <https://huggingface.co/steja/whisper-small-persian> |
| ★ `C1Tech/whisper_small_persian` | Whisper Small; gated-access repository | <https://huggingface.co/C1Tech/whisper_small_persian> |
| ★ `Neurai/NeuraSpeech_WhisperBase` | Whisper Base Persian ASR | <https://huggingface.co/Neurai/NeuraSpeech_WhisperBase> |
| ★ `aictsharif/whisper-tiny-fa` | Whisper Tiny Persian fine-tune | <https://huggingface.co/aictsharif/whisper-tiny-fa> |
| ★ `aictsharif/whisper-base-fa` | Whisper Base Persian fine-tune | <https://huggingface.co/aictsharif/whisper-base-fa> |
| ★ `aictsharif/whisper-small-fa` | Whisper Small Persian fine-tune | <https://huggingface.co/aictsharif/whisper-small-fa> |
| ★ `aictsharif/whisper-medium-fa` | Whisper Medium Persian fine-tune | <https://huggingface.co/aictsharif/whisper-medium-fa> |
| ★ `aictsharif/whisper-large-v2-fa` | Whisper Large-v2 Persian fine-tune | <https://huggingface.co/aictsharif/whisper-large-v2-fa> |
| ★ `SeyedAli/whisper-fa-small-v1` | Whisper Small Persian ASR | <https://huggingface.co/SeyedAli/whisper-fa-small-v1> |
| ★ `MahdinourabadiAI/whisper-medium-fa` | Whisper Medium; Common Voice Persian | <https://huggingface.co/MahdinourabadiAI/whisper-medium-fa> |
| ★ `Yasaman/whisper_fa` | Persian Whisper ASR checkpoint | <https://huggingface.co/Yasaman/whisper_fa> |
| `speechbrain/asr-whisper-large-v2-commonvoice-fa` | SpeechBrain pipeline; Whisper Large-v2 | <https://huggingface.co/speechbrain/asr-whisper-large-v2-commonvoice-fa> |
| `speechbrain/asr-whisper-medium-commonvoice-fa` | SpeechBrain pipeline; Whisper Medium | <https://huggingface.co/speechbrain/asr-whisper-medium-commonvoice-fa> |
| `hezarai/whisper-small-fa` | Whisper Small Persian fine-tune | <https://huggingface.co/hezarai/whisper-small-fa> |
| `Paulwalker4884/whisper-persian` | Whisper Base Persian LoRA/adaptation | <https://huggingface.co/Paulwalker4884/whisper-persian> |
| `MohammadReza-Halakoo/persian-whisper-large-v3-10-percent-17-0-one-epoch` | Whisper Large-v3 Persian experiment | <https://huggingface.co/MohammadReza-Halakoo/persian-whisper-large-v3-10-percent-17-0-one-epoch> |
| `vhdm/whisper-large-fa-v1` | Whisper Large Persian ASR | <https://huggingface.co/vhdm/whisper-large-fa-v1> |
| `SanayAI/whisper-large-v3-persian-common-voice-17` | Whisper Large-v3; Common Voice 17 Persian | <https://huggingface.co/SanayAI/whisper-large-v3-persian-common-voice-17> |
| `nezamisafa/whisper-persian-v4` | Persian Whisper checkpoint | <https://huggingface.co/nezamisafa/whisper-persian-v4> |
| `nezamisafa/whisper-v3-turbo-persian-v1.0` | Whisper Large-v3 Turbo Persian adaptation | <https://huggingface.co/nezamisafa/whisper-v3-turbo-persian-v1.0> |
| `aliyzd95/whisper-small-persian-v1` | Whisper Small Persian ASR | <https://huggingface.co/aliyzd95/whisper-small-persian-v1> |
| `MohammadGholizadeh/whisper-small-fa` | Whisper Small Persian ASR | <https://huggingface.co/MohammadGholizadeh/whisper-small-fa> |
| `hackergeek98/whisper-fa-tinyyy` | Whisper Tiny Persian checkpoint | <https://huggingface.co/hackergeek98/whisper-fa-tinyyy> |
| `taesiri/whisper-small-fa-7` | Whisper Small Persian checkpoint | <https://huggingface.co/taesiri/whisper-small-fa-7> |
| `AmirMohseni/whisper-small-persian` | Whisper Small Persian ASR | <https://huggingface.co/AmirMohseni/whisper-small-persian> |
| `makhataei/Whisper-Small-Common-Voice` | Whisper Small; Common Voice Persian lineage | <https://huggingface.co/makhataei/Whisper-Small-Common-Voice> |
| `makhataei/Whisper-Small-Ctejarat` | Whisper Small; Ctejarat Persian speech data | <https://huggingface.co/makhataei/Whisper-Small-Ctejarat> |
| `alisharifi/whisper-farsi` | Farsi Whisper repository | <https://huggingface.co/alisharifi/whisper-farsi> |
| `rezaFarsh/whisper-small-persian` | Whisper Small Persian ASR | <https://huggingface.co/rezaFarsh/whisper-small-persian> |
| `alism98/whisper-small-persian` | Whisper Small; Common Voice 13 Persian | <https://huggingface.co/alism98/whisper-small-persian> |
| `mjavadf/whisper-small-fa` | Whisper Small Persian ASR | <https://huggingface.co/mjavadf/whisper-small-fa> |
| `mohammadMira/persian-whisper` | Existing Persian Whisper repository; sparse metadata | <https://huggingface.co/mohammadMira/persian-whisper> |

The `alism98` page explicitly exposes the model through `AutoModelForSpeechSeq2Seq` and the Transformers automatic-speech-recognition pipeline, and tags it for Persian and Common Voice 13. The `mohammadMira` repository exists but currently has an empty model card and no ASR task metadata, so it should be treated as a repository to inspect rather than a ready-to-deploy recommendation. citeturn19search0turn19search1

Several of these models have limited or absent evaluation information. Presence on Hugging Face therefore does not imply that a model has been independently benchmarked, that its claimed word error rate is directly comparable with another model’s result, or that all necessary processor/tokenizer files are complete.

## Wav2Vec2, XLS-R, MMS, and CTC repositories

Persian Wav2Vec2 and XLS-R models form the second major family. The established checkpoints include `jonatasgrosman/wav2vec2-large-xlsr-53-persian`, multiple `m3hrdadfi` models, `ghofrani/xls-r-1b-fa-cv8`, and newer 300-million-parameter XLS-R adaptations. Hugging Face’s Persian collections explicitly classify many of these repositories as automatic speech recognition models. citeturn18view1turn18view2turn11search0turn11search3

| Name | Architecture or status | Main Hugging Face URL |
|---|---|---|
| `jonatasgrosman/wav2vec2-large-xlsr-53-persian` | Wav2Vec2 XLSR-53 CTC | <https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-persian> |
| `m3hrdadfi/wav2vec2-large-xlsr-persian` | Wav2Vec2 XLSR Persian CTC | <https://huggingface.co/m3hrdadfi/wav2vec2-large-xlsr-persian> |
| `m3hrdadfi/wav2vec2-large-xlsr-persian-v2` | Wav2Vec2 XLSR Persian, version 2 | <https://huggingface.co/m3hrdadfi/wav2vec2-large-xlsr-persian-v2> |
| `m3hrdadfi/wav2vec2-large-xlsr-persian-v3` | Wav2Vec2 XLSR Persian, version 3 | <https://huggingface.co/m3hrdadfi/wav2vec2-large-xlsr-persian-v3> |
| `m3hrdadfi/wav2vec2-large-xlsr-persian-shemo` | Wav2Vec2 XLSR trained with ShEMO-related data | <https://huggingface.co/m3hrdadfi/wav2vec2-large-xlsr-persian-shemo> |
| `lnxdx/Wav2Vec2-Large-XLSR-Persian-ShEMO` | Wav2Vec2 XLS-R 300M; Persian/ShEMO | <https://huggingface.co/lnxdx/Wav2Vec2-Large-XLSR-Persian-ShEMO> |
| `ghofrani/xls-r-1b-fa-cv8` | XLS-R 1B; Persian Common Voice 8 | <https://huggingface.co/ghofrani/xls-r-1b-fa-cv8> |
| `alifarokh/wav2vec2-xls-r-300m-fa` | XLS-R 300M Persian CTC | <https://huggingface.co/alifarokh/wav2vec2-xls-r-300m-fa> |
| `SLPL/Sharif-wav2vec2` | Sharif Persian Wav2Vec2 ASR | <https://huggingface.co/SLPL/Sharif-wav2vec2> |
| `SeyedAli/Persian-Speech-Transcription-Wav2Vec2-V1` | Persian speech transcription; Wav2Vec2 | <https://huggingface.co/SeyedAli/Persian-Speech-Transcription-Wav2Vec2-V1> |
| `rsd16/wav2vec2-large-xlsr-53-fine-tuned-farsi` | XLSR-53 fine-tuned for Farsi | <https://huggingface.co/rsd16/wav2vec2-large-xlsr-53-fine-tuned-farsi> |
| `rsd16/wav2vec2-large-xlsr-53-fine-tuned-farsi-v2` | Second Farsi XLSR-53 checkpoint | <https://huggingface.co/rsd16/wav2vec2-large-xlsr-53-fine-tuned-farsi-v2> |
| `zoha/wav2vec2-base-common-voice-persian-colab` | Wav2Vec2 Base; Common Voice Persian | <https://huggingface.co/zoha/wav2vec2-base-common-voice-persian-colab> |
| `zoha/wav2vec2-base-common-voice-90p-persian-colab` | Wav2Vec2 Base; 90-percent data experiment | <https://huggingface.co/zoha/wav2vec2-base-common-voice-90p-persian-colab> |
| `zoha/wav2vec2-base-common-voice-80p-persian-colab` | Wav2Vec2 Base; 80-percent data experiment | <https://huggingface.co/zoha/wav2vec2-base-common-voice-80p-persian-colab> |
| `zoha/wav2vec2-base-common-voice-50p-persian-colab` | Wav2Vec2 Base; 50-percent data experiment | <https://huggingface.co/zoha/wav2vec2-base-common-voice-50p-persian-colab> |
| `zoha/wav2vec2-base-common-voice-40p-persian-colab` | Wav2Vec2 Base; 40-percent data experiment | <https://huggingface.co/zoha/wav2vec2-base-common-voice-40p-persian-colab> |
| `zoha/wav2vec2-xlsr-persian` | Persian XLSR checkpoint | <https://huggingface.co/zoha/wav2vec2-xlsr-persian> |
| `zoha/wav2vec2-xlsr-persian-50p` | Persian XLSR partial-data experiment | <https://huggingface.co/zoha/wav2vec2-xlsr-persian-50p> |
| `Rasooli/wav2vec2-large-xls-r-300m-Farsi-colab` | XLS-R 300M Farsi checkpoint | <https://huggingface.co/Rasooli/wav2vec2-large-xls-r-300m-Farsi-colab> |
| `Rasooli3003/wav2vec2-large-xls-r-300m-Farsi-colab` | XLS-R 300M Farsi repository | <https://huggingface.co/Rasooli3003/wav2vec2-large-xls-r-300m-Farsi-colab> |
| `pourzare/wav2vec2-large-xls-r-300m-persian-colab` | XLS-R 300M Persian checkpoint | <https://huggingface.co/pourzare/wav2vec2-large-xls-r-300m-persian-colab> |
| `manifoldix/xlsr-fa-lm` | Persian XLSR ASR with language-model integration | <https://huggingface.co/manifoldix/xlsr-fa-lm> |
| `rtler/wav2vec2-large-xls-r-300m-persian-12-colab` | XLS-R 300M Persian checkpoint | <https://huggingface.co/rtler/wav2vec2-large-xls-r-300m-persian-12-colab> |
| `masoumehb/wav2vec2-large-xlsr-persian-v3` | Persian Wav2Vec2/XLSR checkpoint | <https://huggingface.co/masoumehb/wav2vec2-large-xlsr-persian-v3> |
| `tntchack/wav2vec_lm_farsi` | Farsi Wav2Vec2 plus language-model assets | <https://huggingface.co/tntchack/wav2vec_lm_farsi> |
| `AmirrezaV1/ASR-persian` | Transformers CTC Persian ASR repository | <https://huggingface.co/AmirrezaV1/ASR-persian> |

The `zoha` repositories are useful for reproducing partial-data experiments, but several have minimal cards or lack complete task metadata. The Persian collection nevertheless confirms the existence of the 90-, 80-, 50-, and 40-percent checkpoints, as well as the XLSR variants. citeturn18view2

`manifoldix/xlsr-fa-lm` is distinct from a plain acoustic checkpoint because its repository is oriented toward Persian ASR with language-model decoding. It is therefore potentially useful when comparing greedy CTC decoding against beam search with a Persian language model. citeturn15view3

## NeMo, FastConformer, and Parakeet repositories

This category contains Persian ASR models based on NVIDIA NeMo architectures, particularly FastConformer, hybrid transducer/CTC systems, and Parakeet-derived CTC models. These models are operationally different from Transformers-native Whisper or Wav2Vec2 checkpoints: several are loaded through NeMo rather than `transformers.pipeline`, and their repository files may use `.nemo` archives. citeturn11search2turn10search7turn2search9

| Name | Architecture or status | Main Hugging Face URL |
|---|---|---|
| `nvidia/stt_fa_fastconformer_hybrid_large` | NeMo FastConformer hybrid transducer/CTC | <https://huggingface.co/nvidia/stt_fa_fastconformer_hybrid_large> |
| `Neurai/NeuraSpeech_900h` | Persian FastConformer-TDT ASR | <https://huggingface.co/Neurai/NeuraSpeech_900h> |
| `MohammadGholizadeh/parakeet-ctc-1.1b-persian.nemo` | Persian Parakeet CTC, NeMo format | <https://huggingface.co/MohammadGholizadeh/parakeet-ctc-1.1b-persian.nemo> |
| `alifarokh/nemo-conformer-medium-fa` | NeMo Conformer Medium Persian ASR | <https://huggingface.co/alifarokh/nemo-conformer-medium-fa> |
| `Reza2kn/Shenava-Koochik-v1.0` | Compact Persian NeMo ASR | <https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0> |
| `Reza2kn/Shenava-Koochik-0.9` | Earlier Shenava-Koochik Persian checkpoint | <https://huggingface.co/Reza2kn/Shenava-Koochik-0.9> |
| `Reza2kn/Shenava-Rizeh-v1.0` | FastConformer RNNT/CTC Persian ASR | <https://huggingface.co/Reza2kn/Shenava-Rizeh-v1.0> |
| `Reza2kn/Shenava-Rizeh-Pizeh-v1.0` | Distilled compact FastConformer Persian ASR | <https://huggingface.co/Reza2kn/Shenava-Rizeh-Pizeh-v1.0> |

The Rizeh model card describes a compact FastConformer transducer/CTC model, while the Rizeh-Pizeh repository describes an even smaller distilled model. Their cards also link to separate ONNX, Core ML, and other deployment-oriented exports; those exports were not counted as additional primary models in the total above. citeturn15view0turn15view1

For production comparisons, NeMo models should be benchmarked separately from Whisper and Transformers CTC systems. Their decoding algorithms, tokenization, streaming capabilities, and runtime requirements differ enough that parameter count alone is not a meaningful comparison.

## Multilingual and audio-language ASR models with Persian coverage

These repositories are not Persian-only fine-tunes. They are multilingual foundation ASR models or generative audio-language systems available on Hugging Face and surfaced through Persian-language or ASR discovery paths. They are important baselines because a strong multilingual checkpoint may outperform an older Persian fine-tune, particularly on varied speakers or out-of-domain recordings. Hugging Face’s current Persian-language ASR listings include Whisper, SeamlessM4T, MMS, Qwen3-ASR, and VibeVoice-ASR families. citeturn1search4turn3search0turn3search6turn11search10

| Name | Architecture or scope | Main Hugging Face URL |
|---|---|---|
| `openai/whisper-tiny` | Multilingual Whisper Tiny | <https://huggingface.co/openai/whisper-tiny> |
| `openai/whisper-base` | Multilingual Whisper Base | <https://huggingface.co/openai/whisper-base> |
| `openai/whisper-small` | Multilingual Whisper Small | <https://huggingface.co/openai/whisper-small> |
| `openai/whisper-medium` | Multilingual Whisper Medium | <https://huggingface.co/openai/whisper-medium> |
| `openai/whisper-large-v2` | Multilingual Whisper Large-v2 | <https://huggingface.co/openai/whisper-large-v2> |
| `openai/whisper-large-v3` | Multilingual Whisper Large-v3 | <https://huggingface.co/openai/whisper-large-v3> |
| `openai/whisper-large-v3-turbo` | Faster multilingual Whisper Large-v3 derivative | <https://huggingface.co/openai/whisper-large-v3-turbo> |
| `facebook/seamless-m4t-v2-large` | Multilingual speech-to-text and speech translation | <https://huggingface.co/facebook/seamless-m4t-v2-large> |
| `facebook/mms-1b-all` | Massively multilingual CTC ASR with language adapters | <https://huggingface.co/facebook/mms-1b-all> |
| `Qwen/Qwen3-ASR-0.6B` | Generative multilingual audio-language ASR | <https://huggingface.co/Qwen/Qwen3-ASR-0.6B> |
| `Qwen/Qwen3-ASR-0.6B-hf` | Hugging Face-native Qwen3-ASR 0.6B packaging | <https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf> |
| `Qwen/Qwen3-ASR-1.7B` | Larger generative multilingual audio-language ASR | <https://huggingface.co/Qwen/Qwen3-ASR-1.7B> |
| `Qwen/Qwen3-ASR-1.7B-hf` | Hugging Face-native Qwen3-ASR 1.7B packaging | <https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf> |
| `microsoft/VibeVoice-ASR` | Large multilingual generative ASR/audio model | <https://huggingface.co/microsoft/VibeVoice-ASR> |

For Whisper, the multilingual checkpoints are the versions without the `.en` suffix. Persian decoding should normally be forced or prompted with the Persian language setting when the framework allows it, rather than relying solely on automatic language detection for short, noisy, or code-switched clips.

For MMS, users should verify that the correct Persian language adapter or tokenizer configuration is selected. For SeamlessM4T, the speech-recognition and speech-translation modes should not be conflated: Persian speech-to-Persian-text ASR and Persian-to-English speech translation are different evaluation tasks.

The Qwen3-ASR and VibeVoice entries satisfy the request for newer **audio-language or LLM-style ASR models**. They should nevertheless be evaluated independently for Persian rather than assumed to match their aggregate multilingual performance. Their larger language-model-style decoders may help with contextual transcription, but they can also introduce generative substitutions that are uncommon in frame-aligned CTC systems.

## Corrections, exclusions, and practical caveats

The master list incorporates all of the user-supplied findings, with one URL correction:

| Supplied entry | Verification result |
|---|---|
| `steja/whisper-large-persian-steja` | Exact public URL could not be verified; replaced with the active `steja/whisper-large-persian` repository |
| Remaining ten supplied URLs | Included under their supplied names and URLs |

The `C1Tech/whisper_small_persian` repository is present but gated, meaning users may need to authenticate and accept repository conditions before downloading its files. Gated availability still satisfies presence on Hugging Face, but it is not equivalent to unrestricted public download. citeturn6view1

Several Hugging Face collections contain false positives for a Persian-ASR search. For example, `m3hrdadfi/wav2vec2-xlsr-persian-speech-emotion-recognition` is a **speech emotion classification** model rather than a transcription model, despite appearing with an ASR-related tag in some indexes. Similarly, `aliyzd95/wav2vec2-large-xlsr-persian-KWS` is oriented toward **keyword spotting**, not unrestricted speech-to-text. These were deliberately excluded from the master list. citeturn18view1turn18view2turn11search9

Repositories whose names contain `farsipal` but whose model identifiers contain `el` were also excluded. In Whisper naming and language codes, `el` generally denotes Greek; the repository owner’s Persian-associated name is not sufficient evidence that the model transcribes Farsi. This illustrates why a collection-wide keyword match should not be treated as a validated Persian ASR inventory.

Finally, the list records **repository availability**, not model quality. Before selecting a model, a serious Persian evaluation should normalize Arabic-versus-Persian Unicode characters, punctuation, digits, spacing and half-spaces, and optional diacritics consistently. It should also distinguish clean read speech from telephone audio, conversational speech, broadcast speech, code-switching, dialectal Persian, and domain-specific vocabulary. Models trained principally on Common Voice may rank differently on Iranian broadcast, call-centre, classroom, medical, or spontaneous conversational audio than they do on their published validation set.
