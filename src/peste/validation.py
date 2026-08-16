"""Semantic validation beyond schema parsing."""

import re
from pathlib import Path

from peste.schemas import ModelSpec

CURRENT_RUNTIME_IMAGE = "ghcr.io/armanjr/peste-benchmark:2.1.0"
LEGACY_RUNTIME_IMAGE = "ghcr.io/armanjr/peste-benchmark:2.0.0"
EXPECTED_RUNTIME_DOCKERFILE = "runtimes/Dockerfile"
REPOSITORY_PATTERN = re.compile(r"^[^/\s]+/[^/\s]+$")
LEGACY_2_0_MODELS = {
    "nvidia-fastconformer-fa": (
        "nvidia/stt_fa_fastconformer_hybrid_large",
        "249cf5bf70dda7220a60ddeeecff2f6aad8e1784",
        "nemo-rnnt",
    ),
    "persian-speech-transcription-wav2vec2-v1-seyedali": (
        "SeyedAli/Persian-Speech-Transcription-Wav2Vec2-V1",
        "21623b1ffbdcb4c79bf7bd74737ab30237db4b66",
        "transformers-ctc",
    ),
    "qwen3-asr-0-6b": (
        "Qwen/Qwen3-ASR-0.6B-hf",
        "7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c",
        "transformers-qwen",
    ),
    "qwen3-asr-1-7b": (
        "Qwen/Qwen3-ASR-1.7B-hf",
        "bcd2b5b7f32b480ab5790554cfa8347f246a14f3",
        "transformers-qwen",
    ),
    "shenava-rizeh-v1-0": (
        "Reza2kn/Shenava-Rizeh-v1.0",
        "74c96b7c23d8611dd4d0c775744f43bc4fb9c2ec",
        "nemo-ctc",
    ),
    "visualears-fastconformer-fa-full-ab": (
        "Reza2kn/visualears-fastconformer-fa-full-ab",
        "7f43a9d41d06328605257f0f28542c2f2332ed55",
        "nemo-rnnt",
    ),
    "wav2vec2-large-xlsr-53-persian": (
        "jonatasgrosman/wav2vec2-large-xlsr-53-persian",
        "234714078a1398a9db88194c5a40fefe6f376dc1",
        "transformers-ctc",
    ),
    "wav2vec2-large-xlsr-persian-m3hrdadfi": (
        "m3hrdadfi/wav2vec2-large-xlsr-persian",
        "a6fc7cdc898c6ec218e7f337a4835c3cd1ab8fab",
        "transformers-ctc",
    ),
    "wav2vec2-large-xlsr-persian-shemo-m3hrdadfi": (
        "m3hrdadfi/wav2vec2-large-xlsr-persian-shemo",
        "f9aa526bb0408f48543d0359dca089555adefc05",
        "transformers-ctc",
    ),
    "wav2vec2-large-xlsr-persian-v2-m3hrdadfi": (
        "m3hrdadfi/wav2vec2-large-xlsr-persian-v2",
        "599d7361d87b6ea3ca5d64a993e8ad8c942c48eb",
        "transformers-ctc",
    ),
    "wav2vec2-large-xlsr-persian-v3-masoumehb": (
        "masoumehb/wav2vec2-large-xlsr-persian-v3",
        "918f655ca45ef4b729b496288139114a3fdf2b1a",
        "transformers-ctc",
    ),
    "wav2vec2-xls-r-300m-fa-alifarokh": (
        "alifarokh/wav2vec2-xls-r-300m-fa",
        "79d44772d3bfc1f9000748c8478781662a5fbc64",
        "transformers-ctc",
    ),
    "xls-r-1b-fa-cv8-ghofrani": (
        "ghofrani/xls-r-1b-fa-cv8",
        "c38ce46e838cade8ecadc7ff5ad5fb58fd7cda95",
        "transformers-ctc",
    ),
}


def validate_model_policy(model: ModelSpec, root: Path) -> None:
    if REPOSITORY_PATTERN.fullmatch(model.repository) is None:
        raise ValueError(f"Invalid Hugging Face repository for {model.model_id}")
    if not model.license.strip():
        raise ValueError(f"Model license is empty for {model.model_id}")
    dockerfile = root / model.runtime.dockerfile
    if not dockerfile.is_file():
        raise ValueError(f"Runtime Dockerfile does not exist: {dockerfile}")
    legacy_identity = LEGACY_2_0_MODELS.get(model.model_id)
    if (
        legacy_identity is not None
        and (
            model.repository,
            model.revision,
            model.adapter,
        )
        != legacy_identity
    ):
        raise ValueError(f"Legacy runtime identity mismatch for {model.model_id}")
    expected_image = LEGACY_RUNTIME_IMAGE if legacy_identity is not None else CURRENT_RUNTIME_IMAGE
    if model.runtime.image != expected_image:
        raise ValueError(f"Runtime image mismatch for {model.model_id}")
    if model.runtime.dockerfile != EXPECTED_RUNTIME_DOCKERFILE:
        raise ValueError(f"Runtime Dockerfile policy mismatch for {model.model_id}")
    if model.adapter == "transformers-whisper":
        expected = {"task": "transcribe", "max_new_tokens": 444, "return_timestamps": "auto"}
        if model.language != "fa" or model.generation != expected:
            raise ValueError(f"Whisper policy mismatch for {model.model_id}")
    elif model.adapter == "transformers-qwen":
        if model.language != "fa" or model.generation != {"max_new_tokens": 256}:
            raise ValueError(f"Qwen policy mismatch for {model.model_id}")
    elif model.adapter == "transformers-ctc":
        expected = {
            "decoder": "greedy",
            "external_language_model": False,
            "group_tokens": True,
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
        }
        if model.language != "fa" or model.runtime.name != "modern" or model.generation != expected:
            raise ValueError(f"Transformers CTC policy mismatch for {model.model_id}")
    elif model.adapter == "nemo-rnnt":
        expected = {"decoder": "rnnt", "external_language_model": False}
        if model.generation != expected:
            raise ValueError(f"NeMo RNNT policy mismatch for {model.model_id}")
    elif model.adapter == "nemo-ctc":
        expected = {"decoder": "ctc", "external_language_model": False}
        if model.generation != expected:
            raise ValueError(f"NeMo CTC policy mismatch for {model.model_id}")
    expected_runtime = "nemo" if model.adapter in {"nemo-rnnt", "nemo-ctc"} else "modern"
    if model.runtime.name != expected_runtime:
        raise ValueError(
            f"Runtime {model.runtime.name} does not match adapter {model.adapter} "
            f"for {model.model_id}"
        )
