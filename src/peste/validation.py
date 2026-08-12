"""Semantic validation beyond schema parsing."""

import re
from pathlib import Path

from peste.schemas import ModelSpec

EXPECTED_RUNTIME_IMAGE = "ghcr.io/armanjr/peste-benchmark:2.0.0"
EXPECTED_RUNTIME_DOCKERFILE = "runtimes/Dockerfile"
REPOSITORY_PATTERN = re.compile(r"^[^/\s]+/[^/\s]+$")


def validate_model_policy(model: ModelSpec, root: Path) -> None:
    if REPOSITORY_PATTERN.fullmatch(model.repository) is None:
        raise ValueError(f"Invalid Hugging Face repository for {model.model_id}")
    if not model.license.strip():
        raise ValueError(f"Model license is empty for {model.model_id}")
    dockerfile = root / model.runtime.dockerfile
    if not dockerfile.is_file():
        raise ValueError(f"Runtime Dockerfile does not exist: {dockerfile}")
    if model.runtime.image != EXPECTED_RUNTIME_IMAGE:
        raise ValueError(f"Runtime image mismatch for {model.model_id}")
    if model.runtime.dockerfile != EXPECTED_RUNTIME_DOCKERFILE:
        raise ValueError(f"Runtime Dockerfile policy mismatch for {model.model_id}")
    if model.adapter == "transformers-whisper":
        expected = {"task": "transcribe", "max_new_tokens": 444, "return_timestamps": False}
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
            raise ValueError(f"NeMo policy mismatch for {model.model_id}")
    expected_runtime = "nemo" if model.adapter == "nemo-rnnt" else "modern"
    if model.runtime.name != expected_runtime:
        raise ValueError(
            f"Runtime {model.runtime.name} does not match adapter {model.adapter} "
            f"for {model.model_id}"
        )
