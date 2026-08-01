"""Semantic validation beyond schema parsing."""

from pathlib import Path

from peste.schemas import ModelSpec


def validate_model_policy(model: ModelSpec, root: Path) -> None:
    dockerfile = root / model.runtime.dockerfile
    if not dockerfile.is_file():
        raise ValueError(f"Runtime Dockerfile does not exist: {dockerfile}")
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
            "batch_size": 1,
            "external_language_model": False,
            "group_tokens": True,
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
        }
        if model.language != "fa" or model.runtime.name != "modern" or model.generation != expected:
            raise ValueError(f"Transformers CTC policy mismatch for {model.model_id}")
    elif model.adapter == "vibevoice":
        expected = {"max_new_tokens": 512, "temperature": 0.0, "top_p": 1.0, "num_beams": 1}
        if model.language is not None or model.generation != expected:
            raise ValueError(f"VibeVoice policy mismatch for {model.model_id}")
    elif model.adapter == "nemo-rnnt":
        expected = {"decoder": "rnnt", "batch_size": 1, "external_language_model": False}
        if model.generation != expected:
            raise ValueError(f"NeMo policy mismatch for {model.model_id}")
