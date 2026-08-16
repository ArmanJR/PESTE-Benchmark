"""Semantic model-policy validation tests."""

from pathlib import Path
from typing import Any

import pytest
from conftest import make_model

from peste.schemas import ModelSpec
from peste.validation import validate_model_policy


def _ctc_model(**updates: Any) -> ModelSpec:
    payload = make_model("transformers-ctc", dtype="float32").model_dump(mode="python")
    payload.update(updates)
    return ModelSpec.model_validate(payload)


def _make_runtime(tmp_path: Path) -> None:
    dockerfile = tmp_path / "runtimes" / "Dockerfile"
    dockerfile.parent.mkdir(parents=True)
    dockerfile.touch()


def test_transformers_ctc_policy_accepts_exact_configuration(tmp_path: Path) -> None:
    _make_runtime(tmp_path)

    validate_model_policy(_ctc_model(), tmp_path)


def test_whisper_policy_requires_automatic_timestamps_and_current_runtime(tmp_path: Path) -> None:
    _make_runtime(tmp_path)
    model = make_model()

    validate_model_policy(model, tmp_path)

    invalid_generation = model.model_copy(
        update={
            "generation": {
                "task": "transcribe",
                "max_new_tokens": 444,
                "return_timestamps": False,
            }
        }
    )
    with pytest.raises(ValueError, match="Whisper policy mismatch"):
        validate_model_policy(invalid_generation, tmp_path)


def test_only_grandfathered_models_accept_the_legacy_runtime(tmp_path: Path) -> None:
    _make_runtime(tmp_path)
    current = make_model("transformers-qwen", model_id="new-qwen")
    legacy_runtime = current.runtime.model_copy(
        update={"image": "ghcr.io/armanjr/peste-benchmark:2.0.0"}
    )

    with pytest.raises(ValueError, match="Runtime image mismatch"):
        validate_model_policy(current.model_copy(update={"runtime": legacy_runtime}), tmp_path)

    grandfathered = make_model("transformers-qwen", model_id="qwen3-asr-0-6b").model_copy(
        update={
            "repository": "Qwen/Qwen3-ASR-0.6B-hf",
            "revision": "7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c",
            "runtime": legacy_runtime,
        }
    )
    validate_model_policy(grandfathered, tmp_path)

    with pytest.raises(ValueError, match="Legacy runtime identity mismatch"):
        validate_model_policy(
            grandfathered.model_copy(update={"repository": "other/model"}), tmp_path
        )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("decoder", "beam"),
        ("batch_size", 2),
        ("external_language_model", True),
        ("group_tokens", False),
        ("skip_special_tokens", True),
        ("clean_up_tokenization_spaces", True),
    ],
)
def test_transformers_ctc_policy_rejects_wrong_decoding_configuration(
    tmp_path: Path, field: str, invalid_value: str | int | bool
) -> None:
    _make_runtime(tmp_path)
    model = _ctc_model()
    generation = dict(model.generation)
    generation[field] = invalid_value

    with pytest.raises(ValueError, match="Transformers CTC policy mismatch"):
        validate_model_policy(_ctc_model(generation=generation), tmp_path)


def test_transformers_ctc_policy_rejects_wrong_language(tmp_path: Path) -> None:
    _make_runtime(tmp_path)

    with pytest.raises(ValueError, match="Transformers CTC policy mismatch"):
        validate_model_policy(_ctc_model(language="en"), tmp_path)


def test_transformers_ctc_policy_rejects_wrong_runtime(tmp_path: Path) -> None:
    _make_runtime(tmp_path)
    runtime = make_model("transformers-ctc").runtime.model_copy(update={"name": "nemo"})

    with pytest.raises(ValueError, match="Transformers CTC policy mismatch"):
        validate_model_policy(_ctc_model(runtime=runtime), tmp_path)


def test_nemo_ctc_policy_accepts_only_ctc_decoder(tmp_path: Path) -> None:
    _make_runtime(tmp_path)
    model = make_model("nemo-ctc", dtype="float32")

    validate_model_policy(model, tmp_path)

    invalid = model.model_copy(
        update={"generation": {"decoder": "rnnt", "external_language_model": False}}
    )
    with pytest.raises(ValueError, match="NeMo CTC policy mismatch"):
        validate_model_policy(invalid, tmp_path)
