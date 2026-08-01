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


def _make_modern_runtime(tmp_path: Path) -> None:
    dockerfile = tmp_path / "runtimes" / "modern" / "Dockerfile"
    dockerfile.parent.mkdir(parents=True)
    dockerfile.touch()


def test_transformers_ctc_policy_accepts_exact_configuration(tmp_path: Path) -> None:
    _make_modern_runtime(tmp_path)

    validate_model_policy(_ctc_model(), tmp_path)


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
    _make_modern_runtime(tmp_path)
    model = _ctc_model()
    generation = dict(model.generation)
    generation[field] = invalid_value

    with pytest.raises(ValueError, match="Transformers CTC policy mismatch"):
        validate_model_policy(_ctc_model(generation=generation), tmp_path)


def test_transformers_ctc_policy_rejects_wrong_language(tmp_path: Path) -> None:
    _make_modern_runtime(tmp_path)

    with pytest.raises(ValueError, match="Transformers CTC policy mismatch"):
        validate_model_policy(_ctc_model(language="en"), tmp_path)


def test_transformers_ctc_policy_rejects_wrong_runtime(tmp_path: Path) -> None:
    _make_modern_runtime(tmp_path)
    runtime = make_model("transformers-ctc").runtime.model_copy(update={"name": "nemo"})

    with pytest.raises(ValueError, match="Transformers CTC policy mismatch"):
        validate_model_policy(_ctc_model(runtime=runtime), tmp_path)
