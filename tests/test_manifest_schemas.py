"""Manifest and validated contract tests."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from peste.constants import PROJECT_ROOT
from peste.digests import sha256_file
from peste.manifest import validate_manifest
from peste.schemas import RunBundle, RunStatus
from peste.specs import load_suite


def test_manifest_digest_counts_and_order(
    tiny_suite: tuple[object, Path, list[object]],
) -> None:
    suite, directory, expected_rows = tiny_suite
    assert validate_manifest(suite, directory) == expected_rows  # type: ignore[arg-type]


def test_manifest_tampering_is_rejected(
    tiny_suite: tuple[object, Path, list[object]],
) -> None:
    suite, directory, _ = tiny_suite
    path = directory / "manifest.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert sha256_file(path) != suite.manifest_sha256  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_manifest(suite, directory)  # type: ignore[arg-type]


def test_fleurs_fa_ir_v2_reuses_the_pinned_manifest_with_fa_v2() -> None:
    suite = load_suite("fleurs-fa-ir-v2")

    assert suite.normalization_version == "fa-v2"
    rows = validate_manifest(suite, PROJECT_ROOT / "suites" / suite.suite_id)
    assert len(rows) == sum(suite.expected_split_counts.values())


def test_success_requires_aggregates() -> None:
    payload = {
        "schema_version": 1,
        "run_id": "run",
        "suite_id": "suite",
        "suite_digest": "a" * 64,
        "model_id": "model",
        "model_digest": "b" * 64,
        "status": RunStatus.SUCCESS,
        "environment": {
            "peste_revision": "rev",
            "image_reference": "image",
            "image_digest": "digest",
            "dependency_versions": {},
            "python_version": "3.12",
            "pytorch_version": "2",
            "cuda_version": "12.6",
            "hardware_profile": {},
            "seed": 1,
        },
        "memory": {
            "peak_cuda_reserved_bytes": 1,
            "peak_cuda_allocated_bytes": 1,
            "peak_process_rss_bytes": 1,
            "checkpoint_bytes": 1,
            "parameter_count": 1,
            "native_dtype": "float16",
        },
        "predictions_path": "predictions.jsonl",
        "aggregates": None,
        "logs": {"runner": "runner.jsonl"},
    }
    with pytest.raises(ValidationError, match="successful run requires"):
        RunBundle.model_validate(json.loads(json.dumps(payload)))
