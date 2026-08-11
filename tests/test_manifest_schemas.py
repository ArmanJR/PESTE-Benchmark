"""Manifest and validated contract tests."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from peste.constants import PROJECT_ROOT
from peste.digests import sha256_file
from peste.manifest import validate_manifest
from peste.schemas import RunBundle, RunStatus, SuiteSpec
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
        "schema_version": 2,
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
            "gpu_product_name": "NVIDIA RTX 6000 Ada Generation",
            "driver_version": "580.142",
            "ecc_state": "Disabled",
            "power_limit_watts": 300,
            "cpu_model": "CPU",
            "gpu_uuid": "GPU-test",
            "seed": 1,
        },
        "speed": {
            "valid": True,
            "batch_size": 8,
            "warmup_batches": 2,
            "measured_batches": 1,
            "total_audio_seconds": 10,
            "processing_seconds": 2,
            "audio_throughput_x": 5,
            "rtf": 0.2,
            "timing_artifact": "timing.jsonl",
        },
        "model_facts": {
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

    payload["status"] = RunStatus.FAILED
    with pytest.raises(ValidationError, match="successful run can contain valid speed"):
        RunBundle.model_validate(json.loads(json.dumps(payload)))


def test_schema_one_is_rejected_with_clear_error() -> None:
    payload = json.loads((PROJECT_ROOT / "suites" / "fleurs-fa-ir-v1" / "suite.json").read_text())
    payload["schema_version"] = 1
    with pytest.raises(ValidationError, match="Unsupported schema_version 1; expected 2"):
        SuiteSpec.model_validate(payload)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"audio_throughput_x": 4.9}, "audio_throughput_x"),
        ({"rtf": 0.3}, "RTF"),
        ({"valid": False, "invalidity_reason": None}, "invalidity reason"),
    ],
)
def test_speed_statistics_invariants(changes: dict[str, object], message: str) -> None:
    from peste.schemas import SpeedStatistics

    payload: dict[str, object] = {
        "valid": True,
        "batch_size": 8,
        "warmup_batches": 2,
        "measured_batches": 10,
        "total_audio_seconds": 100.0,
        "processing_seconds": 20.0,
        "audio_throughput_x": 5.0,
        "rtf": 0.2,
        "timing_artifact": "timing.jsonl",
    }
    payload.update(changes)
    with pytest.raises(ValidationError, match=message):
        SpeedStatistics.model_validate(payload)
