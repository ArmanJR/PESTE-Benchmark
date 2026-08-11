"""Accuracy/speed ranking and deterministic generated-output tests."""

import json
from pathlib import Path

import pytest

from peste.leaderboard import accuracy_order, collect_rows, generate_leaderboards, speed_order
from peste.specs import load_model, spec_digest


def _bundle(
    model: str,
    status: str,
    wer: float,
    cer: float,
    throughput: float,
    suite_digest: str = "a" * 64,
    *,
    speed_valid: bool = True,
) -> dict[str, object]:
    aggregates = None
    if status == "success":
        aggregates = {
            "samples": 2,
            "wer": wer,
            "cer": cer,
            "word_errors": round(wer * 10_000),
            "word_reference_units": 10_000,
            "character_errors": round(cer * 10_000),
            "character_reference_units": 10_000,
            "word_accuracy_pct": 100 * max(0, 1 - wer),
        }
    processing_seconds = 100.0 / throughput if throughput else 0.0
    return {
        "schema_version": 2,
        "run_id": f"run-{model}",
        "suite_id": "tiny-suite-v1",
        "suite_digest": suite_digest,
        "model_id": model,
        "model_digest": "b" * 64,
        "status": status,
        "environment": {
            "peste_revision": "rev",
            "image_reference": "image",
            "image_digest": "digest",
            "dependency_versions": {},
            "python_version": "3.12",
            "pytorch_version": "2",
            "cuda_version": "12.9",
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
            "valid": speed_valid,
            "batch_size": 8,
            "warmup_batches": 2,
            "measured_batches": 1 if status == "success" else 0,
            "total_audio_seconds": 100.0 if throughput else 0.0,
            "processing_seconds": processing_seconds,
            "audio_throughput_x": throughput,
            "rtf": 1 / throughput if throughput else 0.0,
            "timing_artifact": "timing.jsonl",
            "invalidity_reason": None if speed_valid else "resumed run",
        },
        "model_facts": {
            "checkpoint_bytes": 4 * 1024**3,
            "parameter_count": 123,
            "native_dtype": "float16",
        },
        "predictions_path": "predictions.jsonl",
        "aggregates": aggregates,
        "logs": {"runner": "runner.jsonl"},
    }


def _write_bundle(directory: Path, bundle: dict[str, object]) -> None:
    (directory / "run.json").write_text(json.dumps(bundle), encoding="utf-8")
    if bundle["status"] != "success":
        return
    aggregates = bundle["aggregates"]
    assert isinstance(aggregates, dict)
    records = []
    for sequence in range(2):
        word_errors = int(aggregates["word_errors"])
        character_errors = int(aggregates["character_errors"])
        records.append(
            {
                "schema_version": 2,
                "sequence": sequence,
                "sample_id": f"test-{sequence:06d}",
                "reference": "متن",
                "prediction": "متن",
                "normalized_reference": "متن",
                "normalized_prediction": "متن",
                "word_substitutions": 0,
                "word_deletions": 0,
                "word_insertions": word_errors // 2 + (word_errors % 2 if sequence == 0 else 0),
                "word_reference_units": 5_000,
                "character_substitutions": 0,
                "character_deletions": 0,
                "character_insertions": (
                    character_errors // 2 + (character_errors % 2 if sequence == 0 else 0)
                ),
                "character_reference_units": 5_000,
                "structured_output": None,
            }
        )
    (directory / "predictions.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_accuracy_and_speed_rank_orders_exclude_invalid_speed(
    tmp_path: Path, tiny_suite: tuple[object, Path, list[object]]
) -> None:
    suite, _, _ = tiny_suite
    suite_digest = spec_digest(suite)  # type: ignore[arg-type]
    results = tmp_path / "results"
    cases = (
        ("zeta", "success", 0.1, 0.1, 20.0, True),
        ("alpha", "success", 0.1, 0.1, 10.0, True),
        ("beta", "success", 0.1, 0.05, 20.0, True),
        ("cer-leader", "success", 0.2, 0.01, 5.0, False),
        ("oom-model", "oom", 0.0, 0.0, 0.0, False),
    )
    for model, status, wer, cer, throughput, valid in cases:
        directory = results / model
        directory.mkdir(parents=True)
        _write_bundle(
            directory,
            _bundle(
                model,
                status,
                wer,
                cer,
                throughput,
                suite_digest,
                speed_valid=valid,
            ),
        )
    rows = collect_rows(suite, results, require_tracked=False, root=tmp_path)  # type: ignore[arg-type]
    assert [row.model_id for row in accuracy_order(rows)] == [
        "cer-leader",
        "beta",
        "alpha",
        "zeta",
    ]
    assert [row.model_id for row in speed_order(rows)] == ["beta", "zeta", "alpha"]


def test_static_speed_outputs_are_deterministic_and_have_no_memory_fields(
    tmp_path: Path, tiny_suite: tuple[object, Path, list[object]]
) -> None:
    suite, _, _ = tiny_suite
    suite_digest = spec_digest(suite)  # type: ignore[arg-type]
    results = tmp_path / "results"
    for model, throughput in (("model-a", 12.5), ("model-b", 10.0)):
        directory = results / model
        directory.mkdir(parents=True)
        _write_bundle(directory, _bundle(model, "success", 0.2, 0.1, throughput, suite_digest))
    (tmp_path / "README.md").write_text(
        "before\n<!-- LEADERBOARD:START -->\nstale\n<!-- LEADERBOARD:END -->\nafter\n",
        encoding="utf-8",
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_leaderboards(suite, results, first, require_tracked=False, root=tmp_path)  # type: ignore[arg-type]
    first_readme = (tmp_path / "README.md").read_bytes()
    generate_leaderboards(suite, results, second, require_tracked=False, root=tmp_path)  # type: ignore[arg-type]
    for name in (
        "leaderboard.json",
        "leaderboard.csv",
        "leaderboard.md",
        "leaderboard-accuracy.svg",
        "leaderboard-speed.svg",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert (tmp_path / "README.md").read_bytes() == first_readme
    payload = json.loads((first / "leaderboard.json").read_text())
    assert payload["schema_version"] == 2
    assert payload["speed"][0]["model_id"] == "model-a"
    for path in first.iterdir():
        assert "memory_efficiency" not in path.read_text(encoding="utf-8")
        assert "peak_cuda" not in path.read_text(encoding="utf-8")


def test_markdown_links_models_and_presents_speed_board(
    tmp_path: Path, tiny_suite: tuple[object, Path, list[object]]
) -> None:
    suite, _, _ = tiny_suite
    results = tmp_path / "results"
    directory = results / "model"
    directory.mkdir(parents=True)
    _write_bundle(
        directory,
        _bundle("model", "success", 0.2, 0.1, 12.5, spec_digest(suite)),  # type: ignore[arg-type]
    )
    models = tmp_path / "models"
    models.mkdir()
    (models / "model.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "model_id": "model",
                "repository": "organization/checkpoint",
                "revision": "a" * 40,
                "adapter": "transformers-whisper",
                "native_dtype": "float16",
                "license": "Apache-2.0",
                "language": "fa",
                "generation": {
                    "task": "transcribe",
                    "max_new_tokens": 444,
                    "return_timestamps": False,
                },
                "runtime": {
                    "name": "modern",
                    "image": "peste-modern:2.0.0",
                    "dockerfile": "runtimes/modern/Dockerfile",
                },
                "speed_profile": {
                    "hardware_profile_id": "rtx-6000-ada-v1",
                    "batch_size": 8,
                },
            }
        ),
        encoding="utf-8",
    )
    bundle = json.loads((directory / "run.json").read_text())
    bundle["model_digest"] = spec_digest(load_model("model", tmp_path))
    (directory / "run.json").write_text(json.dumps(bundle), encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "<!-- LEADERBOARD:START -->\n<!-- LEADERBOARD:END -->\n", encoding="utf-8"
    )
    output = tmp_path / "generated"

    generate_leaderboards(suite, results, output, require_tracked=False, root=tmp_path)  # type: ignore[arg-type]

    expected_link = "[`model`](https://huggingface.co/organization/checkpoint)"
    markdown = (output / "leaderboard.md").read_text()
    assert expected_link in markdown
    assert "## Steady-state speed" in markdown
    assert "| Rank | Model | Batch | Throughput | RTF | Processing s | Audio s |" in markdown
    speed_svg = (output / "leaderboard-speed.svg").read_text()
    assert "PESTE steady-state speed leaderboard" in speed_svg
    assert "https://huggingface.co/organization/checkpoint" in speed_svg


def test_uncertainty_requires_prediction_counts_to_match_aggregates(
    tmp_path: Path, tiny_suite: tuple[object, Path, list[object]]
) -> None:
    suite, _, _ = tiny_suite
    results = tmp_path / "results"
    directory = results / "model"
    directory.mkdir(parents=True)
    bundle = _bundle("model", "success", 0.1, 0.1, 10.0, spec_digest(suite))  # type: ignore[arg-type]
    _write_bundle(directory, bundle)
    aggregates = bundle["aggregates"]
    assert isinstance(aggregates, dict)
    aggregates["cer"] = 0.2
    (directory / "run.json").write_text(json.dumps(bundle), encoding="utf-8")
    with pytest.raises(ValueError, match="CER aggregate does not match predictions"):
        collect_rows(suite, results, require_tracked=False, root=tmp_path)  # type: ignore[arg-type]


def test_stale_and_uncommitted_results_are_excluded(
    tmp_path: Path, tiny_suite: tuple[object, Path, list[object]]
) -> None:
    suite, _, _ = tiny_suite
    results = tmp_path / "results"
    stale_directory = results / "stale"
    stale_directory.mkdir(parents=True)
    _write_bundle(stale_directory, _bundle("stale", "success", 0.1, 0.1, 10.0))
    uncommitted_directory = results / "uncommitted"
    uncommitted_directory.mkdir()
    uncommitted = _bundle(
        "uncommitted",
        "success",
        0.1,
        0.1,
        10.0,
        spec_digest(suite),  # type: ignore[arg-type]
    )
    environment = uncommitted["environment"]
    assert isinstance(environment, dict)
    environment["peste_revision"] = "uncommitted"
    _write_bundle(uncommitted_directory, uncommitted)
    assert collect_rows(suite, results, require_tracked=False, root=tmp_path) == []  # type: ignore[arg-type]
