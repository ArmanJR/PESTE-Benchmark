"""Ranking, OOM exclusion, and generated output tests."""

import json
from pathlib import Path

import pytest

from peste.leaderboard import accuracy_order, collect_rows, efficiency_order, generate_leaderboards
from peste.specs import load_model, spec_digest


def _bundle(
    model: str,
    status: str,
    wer: float,
    cer: float,
    efficiency: float,
    suite_digest: str = "a" * 64,
) -> dict[str, object]:
    aggregates = None
    if status == "success":
        word_reference_units = 10_000
        character_reference_units = 10_000
        aggregates = {
            "samples": 2,
            "wer": wer,
            "cer": cer,
            "word_errors": round(wer * word_reference_units),
            "word_reference_units": word_reference_units,
            "character_errors": round(cer * character_reference_units),
            "character_reference_units": character_reference_units,
            "word_accuracy_pct": 100 * max(0, 1 - wer),
            "memory_efficiency": efficiency,
        }
    return {
        "schema_version": 1,
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
            "cuda_version": "12.6",
            "hardware_profile": {},
            "seed": 1,
        },
        "memory": {
            "peak_cuda_reserved_bytes": 2 * 1024**3,
            "peak_cuda_allocated_bytes": 1024**3,
            "peak_process_rss_bytes": 3 * 1024**3,
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
    word_errors = int(aggregates["word_errors"])
    character_errors = int(aggregates["character_errors"])
    records = []
    for sequence in range(2):
        sample_word_errors = word_errors // 2 + (word_errors % 2 if sequence == 0 else 0)
        sample_character_errors = character_errors // 2 + (
            character_errors % 2 if sequence == 0 else 0
        )
        records.append(
            {
                "schema_version": 1,
                "sequence": sequence,
                "sample_id": f"test-{sequence:06d}",
                "reference": "متن",
                "prediction": "متن",
                "normalized_reference": "متن",
                "normalized_prediction": "متن",
                "word_substitutions": 0,
                "word_deletions": 0,
                "word_insertions": sample_word_errors,
                "word_reference_units": 5_000,
                "character_substitutions": 0,
                "character_deletions": 0,
                "character_insertions": sample_character_errors,
                "character_reference_units": 5_000,
                "structured_output": None,
            }
        )
    (directory / "predictions.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_rank_order_and_oom_exclusion(
    tmp_path: Path, tiny_suite: tuple[object, Path, list[object]]
) -> None:
    suite, _, _ = tiny_suite
    suite_digest = spec_digest(suite)  # type: ignore[arg-type]
    results = tmp_path / "results"
    for model, status, wer, cer, efficiency in (
        ("zeta", "success", 0.1, 0.1, 20.0),
        ("alpha", "success", 0.1, 0.1, 10.0),
        ("beta", "success", 0.1, 0.05, 20.0),
        ("cer-leader", "success", 0.2, 0.01, 5.0),
        ("oom-model", "oom", 0.0, 0.0, 100.0),
    ):
        directory = results / model
        directory.mkdir(parents=True)
        _write_bundle(directory, _bundle(model, status, wer, cer, efficiency, suite_digest))
    rows = collect_rows(suite, results, require_tracked=False, root=tmp_path)  # type: ignore[arg-type]
    assert [row.model_id for row in accuracy_order(rows)] == [
        "cer-leader",
        "beta",
        "alpha",
        "zeta",
    ]
    assert [row.model_id for row in efficiency_order(rows)] == [
        "beta",
        "zeta",
        "alpha",
        "cer-leader",
    ]


def test_static_outputs_are_deterministic(
    tmp_path: Path, tiny_suite: tuple[object, Path, list[object]]
) -> None:
    suite, _, _ = tiny_suite
    suite_digest = spec_digest(suite)  # type: ignore[arg-type]
    results = tmp_path / "results"
    for model, cer in (("model-a", 0.1), ("model-b", 0.1)):
        directory = results / model
        directory.mkdir(parents=True)
        _write_bundle(
            directory,
            _bundle(model, "success", 0.2, cer, 12.5, suite_digest),
        )
    (tmp_path / "README.md").write_text(
        "before\n<!-- LEADERBOARD:START -->\nstale\n<!-- LEADERBOARD:END -->\nafter\n",
        encoding="utf-8",
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_leaderboards(suite, results, first, require_tracked=False, root=tmp_path)  # type: ignore[arg-type]
    first_readme = (tmp_path / "README.md").read_bytes()
    generate_leaderboards(suite, results, second, require_tracked=False, root=tmp_path)  # type: ignore[arg-type]
    assert (first / "leaderboard.json").read_bytes() == (second / "leaderboard.json").read_bytes()
    assert (first / "leaderboard.csv").read_bytes() == (second / "leaderboard.csv").read_bytes()
    assert (first / "leaderboard-accuracy.svg").read_bytes() == (
        second / "leaderboard-accuracy.svg"
    ).read_bytes()
    assert (first / "leaderboard-memory.svg").read_bytes() == (
        second / "leaderboard-memory.svg"
    ).read_bytes()
    assert (tmp_path / "README.md").read_bytes() == first_readme
    payload = json.loads((first / "leaderboard.json").read_text(encoding="utf-8"))
    assert payload["accuracy"][0]["cer_ci_lower"] == pytest.approx(0.1)
    assert payload["accuracy"][0]["cer_ci_upper"] == pytest.approx(0.1)
    assert payload["adjacent_cer_comparisons"][0]["resolved"] is False
    markdown = (first / "leaderboard.md").read_text(encoding="utf-8")
    assert "## Paired adjacent CER comparisons" in markdown
    assert "| No clear difference |" in markdown
    assert "[+" not in markdown


def test_markdown_links_models_to_declared_hugging_face_repositories(
    tmp_path: Path, tiny_suite: tuple[object, Path, list[object]]
) -> None:
    suite, _, _ = tiny_suite
    suite_digest = spec_digest(suite)  # type: ignore[arg-type]
    results = tmp_path / "results"
    directory = results / "model"
    directory.mkdir(parents=True)
    _write_bundle(directory, _bundle("model", "success", 0.2, 0.1, 12.5, suite_digest))
    models = tmp_path / "models"
    models.mkdir()
    (models / "model.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
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
                    "image": "peste-modern:test",
                    "dockerfile": "runtimes/modern/Dockerfile",
                },
            }
        ),
        encoding="utf-8",
    )
    bundle = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    bundle["model_digest"] = spec_digest(load_model("model", tmp_path))
    (directory / "run.json").write_text(json.dumps(bundle), encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "<!-- LEADERBOARD:START -->\n<!-- LEADERBOARD:END -->\n", encoding="utf-8"
    )

    output = tmp_path / "generated"
    generate_leaderboards(suite, results, output, require_tracked=False, root=tmp_path)  # type: ignore[arg-type]

    expected_link = "[`model`](https://huggingface.co/organization/checkpoint)"
    markdown = (output / "leaderboard.md").read_text(encoding="utf-8")
    assert "# PESTE leaderboard — `tiny-suite-v1`" in markdown
    assert "## Normalized accuracy\n\n![Normalized accuracy leaderboard]" in markdown
    assert (
        "## Accuracy per peak CUDA memory\n\n![Accuracy per peak CUDA memory leaderboard]"
        in markdown
    )
    assert "google/fleurs` / `fa_ir` / `test" not in markdown
    assert "Jetson AGX Orin 32GB" not in markdown
    assert expected_link in markdown
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert expected_link in readme
    accuracy_image = "![Normalized accuracy leaderboard](generated/leaderboard-accuracy.svg)"
    accuracy_table = "| Order | Model | CER | WER | Word accuracy |"
    memory_heading = "### Accuracy per peak CUDA memory"
    memory_image = "![Accuracy per peak CUDA memory leaderboard](generated/leaderboard-memory.svg)"
    memory_table = "| Rank | Model | Accuracy / reserved GiB | WER | Peak CUDA reserved GiB |"
    assert readme.index(accuracy_image) < readme.index(accuracy_table)
    assert readme.index(accuracy_table) < readme.index(memory_heading)
    assert readme.index(memory_image) < readme.index(memory_table)
    assert "CER is the primary ranking metric" in readme
    assert "fa-v1 converts ZWNJ to spaces" in readme
    assert "Point-estimate order does not establish statistical significance" in readme
    assert "<sub>95% CI:" in readme
    accuracy_svg = (output / "leaderboard-accuracy.svg").read_text(encoding="utf-8")
    memory_svg = (output / "leaderboard-memory.svg").read_text(encoding="utf-8")
    for svg in (accuracy_svg, memory_svg):
        assert "https://huggingface.co/organization/checkpoint" in svg
        assert ">model</text>" in svg


def test_uncertainty_requires_prediction_counts_to_match_aggregates(
    tmp_path: Path, tiny_suite: tuple[object, Path, list[object]]
) -> None:
    suite, _, _ = tiny_suite
    results = tmp_path / "results"
    directory = results / "model"
    directory.mkdir(parents=True)
    bundle = _bundle(
        "model",
        "success",
        0.1,
        0.1,
        10.0,
        spec_digest(suite),  # type: ignore[arg-type]
    )
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
    uncommitted["environment"]["peste_revision"] = "uncommitted"  # type: ignore[index]
    _write_bundle(uncommitted_directory, uncommitted)

    assert (
        collect_rows(  # type: ignore[arg-type]
            suite, results, require_tracked=False, root=tmp_path
        )
        == []
    )
