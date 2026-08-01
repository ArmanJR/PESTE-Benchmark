"""Ranking, OOM exclusion, and generated output tests."""

import json
from pathlib import Path

from psst.leaderboard import accuracy_order, collect_rows, efficiency_order, generate_leaderboards
from psst.specs import spec_digest


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
        aggregates = {
            "samples": 2,
            "wer": wer,
            "cer": cer,
            "word_errors": 1,
            "word_reference_units": 2,
            "character_errors": 1,
            "character_reference_units": 10,
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
            "psst_revision": "rev",
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
        ("oom-model", "oom", 0.0, 0.0, 100.0),
    ):
        directory = results / model
        directory.mkdir(parents=True)
        (directory / "run.json").write_text(
            json.dumps(_bundle(model, status, wer, cer, efficiency, suite_digest)),
            encoding="utf-8",
        )
    rows = collect_rows(suite, results, require_tracked=False, root=tmp_path)  # type: ignore[arg-type]
    assert [row.model_id for row in accuracy_order(rows)] == ["beta", "alpha", "zeta"]
    assert [row.model_id for row in efficiency_order(rows)] == ["beta", "zeta", "alpha"]


def test_static_outputs_are_deterministic(
    tmp_path: Path, tiny_suite: tuple[object, Path, list[object]]
) -> None:
    suite, _, _ = tiny_suite
    suite_digest = spec_digest(suite)  # type: ignore[arg-type]
    results = tmp_path / "results"
    directory = results / "model"
    directory.mkdir(parents=True)
    (directory / "run.json").write_text(
        json.dumps(_bundle("model", "success", 0.2, 0.1, 12.5, suite_digest)),
        encoding="utf-8",
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
    assert (tmp_path / "README.md").read_bytes() == first_readme


def test_stale_and_uncommitted_results_are_excluded(
    tmp_path: Path, tiny_suite: tuple[object, Path, list[object]]
) -> None:
    suite, _, _ = tiny_suite
    results = tmp_path / "results"
    stale_directory = results / "stale"
    stale_directory.mkdir(parents=True)
    (stale_directory / "run.json").write_text(
        json.dumps(_bundle("stale", "success", 0.1, 0.1, 10.0)), encoding="utf-8"
    )
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
    uncommitted["environment"]["psst_revision"] = "uncommitted"  # type: ignore[index]
    (uncommitted_directory / "run.json").write_text(json.dumps(uncommitted), encoding="utf-8")

    assert (
        collect_rows(  # type: ignore[arg-type]
            suite, results, require_tracked=False, root=tmp_path
        )
        == []
    )
