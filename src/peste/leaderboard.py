"""Deterministic static leaderboard generation from immutable run bundles."""

import csv
import io
import logging
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from peste.constants import PROJECT_ROOT
from peste.digests import canonical_json
from peste.plotting import render_accuracy_svg, render_memory_svg
from peste.schemas import RunBundle, RunStatus, SuiteSpec
from peste.specs import discover_models, spec_digest

LOGGER = logging.getLogger(__name__)
GIB = 1024**3


@dataclass(frozen=True, slots=True)
class LeaderboardRow:
    model_id: str
    run_id: str
    wer: float
    cer: float
    word_accuracy_pct: float
    memory_efficiency: float
    peak_cuda_reserved_gib: float
    peak_cuda_allocated_gib: float
    peak_process_rss_gib: float
    checkpoint_gib: float
    parameter_count: int
    native_dtype: str


def _is_tracked(path: Path, root: Path) -> bool:
    try:
        relative = path.resolve().relative_to(root.resolve())
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", str(relative)],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=5,
        )
        return True
    except (ValueError, subprocess.SubprocessError, FileNotFoundError):
        return False


def collect_rows(
    suite: SuiteSpec,
    results_directory: Path,
    *,
    require_tracked: bool = True,
    root: Path = PROJECT_ROOT,
) -> list[LeaderboardRow]:
    rows: list[LeaderboardRow] = []
    suite_digest = spec_digest(suite)
    model_digests = {model.model_id: spec_digest(model) for model in discover_models(root)}
    for path in sorted(results_directory.glob("*/run.json")):
        if require_tracked and not _is_tracked(path, root):
            LOGGER.warning("Ignoring uncommitted result bundle", extra={"path": str(path)})
            continue
        bundle = RunBundle.model_validate_json(path.read_text(encoding="utf-8"))
        if bundle.suite_id != suite.suite_id or bundle.status != RunStatus.SUCCESS:
            continue
        if bundle.suite_digest != suite_digest:
            LOGGER.warning("Ignoring result with a stale suite digest", extra={"path": str(path)})
            continue
        expected_model_digest = model_digests.get(bundle.model_id)
        if expected_model_digest is not None and bundle.model_digest != expected_model_digest:
            LOGGER.warning("Ignoring result with a stale model digest", extra={"path": str(path)})
            continue
        if bundle.environment.peste_revision == "uncommitted":
            LOGGER.warning("Ignoring result from uncommitted source", extra={"path": str(path)})
            continue
        if bundle.aggregates is None:
            continue
        if bundle.aggregates.samples != suite.expected_split_counts[suite.evaluation_split]:
            LOGGER.warning("Ignoring incomplete result bundle", extra={"path": str(path)})
            continue
        memory = bundle.memory
        rows.append(
            LeaderboardRow(
                model_id=bundle.model_id,
                run_id=bundle.run_id,
                wer=bundle.aggregates.wer,
                cer=bundle.aggregates.cer,
                word_accuracy_pct=bundle.aggregates.word_accuracy_pct,
                memory_efficiency=bundle.aggregates.memory_efficiency,
                peak_cuda_reserved_gib=memory.peak_cuda_reserved_bytes / GIB,
                peak_cuda_allocated_gib=memory.peak_cuda_allocated_bytes / GIB,
                peak_process_rss_gib=memory.peak_process_rss_bytes / GIB,
                checkpoint_gib=memory.checkpoint_bytes / GIB,
                parameter_count=memory.parameter_count,
                native_dtype=memory.native_dtype,
            )
        )
    successful_by_model: dict[str, list[LeaderboardRow]] = {}
    for row in rows:
        successful_by_model.setdefault(row.model_id, []).append(row)
    duplicates = [model_id for model_id, runs in successful_by_model.items() if len(runs) > 1]
    if duplicates:
        raise ValueError(f"Multiple official successful runs for models: {', '.join(duplicates)}")
    return rows


def accuracy_order(rows: list[LeaderboardRow]) -> list[LeaderboardRow]:
    return sorted(rows, key=lambda row: (row.wer, row.cer, row.model_id))


def efficiency_order(rows: list[LeaderboardRow]) -> list[LeaderboardRow]:
    return sorted(rows, key=lambda row: (-row.memory_efficiency, row.wer, row.model_id))


def _model_cell(model_id: str, repositories: dict[str, str]) -> str:
    repository = repositories.get(model_id)
    if repository is None:
        return f"`{model_id}`"
    return f"[`{model_id}`](https://huggingface.co/{repository})"


def _table(rows: list[LeaderboardRow], efficiency: bool, repositories: dict[str, str]) -> str:
    if efficiency:
        headings = "| Rank | Model | Accuracy / reserved GiB | WER | Peak CUDA reserved GiB |"
        separator = "|---:|---|---:|---:|---:|"
        values = [
            f"| {rank} | {_model_cell(row.model_id, repositories)} | "
            f"{row.memory_efficiency:.4f} | "
            f"{row.wer:.4f} | {row.peak_cuda_reserved_gib:.3f} |"
            for rank, row in enumerate(rows, start=1)
        ]
    else:
        headings = "| Rank | Model | WER | CER | Word accuracy |"
        separator = "|---:|---|---:|---:|---:|"
        values = [
            f"| {rank} | {_model_cell(row.model_id, repositories)} | "
            f"{row.wer:.4f} | {row.cer:.4f} | "
            f"{row.word_accuracy_pct:.2f}% |"
            for rank, row in enumerate(rows, start=1)
        ]
    if not values:
        values = ["| — | No complete official results yet | — | — | — |"]
    return "\n".join([headings, separator, *values])


def render_markdown(
    suite: SuiteSpec,
    rows: list[LeaderboardRow],
    repositories: dict[str, str] | None = None,
    *,
    image_prefix: str = "",
    heading_level: int = 2,
) -> str:
    model_repositories = repositories or {}
    heading = "#" * heading_level
    return (
        f"# PESTE leaderboard — `{suite.suite_id}`\n\n"
        f"{heading} Normalized accuracy\n\n"
        f"![Normalized accuracy leaderboard]({image_prefix}leaderboard-accuracy.svg)\n\n"
        + _table(accuracy_order(rows), efficiency=False, repositories=model_repositories)
        + f"\n\n{heading} Accuracy per peak CUDA memory\n\n"
        f"![Accuracy per peak CUDA memory leaderboard]({image_prefix}leaderboard-memory.svg)\n\n"
        + _table(efficiency_order(rows), efficiency=True, repositories=model_repositories)
        + "\n\nPeak CUDA memory is unified system/GPU memory and is not directly comparable "
        "with process VRAM reported on discrete GPUs.\n"
    )


def _as_dict(row: LeaderboardRow, rank: int) -> dict[str, str | int | float]:
    return {"rank": rank, **asdict(row)}


def generate_leaderboards(
    suite: SuiteSpec,
    results_directory: Path,
    generated_directory: Path,
    *,
    require_tracked: bool = True,
    root: Path = PROJECT_ROOT,
) -> None:
    rows = collect_rows(suite, results_directory, require_tracked=require_tracked, root=root)
    accuracy = accuracy_order(rows)
    efficiency = efficiency_order(rows)
    repositories = {model.model_id: model.repository for model in discover_models(root)}
    markdown = render_markdown(suite, rows, repositories)
    generated_directory.mkdir(parents=True, exist_ok=True)
    (generated_directory / "leaderboard.md").write_text(markdown, encoding="utf-8")
    (generated_directory / "leaderboard-accuracy.svg").write_text(
        render_accuracy_svg(suite.suite_id, rows, repositories), encoding="utf-8"
    )
    (generated_directory / "leaderboard-memory.svg").write_text(
        render_memory_svg(suite.suite_id, rows, repositories), encoding="utf-8"
    )
    payload = {
        "schema_version": 1,
        "suite_id": suite.suite_id,
        "accuracy": [_as_dict(row, rank) for rank, row in enumerate(accuracy, start=1)],
        "memory_efficiency": [_as_dict(row, rank) for rank, row in enumerate(efficiency, start=1)],
    }
    (generated_directory / "leaderboard.json").write_bytes(canonical_json(payload))
    columns = ["board", "rank", *LeaderboardRow.__dataclass_fields__]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for board, ordered in (("accuracy", accuracy), ("memory_efficiency", efficiency)):
        for rank, row in enumerate(ordered, start=1):
            writer.writerow({"board": board, **_as_dict(row, rank)})
    (generated_directory / "leaderboard.csv").write_text(buffer.getvalue(), encoding="utf-8")
    readme_path = root / "README.md"
    if readme_path.exists():
        readme = readme_path.read_text(encoding="utf-8")
        start_marker = "<!-- LEADERBOARD:START -->"
        end_marker = "<!-- LEADERBOARD:END -->"
        if readme.count(start_marker) != 1 or readme.count(end_marker) != 1:
            raise ValueError("README leaderboard markers are missing or duplicated")
        before, remainder = readme.split(start_marker)
        _, after = remainder.split(end_marker)
        embedded = render_markdown(
            suite,
            rows,
            repositories,
            image_prefix="generated/",
            heading_level=3,
        ).removeprefix(f"# PESTE leaderboard — `{suite.suite_id}`\n\n")
        readme_path.write_text(
            f"{before}{start_marker}\n\n{embedded}\n{end_marker}{after}", encoding="utf-8"
        )
    LOGGER.info(
        "Generated static leaderboards",
        extra={"suite": suite.suite_id, "ranked_models": len(rows)},
    )
