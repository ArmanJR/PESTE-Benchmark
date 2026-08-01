"""Deterministic static leaderboard generation from immutable run bundles."""

import csv
import io
import logging
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from psst.constants import PROJECT_ROOT
from psst.digests import canonical_json
from psst.schemas import RunBundle, RunStatus, SuiteSpec
from psst.specs import discover_models, spec_digest

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
        if bundle.environment.psst_revision == "uncommitted":
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


def _table(rows: list[LeaderboardRow], efficiency: bool) -> str:
    if efficiency:
        headings = "| Rank | Model | Accuracy / reserved GiB | WER | Peak CUDA reserved GiB |"
        separator = "|---:|---|---:|---:|---:|"
        values = [
            f"| {rank} | `{row.model_id}` | {row.memory_efficiency:.4f} | "
            f"{row.wer:.4f} | {row.peak_cuda_reserved_gib:.3f} |"
            for rank, row in enumerate(rows, start=1)
        ]
    else:
        headings = "| Rank | Model | WER | CER | Word accuracy |"
        separator = "|---:|---|---:|---:|---:|"
        values = [
            f"| {rank} | `{row.model_id}` | {row.wer:.4f} | {row.cer:.4f} | "
            f"{row.word_accuracy_pct:.2f}% |"
            for rank, row in enumerate(rows, start=1)
        ]
    if not values:
        values = ["| — | No complete official results yet | — | — | — |"]
    return "\n".join([headings, separator, *values])


def render_markdown(rows: list[LeaderboardRow]) -> str:
    return (
        "# PSST v1 leaderboards\n\n"
        "## Normalized accuracy — FLEURS Persian test split\n\n"
        + _table(accuracy_order(rows), efficiency=False)
        + "\n\n## Accuracy per peak CUDA memory — Jetson AGX Orin 32GB\n\n"
        + _table(efficiency_order(rows), efficiency=True)
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
    generated_directory.mkdir(parents=True, exist_ok=True)
    (generated_directory / "leaderboard.md").write_text(render_markdown(rows), encoding="utf-8")
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
        embedded = render_markdown(rows).removeprefix("# PSST v1 leaderboards\n\n")
        readme_path.write_text(
            f"{before}{start_marker}\n\n{embedded}\n{end_marker}{after}", encoding="utf-8"
        )
    LOGGER.info(
        "Generated static leaderboards",
        extra={"suite": suite.suite_id, "ranked_models": len(rows)},
    )
