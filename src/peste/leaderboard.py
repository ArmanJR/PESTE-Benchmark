"""Deterministic static leaderboard generation from immutable run bundles."""

import csv
import io
import logging
import math
import subprocess
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

from peste.constants import DEFAULT_SEED, PROJECT_ROOT
from peste.digests import canonical_json
from peste.plotting import render_accuracy_svg, render_pareto_svg, render_speed_svg
from peste.schemas import PredictionRecord, RunBundle, RunStatus, SuiteSpec
from peste.specs import discover_models, spec_digest
from peste.uncertainty import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_CONFIDENCE_LEVEL,
    bootstrap_paired_rate_difference,
    bootstrap_rate,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LeaderboardRow:
    model_id: str
    run_id: str
    wer: float
    wer_ci_lower: float
    wer_ci_upper: float
    cer: float
    cer_ci_lower: float
    cer_ci_upper: float
    word_accuracy_pct: float
    speed_valid: bool
    batch_size: int
    audio_throughput_x: float
    rtf: float
    processing_seconds: float
    total_audio_seconds: float
    checkpoint_gib: float
    parameter_count: int
    native_dtype: str


@dataclass(frozen=True, slots=True)
class PredictionCounts:
    sample_ids: tuple[str, ...]
    word_errors: tuple[int, ...]
    word_reference_units: tuple[int, ...]
    character_errors: tuple[int, ...]
    character_reference_units: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PairedComparison:
    first_model_id: str
    second_model_id: str
    cer_difference: float
    ci_lower: float
    ci_upper: float
    resolved: bool


@dataclass(frozen=True, slots=True)
class ParetoDominance:
    dominant_model_id: str
    dominated_model_id: str
    cer_difference: float
    ci_lower: float
    ci_upper: float
    statistically_supported: bool


@dataclass(frozen=True, slots=True)
class ParetoEfficiency:
    row: LeaderboardRow
    pareto_efficient: bool
    statistically_pareto_efficient: bool
    dominated_by: tuple[str, ...]
    statistically_dominated_by: tuple[str, ...]


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


def _load_prediction_counts(
    run_path: Path, bundle: RunBundle, expected_samples: int
) -> PredictionCounts:
    prediction_path = run_path.parent / bundle.predictions_path
    records: list[PredictionRecord] = []
    with prediction_path.open(encoding="utf-8") as predictions:
        for line_number, line in enumerate(predictions, start=1):
            try:
                record = PredictionRecord.model_validate_json(line)
            except ValueError as error:
                raise ValueError(
                    f"Invalid prediction at {prediction_path}:{line_number}: {error}"
                ) from error
            if record.sequence != len(records):
                raise ValueError(f"Non-contiguous prediction sequence in {prediction_path}")
            records.append(record)
    if len(records) != expected_samples:
        raise ValueError(
            f"Prediction count mismatch for {bundle.run_id}: "
            f"expected {expected_samples}, got {len(records)}"
        )
    counts = PredictionCounts(
        sample_ids=tuple(record.sample_id for record in records),
        word_errors=tuple(
            record.word_substitutions + record.word_deletions + record.word_insertions
            for record in records
        ),
        word_reference_units=tuple(record.word_reference_units for record in records),
        character_errors=tuple(
            record.character_substitutions
            + record.character_deletions
            + record.character_insertions
            for record in records
        ),
        character_reference_units=tuple(record.character_reference_units for record in records),
    )
    LOGGER.debug(
        "Loaded prediction counts for uncertainty estimation",
        extra={"run": bundle.run_id, "model": bundle.model_id, "samples": len(records)},
    )
    return counts


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
        counts = _load_prediction_counts(
            path, bundle, suite.expected_split_counts[suite.evaluation_split]
        )
        wer_estimate = bootstrap_rate(
            counts.word_errors,
            counts.word_reference_units,
            seed=DEFAULT_SEED,
        )
        cer_estimate = bootstrap_rate(
            counts.character_errors,
            counts.character_reference_units,
            seed=DEFAULT_SEED,
        )
        if not math.isclose(wer_estimate.point, bundle.aggregates.wer, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"WER aggregate does not match predictions for {bundle.run_id}")
        if not math.isclose(cer_estimate.point, bundle.aggregates.cer, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"CER aggregate does not match predictions for {bundle.run_id}")
        speed = bundle.speed
        facts = bundle.model_facts
        rows.append(
            LeaderboardRow(
                model_id=bundle.model_id,
                run_id=bundle.run_id,
                wer=bundle.aggregates.wer,
                wer_ci_lower=wer_estimate.lower,
                wer_ci_upper=wer_estimate.upper,
                cer=bundle.aggregates.cer,
                cer_ci_lower=cer_estimate.lower,
                cer_ci_upper=cer_estimate.upper,
                word_accuracy_pct=bundle.aggregates.word_accuracy_pct,
                speed_valid=speed.valid,
                batch_size=speed.batch_size,
                audio_throughput_x=speed.audio_throughput_x,
                rtf=speed.rtf,
                processing_seconds=speed.processing_seconds,
                total_audio_seconds=speed.total_audio_seconds,
                checkpoint_gib=facts.checkpoint_bytes / 1024**3,
                parameter_count=facts.parameter_count,
                native_dtype=facts.native_dtype,
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
    return sorted(rows, key=lambda row: (row.cer, row.wer, row.model_id))


def speed_order(rows: list[LeaderboardRow]) -> list[LeaderboardRow]:
    return sorted(
        (row for row in rows if row.speed_valid),
        key=lambda row: (-row.audio_throughput_x, row.cer, row.wer, row.model_id),
    )


def _point_dominates(first: LeaderboardRow, second: LeaderboardRow) -> bool:
    """Return whether first is no worse on both objectives and strictly better on one."""
    no_less_accurate = first.cer <= second.cer
    no_slower = first.audio_throughput_x >= second.audio_throughput_x
    strictly_better = first.cer < second.cer or first.audio_throughput_x > second.audio_throughput_x
    return no_less_accurate and no_slower and strictly_better


def pareto_dominators(rows: list[LeaderboardRow]) -> dict[str, tuple[str, ...]]:
    """Return deterministic point-estimate dominators for every speed-valid model."""
    ordered = speed_order(rows)
    return {
        target.model_id: tuple(
            candidate.model_id
            for candidate in ordered
            if candidate.model_id != target.model_id and _point_dominates(candidate, target)
        )
        for target in ordered
    }


def _prediction_counts_by_model(
    rows: list[LeaderboardRow], results_directory: Path, expected_samples: int
) -> dict[str, PredictionCounts]:
    run_paths: dict[str, tuple[Path, RunBundle]] = {}
    for run_path in results_directory.glob("*/run.json"):
        bundle = RunBundle.model_validate_json(run_path.read_text(encoding="utf-8"))
        run_paths[bundle.run_id] = (run_path, bundle)
    counts_by_model: dict[str, PredictionCounts] = {}
    for row in rows:
        try:
            run_path, bundle = run_paths[row.run_id]
        except KeyError as error:
            raise ValueError(f"Missing result bundle for run {row.run_id}") from error
        counts_by_model[row.model_id] = _load_prediction_counts(run_path, bundle, expected_samples)
    return counts_by_model


def _validate_paired_counts(
    first_model_id: str,
    second_model_id: str,
    first_counts: PredictionCounts,
    second_counts: PredictionCounts,
) -> None:
    if (
        first_counts.sample_ids != second_counts.sample_ids
        or first_counts.character_reference_units != second_counts.character_reference_units
    ):
        raise ValueError(f"Cannot pair predictions for {first_model_id} and {second_model_id}")


def paired_cer_comparisons(
    rows: list[LeaderboardRow], results_directory: Path, expected_samples: int
) -> list[PairedComparison]:
    comparisons: list[PairedComparison] = []
    ordered = accuracy_order(rows)
    counts_by_model = _prediction_counts_by_model(ordered, results_directory, expected_samples)
    for first, second in pairwise(ordered):
        first_counts = counts_by_model[first.model_id]
        second_counts = counts_by_model[second.model_id]
        _validate_paired_counts(first.model_id, second.model_id, first_counts, second_counts)
        estimate = bootstrap_paired_rate_difference(
            first_counts.character_errors,
            second_counts.character_errors,
            first_counts.character_reference_units,
            seed=DEFAULT_SEED,
        )
        comparisons.append(
            PairedComparison(
                first_model_id=first.model_id,
                second_model_id=second.model_id,
                cer_difference=estimate.point,
                ci_lower=estimate.lower,
                ci_upper=estimate.upper,
                resolved=estimate.upper < 0 or estimate.lower > 0,
            )
        )
    return comparisons


def pareto_efficiency(
    rows: list[LeaderboardRow], results_directory: Path, expected_samples: int
) -> tuple[list[ParetoEfficiency], list[ParetoDominance]]:
    """Classify point-estimate efficiency and test each dominance edge with paired CER."""
    ordered = speed_order(rows)
    rows_by_model = {row.model_id: row for row in ordered}
    dominators = pareto_dominators(ordered)
    counts_by_model = _prediction_counts_by_model(ordered, results_directory, expected_samples)
    comparisons: list[ParetoDominance] = []
    supported_by_model: dict[str, list[str]] = {row.model_id: [] for row in ordered}
    for dominated in ordered:
        dominated_counts = counts_by_model[dominated.model_id]
        for dominant_model_id in dominators[dominated.model_id]:
            dominant = rows_by_model[dominant_model_id]
            dominant_counts = counts_by_model[dominant_model_id]
            _validate_paired_counts(
                dominant.model_id,
                dominated.model_id,
                dominant_counts,
                dominated_counts,
            )
            estimate = bootstrap_paired_rate_difference(
                dominant_counts.character_errors,
                dominated_counts.character_errors,
                dominant_counts.character_reference_units,
                seed=DEFAULT_SEED,
            )
            statistically_supported = estimate.upper < 0
            comparisons.append(
                ParetoDominance(
                    dominant_model_id=dominant.model_id,
                    dominated_model_id=dominated.model_id,
                    cer_difference=estimate.point,
                    ci_lower=estimate.lower,
                    ci_upper=estimate.upper,
                    statistically_supported=statistically_supported,
                )
            )
            if statistically_supported:
                supported_by_model[dominated.model_id].append(dominant.model_id)
    entries = [
        ParetoEfficiency(
            row=row,
            pareto_efficient=not dominators[row.model_id],
            statistically_pareto_efficient=not supported_by_model[row.model_id],
            dominated_by=dominators[row.model_id],
            statistically_dominated_by=tuple(supported_by_model[row.model_id]),
        )
        for row in ordered
    ]
    entries.sort(
        key=lambda entry: (
            not entry.pareto_efficient,
            -entry.row.audio_throughput_x,
            entry.row.cer,
            entry.row.wer,
            entry.row.model_id,
        )
    )
    LOGGER.info(
        "Calculated accuracy-speed Pareto efficiency",
        extra={
            "eligible_models": len(entries),
            "frontier_models": sum(entry.pareto_efficient for entry in entries),
            "dominance_edges": len(comparisons),
            "statistically_supported_edges": sum(
                comparison.statistically_supported for comparison in comparisons
            ),
        },
    )
    return entries, comparisons


def _model_cell(model_id: str, repositories: dict[str, str]) -> str:
    repository = repositories.get(model_id)
    if repository is None:
        return model_id
    return f"[{model_id}](https://huggingface.co/{repository})"


def _rate_with_interval(point: float, lower: float, upper: float) -> str:
    return f"{point:.4f}<br><sub>95% CI: {lower:.4f}–{upper:.4f}</sub>"


def _percentage_points(value: float) -> str:
    return f"{value * 100:.2f}".replace("-", "−")


def _table(rows: list[LeaderboardRow], speed: bool, repositories: dict[str, str]) -> str:
    if speed:
        headings = "| Rank | Model | Batch | Throughput | RTF | Processing s | Audio s |"
        separator = "|---:|---|---:|---:|---:|---:|---:|"
        values = [
            f"| {rank} | {_model_cell(row.model_id, repositories)} | "
            f"{row.batch_size} | {row.audio_throughput_x:.3f}× | {row.rtf:.5f} | "
            f"{row.processing_seconds:.3f} | {row.total_audio_seconds:.3f} |"
            for rank, row in enumerate(rows, start=1)
        ]
    else:
        headings = "| Order | Model | CER | WER | Word accuracy |"
        separator = "|---:|---|---:|---:|---:|"
        values = [
            f"| {rank} | {_model_cell(row.model_id, repositories)} | "
            f"{_rate_with_interval(row.cer, row.cer_ci_lower, row.cer_ci_upper)} | "
            f"{_rate_with_interval(row.wer, row.wer_ci_lower, row.wer_ci_upper)} | "
            f"{row.word_accuracy_pct:.2f}% |"
            for rank, row in enumerate(rows, start=1)
        ]
    if not values:
        columns = 7 if speed else 5
        values = [
            "| — | No complete official results yet | "
            + " | ".join("—" for _ in range(columns - 2))
            + " |"
        ]
    return "\n".join([headings, separator, *values])


def _comparison_table(comparisons: list[PairedComparison], repositories: dict[str, str]) -> str:
    headings = "| Adjacent models | ΔCER | Paired 95% range | Evidence |"
    separator = "|---|---:|---:|---|"
    values = [
        f"| {_model_cell(comparison.first_model_id, repositories)} − "
        f"{_model_cell(comparison.second_model_id, repositories)} | "
        f"{_percentage_points(comparison.cer_difference)} pp | "
        f"{_percentage_points(comparison.ci_lower)} to "
        f"{_percentage_points(comparison.ci_upper)} pp | "
        f"{'First model has lower CER' if comparison.resolved else 'No clear difference'} |"
        for comparison in comparisons
    ]
    return "\n".join([headings, separator, *values])


def _model_list(model_ids: tuple[str, ...], repositories: dict[str, str]) -> str:
    if not model_ids:
        return "—"
    return "<br>".join(_model_cell(model_id, repositories) for model_id in model_ids)


def _pareto_table(entries: list[ParetoEfficiency], repositories: dict[str, str]) -> str:
    headings = (
        "| Order | Model | CER | Throughput | Point frontier | Supported frontier | "
        "Point dominators | Supported dominators |"
    )
    separator = "|---:|---|---:|---:|---|---|---|---|"
    values = [
        f"| {order} | {_model_cell(entry.row.model_id, repositories)} | "
        f"{_rate_with_interval(entry.row.cer, entry.row.cer_ci_lower, entry.row.cer_ci_upper)} | "
        f"{entry.row.audio_throughput_x:.3f}× | "
        f"{'Yes' if entry.pareto_efficient else 'No'} | "
        f"{'Yes' if entry.statistically_pareto_efficient else 'No'} | "
        f"{_model_list(entry.dominated_by, repositories)} | "
        f"{_model_list(entry.statistically_dominated_by, repositories)} |"
        for order, entry in enumerate(entries, start=1)
    ]
    if not values:
        values = ["| — | No complete speed-valid results yet | — | — | — | — | — | — |"]
    return "\n".join([headings, separator, *values])


def render_markdown(
    suite: SuiteSpec,
    rows: list[LeaderboardRow],
    repositories: dict[str, str] | None = None,
    comparisons: list[PairedComparison] | None = None,
    pareto_entries: list[ParetoEfficiency] | None = None,
    *,
    image_prefix: str = "",
    heading_level: int = 2,
) -> str:
    model_repositories = repositories or {}
    paired_comparisons = comparisons or []
    efficiency_entries = pareto_entries or []
    heading = "#" * heading_level
    subheading = "#" * (heading_level + 1)
    comparison_section = ""
    if paired_comparisons:
        comparison_section = (
            f"\n\n{subheading} Paired adjacent CER comparisons\n\n"
            + _comparison_table(paired_comparisons, model_repositories)
        )
    return (
        f"# PESTE leaderboard — `{suite.suite_id}`\n\n"
        f"{heading} Normalized accuracy\n\n"
        f"![Normalized accuracy leaderboard]({image_prefix}leaderboard-accuracy.svg)\n\n"
        + _table(accuracy_order(rows), speed=False, repositories=model_repositories)
        + "\n\nCER is the primary ranking metric because Persian WER is orthography-sensitive: "
        "fa-v1 converts ZWNJ to spaces, while CER ignores normalized whitespace. WER and "
        "derived word accuracy remain complementary, segmentation-sensitive measurements."
        "\n\nPoint-estimate order does not establish statistical significance. Intervals use "
        f"a deterministic {DEFAULT_BOOTSTRAP_REPLICATES:,}-replicate utterance-level percentile "
        f"bootstrap at {DEFAULT_CONFIDENCE_LEVEL:.0%} confidence with seed {DEFAULT_SEED}. "
        "Paired intervals containing zero are reported as no clear difference; these intervals "
        "measure test-set sampling uncertainty only."
        + comparison_section
        + f"\n\n{heading} Steady-state speed\n\n"
        f"![Steady-state speed leaderboard]({image_prefix}leaderboard-speed.svg)\n\n"
        + _table(speed_order(rows), speed=True, repositories=model_repositories)
        + "\n\nThroughput is total audio seconds divided by measured processing seconds; "
        "RTF is its reciprocal. Resumed runs retain accuracy but are excluded here.\n"
        + f"\n{heading} Accuracy-speed Pareto efficiency\n\n"
        f"![Accuracy-speed Pareto efficiency]({image_prefix}leaderboard-pareto.svg)\n\n"
        + _pareto_table(efficiency_entries, model_repositories)
        + "\n\nA speed-valid model is Pareto-efficient when no other model has both equal-or-lower "
        "CER and equal-or-higher throughput, with at least one strict advantage. Point dominators "
        "use the published estimates. Supported dominators additionally require the paired 95% "
        "CER-difference interval to remain below zero. CER intervals measure test-set sampling "
        "uncertainty; speed is a single deterministic run without a confidence interval. The plot "
        "inverts its logarithmic CER axis so visually better directions are up and right while "
        "tick labels remain raw CER. Pareto status is a trade-off classification, not a composite "
        "score.\n"
    )


def _as_dict(row: LeaderboardRow, rank: int) -> dict[str, str | int | float]:
    return {"rank": rank, **asdict(row)}


def _pareto_as_dict(entry: ParetoEfficiency, order: int) -> dict[str, object]:
    return {
        "order": order,
        **asdict(entry.row),
        "pareto_efficient": entry.pareto_efficient,
        "statistically_pareto_efficient": entry.statistically_pareto_efficient,
        "dominated_by": list(entry.dominated_by),
        "dominated_by_count": len(entry.dominated_by),
        "statistically_dominated_by": list(entry.statistically_dominated_by),
        "statistically_dominated_by_count": len(entry.statistically_dominated_by),
    }


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
    speed = speed_order(rows)
    comparisons = paired_cer_comparisons(
        accuracy,
        results_directory,
        suite.expected_split_counts[suite.evaluation_split],
    )
    pareto_entries, pareto_comparisons = pareto_efficiency(
        rows,
        results_directory,
        suite.expected_split_counts[suite.evaluation_split],
    )
    repositories = {model.model_id: model.repository for model in discover_models(root)}
    markdown = render_markdown(suite, rows, repositories, comparisons, pareto_entries)
    generated_directory.mkdir(parents=True, exist_ok=True)
    (generated_directory / "leaderboard.md").write_text(markdown, encoding="utf-8")
    (generated_directory / "leaderboard-accuracy.svg").write_text(
        render_accuracy_svg(suite.suite_id, rows, repositories), encoding="utf-8"
    )
    (generated_directory / "leaderboard-speed.svg").write_text(
        render_speed_svg(suite.suite_id, rows, repositories), encoding="utf-8"
    )
    frontier_model_ids = {entry.row.model_id for entry in pareto_entries if entry.pareto_efficient}
    statistically_dominated_model_ids = {
        entry.row.model_id for entry in pareto_entries if entry.statistically_dominated_by
    }
    (generated_directory / "leaderboard-pareto.svg").write_text(
        render_pareto_svg(
            suite.suite_id,
            rows,
            frontier_model_ids,
            statistically_dominated_model_ids,
            repositories,
        ),
        encoding="utf-8",
    )
    payload = {
        "schema_version": 2,
        "suite_id": suite.suite_id,
        "uncertainty": {
            "method": "utterance-level percentile bootstrap",
            "confidence_level": DEFAULT_CONFIDENCE_LEVEL,
            "replicates": DEFAULT_BOOTSTRAP_REPLICATES,
            "seed": DEFAULT_SEED,
        },
        "accuracy": [_as_dict(row, rank) for rank, row in enumerate(accuracy, start=1)],
        "adjacent_cer_comparisons": [asdict(comparison) for comparison in comparisons],
        "speed": [_as_dict(row, rank) for rank, row in enumerate(speed, start=1)],
        "pareto": [
            _pareto_as_dict(entry, order) for order, entry in enumerate(pareto_entries, start=1)
        ],
        "pareto_dominance_comparisons": [asdict(comparison) for comparison in pareto_comparisons],
    }
    (generated_directory / "leaderboard.json").write_bytes(canonical_json(payload))
    pareto_columns = (
        "pareto_order",
        "pareto_efficient",
        "statistically_pareto_efficient",
        "dominated_by",
        "dominated_by_count",
        "statistically_dominated_by",
        "statistically_dominated_by_count",
    )
    columns = ["board", "rank", *LeaderboardRow.__dataclass_fields__, *pareto_columns]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for board, ordered in (("accuracy", accuracy), ("speed", speed)):
        for rank, row in enumerate(ordered, start=1):
            writer.writerow({"board": board, **_as_dict(row, rank)})
    for order, entry in enumerate(pareto_entries, start=1):
        writer.writerow(
            {
                "board": "pareto",
                "rank": "",
                "pareto_order": order,
                **asdict(entry.row),
                "pareto_efficient": entry.pareto_efficient,
                "statistically_pareto_efficient": entry.statistically_pareto_efficient,
                "dominated_by": ";".join(entry.dominated_by),
                "dominated_by_count": len(entry.dominated_by),
                "statistically_dominated_by": ";".join(entry.statistically_dominated_by),
                "statistically_dominated_by_count": len(entry.statistically_dominated_by),
            }
        )
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
            comparisons,
            pareto_entries,
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
