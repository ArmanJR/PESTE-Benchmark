"""Deterministic, resumable, steady-state batched benchmark runner."""

import importlib.metadata
import json
import logging
import os
import platform
import random
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from peste.adapters import create_adapter
from peste.adapters.base import ASRAdapter, Transcription, require_batch_cardinality
from peste.digests import canonical_json
from peste.logging import configure_logging
from peste.manifest import validate_manifest
from peste.metrics import EditCounts, SampleScore, aggregate_scores, score_sample
from peste.schemas import (
    AggregateMetrics,
    EnvironmentFingerprint,
    LogReferences,
    ManifestRow,
    ModelFacts,
    ModelSpec,
    PredictionRecord,
    RunBundle,
    RunRequest,
    RunStatus,
    SpeedStatistics,
    SuiteSpec,
)
from peste.specs import spec_digest

LOGGER = logging.getLogger(__name__)
WARMUP_BATCHES = 2
TIMING_ARTIFACT = "timing.jsonl"


def _seed_runtime(seed: int) -> Any:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return torch


def _checkpoint_bytes(model: ModelSpec, cache_directory: Path) -> int:
    snapshot = (
        cache_directory
        / f"models--{model.repository.replace('/', '--')}"
        / "snapshots"
        / model.revision
    )
    if not snapshot.exists():
        LOGGER.warning("Pinned snapshot directory not found", extra={"path": str(snapshot)})
        return 0
    seen: set[tuple[int, int]] = set()
    total = 0
    for path in snapshot.rglob("*"):
        if not path.is_file():
            continue
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity not in seen:
            total += stat.st_size
            seen.add(identity)
    return total


def _source_revision() -> str:
    explicit = os.environ.get("PESTE_SOURCE_REVISION")
    if explicit:
        return explicit
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "uncommitted"


def _environment(seed: int) -> EnvironmentFingerprint:
    import torch

    versions = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    }
    try:
        hardware = json.loads(os.environ.get("PESTE_HARDWARE_PROFILE_JSON", "{}"))
    except json.JSONDecodeError as error:
        raise ValueError("PESTE_HARDWARE_PROFILE_JSON is not valid JSON") from error
    provenance = hardware.get("cloud_provenance")
    return EnvironmentFingerprint(
        peste_revision=_source_revision(),
        image_reference=os.environ.get("PESTE_IMAGE_REFERENCE", "unknown"),
        image_digest=os.environ.get("PESTE_IMAGE_DIGEST", "unknown"),
        dependency_versions=versions,
        python_version=platform.python_version(),
        pytorch_version=torch.__version__,
        cuda_version=str(torch.version.cuda),
        hardware_profile=hardware,
        gpu_product_name=str(hardware.get("gpu_product_name", "unknown")),
        driver_version=str(hardware.get("driver_version", "unknown")),
        ecc_state=str(hardware.get("ecc_state", "unknown")),
        power_limit_watts=float(hardware.get("power_limit_watts", 0)),
        cpu_model=str(hardware.get("cpu_model", "unknown")),
        gpu_uuid=str(hardware.get("gpu_uuid", "unknown")),
        cloud_provenance=None if provenance is None else str(provenance),
        seed=seed,
    )


def _record(
    sequence: int,
    sample_id: str,
    reference: str,
    transcription: Transcription,
    normalization_version: str,
) -> PredictionRecord:
    score = score_sample(reference, transcription.text, normalization_version)
    return PredictionRecord(
        schema_version=2,
        sequence=sequence,
        sample_id=sample_id,
        reference=reference,
        prediction=transcription.text,
        normalized_reference=score.normalized_reference,
        normalized_prediction=score.normalized_prediction,
        word_substitutions=score.words.substitutions,
        word_deletions=score.words.deletions,
        word_insertions=score.words.insertions,
        word_reference_units=score.words.reference_units,
        character_substitutions=score.characters.substitutions,
        character_deletions=score.characters.deletions,
        character_insertions=score.characters.insertions,
        character_reference_units=score.characters.reference_units,
        structured_output=transcription.structured_output,
    )


def _score_from_record(record: PredictionRecord) -> SampleScore:
    return SampleScore(
        normalized_reference=record.normalized_reference,
        normalized_prediction=record.normalized_prediction,
        words=EditCounts(
            record.word_substitutions,
            record.word_deletions,
            record.word_insertions,
            record.word_reference_units,
        ),
        characters=EditCounts(
            record.character_substitutions,
            record.character_deletions,
            record.character_insertions,
            record.character_reference_units,
        ),
    )


def duration_order(rows: Sequence[ManifestRow]) -> list[tuple[int, ManifestRow]]:
    """Return deterministic measured order without changing manifest sequence identity."""
    return sorted(enumerate(rows), key=lambda item: (item[1].duration_seconds, item[0]))


def duration_batches(
    rows: Sequence[ManifestRow], batch_size: int
) -> list[list[tuple[int, ManifestRow]]]:
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    ordered = duration_order(rows)
    return [ordered[index : index + batch_size] for index in range(0, len(ordered), batch_size)]


def _prime_audio(paths: Sequence[Path]) -> None:
    """Validate paths and populate the OS page cache outside the timed region."""
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Canonical audio is missing: {path}")
        with path.open("rb") as audio:
            while audio.read(1024 * 1024):
                pass


def _timed_transcribe_batch(
    adapter: ASRAdapter,
    audio_paths: list[Path],
    torch: Any,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[list[Transcription], float]:
    """Synchronize at the exact boundaries of one end-to-end adapter call."""
    torch.cuda.synchronize()
    started = clock()
    transcriptions = adapter.transcribe_batch(audio_paths)
    torch.cuda.synchronize()
    elapsed = clock() - started
    if elapsed <= 0:
        raise RuntimeError("Measured batch processing time must be positive")
    return require_batch_cardinality(adapter.spec.model_id, audio_paths, transcriptions), elapsed


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json(value))


def _write_predictions(path: Path, records: Sequence[PredictionRecord]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as output:
        for record in sorted(records, key=lambda item: item.sequence):
            output.write(canonical_json(record.model_dump(mode="json")))
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def _load_journal(
    path: Path, batches: Sequence[Sequence[tuple[int, ManifestRow]]]
) -> tuple[list[PredictionRecord], float, float]:
    records: list[PredictionRecord] = []
    total_audio_seconds = 0.0
    processing_seconds = 0.0
    if not path.exists():
        return records, total_audio_seconds, processing_seconds
    with path.open(encoding="utf-8") as journal:
        for line_number, line in enumerate(journal, start=1):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid timing journal JSON at line {line_number}: {error}"
                ) from error
            if entry.get("schema_version") != 2:
                raise ValueError(
                    f"Unsupported timing journal schema_version at line {line_number}; expected 2"
                )
            batch_index = entry.get("batch_index")
            if batch_index != line_number - 1 or batch_index >= len(batches):
                raise ValueError(f"Non-contiguous timing journal batch at line {line_number}")
            expected = batches[batch_index]
            expected_sequences = [sequence for sequence, _ in expected]
            expected_sample_ids = [row.sample_id for _, row in expected]
            if entry.get("sequences") != expected_sequences:
                raise ValueError(f"Timing journal sequence mismatch at line {line_number}")
            if entry.get("sample_ids") != expected_sample_ids:
                raise ValueError(f"Timing journal sample mismatch at line {line_number}")
            batch_records = [PredictionRecord.model_validate(value) for value in entry["records"]]
            if [record.sequence for record in batch_records] != expected_sequences:
                raise ValueError(f"Timing journal prediction order mismatch at line {line_number}")
            records.extend(batch_records)
            total_audio_seconds += float(entry["audio_seconds"])
            processing_seconds += float(entry["processing_seconds"])
    return records, total_audio_seconds, processing_seconds


def _append_journal_entry(
    path: Path,
    batch_index: int,
    batch: Sequence[tuple[int, ManifestRow]],
    records: Sequence[PredictionRecord],
    processing_seconds: float,
) -> None:
    audio_seconds = sum(row.duration_seconds for _, row in batch)
    entry = {
        "schema_version": 2,
        "batch_index": batch_index,
        "sequences": [sequence for sequence, _ in batch],
        "sample_ids": [row.sample_id for _, row in batch],
        "audio_seconds": audio_seconds,
        "processing_seconds": processing_seconds,
        "records": [record.model_dump(mode="json") for record in records],
    }
    with path.open("ab") as journal:
        journal.write(canonical_json(entry))
        journal.flush()
        os.fsync(journal.fileno())


def _warmup_paths(
    batches: Sequence[Sequence[tuple[int, ManifestRow]]], dataset_cache: Path
) -> list[Path]:
    representative = batches[len(batches) // 2]
    return [dataset_cache / row.audio_path for _, row in representative]


def _invalid_speed(
    model: ModelSpec,
    measured_batches: int,
    total_audio_seconds: float,
    processing_seconds: float,
    reason: str,
) -> SpeedStatistics:
    throughput = (
        total_audio_seconds / processing_seconds
        if total_audio_seconds > 0 and processing_seconds > 0
        else 0.0
    )
    rtf = (
        processing_seconds / total_audio_seconds
        if total_audio_seconds > 0 and processing_seconds > 0
        else 0.0
    )
    return SpeedStatistics(
        valid=False,
        batch_size=model.speed_profile.batch_size,
        warmup_batches=WARMUP_BATCHES,
        measured_batches=measured_batches,
        total_audio_seconds=total_audio_seconds,
        processing_seconds=processing_seconds,
        audio_throughput_x=throughput,
        rtf=rtf,
        timing_artifact=TIMING_ARTIFACT,
        invalidity_reason=reason,
    )


def run_benchmark(
    request: RunRequest,
    suite: SuiteSpec,
    model: ModelSpec,
    suite_directory: Path,
    adapter: ASRAdapter | None = None,
) -> RunBundle:
    """Execute or resume one official evaluation run."""
    if request.suite_digest != spec_digest(suite) or request.model_digest != spec_digest(model):
        raise ValueError("Run request spec digests do not match the loaded specifications")
    output = request.output_directory
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "runner.jsonl"
    configure_logging(log_path=log_path)
    request_path = output / "request.json"
    request_payload = request.model_copy(update={"resume": None}).model_dump(mode="json")
    if request_path.exists():
        existing = json.loads(request_path.read_text(encoding="utf-8"))
        if existing != request_payload:
            raise ValueError("Resume request differs from the original immutable run specification")
        if request.resume is None:
            raise FileExistsError("Run output already exists; pass an explicit resume state")
    else:
        _write_json(request_path, request_payload)

    manifest_rows = validate_manifest(suite, suite_directory)
    ranked_rows = [row for row in manifest_rows if row.split == suite.evaluation_split]
    expected_samples = suite.expected_split_counts[suite.evaluation_split]
    if len(ranked_rows) != expected_samples:
        raise ValueError(
            f"Ranked split contains {len(ranked_rows)} rows; expected {expected_samples}"
        )
    batches = duration_batches(ranked_rows, model.speed_profile.batch_size)
    timing_path = output / TIMING_ARTIFACT
    timing_path.touch(exist_ok=True)
    records, total_audio_seconds, processing_seconds = _load_journal(timing_path, batches)
    completed_sequences = {record.sequence for record in records}
    completed_batches = sum(
        1 for batch in batches if all(sequence in completed_sequences for sequence, _ in batch)
    )
    if request.resume is None and records:
        raise ValueError("Existing timing progress requires an explicit resume state")
    if request.resume is not None and (
        request.resume.completed_samples != len(records)
        or request.resume.completed_batches != completed_batches
    ):
        raise ValueError("Resume state does not match append-only batch progress")
    predictions_path = output / "predictions.jsonl"
    if records:
        _write_predictions(predictions_path, records)
    else:
        predictions_path.touch(exist_ok=True)

    torch = _seed_runtime(request.seed)
    torch.cuda.empty_cache()
    active_adapter = adapter or create_adapter(model, request.model_cache)
    status = RunStatus.RUNNING
    error_message: str | None = None
    parameter_count = 0
    try:
        LOGGER.info(
            "Starting batched benchmark run",
            extra={
                "run": request.run_id,
                "model": model.model_id,
                "batch_size": model.speed_profile.batch_size,
                "completed_batches": completed_batches,
                "completed_samples": len(records),
            },
        )
        all_audio_paths = [request.dataset_cache / row.audio_path for row in ranked_rows]
        _prime_audio(all_audio_paths)
        active_adapter.load()
        parameter_count = active_adapter.parameter_count
        warmup_paths = _warmup_paths(batches, request.dataset_cache)
        for warmup_index in range(WARMUP_BATCHES):
            warmup_outputs = active_adapter.transcribe_batch(warmup_paths)
            require_batch_cardinality(model.model_id, warmup_paths, warmup_outputs)
            torch.cuda.synchronize()
            LOGGER.info(
                "Completed excluded warmup batch",
                extra={"run": request.run_id, "warmup": warmup_index + 1},
            )
        for batch_index, batch in enumerate(batches[completed_batches:], start=completed_batches):
            audio_paths = [request.dataset_cache / row.audio_path for _, row in batch]
            transcriptions, elapsed = _timed_transcribe_batch(active_adapter, audio_paths, torch)
            batch_records = [
                _record(
                    sequence,
                    row.sample_id,
                    row.transcription,
                    transcription,
                    suite.normalization_version,
                )
                for (sequence, row), transcription in zip(batch, transcriptions, strict=True)
            ]
            _append_journal_entry(timing_path, batch_index, batch, batch_records, elapsed)
            records.extend(batch_records)
            total_audio_seconds += sum(row.duration_seconds for _, row in batch)
            processing_seconds += elapsed
            _write_predictions(predictions_path, records)
            LOGGER.info(
                "Completed measured batch",
                extra={
                    "run": request.run_id,
                    "model": model.model_id,
                    "batch": batch_index + 1,
                    "completed_samples": len(records),
                    "total_samples": expected_samples,
                    "processing_seconds": elapsed,
                },
            )
        status = RunStatus.SUCCESS
    except Exception as error:
        error_message = f"{type(error).__name__}: {error}"
        is_cuda_oom = isinstance(error, torch.cuda.OutOfMemoryError) or (
            isinstance(error, RuntimeError)
            and "cuda" in str(error).lower()
            and "out of memory" in str(error).lower()
        )
        if is_cuda_oom:
            status = RunStatus.OOM
            LOGGER.exception(
                "Native-precision CUDA OOM; no fallback will be attempted",
                extra={"run": request.run_id, "model": model.model_id},
            )
        else:
            status = RunStatus.FAILED
            LOGGER.exception(
                "Benchmark run failed",
                extra={"run": request.run_id, "model": model.model_id},
            )
    finally:
        try:
            active_adapter.close()
        except Exception as close_error:
            LOGGER.exception(
                "Adapter cleanup failed",
                extra={"run": request.run_id, "model": model.model_id},
            )
            if status == RunStatus.SUCCESS:
                status = RunStatus.FAILED
                error_message = f"{type(close_error).__name__}: {close_error}"

    aggregates = None
    if status == RunStatus.SUCCESS:
        if len(records) != expected_samples:
            status = RunStatus.FAILED
            error_message = f"Incomplete run: {len(records)} of {expected_samples} samples"
        else:
            corpus = aggregate_scores([_score_from_record(record) for record in records])
            aggregates = AggregateMetrics(
                samples=len(records),
                wer=corpus.wer,
                cer=corpus.cer,
                word_errors=corpus.words.errors,
                word_reference_units=corpus.words.reference_units,
                character_errors=corpus.characters.errors,
                character_reference_units=corpus.characters.reference_units,
                word_accuracy_pct=100.0 * max(0.0, 1.0 - corpus.wer),
            )

    measured_sequences = {record.sequence for record in records}
    measured_batches = sum(
        1 for batch in batches if all(sequence in measured_sequences for sequence, _ in batch)
    )
    if status == RunStatus.SUCCESS and request.resume is None:
        speed = SpeedStatistics(
            valid=True,
            batch_size=model.speed_profile.batch_size,
            warmup_batches=WARMUP_BATCHES,
            measured_batches=measured_batches,
            total_audio_seconds=total_audio_seconds,
            processing_seconds=processing_seconds,
            audio_throughput_x=total_audio_seconds / processing_seconds,
            rtf=processing_seconds / total_audio_seconds,
            timing_artifact=TIMING_ARTIFACT,
        )
    else:
        speed_reason = (
            "Run resumed after an interruption; speed requires a fresh uninterrupted run"
            if status == RunStatus.SUCCESS and request.resume is not None
            else error_message or f"Run status is {status.value}"
        )
        speed = _invalid_speed(
            model,
            measured_batches,
            total_audio_seconds,
            processing_seconds,
            speed_reason,
        )

    bundle = RunBundle(
        schema_version=2,
        run_id=request.run_id,
        suite_id=suite.suite_id,
        suite_digest=request.suite_digest,
        model_id=model.model_id,
        model_digest=request.model_digest,
        status=status,
        environment=_environment(request.seed),
        speed=speed,
        model_facts=ModelFacts(
            checkpoint_bytes=_checkpoint_bytes(model, request.model_cache),
            parameter_count=parameter_count,
            native_dtype=model.native_dtype,
        ),
        predictions_path=predictions_path.name,
        aggregates=aggregates,
        logs=LogReferences(runner=log_path.name, container="container.jsonl"),
        error=error_message,
    )
    _write_json(output / "run.json", bundle.model_dump(mode="json"))
    LOGGER.info(
        "Benchmark run finalized",
        extra={
            "run": request.run_id,
            "model": model.model_id,
            "status": status.value,
            "speed_valid": speed.valid,
        },
    )
    return bundle
