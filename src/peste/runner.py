"""Deterministic, resumable, batch-size-one benchmark runner."""

import importlib.metadata
import json
import logging
import os
import platform
import random
import resource
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from peste.adapters import create_adapter
from peste.adapters.base import ASRAdapter
from peste.digests import canonical_json
from peste.logging import configure_logging
from peste.manifest import validate_manifest
from peste.metrics import EditCounts, SampleScore, aggregate_scores, memory_efficiency, score_sample
from peste.schemas import (
    AggregateMetrics,
    EnvironmentFingerprint,
    LogReferences,
    MemoryStatistics,
    ModelSpec,
    PredictionRecord,
    RunBundle,
    RunRequest,
    RunStatus,
    SuiteSpec,
)
from peste.specs import spec_digest

LOGGER = logging.getLogger(__name__)
GIB = 1024**3


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


def _peak_rss_bytes() -> int:
    status_path = Path("/proc/self/status")
    if status_path.exists():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    scale = 1 if sys.platform == "darwin" else 1024
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * scale


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
    return EnvironmentFingerprint(
        peste_revision=_source_revision(),
        image_reference=os.environ.get("PESTE_IMAGE_REFERENCE", "unknown"),
        image_digest=os.environ.get("PESTE_IMAGE_DIGEST", "unknown"),
        dependency_versions=versions,
        python_version=platform.python_version(),
        pytorch_version=torch.__version__,
        cuda_version=str(torch.version.cuda),
        hardware_profile=hardware,
        seed=seed,
    )


def _record(sequence: int, sample_id: str, reference: str, transcription: Any) -> PredictionRecord:
    score = score_sample(reference, transcription.text)
    return PredictionRecord(
        schema_version=1,
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


def _load_resume_records(path: Path, expected_sample_ids: list[str]) -> list[PredictionRecord]:
    records: list[PredictionRecord] = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = PredictionRecord.model_validate_json(line)
            except ValueError as error:
                raise ValueError(
                    f"Invalid resume prediction at line {line_number}: {error}"
                ) from error
            if record.sequence != len(records):
                raise ValueError(f"Non-contiguous resume sequence at line {line_number}")
            if record.sample_id != expected_sample_ids[len(records)]:
                raise ValueError(f"Resume sample mismatch at line {line_number}")
            records.append(record)
    return records


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json(value))


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
    predictions_path = output / "predictions.jsonl"
    records = _load_resume_records(predictions_path, [row.sample_id for row in ranked_rows])
    if request.resume is None and records:
        raise ValueError("Existing predictions require an explicit resume state")
    if request.resume is not None and request.resume.completed_samples != len(records):
        raise ValueError("Resume state does not match append-only prediction count")

    torch = _seed_runtime(request.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    peak_reserved = request.resume.peak_cuda_reserved_bytes if request.resume else 0
    peak_allocated = request.resume.peak_cuda_allocated_bytes if request.resume else 0
    peak_rss = request.resume.peak_process_rss_bytes if request.resume else 0
    active_adapter = adapter or create_adapter(model, request.model_cache)
    status = RunStatus.RUNNING
    error_message: str | None = None
    parameter_count = 0
    try:
        LOGGER.info(
            "Starting benchmark run",
            extra={"run": request.run_id, "model": model.model_id, "completed": len(records)},
        )
        active_adapter.load()
        parameter_count = active_adapter.parameter_count
        with predictions_path.open("a", encoding="utf-8") as prediction_log:
            for sequence, row in enumerate(ranked_rows[len(records) :], start=len(records)):
                audio_path = request.dataset_cache / row.audio_path
                if not audio_path.exists():
                    raise FileNotFoundError(f"Canonical audio is missing: {audio_path}")
                transcription = active_adapter.transcribe(audio_path)
                record = _record(sequence, row.sample_id, row.transcription, transcription)
                prediction_log.write(record.model_dump_json() + "\n")
                prediction_log.flush()
                os.fsync(prediction_log.fileno())
                records.append(record)
                peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved())
                peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated())
                peak_rss = max(peak_rss, _peak_rss_bytes())
                LOGGER.info(
                    "Completed sample",
                    extra={
                        "run": request.run_id,
                        "model": model.model_id,
                        "sample": row.sample_id,
                        "completed": sequence + 1,
                        "total": expected_samples,
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
        peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved())
        peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated())
        peak_rss = max(peak_rss, _peak_rss_bytes())
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
            reserved_gib = peak_reserved / GIB
            if reserved_gib <= 0:
                status = RunStatus.FAILED
                error_message = "Peak CUDA reserved memory was not measured"
            else:
                aggregates = AggregateMetrics(
                    samples=len(records),
                    wer=corpus.wer,
                    cer=corpus.cer,
                    word_errors=corpus.words.errors,
                    word_reference_units=corpus.words.reference_units,
                    character_errors=corpus.characters.errors,
                    character_reference_units=corpus.characters.reference_units,
                    word_accuracy_pct=100.0 * max(0.0, 1.0 - corpus.wer),
                    memory_efficiency=memory_efficiency(corpus.wer, reserved_gib),
                )
    bundle = RunBundle(
        schema_version=1,
        run_id=request.run_id,
        suite_id=suite.suite_id,
        suite_digest=request.suite_digest,
        model_id=model.model_id,
        model_digest=request.model_digest,
        status=status,
        environment=_environment(request.seed),
        memory=MemoryStatistics(
            peak_cuda_reserved_bytes=peak_reserved,
            peak_cuda_allocated_bytes=peak_allocated,
            peak_process_rss_bytes=peak_rss,
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
        extra={"run": request.run_id, "model": model.model_id, "status": status.value},
    )
    return bundle
