"""Deterministic model batch-size calibration for the RTX speed profile."""

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from peste.adapters.base import (
    AdapterOutputError,
    ASRAdapter,
    Transcription,
    require_batch_cardinality,
)
from peste.normalization import normalize
from peste.schemas import ManifestRow

LOGGER = logging.getLogger(__name__)
CANDIDATE_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128)
MAX_BATCH_SIZE_BY_ADAPTER = {"nemo-ctc": 32}
CALIBRATION_SAMPLES = 128
CONFORMANCE_SAMPLES = 16
WARMUP_PASSES = 2
MEASURED_PASSES = 3
MAX_VRAM_FRACTION = 0.85
KNEE_FRACTION = 0.95


@dataclass(frozen=True, slots=True)
class CandidateResult:
    batch_size: int
    safe: bool
    throughput_x: float | None
    peak_vram_fraction: float | None
    rejection_reason: str | None = None


@dataclass(frozen=True, slots=True)
class SpeedProfileResult:
    selected_batch_size: int
    best_throughput_x: float
    candidates: tuple[CandidateResult, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_batch_size": self.selected_batch_size,
            "best_throughput_x": self.best_throughput_x,
            "candidates": [asdict(candidate) for candidate in self.candidates],
        }


def quantile_rows(rows: Sequence[ManifestRow], count: int) -> list[ManifestRow]:
    """Select deterministic duration quantiles, always including both extremes."""
    if count <= 0:
        raise ValueError("Quantile sample count must be positive")
    ordered = sorted(rows, key=lambda row: (row.duration_seconds, row.upstream_row_index))
    if len(ordered) < count:
        raise ValueError(f"Need at least {count} rows for calibration; received {len(ordered)}")
    if count == 1:
        return [ordered[0]]
    indices = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    if len(set(indices)) != count:
        raise AssertionError("Quantile selection produced duplicate rows")
    return [ordered[index] for index in indices]


def _chunks[T](values: Sequence[T], size: int) -> list[list[T]]:
    return [list(values[index : index + size]) for index in range(0, len(values), size)]


def _is_oom(error: Exception, torch: Any) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError)
        and "cuda" in str(error).lower()
        and "out of memory" in str(error).lower()
    )


def _is_capacity_error(error: Exception, torch: Any) -> bool:
    if _is_oom(error, torch):
        return True
    return isinstance(error, RuntimeError) and "canuse32bitindexmath" in str(error).casefold()


def candidate_batch_sizes(adapter: ASRAdapter) -> tuple[int, ...]:
    maximum = MAX_BATCH_SIZE_BY_ADAPTER.get(adapter.spec.adapter)
    if maximum is None:
        return CANDIDATE_BATCH_SIZES
    return tuple(size for size in CANDIDATE_BATCH_SIZES if size <= maximum)


def _transcribe_checked(adapter: ASRAdapter, paths: list[Path]) -> list[Transcription]:
    outputs = adapter.transcribe_batch(paths)
    return require_batch_cardinality(adapter.spec.model_id, paths, outputs)


def _normalized_outputs(
    adapter: ASRAdapter, paths: Sequence[Path], batch_size: int, normalization_version: str
) -> list[str]:
    outputs: list[str] = []
    for batch in _chunks(paths, batch_size):
        outputs.extend(
            normalize(item.text, normalization_version)
            for item in _transcribe_checked(adapter, batch)
        )
    return outputs


def calibrate_batch_size(
    adapter: ASRAdapter,
    rows: Sequence[ManifestRow],
    dataset_cache: Path,
    normalization_version: str,
    torch: Any,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> SpeedProfileResult:
    """Measure safe candidates and select the smallest point on the 95% throughput knee."""
    calibration = quantile_rows(rows, CALIBRATION_SAMPLES)
    conformance = quantile_rows(rows, CONFORMANCE_SAMPLES)
    calibration_paths = [dataset_cache / row.audio_path for row in calibration]
    conformance_paths = [dataset_cache / row.audio_path for row in conformance]
    missing = [str(path) for path in calibration_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Calibration audio is missing: {missing[0]}")
    singleton = _normalized_outputs(adapter, conformance_paths, 1, normalization_version)
    total_audio_seconds = sum(row.duration_seconds for row in calibration)
    total_vram = int(torch.cuda.get_device_properties(0).total_memory)
    if total_vram <= 0:
        raise RuntimeError("CUDA device reported no VRAM")

    results: list[CandidateResult] = []
    ordered_rows = sorted(rows, key=lambda row: (row.duration_seconds, row.upstream_row_index))
    for candidate in candidate_batch_sizes(adapter):
        LOGGER.info(
            "Profiling batch-size candidate",
            extra={"model": adapter.spec.model_id, "batch_size": candidate},
        )
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            stress_paths = (
                [dataset_cache / row.audio_path for row in ordered_rows[-candidate:]]
                if adapter.spec.adapter == "nemo-ctc"
                else calibration_paths[-candidate:]
            )
            _transcribe_checked(adapter, stress_paths)
            torch.cuda.synchronize()
            peak_fraction = float(torch.cuda.max_memory_reserved()) / total_vram
            if peak_fraction > MAX_VRAM_FRACTION:
                results.append(
                    CandidateResult(
                        batch_size=candidate,
                        safe=False,
                        throughput_x=None,
                        peak_vram_fraction=peak_fraction,
                        rejection_reason="peak VRAM exceeds 85% on longest-duration stress batch",
                    )
                )
                continue
            candidate_conformance = _normalized_outputs(
                adapter, conformance_paths, candidate, normalization_version
            )
            if candidate_conformance != singleton:
                results.append(
                    CandidateResult(
                        batch_size=candidate,
                        safe=False,
                        throughput_x=None,
                        peak_vram_fraction=peak_fraction,
                        rejection_reason="normalized output diverges from singleton conformance",
                    )
                )
                continue
            representative = calibration_paths[
                max(0, len(calibration_paths) // 2 - candidate // 2) :
            ][:candidate]
            for _ in range(WARMUP_PASSES):
                _transcribe_checked(adapter, representative)
                torch.cuda.synchronize()
            processing_seconds = 0.0
            for _ in range(MEASURED_PASSES):
                for batch in _chunks(calibration_paths, candidate):
                    torch.cuda.synchronize()
                    started = clock()
                    _transcribe_checked(adapter, batch)
                    torch.cuda.synchronize()
                    processing_seconds += clock() - started
            if processing_seconds <= 0:
                raise RuntimeError("Candidate processing time must be positive")
            throughput = total_audio_seconds * MEASURED_PASSES / processing_seconds
            results.append(
                CandidateResult(
                    batch_size=candidate,
                    safe=True,
                    throughput_x=throughput,
                    peak_vram_fraction=peak_fraction,
                )
            )
        except AdapterOutputError as error:
            results.append(
                CandidateResult(
                    batch_size=candidate,
                    safe=False,
                    throughput_x=None,
                    peak_vram_fraction=None,
                    rejection_reason=f"output cardinality violation: {error}",
                )
            )
        except Exception as error:
            if not _is_capacity_error(error, torch):
                raise
            LOGGER.warning(
                "Rejected capacity-limited batch-size candidate",
                extra={
                    "model": adapter.spec.model_id,
                    "batch_size": candidate,
                    "error": str(error),
                },
            )
            results.append(
                CandidateResult(
                    batch_size=candidate,
                    safe=False,
                    throughput_x=None,
                    peak_vram_fraction=None,
                    rejection_reason=f"Runtime capacity limit: {error}",
                )
            )
            torch.cuda.empty_cache()

    safe = [result for result in results if result.safe and result.throughput_x is not None]
    if not safe:
        raise RuntimeError("No safe batch-size candidate completed calibration")
    best = max(result.throughput_x for result in safe if result.throughput_x is not None)
    selected = min(
        result.batch_size
        for result in safe
        if result.throughput_x is not None and result.throughput_x >= KNEE_FRACTION * best
    )
    LOGGER.info(
        "Selected deterministic speed profile",
        extra={
            "model": adapter.spec.model_id,
            "batch_size": selected,
            "best_throughput_x": best,
        },
    )
    return SpeedProfileResult(selected, best, tuple(results))
