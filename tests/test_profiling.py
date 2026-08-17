"""Deterministic speed-profile calibration tests."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from conftest import make_model

import peste.profiling as profiling
from peste.adapters.base import ASRAdapter, Transcription
from peste.schemas import ManifestRow


class FakeCuda:
    class OutOfMemoryError(RuntimeError):
        pass

    def __init__(self, adapter: "ProfileAdapter", bytes_per_item: int = 20) -> None:
        self.adapter = adapter
        self.bytes_per_item = bytes_per_item

    def get_device_properties(self, index: int) -> Any:
        return SimpleNamespace(total_memory=100)

    def empty_cache(self) -> None:
        pass

    def reset_peak_memory_stats(self) -> None:
        pass

    def max_memory_reserved(self) -> int:
        return self.adapter.last_batch_size * self.bytes_per_item

    def synchronize(self) -> None:
        pass


class ProfileAdapter(ASRAdapter):
    def __init__(
        self,
        *args: Any,
        oom_at: int | None = None,
        indexing_limit_at: int | None = None,
        bad_cardinality_at: int | None = None,
        diverge_at: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.oom_at = oom_at
        self.indexing_limit_at = indexing_limit_at
        self.bad_cardinality_at = bad_cardinality_at
        self.diverge_at = diverge_at
        self.last_batch_size = 0
        self.batches_seen: list[tuple[str, ...]] = []

    def load(self) -> None:
        pass

    def transcribe_batch(self, audio_paths: list[Path]) -> list[Transcription]:
        self.last_batch_size = len(audio_paths)
        self.batches_seen.append(tuple(path.stem for path in audio_paths))
        if self.oom_at == len(audio_paths):
            raise FakeCuda.OutOfMemoryError("test OOM")
        if self.indexing_limit_at == len(audio_paths):
            raise RuntimeError("Expected canUse32BitIndexMath(input) to be true")
        outputs = [Transcription(path.stem) for path in audio_paths]
        if self.bad_cardinality_at == len(audio_paths):
            return outputs[:-1]
        if self.diverge_at == len(audio_paths):
            return [Transcription(f"different-{index}") for index, _ in enumerate(audio_paths)]
        return outputs

    def close(self) -> None:
        pass

    @property
    def parameter_count(self) -> int:
        return 1


def _rows_and_audio(tmp_path: Path) -> list[ManifestRow]:
    rows = []
    for index in range(4):
        path = tmp_path / f"audio-{index}.wav"
        path.touch()
        rows.append(
            ManifestRow(
                schema_version=2,
                sample_id=f"test-{index:06d}",
                split="test",
                upstream_row_index=index,
                upstream_row_id=index,
                transcription="متن",
                duration_seconds=float(index + 1),
                audio_sha256="a" * 64,
                audio_path=path.name,
                source_repository="google/fleurs",
                source_revision="b" * 40,
                source_license="CC-BY-4.0",
            )
        )
    return rows


def _configure_small_profile(monkeypatch: Any) -> None:
    monkeypatch.setattr(profiling, "CALIBRATION_SAMPLES", 4)
    monkeypatch.setattr(profiling, "CONFORMANCE_SAMPLES", 2)
    monkeypatch.setattr(profiling, "CANDIDATE_BATCH_SIZES", (1, 2, 4))
    monkeypatch.setattr(profiling, "WARMUP_PASSES", 0)
    monkeypatch.setattr(profiling, "MEASURED_PASSES", 1)


def _clock() -> Any:
    value = 0.0

    def advance() -> float:
        nonlocal value
        value += 1.0
        return value

    return advance


def test_profiler_selects_smallest_candidate_at_95_percent_knee(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _configure_small_profile(monkeypatch)
    adapter = ProfileAdapter(make_model(), tmp_path)
    result = profiling.calibrate_batch_size(
        adapter,
        _rows_and_audio(tmp_path),
        tmp_path,
        "fa-v1",
        SimpleNamespace(cuda=FakeCuda(adapter)),
        clock=_clock(),
    )
    assert result.selected_batch_size == 4
    assert [candidate.safe for candidate in result.candidates] == [True, True, True]
    assert ("audio-0", "audio-3", "audio-0", "audio-3") in adapter.batches_seen


def test_profiler_rejects_oom_and_uses_next_best_candidate(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _configure_small_profile(monkeypatch)
    adapter = ProfileAdapter(make_model(), tmp_path, oom_at=4)
    result = profiling.calibrate_batch_size(
        adapter,
        _rows_and_audio(tmp_path),
        tmp_path,
        "fa-v1",
        SimpleNamespace(cuda=FakeCuda(adapter)),
        clock=_clock(),
    )
    assert result.selected_batch_size == 2
    assert "OOM" in (result.candidates[-1].rejection_reason or "")


def test_nemo_ctc_profiler_caps_candidates_and_rejects_indexing_limit(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _configure_small_profile(monkeypatch)
    monkeypatch.setattr(profiling, "MAX_BATCH_SIZE_BY_ADAPTER", {"nemo-ctc": 2})
    adapter = ProfileAdapter(make_model("nemo-ctc", dtype="float32"), tmp_path, indexing_limit_at=2)
    result = profiling.calibrate_batch_size(
        adapter,
        _rows_and_audio(tmp_path),
        tmp_path,
        "fa-v1",
        SimpleNamespace(cuda=FakeCuda(adapter)),
        clock=_clock(),
    )

    assert [candidate.batch_size for candidate in result.candidates] == [1, 2]
    assert result.selected_batch_size == 1
    assert "canUse32BitIndexMath" in (result.candidates[-1].rejection_reason or "")


def test_profiler_rejects_candidate_above_vram_headroom(monkeypatch: Any, tmp_path: Path) -> None:
    _configure_small_profile(monkeypatch)
    adapter = ProfileAdapter(make_model(), tmp_path)
    result = profiling.calibrate_batch_size(
        adapter,
        _rows_and_audio(tmp_path),
        tmp_path,
        "fa-v1",
        SimpleNamespace(cuda=FakeCuda(adapter, bytes_per_item=30)),
        clock=_clock(),
    )
    assert result.selected_batch_size == 2
    assert result.candidates[-1].peak_vram_fraction == pytest.approx(1.2)
    assert "85%" in (result.candidates[-1].rejection_reason or "")


def test_profiler_rejects_cardinality_and_singleton_divergence(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _configure_small_profile(monkeypatch)
    cardinality_adapter = ProfileAdapter(make_model(), tmp_path, bad_cardinality_at=4)
    cardinality = profiling.calibrate_batch_size(
        cardinality_adapter,
        _rows_and_audio(tmp_path),
        tmp_path,
        "fa-v1",
        SimpleNamespace(cuda=FakeCuda(cardinality_adapter)),
        clock=_clock(),
    )
    assert "cardinality" in (cardinality.candidates[-1].rejection_reason or "")

    divergence_adapter = ProfileAdapter(make_model(), tmp_path, diverge_at=2)
    divergence = profiling.calibrate_batch_size(
        divergence_adapter,
        _rows_and_audio(tmp_path),
        tmp_path,
        "fa-v1",
        SimpleNamespace(cuda=FakeCuda(divergence_adapter)),
        clock=_clock(),
    )
    assert any(
        "diverges" in (candidate.rejection_reason or "") for candidate in divergence.candidates
    )
    assert [candidate.safe for candidate in divergence.candidates] == [True, False, False]
    assert "smaller batch-size candidate 2" in (divergence.candidates[-1].rejection_reason or "")
