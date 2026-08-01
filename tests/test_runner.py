"""Full runner path with generated WAVs and a fake adapter."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import soundfile as sf
from conftest import make_model

import peste.runner as runner
from peste.adapters.base import ASRAdapter, Transcription
from peste.schemas import ResumeState, RunRequest, RunStatus, SuiteSpec
from peste.specs import spec_digest


class FakeCuda:
    class OutOfMemoryError(RuntimeError):
        pass

    def __init__(self) -> None:
        self.reserved = 2 * 1024**3
        self.allocated = 1024**3

    def empty_cache(self) -> None:
        pass

    def reset_peak_memory_stats(self) -> None:
        pass

    def max_memory_reserved(self) -> int:
        return self.reserved

    def max_memory_allocated(self) -> int:
        return self.allocated


class FakeAdapter(ASRAdapter):
    def __init__(
        self,
        *args: Any,
        fail_after: int | None = None,
        oom: bool = False,
        driver_oom: bool = False,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.calls = 0
        self.fail_after = fail_after
        self.oom = oom
        self.driver_oom = driver_oom
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def transcribe(self, audio_path: Path) -> Transcription:
        if self.oom:
            raise FakeCuda.OutOfMemoryError("simulated allocation failure")
        if self.driver_oom:
            raise RuntimeError("CUDA driver error: out of memory")
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("simulated interruption")
        self.calls += 1
        return Transcription(text=f"متن {self.calls - 1}")

    def close(self) -> None:
        self.loaded = False

    @property
    def parameter_count(self) -> int:
        return 42


class NumberWordAdapter(FakeAdapter):
    def transcribe(self, audio_path: Path) -> Transcription:
        transcription = Transcription(text=f"متن {('صفر', 'یک')[self.calls]}")
        self.calls += 1
        return transcription


def _request(tmp_path: Path, suite: SuiteSpec, resume: ResumeState | None = None) -> RunRequest:
    model = make_model()
    return RunRequest(
        schema_version=1,
        run_id="test-run",
        suite_id=suite.suite_id,
        suite_digest=spec_digest(suite),
        model_id=model.model_id,
        model_digest=spec_digest(model),
        seed=7,
        dataset_cache=tmp_path / "cache",
        model_cache=tmp_path / "models",
        output_directory=tmp_path / "output",
        resume=resume,
    )


def _create_audio(tmp_path: Path, rows: list[Any]) -> None:
    for row in rows:
        path = tmp_path / "cache" / row.audio_path
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, np.zeros(160, dtype=np.float32), 16_000, subtype="PCM_16")


def _patch_runtime(monkeypatch: Any, fake_cuda: FakeCuda) -> None:
    from peste.schemas import EnvironmentFingerprint

    monkeypatch.setattr(runner, "_seed_runtime", lambda seed: SimpleNamespace(cuda=fake_cuda))
    monkeypatch.setattr(
        runner,
        "_environment",
        lambda seed: EnvironmentFingerprint(
            peste_revision="test",
            image_reference="test",
            image_digest="test",
            dependency_versions={},
            python_version="3.12",
            pytorch_version="test",
            cuda_version="test",
            hardware_profile={},
            seed=seed,
        ),
    )


def test_full_fake_adapter_orchestration(
    monkeypatch: Any,
    tmp_path: Path,
    tiny_suite: tuple[SuiteSpec, Path, list[Any]],
) -> None:
    suite, suite_directory, rows = tiny_suite
    _create_audio(tmp_path, rows)
    fake_cuda = FakeCuda()
    _patch_runtime(monkeypatch, fake_cuda)
    request = _request(tmp_path, suite)
    model = make_model()
    adapter = FakeAdapter(model, request.model_cache)
    bundle = runner.run_benchmark(request, suite, model, suite_directory, adapter)
    assert bundle.status == RunStatus.SUCCESS
    assert bundle.aggregates is not None
    assert bundle.aggregates.samples == 2
    assert bundle.aggregates.wer == 0
    assert bundle.memory.peak_cuda_reserved_bytes == 2 * 1024**3
    predictions = (request.output_directory / "predictions.jsonl").read_text(encoding="utf-8")
    assert len(predictions.splitlines()) == 2
    assert json.loads(predictions.splitlines()[0])["sample_id"] == "test-000000"


def test_runner_scores_with_the_suite_normalization_version(
    monkeypatch: Any,
    tmp_path: Path,
    tiny_suite: tuple[SuiteSpec, Path, list[Any]],
) -> None:
    suite, suite_directory, rows = tiny_suite
    suite = suite.model_copy(update={"normalization_version": "fa-v2"})
    _create_audio(tmp_path, rows)
    _patch_runtime(monkeypatch, FakeCuda())
    request = _request(tmp_path, suite)
    model = make_model()

    bundle = runner.run_benchmark(
        request,
        suite,
        model,
        suite_directory,
        NumberWordAdapter(model, request.model_cache),
    )

    assert bundle.status == RunStatus.SUCCESS
    assert bundle.aggregates is not None
    assert bundle.aggregates.wer == 0
    records = [
        json.loads(line)
        for line in (request.output_directory / "predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert records[0]["normalized_reference"] == "متن صفر"
    assert records[0]["normalized_prediction"] == "متن صفر"


def test_resume_validates_request_and_preserves_peak_memory(
    monkeypatch: Any,
    tmp_path: Path,
    tiny_suite: tuple[SuiteSpec, Path, list[Any]],
) -> None:
    suite, suite_directory, rows = tiny_suite
    _create_audio(tmp_path, rows)
    fake_cuda = FakeCuda()
    _patch_runtime(monkeypatch, fake_cuda)
    model = make_model()
    initial = _request(tmp_path, suite)
    failed = runner.run_benchmark(
        initial,
        suite,
        model,
        suite_directory,
        FakeAdapter(model, initial.model_cache, fail_after=1),
    )
    assert failed.status == RunStatus.FAILED
    assert len((initial.output_directory / "predictions.jsonl").read_text().splitlines()) == 1

    fake_cuda.reserved = 1024**3
    resume_state = ResumeState(
        completed_samples=1,
        peak_cuda_reserved_bytes=failed.memory.peak_cuda_reserved_bytes,
        peak_cuda_allocated_bytes=failed.memory.peak_cuda_allocated_bytes,
        peak_process_rss_bytes=failed.memory.peak_process_rss_bytes,
    )
    resumed = runner.run_benchmark(
        _request(tmp_path, suite, resume_state),
        suite,
        model,
        suite_directory,
        FakeAdapter(model, initial.model_cache),
    )
    assert resumed.status == RunStatus.SUCCESS
    assert resumed.memory.peak_cuda_reserved_bytes == 2 * 1024**3
    assert len((initial.output_directory / "predictions.jsonl").read_text().splitlines()) == 2


def test_native_precision_oom_is_unranked(
    monkeypatch: Any,
    tmp_path: Path,
    tiny_suite: tuple[SuiteSpec, Path, list[Any]],
) -> None:
    suite, suite_directory, rows = tiny_suite
    _create_audio(tmp_path, rows)
    fake_cuda = FakeCuda()
    _patch_runtime(monkeypatch, fake_cuda)
    request = _request(tmp_path, suite)
    model = make_model()
    bundle = runner.run_benchmark(
        request,
        suite,
        model,
        suite_directory,
        FakeAdapter(model, request.model_cache, oom=True),
    )
    assert bundle.status == RunStatus.OOM
    assert bundle.aggregates is None
    assert "simulated allocation failure" in (bundle.error or "")


def test_cuda_driver_runtime_oom_is_unranked(
    monkeypatch: Any,
    tmp_path: Path,
    tiny_suite: tuple[SuiteSpec, Path, list[Any]],
) -> None:
    suite, suite_directory, rows = tiny_suite
    _create_audio(tmp_path, rows)
    fake_cuda = FakeCuda()
    _patch_runtime(monkeypatch, fake_cuda)
    request = _request(tmp_path, suite)
    model = make_model()

    bundle = runner.run_benchmark(
        request,
        suite,
        model,
        suite_directory,
        FakeAdapter(model, request.model_cache, driver_oom=True),
    )

    assert bundle.status == RunStatus.OOM
    assert bundle.aggregates is None
    assert "CUDA driver error: out of memory" in (bundle.error or "")
