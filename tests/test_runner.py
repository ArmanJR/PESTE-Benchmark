"""Full batched runner path with generated WAVs and fake CUDA inference."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from conftest import make_model

import peste.runner as runner
from peste.adapters.base import ASRAdapter, Transcription
from peste.digests import canonical_json, sha256_bytes
from peste.schemas import EnvironmentFingerprint, ResumeState, RunRequest, RunStatus, SuiteSpec
from peste.specs import spec_digest


class FakeCuda:
    class OutOfMemoryError(RuntimeError):
        pass

    def __init__(self) -> None:
        self.synchronizations = 0

    def empty_cache(self) -> None:
        pass

    def synchronize(self) -> None:
        self.synchronizations += 1


class FakeAdapter(ASRAdapter):
    def __init__(
        self,
        *args: Any,
        fail_after_calls: int | None = None,
        oom: bool = False,
        driver_oom: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.calls: list[list[str]] = []
        self.fail_after_calls = fail_after_calls
        self.oom = oom
        self.driver_oom = driver_oom
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def transcribe_batch(self, audio_paths: list[Path]) -> list[Transcription]:
        if self.oom:
            raise FakeCuda.OutOfMemoryError("simulated allocation failure")
        if self.driver_oom:
            raise RuntimeError("CUDA driver error: out of memory")
        if self.fail_after_calls is not None and len(self.calls) >= self.fail_after_calls:
            raise RuntimeError("simulated interruption")
        self.calls.append([path.name for path in audio_paths])
        return [Transcription(text=f"متن {int(path.stem)}") for path in audio_paths]

    def close(self) -> None:
        self.loaded = False

    @property
    def parameter_count(self) -> int:
        return 42


class NumberWordAdapter(FakeAdapter):
    def transcribe_batch(self, audio_paths: list[Path]) -> list[Transcription]:
        self.calls.append([path.name for path in audio_paths])
        return [Transcription(text=f"متن {('صفر', 'یک')[int(path.stem)]}") for path in audio_paths]


def _request(
    tmp_path: Path,
    suite: SuiteSpec,
    model: Any,
    resume: ResumeState | None = None,
) -> RunRequest:
    return RunRequest(
        schema_version=2,
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
            gpu_product_name="NVIDIA RTX 6000 Ada Generation",
            driver_version="580.142",
            ecc_state="Disabled",
            power_limit_watts=300,
            cpu_model="test CPU",
            gpu_uuid="GPU-test",
            seed=seed,
        ),
    )


def test_batched_run_restores_manifest_order_and_computes_exact_speed(
    monkeypatch: Any,
    tmp_path: Path,
    tiny_suite: tuple[SuiteSpec, Path, list[Any]],
) -> None:
    suite, suite_directory, rows = tiny_suite
    test_rows = [row for row in rows if row.split == "test"]
    rows[rows.index(test_rows[0])] = test_rows[0].model_copy(update={"duration_seconds": 2.0})
    encoded = b"".join(canonical_json(row.model_dump(mode="json")) for row in rows)
    (suite_directory / "manifest.jsonl").write_bytes(encoded)
    suite = suite.model_copy(update={"manifest_sha256": sha256_bytes(encoded)})
    _create_audio(tmp_path, rows)
    fake_cuda = FakeCuda()
    _patch_runtime(monkeypatch, fake_cuda)
    model = make_model()
    request = _request(tmp_path, suite, model)
    adapter = FakeAdapter(model, request.model_cache)

    bundle = runner.run_benchmark(request, suite, model, suite_directory, adapter)

    assert bundle.status == RunStatus.SUCCESS
    assert bundle.aggregates is not None and bundle.aggregates.wer == 0
    assert bundle.speed.valid is True
    assert bundle.speed.batch_size == 2
    assert bundle.speed.measured_batches == 1
    assert bundle.speed.audio_throughput_x == pytest.approx(
        bundle.speed.total_audio_seconds / bundle.speed.processing_seconds
    )
    assert bundle.speed.rtf == pytest.approx(1 / bundle.speed.audio_throughput_x)
    assert adapter.calls[-1] == ["000001.wav", "000000.wav"]
    records = [
        json.loads(line)
        for line in (request.output_directory / "predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["sample_id"] for record in records] == ["test-000000", "test-000001"]
    journal = (request.output_directory / "timing.jsonl").read_text().splitlines()
    assert len(journal) == 1
    assert json.loads(journal[0])["sequences"] == [1, 0]


def test_final_partial_batch_is_measured() -> None:
    rows = [
        SimpleNamespace(duration_seconds=float(index), upstream_row_index=index)
        for index in range(1, 6)
    ]
    batches = runner.duration_batches(rows, 2)
    assert [len(batch) for batch in batches] == [2, 2, 1]


def test_runner_scores_with_suite_normalization_version(
    monkeypatch: Any,
    tmp_path: Path,
    tiny_suite: tuple[SuiteSpec, Path, list[Any]],
) -> None:
    suite, suite_directory, rows = tiny_suite
    suite = suite.model_copy(update={"normalization_version": "fa-v2"})
    _create_audio(tmp_path, rows)
    _patch_runtime(monkeypatch, FakeCuda())
    model = make_model()
    request = _request(tmp_path, suite, model)

    bundle = runner.run_benchmark(
        request,
        suite,
        model,
        suite_directory,
        NumberWordAdapter(model, request.model_cache),
    )

    assert bundle.status == RunStatus.SUCCESS
    assert bundle.aggregates is not None and bundle.aggregates.wer == 0


def test_journal_resume_preserves_accuracy_and_invalidates_speed(
    monkeypatch: Any,
    tmp_path: Path,
    tiny_suite: tuple[SuiteSpec, Path, list[Any]],
) -> None:
    suite, suite_directory, rows = tiny_suite
    _create_audio(tmp_path, rows)
    _patch_runtime(monkeypatch, FakeCuda())
    model = make_model().model_copy(
        update={"speed_profile": make_model().speed_profile.model_copy(update={"batch_size": 1})}
    )
    initial = _request(tmp_path, suite, model)
    failed = runner.run_benchmark(
        initial,
        suite,
        model,
        suite_directory,
        FakeAdapter(model, initial.model_cache, fail_after_calls=3),
    )
    assert failed.status == RunStatus.FAILED
    assert len((initial.output_directory / "timing.jsonl").read_text().splitlines()) == 1

    resumed = runner.run_benchmark(
        _request(
            tmp_path,
            suite,
            model,
            ResumeState(completed_batches=1, completed_samples=1),
        ),
        suite,
        model,
        suite_directory,
        FakeAdapter(model, initial.model_cache),
    )

    assert resumed.status == RunStatus.SUCCESS
    assert resumed.aggregates is not None and resumed.aggregates.samples == 2
    assert resumed.speed.valid is False
    assert "resumed" in (resumed.speed.invalidity_reason or "").lower()
    assert len((initial.output_directory / "predictions.jsonl").read_text().splitlines()) == 2


@pytest.mark.parametrize("driver_oom", [False, True])
def test_native_precision_oom_is_unranked(
    monkeypatch: Any,
    tmp_path: Path,
    tiny_suite: tuple[SuiteSpec, Path, list[Any]],
    driver_oom: bool,
) -> None:
    suite, suite_directory, rows = tiny_suite
    _create_audio(tmp_path, rows)
    _patch_runtime(monkeypatch, FakeCuda())
    model = make_model()
    request = _request(tmp_path, suite, model)
    bundle = runner.run_benchmark(
        request,
        suite,
        model,
        suite_directory,
        FakeAdapter(model, request.model_cache, oom=not driver_oom, driver_oom=driver_oom),
    )
    assert bundle.status == RunStatus.OOM
    assert bundle.aggregates is None
    assert bundle.speed.valid is False


def test_timer_boundaries_synchronize_around_adapter_call(tmp_path: Path) -> None:
    events: list[str] = []

    class Cuda:
        def synchronize(self) -> None:
            events.append("synchronize")

    class Adapter(FakeAdapter):
        def transcribe_batch(self, audio_paths: list[Path]) -> list[Transcription]:
            events.append("adapter")
            return [Transcription("ok") for _ in audio_paths]

    times = iter((10.0, 12.5))
    adapter = Adapter(make_model(), tmp_path)
    _, elapsed = runner._timed_transcribe_batch(
        adapter,
        [tmp_path / "audio.wav"],
        SimpleNamespace(cuda=Cuda()),
        clock=lambda: next(times),
    )
    assert events == ["synchronize", "adapter", "synchronize"]
    assert elapsed == 2.5
