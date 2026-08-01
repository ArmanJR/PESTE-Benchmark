"""Real-adapter smoke orchestration with fake inference."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import soundfile as sf
from conftest import make_model

import psst.smoke as smoke
from psst.adapters.base import ASRAdapter, Transcription
from psst.schemas import SuiteSpec


class SmokeAdapter(ASRAdapter):
    def __init__(self, *args: Any, outputs: list[str], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.outputs = iter(outputs)

    def load(self) -> None:
        pass

    def transcribe(self, audio_path: Path) -> Transcription:
        return Transcription(next(self.outputs))

    def close(self) -> None:
        pass

    @property
    def parameter_count(self) -> int:
        return 10


def _run_smoke(
    monkeypatch: Any,
    tmp_path: Path,
    tiny_suite: tuple[SuiteSpec, Path, list[Any]],
    outputs: list[str],
) -> None:
    suite, suite_directory, rows = tiny_suite
    test_row = next(row for row in rows if row.split == "test")
    audio_path = tmp_path / "cache" / test_row.audio_path
    audio_path.parent.mkdir(parents=True)
    sf.write(audio_path, [0.0] * 160, 16_000)
    cuda = SimpleNamespace(
        empty_cache=lambda: None,
        reset_peak_memory_stats=lambda: None,
        max_memory_reserved=lambda: 100,
        max_memory_allocated=lambda: 50,
    )
    monkeypatch.setattr(smoke, "_seed_runtime", lambda seed: SimpleNamespace(cuda=cuda))
    model = make_model()
    monkeypatch.setattr(
        smoke,
        "create_adapter",
        lambda spec, cache: SmokeAdapter(model, cache, outputs=outputs),
    )
    smoke.smoke_adapter(
        suite,
        model,
        suite_directory,
        tmp_path / "cache",
        tmp_path / "models",
        7,
    )


def test_smoke_accepts_identical_normalized_output(
    monkeypatch: Any,
    tmp_path: Path,
    tiny_suite: tuple[SuiteSpec, Path, list[Any]],
) -> None:
    _run_smoke(monkeypatch, tmp_path, tiny_suite, ["مي روم", "می\u200cروم"])


def test_smoke_rejects_nondeterministic_output(
    monkeypatch: Any,
    tmp_path: Path,
    tiny_suite: tuple[SuiteSpec, Path, list[Any]],
) -> None:
    with pytest.raises(RuntimeError, match="different normalized text"):
        _run_smoke(monkeypatch, tmp_path, tiny_suite, ["الف", "ب"])
