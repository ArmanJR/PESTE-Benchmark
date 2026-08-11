"""Adapter protocol shared by isolated inference runtimes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from peste.schemas import ModelSpec


@dataclass(frozen=True, slots=True)
class Transcription:
    text: str
    structured_output: dict[str, Any] | list[Any] | None = None


class AdapterOutputError(RuntimeError):
    """Raised when an adapter violates the ordered batch output contract."""


def require_batch_cardinality(
    model_id: str, audio_paths: list[Path], transcriptions: list[Transcription]
) -> list[Transcription]:
    """Require exactly one ordered transcription for every input path."""
    if len(transcriptions) != len(audio_paths):
        raise AdapterOutputError(
            f"Adapter for {model_id} received {len(audio_paths)} audio paths but returned "
            f"{len(transcriptions)} transcriptions"
        )
    return transcriptions


class ASRAdapter(ABC):
    """Minimal interface for real, ordered batched ASR inference."""

    def __init__(self, spec: ModelSpec, cache_directory: Path) -> None:
        self.spec = spec
        self.cache_directory = cache_directory

    @abstractmethod
    def load(self) -> None:
        """Load the pinned checkpoint in its native benchmark precision."""

    @abstractmethod
    def transcribe_batch(self, audio_paths: list[Path]) -> list[Transcription]:
        """Transcribe canonical WAVs with one ordered output per input."""

    @abstractmethod
    def close(self) -> None:
        """Release model resources."""

    @property
    @abstractmethod
    def parameter_count(self) -> int:
        """Return the measured number of checkpoint parameters."""
