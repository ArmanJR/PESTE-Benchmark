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


class ASRAdapter(ABC):
    """Minimal interface for batch-size-one ASR inference."""

    def __init__(self, spec: ModelSpec, cache_directory: Path) -> None:
        self.spec = spec
        self.cache_directory = cache_directory

    @abstractmethod
    def load(self) -> None:
        """Load the pinned checkpoint in its native benchmark precision."""

    @abstractmethod
    def transcribe(self, audio_path: Path) -> Transcription:
        """Transcribe one canonical 16-kHz mono WAV."""

    @abstractmethod
    def close(self) -> None:
        """Release model resources."""

    @property
    @abstractmethod
    def parameter_count(self) -> int:
        """Return the measured number of checkpoint parameters."""
