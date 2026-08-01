"""Declarative ASR adapter registry."""

from pathlib import Path

from peste.adapters.base import ASRAdapter
from peste.adapters.nemo import NemoRnntAdapter
from peste.adapters.transformers import TransformersQwenAdapter, TransformersWhisperAdapter
from peste.adapters.vibevoice import VibeVoiceAdapter
from peste.schemas import ModelSpec

ADAPTERS: dict[str, type[ASRAdapter]] = {
    "transformers-whisper": TransformersWhisperAdapter,
    "transformers-qwen": TransformersQwenAdapter,
    "vibevoice": VibeVoiceAdapter,
    "nemo-rnnt": NemoRnntAdapter,
}


def create_adapter(spec: ModelSpec, cache_directory: Path) -> ASRAdapter:
    return ADAPTERS[spec.adapter](spec, cache_directory)
