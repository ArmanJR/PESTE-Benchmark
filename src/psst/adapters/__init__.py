"""Declarative ASR adapter registry."""

from pathlib import Path

from psst.adapters.base import ASRAdapter
from psst.adapters.nemo import NemoRnntAdapter
from psst.adapters.transformers import TransformersQwenAdapter, TransformersWhisperAdapter
from psst.adapters.vibevoice import VibeVoiceAdapter
from psst.schemas import ModelSpec

ADAPTERS: dict[str, type[ASRAdapter]] = {
    "transformers-whisper": TransformersWhisperAdapter,
    "transformers-qwen": TransformersQwenAdapter,
    "vibevoice": VibeVoiceAdapter,
    "nemo-rnnt": NemoRnntAdapter,
}


def create_adapter(spec: ModelSpec, cache_directory: Path) -> ASRAdapter:
    return ADAPTERS[spec.adapter](spec, cache_directory)
