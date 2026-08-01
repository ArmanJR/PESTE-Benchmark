"""Modern Transformers Whisper and Qwen3-ASR adapters."""

import logging
from pathlib import Path
from typing import Any

import soundfile as sf

from peste.adapters.base import ASRAdapter, Transcription
from peste.prefetch import pinned_snapshot_directory

LOGGER = logging.getLogger(__name__)


def _torch_dtype(name: str) -> Any:
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _configure_distributed_disabled_loading() -> None:
    """Complete the narrow runtime shim used by distributed-disabled Jetson wheels."""
    import torch

    distributed = getattr(torch, "distributed", None)
    if distributed is None or distributed.is_available():
        return

    from transformers import core_model_loading
    from transformers.distributed.sharding_utils import DTensor

    if not hasattr(core_model_loading, "DTensor"):
        core_model_loading.DTensor = DTensor
        LOGGER.debug("Configured Transformers for non-distributed checkpoint loading")


def _upgrade_legacy_whisper_generation_config(
    processor: Any, model: Any, language: str, model_id: str
) -> None:
    """Supply modern Whisper generation metadata absent from older checkpoints."""
    generation_config = model.generation_config
    if hasattr(generation_config, "lang_to_id") and hasattr(generation_config, "task_to_id"):
        return

    tokenizer = processor.tokenizer
    language_token = f"<|{language}|>"
    language_token_id = tokenizer.convert_tokens_to_ids(language_token)
    task_token_id = tokenizer.convert_tokens_to_ids("<|transcribe|>")
    no_timestamps_token_id = tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    invalid_id = tokenizer.unk_token_id
    if invalid_id in {language_token_id, task_token_id, no_timestamps_token_id}:
        raise ValueError(f"Legacy Whisper tokenizer for {model_id} lacks required decoder tokens")

    generation_config.is_multilingual = True
    generation_config.lang_to_id = {language_token: language_token_id}
    generation_config.task_to_id = {"transcribe": task_token_id}
    generation_config.no_timestamps_token_id = no_timestamps_token_id
    LOGGER.warning(
        "Supplied missing in-memory Whisper generation metadata",
        extra={"model": model_id, "language": language},
    )


class TransformersWhisperAdapter(ASRAdapter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.processor: Any = None
        self.model: Any = None

    def load(self) -> None:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        _configure_distributed_disabled_loading()
        LOGGER.info("Loading Whisper checkpoint", extra={"model": self.spec.model_id})
        snapshot = pinned_snapshot_directory(self.spec, self.cache_directory)
        common = {"local_files_only": True}
        self.processor = AutoProcessor.from_pretrained(str(snapshot), **common)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            str(snapshot),
            dtype=_torch_dtype(self.spec.native_dtype),
            low_cpu_mem_usage=True,
            **common,
        ).to("cuda")
        self.model.eval()
        if self.spec.language is None:
            raise ValueError(f"Whisper model {self.spec.model_id} has no configured language")
        _upgrade_legacy_whisper_generation_config(
            self.processor, self.model, self.spec.language, self.spec.model_id
        )

    def transcribe(self, audio_path: Path) -> Transcription:
        import torch

        audio, sample_rate = sf.read(audio_path, dtype="float32")
        inputs = self.processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        input_features = inputs.input_features.to(
            "cuda", dtype=_torch_dtype(self.spec.native_dtype)
        )
        with torch.inference_mode():
            generated = self.model.generate(
                input_features,
                language=self.spec.language,
                task="transcribe",
                max_new_tokens=int(self.spec.generation["max_new_tokens"]),
                return_timestamps=False,
            )
        text = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        return Transcription(text=text)

    def close(self) -> None:
        self.model = None
        self.processor = None

    @property
    def parameter_count(self) -> int:
        if self.model is None:
            return 0
        return sum(parameter.numel() for parameter in self.model.parameters())


class TransformersQwenAdapter(ASRAdapter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.processor: Any = None
        self.model: Any = None

    def load(self) -> None:
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        _configure_distributed_disabled_loading()
        LOGGER.info("Loading Qwen3-ASR checkpoint", extra={"model": self.spec.model_id})
        snapshot = pinned_snapshot_directory(self.spec, self.cache_directory)
        common = {"local_files_only": True}
        self.processor = AutoProcessor.from_pretrained(str(snapshot), **common)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            str(snapshot),
            dtype=_torch_dtype(self.spec.native_dtype),
            low_cpu_mem_usage=True,
            **common,
        ).to("cuda")
        self.model.eval()

    def transcribe(self, audio_path: Path) -> Transcription:
        import torch

        request = self.processor.apply_transcription_request(
            audio=str(audio_path), language=self.spec.language
        ).to(self.model.device, self.model.dtype)
        with torch.inference_mode():
            generated = self.model.generate(
                **request,
                max_new_tokens=int(self.spec.generation["max_new_tokens"]),
            )
        input_length = request["input_ids"].shape[-1]
        decoded = self.processor.decode(
            generated[:, input_length:], return_format="transcription_only"
        )[0]
        return Transcription(text=decoded)

    def close(self) -> None:
        self.model = None
        self.processor = None

    @property
    def parameter_count(self) -> int:
        if self.model is None:
            return 0
        return sum(parameter.numel() for parameter in self.model.parameters())
