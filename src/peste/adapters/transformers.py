"""Modern Transformers Whisper, Qwen3-ASR, and CTC adapters."""

import logging
from pathlib import Path
from typing import Any

import soundfile as sf

from peste.adapters.base import (
    AdapterOutputError,
    ASRAdapter,
    Transcription,
    require_batch_cardinality,
)
from peste.prefetch import pinned_snapshot_directory

LOGGER = logging.getLogger(__name__)


class TransformersCTCOutputError(AdapterOutputError):
    """Raised when a CTC processor violates the batched decode contract."""


def _torch_dtype(name: str) -> Any:
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _upgrade_legacy_whisper_generation_config(
    processor: Any, model: Any, language: str, model_id: str
) -> None:
    """Supply modern Whisper generation metadata absent from older checkpoints."""
    generation_config = model.generation_config
    tokenizer = processor.tokenizer
    language_token = f"<|{language}|>"
    language_token_id = tokenizer.convert_tokens_to_ids(language_token)
    task_token_id = tokenizer.convert_tokens_to_ids("<|transcribe|>")
    no_timestamps_token_id = tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    invalid_id = tokenizer.unk_token_id
    if invalid_id in {language_token_id, task_token_id, no_timestamps_token_id}:
        raise ValueError(f"Legacy Whisper tokenizer for {model_id} lacks required decoder tokens")

    language_ids = getattr(generation_config, "lang_to_id", {})
    task_ids = getattr(generation_config, "task_to_id", {})
    if not isinstance(language_ids, dict):
        language_ids = {}
    if not isinstance(task_ids, dict):
        task_ids = {}
    if (
        language_ids.get(language_token) == language_token_id
        and task_ids.get("transcribe") == task_token_id
        and getattr(generation_config, "no_timestamps_token_id", None) == no_timestamps_token_id
    ):
        return

    generation_config.is_multilingual = True
    generation_config.lang_to_id = {**language_ids, language_token: language_token_id}
    generation_config.task_to_id = {**task_ids, "transcribe": task_token_id}
    generation_config.no_timestamps_token_id = no_timestamps_token_id
    LOGGER.warning(
        "Supplied missing in-memory Whisper generation metadata",
        extra={
            "model": model_id,
            "language": language,
            "required_fields": ["lang_to_id", "task_to_id", "no_timestamps_token_id"],
        },
    )


def _whisper_segment_frames(model: Any) -> int:
    """Return the maximum mel frames accepted by one native Whisper segment."""
    encoder = model.get_encoder()
    input_stride = int(encoder.conv1.stride[0]) * int(encoder.conv2.stride[0])
    segment_frames = input_stride * int(model.config.max_source_positions)
    if segment_frames <= 0:
        raise ValueError("Whisper model reported a non-positive segment-frame limit")
    return segment_frames


def _requires_whisper_long_form(model: Any, input_features: Any) -> bool:
    """Select native sequential decoding only when a batch exceeds one segment."""
    return int(input_features.shape[-1]) > _whisper_segment_frames(model)


class TransformersWhisperAdapter(ASRAdapter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.processor: Any = None
        self.model: Any = None

    def load(self) -> None:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

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
        LOGGER.info(
            "Configured automatic Whisper long-form decoding",
            extra={
                "model": self.spec.model_id,
                "segment_frames": _whisper_segment_frames(self.model),
                "timestamp_policy": self.spec.generation["return_timestamps"],
                "attention_mask": True,
            },
        )

    def transcribe_batch(self, audio_paths: list[Path]) -> list[Transcription]:
        import torch

        if not audio_paths:
            raise ValueError("Whisper batch must contain at least one audio path")
        decoded_audio = [sf.read(path, dtype="float32") for path in audio_paths]
        sample_rates = {sample_rate for _, sample_rate in decoded_audio}
        if len(sample_rates) != 1:
            raise ValueError("Whisper batch audio must use one sample rate")
        sample_rate = sample_rates.pop()
        inputs = self.processor(
            [audio for audio, _ in decoded_audio],
            sampling_rate=sample_rate,
            return_tensors="pt",
            truncation=False,
            padding="longest",
            return_attention_mask=True,
        )
        if "input_features" not in inputs:
            raise AdapterOutputError(
                f"Whisper processor for {self.spec.model_id} returned no input_features"
            )
        if "attention_mask" not in inputs:
            raise AdapterOutputError(
                f"Whisper processor for {self.spec.model_id} returned no attention_mask"
            )
        return_timestamps = _requires_whisper_long_form(self.model, inputs["input_features"])
        model_inputs = {
            name: tensor.to("cuda", dtype=_torch_dtype(self.spec.native_dtype))
            if tensor.is_floating_point()
            else tensor.to("cuda")
            for name, tensor in inputs.items()
        }
        with torch.inference_mode():
            generated = self.model.generate(
                **model_inputs,
                language=self.spec.language,
                task=str(self.spec.generation["task"]),
                max_new_tokens=int(self.spec.generation["max_new_tokens"]),
                return_timestamps=return_timestamps,
            )
        decoded = self.processor.batch_decode(generated, skip_special_tokens=True)
        transcriptions = [Transcription(text=text) for text in decoded]
        return require_batch_cardinality(self.spec.model_id, audio_paths, transcriptions)

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

    def transcribe_batch(self, audio_paths: list[Path]) -> list[Transcription]:
        import torch

        if not audio_paths:
            raise ValueError("Qwen batch must contain at least one audio path")
        request = self.processor.apply_transcription_request(
            audio=[str(path) for path in audio_paths],
            language=self.spec.language,
            padding=True,
        ).to(self.model.device, self.model.dtype)
        with torch.inference_mode():
            generated = self.model.generate(
                **request,
                max_new_tokens=int(self.spec.generation["max_new_tokens"]),
            )
        input_length = request["input_ids"].shape[-1]
        decoded = [
            self.processor.decode(output[input_length:], return_format="transcription_only")
            for output in generated
        ]
        if not all(isinstance(text, str) for text in decoded):
            raise AdapterOutputError(f"Qwen model {self.spec.model_id} returned non-text output")
        transcriptions = [Transcription(text=text) for text in decoded]
        return require_batch_cardinality(self.spec.model_id, audio_paths, transcriptions)

    def close(self) -> None:
        self.model = None
        self.processor = None

    @property
    def parameter_count(self) -> int:
        if self.model is None:
            return 0
        return sum(parameter.numel() for parameter in self.model.parameters())


class TransformersCTCAdapter(ASRAdapter):
    """Standard Transformers CTC inference with fixed batched greedy decoding."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.processor: Any = None
        self.model: Any = None

    def load(self) -> None:
        from transformers import AutoModelForCTC, AutoProcessor

        LOGGER.info(
            "Loading Transformers CTC checkpoint",
            extra={
                "model": self.spec.model_id,
                "revision": self.spec.revision,
                "native_dtype": self.spec.native_dtype,
            },
        )
        snapshot = pinned_snapshot_directory(self.spec, self.cache_directory)
        LOGGER.debug(
            "Resolved pinned Transformers CTC snapshot",
            extra={"model": self.spec.model_id, "snapshot": str(snapshot)},
        )
        common = {"local_files_only": True}
        self.processor = AutoProcessor.from_pretrained(str(snapshot), **common)
        self.model = AutoModelForCTC.from_pretrained(
            str(snapshot),
            dtype=_torch_dtype(self.spec.native_dtype),
            low_cpu_mem_usage=True,
            **common,
        ).to("cuda")
        self.model.eval()
        LOGGER.info(
            "Loaded Transformers CTC checkpoint",
            extra={
                "model": self.spec.model_id,
                "device": str(self.model.device),
                "dtype": str(self.model.dtype),
                "parameter_count": self.parameter_count,
            },
        )

    def transcribe_batch(self, audio_paths: list[Path]) -> list[Transcription]:
        import torch

        if not audio_paths:
            raise ValueError("CTC batch must contain at least one audio path")
        decoded_audio = [sf.read(path, dtype="float32") for path in audio_paths]
        sample_rates = {sample_rate for _, sample_rate in decoded_audio}
        if len(sample_rates) != 1:
            raise ValueError("CTC batch audio must use one sample rate")
        sample_rate = sample_rates.pop()
        LOGGER.debug(
            "Preparing Transformers CTC inference",
            extra={
                "model": self.spec.model_id,
                "sample_rate": sample_rate,
                "batch_size": len(audio_paths),
            },
        )
        processor_inputs = self.processor(
            [audio for audio, _ in decoded_audio],
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        model_inputs = {
            name: tensor.to(self.model.device, dtype=self.model.dtype)
            if tensor.is_floating_point()
            else tensor.to(self.model.device)
            for name, tensor in processor_inputs.items()
        }
        LOGGER.debug(
            "Running Transformers CTC model",
            extra={"model": self.spec.model_id, "input_names": sorted(model_inputs)},
        )
        with torch.inference_mode():
            logits = self.model(**model_inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is None:
            raise TransformersCTCOutputError(
                f"Transformers CTC model {self.spec.model_id} requires an attention mask "
                "for padding-safe batched decoding"
            )
        output_lengths = (
            self.model._get_feat_extract_output_lengths(attention_mask.sum(dim=-1))
            .detach()
            .cpu()
            .tolist()
        )
        token_sequences = [
            row[: int(output_length)].detach().cpu().tolist()
            for row, output_length in zip(predicted_ids, output_lengths, strict=True)
        ]
        decoded = self.processor.batch_decode(
            token_sequences,
            group_tokens=True,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not isinstance(decoded, list | tuple) or len(decoded) != len(audio_paths):
            output_count = len(decoded) if isinstance(decoded, list | tuple) else "non-sequence"
            raise TransformersCTCOutputError(
                f"Transformers CTC model {self.spec.model_id} expected {len(audio_paths)} "
                f"decoded results, received {output_count}"
            )
        if not all(isinstance(text, str) for text in decoded):
            raise TransformersCTCOutputError(
                f"Transformers CTC model {self.spec.model_id} returned non-text decoded results"
            )
        LOGGER.debug(
            "Completed Transformers CTC inference",
            extra={"model": self.spec.model_id, "decoded_results": len(decoded)},
        )
        transcriptions = [Transcription(text=text) for text in decoded]
        return require_batch_cardinality(self.spec.model_id, audio_paths, transcriptions)

    def close(self) -> None:
        LOGGER.info("Closing Transformers CTC adapter", extra={"model": self.spec.model_id})
        self.model = None
        self.processor = None
        LOGGER.debug(
            "Released Transformers CTC model resources", extra={"model": self.spec.model_id}
        )

    @property
    def parameter_count(self) -> int:
        if self.model is None:
            return 0
        return sum(parameter.numel() for parameter in self.model.parameters())
