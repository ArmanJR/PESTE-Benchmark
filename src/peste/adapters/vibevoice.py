"""VibeVoice-ASR adapter and deterministic segment flattening."""

import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from peste.adapters.base import ASRAdapter, Transcription
from peste.prefetch import (
    VIBEVOICE_TOKENIZER_REPOSITORY,
    VIBEVOICE_TOKENIZER_REVISION,
    pinned_snapshot_directory,
)

LOGGER = logging.getLogger(__name__)


def flatten_segments(value: Any) -> str:
    """Concatenate textual VibeVoice segment payloads in reported order."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("segments", "transcription", "result", "output"):
            if key in value:
                flattened = flatten_segments(value[key])
                if flattened:
                    return flattened
        text = value.get("text")
        return text.strip() if isinstance(text, str) else ""
    if isinstance(value, Iterable):
        return " ".join(filter(None, (flatten_segments(item) for item in value))).strip()
    return ""


class VibeVoiceAdapter(ASRAdapter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.processor: Any = None
        self.model: Any = None

    def load(self) -> None:
        import torch
        from vibevoice.modular.modeling_vibevoice_asr import (
            VibeVoiceASRForConditionalGeneration,
        )
        from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

        model_snapshot = pinned_snapshot_directory(self.spec, self.cache_directory)
        model_common = {"local_files_only": True}
        tokenizer_snapshot = (
            self.cache_directory
            / f"models--{VIBEVOICE_TOKENIZER_REPOSITORY.replace('/', '--')}"
            / "snapshots"
            / VIBEVOICE_TOKENIZER_REVISION
        )
        if not tokenizer_snapshot.is_dir():
            raise FileNotFoundError(f"Pinned VibeVoice tokenizer is missing: {tokenizer_snapshot}")
        LOGGER.info("Loading VibeVoice checkpoint", extra={"model": self.spec.model_id})
        self.processor = VibeVoiceASRProcessor.from_pretrained(
            str(model_snapshot),
            language_model_pretrained_name=str(tokenizer_snapshot),
            **model_common,
        )
        self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            str(model_snapshot),
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **model_common,
        ).to("cuda")
        self.model.eval()

    def transcribe(self, audio_path: Path) -> Transcription:
        import torch

        request = self.processor(
            audio=[str(audio_path)],
            sampling_rate=None,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
        )
        request = {key: value.to("cuda") for key, value in request.items()}
        with torch.inference_mode():
            generated = self.model.generate(
                **request,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                num_beams=1,
                pad_token_id=self.processor.pad_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
        input_length = request["input_ids"].shape[1]
        decoded = self.processor.decode(generated[0, input_length:], skip_special_tokens=True)
        parsed = self.processor.post_process_transcription(decoded)
        structured = {"raw_text": decoded, "segments": parsed}
        return Transcription(text=flatten_segments(parsed), structured_output=structured)

    def close(self) -> None:
        self.model = None
        self.processor = None

    @property
    def parameter_count(self) -> int:
        if self.model is None:
            return 0
        return sum(parameter.numel() for parameter in self.model.parameters())
