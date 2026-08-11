"""NeMo default-RNNT Persian FastConformer adapter."""

import logging
from pathlib import Path
from typing import Any

from peste.adapters.base import ASRAdapter, Transcription, require_batch_cardinality

LOGGER = logging.getLogger(__name__)


class NemoRnntAdapter(ASRAdapter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model: Any = None

    def load(self) -> None:
        from nemo.collections.asr.models import ASRModel

        LOGGER.info("Loading NeMo checkpoint", extra={"model": self.spec.model_id})
        snapshot = (
            self.cache_directory
            / f"models--{self.spec.repository.replace('/', '--')}"
            / "snapshots"
            / self.spec.revision
        )
        checkpoints = list(snapshot.glob("*.nemo"))
        if len(checkpoints) != 1:
            raise FileNotFoundError(
                f"Expected one pinned NeMo checkpoint in {snapshot}, found {len(checkpoints)}"
            )
        self.model = ASRModel.restore_from(str(checkpoints[0]), map_location="cuda")
        self.model.change_decoding_strategy(decoder_type="rnnt")
        decoder = getattr(getattr(self.model, "decoding", None), "decoding", None)
        disable_cuda_graphs = getattr(decoder, "disable_cuda_graphs", None)
        if not callable(disable_cuda_graphs):
            raise RuntimeError(
                f"NeMo model {self.spec.model_id} does not expose CUDA graph decoder control"
            )
        decoder_changed = bool(disable_cuda_graphs())
        LOGGER.info(
            "Disabled NeMo CUDA graph decoder for variable-length batched inference",
            extra={"model": self.spec.model_id, "decoder_changed": decoder_changed},
        )
        self.model.eval()

    def transcribe_batch(self, audio_paths: list[Path]) -> list[Transcription]:
        if not audio_paths:
            raise ValueError("NeMo batch must contain at least one audio path")
        hypotheses = self.model.transcribe(
            [str(audio_path) for audio_path in audio_paths],
            batch_size=len(audio_paths),
            verbose=False,
        )
        transcriptions = [
            Transcription(text=hypothesis.text if hasattr(hypothesis, "text") else str(hypothesis))
            for hypothesis in hypotheses
        ]
        return require_batch_cardinality(self.spec.model_id, audio_paths, transcriptions)

    def close(self) -> None:
        self.model = None

    @property
    def parameter_count(self) -> int:
        if self.model is None:
            return 0
        return sum(parameter.numel() for parameter in self.model.parameters())
