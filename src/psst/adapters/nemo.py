"""NeMo default-RNNT Persian FastConformer adapter."""

import logging
from pathlib import Path
from typing import Any

from psst.adapters.base import ASRAdapter, Transcription

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
        self.model.eval()

    def transcribe(self, audio_path: Path) -> Transcription:
        hypotheses = self.model.transcribe([str(audio_path)], batch_size=1)
        first = hypotheses[0]
        text = first.text if hasattr(first, "text") else str(first)
        return Transcription(text=text)

    def close(self) -> None:
        self.model = None

    @property
    def parameter_count(self) -> int:
        if self.model is None:
            return 0
        return sum(parameter.numel() for parameter in self.model.parameters())
