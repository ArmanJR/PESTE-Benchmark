"""Network-enabled checkpoint prefetch, separated from offline scoring."""

import logging
from pathlib import Path

from huggingface_hub import snapshot_download

from psst.schemas import ModelSpec

LOGGER = logging.getLogger(__name__)
VIBEVOICE_TOKENIZER_REPOSITORY = "Qwen/Qwen2.5-7B"
VIBEVOICE_TOKENIZER_REVISION = "d149729398750b98c0af14eb82c78cfe92750796"


def pinned_snapshot_directory(model: ModelSpec, cache_directory: Path) -> Path:
    snapshot = (
        cache_directory
        / f"models--{model.repository.replace('/', '--')}"
        / "snapshots"
        / model.revision
    )
    if not snapshot.is_dir():
        raise FileNotFoundError(
            f"Pinned snapshot for {model.model_id} is missing from the offline cache: {snapshot}"
        )
    return snapshot


def prefetch_model(model: ModelSpec, cache_directory: Path) -> None:
    LOGGER.info(
        "Prefetching pinned model snapshot",
        extra={"model": model.model_id, "revision": model.revision},
    )
    snapshot_download(repo_id=model.repository, revision=model.revision, cache_dir=cache_directory)
    pinned_snapshot_directory(model, cache_directory)
    if model.adapter == "vibevoice":
        snapshot_download(
            repo_id=VIBEVOICE_TOKENIZER_REPOSITORY,
            revision=VIBEVOICE_TOKENIZER_REVISION,
            cache_dir=cache_directory,
        )
