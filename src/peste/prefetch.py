"""Network-enabled checkpoint prefetch, separated from offline scoring."""

import logging
from pathlib import Path

from huggingface_hub import snapshot_download

from peste.schemas import ModelSpec

LOGGER = logging.getLogger(__name__)


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
