"""Real-adapter deterministic one-sample validation."""

import logging
from pathlib import Path

from psst.adapters import create_adapter
from psst.digests import sha256_bytes
from psst.manifest import validate_manifest
from psst.normalization import normalize
from psst.runner import _seed_runtime
from psst.schemas import ModelSpec, SuiteSpec

LOGGER = logging.getLogger(__name__)


def smoke_adapter(
    suite: SuiteSpec,
    model: ModelSpec,
    suite_directory: Path,
    dataset_cache: Path,
    model_cache: Path,
    seed: int,
) -> None:
    """Load one adapter and require identical normalized text from two passes."""
    rows = validate_manifest(suite, suite_directory)
    sample = next(row for row in rows if row.split == suite.evaluation_split)
    audio_path = dataset_cache / sample.audio_path
    if not audio_path.is_file():
        raise FileNotFoundError(f"Smoke-test audio is missing: {audio_path}")
    torch = _seed_runtime(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    adapter = create_adapter(model, model_cache)
    try:
        LOGGER.info(
            "Starting real adapter smoke test",
            extra={"model": model.model_id, "sample": sample.sample_id, "seed": seed},
        )
        adapter.load()
        _seed_runtime(seed)
        first = normalize(adapter.transcribe(audio_path).text, suite.normalization_version)
        _seed_runtime(seed)
        second = normalize(adapter.transcribe(audio_path).text, suite.normalization_version)
        if first != second:
            LOGGER.error(
                "Repeated smoke transcriptions diverged",
                extra={
                    "model": model.model_id,
                    "sample": sample.sample_id,
                    "first_characters": len(first),
                    "second_characters": len(second),
                    "first_sha256": sha256_bytes(first.encode("utf-8")),
                    "second_sha256": sha256_bytes(second.encode("utf-8")),
                },
            )
            raise RuntimeError("Repeated smoke transcriptions produced different normalized text")
        output_digest = sha256_bytes(first.encode("utf-8"))
        LOGGER.info(
            "Real adapter smoke test passed",
            extra={
                "model": model.model_id,
                "sample": sample.sample_id,
                "normalized_characters": len(first),
                "normalized_sha256": output_digest,
                "parameter_count": adapter.parameter_count,
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            },
        )
    except Exception:
        LOGGER.exception(
            "Real adapter smoke test failed",
            extra={"model": model.model_id, "sample": sample.sample_id},
        )
        raise
    finally:
        adapter.close()
