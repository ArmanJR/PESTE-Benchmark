"""Real-adapter deterministic one-sample validation."""

import logging
from pathlib import Path

from peste.adapters import create_adapter
from peste.adapters.base import require_batch_cardinality
from peste.digests import sha256_bytes
from peste.manifest import validate_manifest
from peste.normalization import normalize
from peste.runner import _seed_runtime
from peste.schemas import ModelSpec, SuiteSpec

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
    evaluation_rows = [row for row in rows if row.split == suite.evaluation_split]
    if model.adapter == "transformers-whisper":
        ordered = sorted(
            evaluation_rows, key=lambda row: (row.duration_seconds, row.upstream_row_index)
        )
        samples = [ordered[0], ordered[-1]]
    else:
        samples = [evaluation_rows[0]]
    audio_paths = [dataset_cache / sample.audio_path for sample in samples]
    missing = [path for path in audio_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Smoke-test audio is missing: {missing[0]}")
    torch = _seed_runtime(seed)
    torch.cuda.empty_cache()
    adapter = create_adapter(model, model_cache)
    try:
        LOGGER.info(
            "Starting real adapter smoke test",
            extra={
                "model": model.model_id,
                "samples": [sample.sample_id for sample in samples],
                "durations_seconds": [sample.duration_seconds for sample in samples],
                "seed": seed,
            },
        )
        adapter.load()
        _seed_runtime(seed)
        first = [
            normalize(output.text, suite.normalization_version)
            for output in require_batch_cardinality(
                model.model_id, audio_paths, adapter.transcribe_batch(audio_paths)
            )
        ]
        _seed_runtime(seed)
        second = [
            normalize(output.text, suite.normalization_version)
            for output in require_batch_cardinality(
                model.model_id, audio_paths, adapter.transcribe_batch(audio_paths)
            )
        ]
        if first != second:
            LOGGER.error(
                "Repeated smoke transcriptions diverged",
                extra={
                    "model": model.model_id,
                    "samples": [sample.sample_id for sample in samples],
                    "first_characters": [len(text) for text in first],
                    "second_characters": [len(text) for text in second],
                    "first_sha256": [sha256_bytes(text.encode("utf-8")) for text in first],
                    "second_sha256": [sha256_bytes(text.encode("utf-8")) for text in second],
                },
            )
            raise RuntimeError("Repeated smoke transcriptions produced different normalized text")
        LOGGER.info(
            "Real adapter smoke test passed",
            extra={
                "model": model.model_id,
                "samples": [sample.sample_id for sample in samples],
                "normalized_characters": [len(text) for text in first],
                "normalized_sha256": [sha256_bytes(text.encode("utf-8")) for text in first],
                "parameter_count": adapter.parameter_count,
            },
        )
    except Exception:
        LOGGER.exception(
            "Real adapter smoke test failed",
            extra={"model": model.model_id, "samples": [sample.sample_id for sample in samples]},
        )
        raise
    finally:
        adapter.close()
