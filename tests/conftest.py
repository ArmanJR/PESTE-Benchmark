"""Shared test contract builders."""

import json
from pathlib import Path

import pytest

from psst.digests import canonical_json, sha256_bytes
from psst.schemas import DatasetSource, ManifestRow, ModelSpec, RuntimeSpec, SuiteSpec


def make_model(
    adapter: str = "transformers-whisper",
    *,
    model_id: str = "fake-model",
    dtype: str = "float16",
) -> ModelSpec:
    language = None if adapter == "vibevoice" else "fa"
    generations: dict[str, dict[str, str | int | float | bool]] = {
        "transformers-whisper": {
            "task": "transcribe",
            "max_new_tokens": 444,
            "return_timestamps": False,
        },
        "transformers-qwen": {"max_new_tokens": 256},
        "vibevoice": {
            "max_new_tokens": 512,
            "temperature": 0.0,
            "top_p": 1.0,
            "num_beams": 1,
        },
        "nemo-rnnt": {"decoder": "rnnt", "batch_size": 1, "external_language_model": False},
    }
    runtime_name = {
        "transformers-whisper": "modern",
        "transformers-qwen": "modern",
        "vibevoice": "vibevoice",
        "nemo-rnnt": "nemo",
    }[adapter]
    return ModelSpec.model_validate(
        {
            "schema_version": 1,
            "model_id": model_id,
            "repository": "organization/model",
            "revision": "a" * 40,
            "adapter": adapter,
            "native_dtype": dtype,
            "license": "Apache-2.0",
            "language": language,
            "generation": generations[adapter],
            "runtime": RuntimeSpec(
                name=runtime_name,
                image=f"psst-{runtime_name}:test",
                dockerfile=f"runtimes/{runtime_name}/Dockerfile",
            ),
        }
    )


@pytest.fixture
def tiny_suite(tmp_path: Path) -> tuple[SuiteSpec, Path, list[ManifestRow]]:
    suite_directory = tmp_path / "suite"
    suite_directory.mkdir()
    rows: list[ManifestRow] = []
    for split, count in (("train", 1), ("validation", 1), ("test", 2)):
        for index in range(count):
            rows.append(
                ManifestRow(
                    schema_version=1,
                    sample_id=f"{split}-{index:06d}",
                    split=split,
                    upstream_row_index=index,
                    upstream_row_id=index,
                    transcription=f"متن {index}",
                    duration_seconds=1.0,
                    audio_sha256="b" * 64,
                    audio_path=f"audio/{split}/{index:06d}.wav",
                    source_repository="google/fleurs",
                    source_revision="c" * 40,
                    source_license="CC-BY-4.0",
                )
            )
    encoded = b"".join(canonical_json(row.model_dump(mode="json")) for row in rows)
    (suite_directory / "manifest.jsonl").write_bytes(encoded)
    suite = SuiteSpec(
        schema_version=1,
        suite_id="tiny-suite-v1",
        dataset=DatasetSource(
            repository="google/fleurs",
            revision="c" * 40,
            config="fa_ir",
            license="CC-BY-4.0",
        ),
        evaluation_split="test",
        normalization_version="fa-v1",
        manifest_path="manifest.jsonl",
        manifest_sha256=sha256_bytes(encoded),
        expected_split_counts={"train": 1, "validation": 1, "test": 2},
    )
    (suite_directory / "suite.json").write_text(
        json.dumps(suite.model_dump(mode="json")), encoding="utf-8"
    )
    return suite, suite_directory, rows
