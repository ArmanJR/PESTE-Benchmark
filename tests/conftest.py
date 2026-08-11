"""Shared test contract builders."""

import json
from pathlib import Path

import pytest

from peste.digests import canonical_json, sha256_bytes
from peste.schemas import (
    DatasetSource,
    ManifestRow,
    ModelSpec,
    RuntimeSpec,
    SpeedProfile,
    SuiteSpec,
)


def make_model(
    adapter: str = "transformers-whisper",
    *,
    model_id: str = "fake-model",
    dtype: str = "float16",
) -> ModelSpec:
    generations: dict[str, dict[str, str | int | float | bool]] = {
        "transformers-whisper": {
            "task": "transcribe",
            "max_new_tokens": 444,
            "return_timestamps": False,
        },
        "transformers-qwen": {"max_new_tokens": 256},
        "transformers-ctc": {
            "decoder": "greedy",
            "external_language_model": False,
            "group_tokens": True,
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
        },
        "nemo-rnnt": {"decoder": "rnnt", "external_language_model": False},
    }
    runtime_name = {
        "transformers-whisper": "modern",
        "transformers-qwen": "modern",
        "transformers-ctc": "modern",
        "nemo-rnnt": "nemo",
    }[adapter]
    return ModelSpec.model_validate(
        {
            "schema_version": 2,
            "model_id": model_id,
            "repository": "organization/model",
            "revision": "a" * 40,
            "adapter": adapter,
            "native_dtype": dtype,
            "license": "Apache-2.0",
            "language": "fa",
            "generation": generations[adapter],
            "runtime": RuntimeSpec(
                name=runtime_name,
                image="ghcr.io/armanjr/peste-benchmark:2.0.0",
                dockerfile="runtimes/Dockerfile",
            ),
            "speed_profile": SpeedProfile(
                hardware_profile_id="rtx-6000-ada-v1",
                batch_size=2,
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
                    schema_version=2,
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
        schema_version=2,
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
