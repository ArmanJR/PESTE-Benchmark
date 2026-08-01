"""Immutable suite manifest loading and validation."""

import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from psst.digests import sha256_file
from psst.normalization import normalize
from psst.schemas import ManifestRow, SuiteSpec


def iter_manifest(path: Path) -> Iterator[ManifestRow]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank manifest line at {line_number}")
            try:
                yield ManifestRow.model_validate(json.loads(line))
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"Invalid manifest line {line_number}: {error}") from error


def validate_manifest(spec: SuiteSpec, suite_directory: Path) -> list[ManifestRow]:
    path = suite_directory / spec.manifest_path
    actual_digest = sha256_file(path)
    if actual_digest != spec.manifest_sha256:
        raise ValueError(
            f"Manifest digest mismatch: expected {spec.manifest_sha256}, got {actual_digest}"
        )
    rows = list(iter_manifest(path))
    counts = Counter(row.split for row in rows)
    if dict(counts) != spec.expected_split_counts:
        raise ValueError(
            f"Split counts mismatch: expected {spec.expected_split_counts}, got {counts}"
        )
    sample_ids = {row.sample_id for row in rows}
    if len(sample_ids) != len(rows):
        raise ValueError("Manifest sample IDs must be unique")
    expected_order = [
        (split, index)
        for split in ("train", "validation", "test")
        for index in range(spec.expected_split_counts[split])
    ]
    actual_order = [(row.split, row.upstream_row_index) for row in rows]
    if actual_order != expected_order:
        raise ValueError("Manifest rows are not in canonical split/index order")
    for row in rows:
        if row.source_repository != spec.dataset.repository:
            raise ValueError(f"Unexpected source repository for {row.sample_id}")
        if row.source_revision != spec.dataset.revision:
            raise ValueError(f"Unexpected source revision for {row.sample_id}")
        if not normalize(row.transcription, spec.normalization_version):
            raise ValueError(f"Empty normalized reference for {row.sample_id}")
    return rows
