"""Pinned FLEURS acquisition and canonical audio preparation."""

import io
import logging
import os
import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

from peste.digests import canonical_json, sha256_bytes, sha256_file
from peste.manifest import validate_manifest
from peste.normalization import normalize
from peste.schemas import ManifestRow, SuiteSpec

LOGGER = logging.getLogger(__name__)
CANONICAL_SAMPLE_RATE = 16_000


def _decode_audio(audio: Mapping[str, Any]) -> tuple[np.ndarray[Any, np.dtype[np.float32]], int]:
    source: str | io.BytesIO
    audio_bytes = audio.get("bytes")
    if isinstance(audio_bytes, bytes):
        source = io.BytesIO(audio_bytes)
    else:
        audio_path = audio.get("path")
        if not isinstance(audio_path, str):
            raise ValueError("Dataset audio row has neither bytes nor a path")
        source = audio_path
    samples, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    mono = np.mean(samples, axis=1, dtype=np.float32)
    if sample_rate != CANONICAL_SAMPLE_RATE:
        target_length = round(len(mono) * CANONICAL_SAMPLE_RATE / sample_rate)
        source_positions = np.arange(len(mono), dtype=np.float64)
        target_positions = (
            np.arange(target_length, dtype=np.float64) * sample_rate / CANONICAL_SAMPLE_RATE
        )
        mono = np.interp(target_positions, source_positions, mono).astype(np.float32)
    return mono, CANONICAL_SAMPLE_RATE


def _write_canonical_wav(destination: Path, audio: Mapping[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    samples, sample_rate = _decode_audio(audio)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        sf.write(temporary_path, samples, sample_rate, format="WAV", subtype="PCM_16")
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def materialize_rows(spec: SuiteSpec, cache_directory: Path) -> list[ManifestRow]:
    """Download pinned FLEURS rows and decode each recording exactly once."""
    dataset_cache = cache_directory / "huggingface"
    rows: list[ManifestRow] = []
    for split in ("train", "validation", "test"):
        LOGGER.info("Loading pinned FLEURS split", extra={"suite": spec.suite_id, "split": split})
        dataset = load_dataset(
            spec.dataset.repository,
            spec.dataset.config,
            split=split,
            revision=spec.dataset.revision,
            cache_dir=str(dataset_cache),
        ).cast_column("audio", Audio(decode=False))
        expected_count = spec.expected_split_counts[split]
        if len(dataset) != expected_count:
            raise ValueError(f"{split} has {len(dataset)} rows; expected {expected_count}")
        for index, sample in enumerate(dataset):
            sample_id = f"{split}-{index:06d}"
            relative_audio_path = Path("audio") / split / f"{index:06d}.wav"
            destination = cache_directory / relative_audio_path
            if not destination.exists():
                _write_canonical_wav(destination, sample["audio"])
            with sf.SoundFile(destination) as wav:
                if (
                    wav.samplerate != CANONICAL_SAMPLE_RATE
                    or wav.channels != 1
                    or wav.subtype != "PCM_16"
                ):
                    raise ValueError(f"Non-canonical cached WAV: {destination}")
                duration = len(wav) / CANONICAL_SAMPLE_RATE
            transcription = str(sample["transcription"])
            if not normalize(transcription, spec.normalization_version):
                raise ValueError(f"Empty normalized reference for {sample_id}")
            rows.append(
                ManifestRow(
                    schema_version=2,
                    sample_id=sample_id,
                    split=split,
                    upstream_row_index=index,
                    upstream_row_id=sample.get("id", index),
                    transcription=transcription,
                    duration_seconds=round(duration, 6),
                    audio_sha256=sha256_file(destination),
                    audio_path=relative_audio_path.as_posix(),
                    source_repository=spec.dataset.repository,
                    source_revision=spec.dataset.revision,
                    source_license=spec.dataset.license,
                )
            )
            if (index + 1) % 100 == 0 or index + 1 == expected_count:
                LOGGER.info(
                    "Prepared canonical audio",
                    extra={
                        "suite": spec.suite_id,
                        "split": split,
                        "sample": sample_id,
                        "completed": index + 1,
                        "total": expected_count,
                    },
                )
    return rows


def encoded_manifest(rows: list[ManifestRow]) -> bytes:
    return b"".join(canonical_json(row.model_dump(mode="json")) for row in rows)


def prepare_dataset(spec: SuiteSpec, suite_directory: Path, cache_directory: Path) -> None:
    """Materialize audio and verify it byte-for-byte against the immutable manifest."""
    committed_rows = validate_manifest(spec, suite_directory)
    materialized_rows = materialize_rows(spec, cache_directory)
    if materialized_rows != committed_rows:
        for expected, actual in zip(committed_rows, materialized_rows, strict=True):
            if expected != actual:
                raise ValueError(
                    f"Prepared data differs from immutable manifest at {expected.sample_id}"
                )
        raise ValueError("Prepared data differs from immutable manifest")
    LOGGER.info(
        "Dataset preparation complete",
        extra={"suite": spec.suite_id, "rows": len(materialized_rows)},
    )


def write_initial_manifest(spec: SuiteSpec, suite_directory: Path, cache_directory: Path) -> str:
    """Maintainer-only helper that creates a manifest once and refuses overwrites."""
    manifest_path = suite_directory / spec.manifest_path
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to replace immutable manifest: {manifest_path}")
    rows = materialize_rows(spec, cache_directory)
    counts = Counter(row.split for row in rows)
    if dict(counts) != spec.expected_split_counts:
        raise ValueError(f"Unexpected split counts: {counts}")
    encoded = encoded_manifest(rows)
    manifest_path.write_bytes(encoded)
    digest = sha256_bytes(encoded)
    LOGGER.info(
        "Wrote initial immutable suite manifest",
        extra={"path": str(manifest_path), "sha256": digest, "rows": len(rows)},
    )
    return digest
