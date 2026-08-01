"""Validated public benchmark contracts."""

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Sha256 = str


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetSource(ContractModel):
    repository: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    config: str
    license: str


class SuiteSpec(ContractModel):
    schema_version: Literal[1]
    suite_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    dataset: DatasetSource
    evaluation_split: str
    normalization_version: str
    manifest_path: str
    manifest_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    expected_split_counts: dict[str, int]

    @model_validator(mode="after")
    def evaluation_split_exists(self) -> Self:
        if self.evaluation_split not in self.expected_split_counts:
            raise ValueError("Evaluation split is missing from expected_split_counts")
        if any(count <= 0 for count in self.expected_split_counts.values()):
            raise ValueError("Every expected split count must be positive")
        return self


class RuntimeSpec(ContractModel):
    name: Literal["modern", "vibevoice", "nemo"]
    image: str
    dockerfile: str


class ModelSpec(ContractModel):
    schema_version: Literal[1]
    model_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    repository: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    adapter: Literal["transformers-whisper", "transformers-qwen", "vibevoice", "nemo-rnnt"]
    native_dtype: Literal["float16", "bfloat16", "float32"]
    license: str
    language: str | None
    generation: dict[str, str | int | float | bool]
    runtime: RuntimeSpec


class ManifestRow(ContractModel):
    schema_version: Literal[1]
    sample_id: str
    split: Literal["train", "validation", "test"]
    upstream_row_index: int = Field(ge=0)
    upstream_row_id: int | str
    transcription: str
    duration_seconds: float = Field(gt=0)
    audio_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    audio_path: str
    source_repository: str
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_license: str


class ResumeState(ContractModel):
    completed_samples: int = Field(ge=0)
    peak_cuda_reserved_bytes: int = Field(ge=0)
    peak_cuda_allocated_bytes: int = Field(ge=0)
    peak_process_rss_bytes: int = Field(ge=0)


class RunRequest(ContractModel):
    schema_version: Literal[1]
    run_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
    suite_id: str
    suite_digest: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    model_id: str
    model_digest: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    device: Literal["cuda"] = "cuda"
    seed: int = Field(ge=0)
    dataset_cache: Path
    model_cache: Path
    output_directory: Path
    resume: ResumeState | None = None


class RunStatus(StrEnum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    OOM = "oom"
    KILLED = "killed"


class EnvironmentFingerprint(ContractModel):
    psst_revision: str
    image_reference: str
    image_digest: str
    dependency_versions: dict[str, str]
    python_version: str
    pytorch_version: str
    cuda_version: str
    hardware_profile: dict[str, str | int | float | bool]
    seed: int


class MemoryStatistics(ContractModel):
    peak_cuda_reserved_bytes: int = Field(ge=0)
    peak_cuda_allocated_bytes: int = Field(ge=0)
    peak_process_rss_bytes: int = Field(ge=0)
    checkpoint_bytes: int = Field(ge=0)
    parameter_count: int = Field(ge=0)
    native_dtype: str


class PredictionRecord(ContractModel):
    schema_version: Literal[1]
    sequence: int = Field(ge=0)
    sample_id: str
    reference: str
    prediction: str
    normalized_reference: str
    normalized_prediction: str
    word_substitutions: int = Field(ge=0)
    word_deletions: int = Field(ge=0)
    word_insertions: int = Field(ge=0)
    word_reference_units: int = Field(gt=0)
    character_substitutions: int = Field(ge=0)
    character_deletions: int = Field(ge=0)
    character_insertions: int = Field(ge=0)
    character_reference_units: int = Field(gt=0)
    structured_output: dict[str, Any] | list[Any] | None = None


class AggregateMetrics(ContractModel):
    samples: int = Field(gt=0)
    wer: float = Field(ge=0)
    cer: float = Field(ge=0)
    word_errors: int = Field(ge=0)
    word_reference_units: int = Field(gt=0)
    character_errors: int = Field(ge=0)
    character_reference_units: int = Field(gt=0)
    word_accuracy_pct: float = Field(ge=0, le=100)
    memory_efficiency: float = Field(ge=0)


class LogReferences(ContractModel):
    runner: str
    container: str | None = None
    diagnostics: str | None = None


class RunBundle(ContractModel):
    schema_version: Literal[1]
    run_id: str
    suite_id: str
    suite_digest: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    model_id: str
    model_digest: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    status: RunStatus
    environment: EnvironmentFingerprint
    memory: MemoryStatistics
    predictions_path: str
    aggregates: AggregateMetrics | None
    logs: LogReferences
    error: str | None = None

    @model_validator(mode="after")
    def successful_run_is_complete(self) -> Self:
        if self.status == RunStatus.SUCCESS and self.aggregates is None:
            raise ValueError("A successful run requires aggregate metrics")
        if self.status != RunStatus.SUCCESS and self.aggregates is not None:
            raise ValueError("Failed, OOM, killed, and running runs cannot be ranked")
        return self
