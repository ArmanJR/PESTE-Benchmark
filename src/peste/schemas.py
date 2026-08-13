"""Validated public benchmark contracts."""

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Sha256 = str


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SchemaV2Model(ContractModel):
    """Base for persisted v2 contracts with an actionable version error."""

    schema_version: Literal[2]

    @model_validator(mode="before")
    @classmethod
    def reject_unsupported_schema_version(cls, value: Any) -> Any:
        if isinstance(value, dict) and value.get("schema_version") != 2:
            version = value.get("schema_version", "missing")
            raise ValueError(f"Unsupported schema_version {version}; expected 2")
        return value


class DatasetSource(ContractModel):
    repository: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    config: str
    license: str


class SuiteSpec(SchemaV2Model):
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
    name: Literal["modern", "nemo"]
    image: str
    dockerfile: str


class SpeedProfile(ContractModel):
    hardware_profile_id: Literal["rtx-6000-ada-v1"]
    batch_size: int = Field(gt=0)


class ModelSpec(SchemaV2Model):
    model_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    repository: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    adapter: Literal[
        "transformers-whisper",
        "transformers-qwen",
        "transformers-ctc",
        "nemo-rnnt",
        "nemo-ctc",
    ]
    native_dtype: Literal["float16", "bfloat16", "float32"]
    license: str
    language: str | None
    generation: dict[str, str | int | float | bool]
    runtime: RuntimeSpec
    speed_profile: SpeedProfile


class CampaignCandidate(ContractModel):
    """Immutable identity for one model considered by a benchmark campaign."""

    model_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    repository: str = Field(pattern=r"^[^/\s]+/[^/\s]+$")
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    adapter: Literal[
        "transformers-whisper",
        "transformers-qwen",
        "transformers-ctc",
        "nemo-rnnt",
        "nemo-ctc",
    ]


class CampaignSpec(SchemaV2Model):
    """Tracked definition of an exact, ordered model qualification campaign."""

    campaign_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    suite_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    candidates: tuple[CampaignCandidate, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def candidates_are_unique(self) -> Self:
        model_ids = [candidate.model_id for candidate in self.candidates]
        identities = [(candidate.repository, candidate.revision) for candidate in self.candidates]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("Campaign model IDs must be unique")
        if len(identities) != len(set(identities)):
            raise ValueError("Campaign repository revisions must be unique")
        return self


class ManifestRow(SchemaV2Model):
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
    completed_batches: int = Field(ge=0)
    completed_samples: int = Field(ge=0)


class RunRequest(SchemaV2Model):
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
    peste_revision: str
    image_reference: str
    image_digest: str
    dependency_versions: dict[str, str]
    python_version: str
    pytorch_version: str
    cuda_version: str
    hardware_profile: dict[str, str | int | float | bool]
    gpu_product_name: str
    driver_version: str
    ecc_state: str
    power_limit_watts: float = Field(ge=0)
    cpu_model: str
    gpu_uuid: str
    cloud_provenance: str | None = None
    seed: int


class ModelFacts(ContractModel):
    checkpoint_bytes: int = Field(ge=0)
    parameter_count: int = Field(ge=0)
    native_dtype: Literal["float16", "bfloat16", "float32"]


class SpeedStatistics(ContractModel):
    valid: bool
    batch_size: int = Field(gt=0)
    warmup_batches: int = Field(ge=0)
    measured_batches: int = Field(ge=0)
    total_audio_seconds: float = Field(ge=0, allow_inf_nan=False)
    processing_seconds: float = Field(ge=0, allow_inf_nan=False)
    audio_throughput_x: float = Field(ge=0, allow_inf_nan=False)
    rtf: float = Field(ge=0, allow_inf_nan=False)
    timing_artifact: str = Field(min_length=1)
    invalidity_reason: str | None = None

    @model_validator(mode="after")
    def timing_is_consistent(self) -> Self:
        import math

        has_audio = self.total_audio_seconds > 0
        has_processing = self.processing_seconds > 0
        if has_audio != has_processing:
            raise ValueError("Audio and processing times must both be zero or both be positive")
        if has_audio:
            throughput = self.total_audio_seconds / self.processing_seconds
            rtf = self.processing_seconds / self.total_audio_seconds
            if not math.isclose(self.audio_throughput_x, throughput, rel_tol=1e-12):
                raise ValueError("audio_throughput_x does not match audio/processing time")
            if not math.isclose(self.rtf, rtf, rel_tol=1e-12):
                raise ValueError("RTF does not match processing/audio time")
            if not math.isclose(self.audio_throughput_x * self.rtf, 1.0, rel_tol=1e-12):
                raise ValueError("audio throughput and RTF must be reciprocal")
        elif self.audio_throughput_x != 0 or self.rtf != 0:
            raise ValueError("Zero timing values require zero throughput and RTF")
        if self.valid:
            if self.invalidity_reason is not None:
                raise ValueError("Valid speed statistics cannot have an invalidity reason")
            if self.measured_batches == 0:
                raise ValueError("Valid speed statistics require at least one measured batch")
            if self.total_audio_seconds <= 0 or self.processing_seconds <= 0:
                raise ValueError(
                    "Valid speed statistics require positive audio and processing time"
                )
        elif not self.invalidity_reason:
            raise ValueError("Invalid speed statistics require an invalidity reason")
        return self


class PredictionRecord(SchemaV2Model):
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


class LogReferences(ContractModel):
    runner: str
    container: str | None = None
    diagnostics: str | None = None


class RunBundle(SchemaV2Model):
    run_id: str
    suite_id: str
    suite_digest: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    model_id: str
    model_digest: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    status: RunStatus
    environment: EnvironmentFingerprint
    speed: SpeedStatistics
    model_facts: ModelFacts
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
        if self.status != RunStatus.SUCCESS and self.speed.valid:
            raise ValueError("Only a successful run can contain valid speed statistics")
        return self
