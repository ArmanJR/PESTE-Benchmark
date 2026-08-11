"""SSH orchestration for doctor-approved discrete-GPU container hosts."""

import json
import logging
import math
import os
import re
import shlex
import subprocess
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, cast
from urllib.parse import urlparse

from peste.constants import DEFAULT_SEED, PROJECT_ROOT
from peste.digests import canonical_json
from peste.schemas import (
    EnvironmentFingerprint,
    LogReferences,
    ModelFacts,
    ModelSpec,
    ResumeState,
    RunBundle,
    RunRequest,
    RunStatus,
    SpeedStatistics,
    SuiteSpec,
)
from peste.specs import spec_digest

LOGGER = logging.getLogger(__name__)
HARDWARE_PROFILE_PATH = PROJECT_ROOT / "hardware" / "rtx-6000-ada-v1.json"
REMOTE_CONTROLLER = "/opt/venvs/modern/bin/peste"
GPU_IDLE_CLOCK_EVENT_MASK = 0x1


def has_performance_clock_event(throttle_state: str) -> bool:
    """Return whether NVML reports any clock event beyond the benign idle state."""
    normalized = throttle_state.strip().casefold()
    if normalized in {"not active", "none"}:
        return False
    try:
        active_mask = int(normalized, 0)
    except ValueError:
        return True
    return active_mask & ~GPU_IDLE_CLOCK_EVENT_MASK != 0


def load_hardware_profile(path: Path = HARDWARE_PROFILE_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as profile_file:
        profile = cast(dict[str, Any], json.load(profile_file))
    if profile.get("profile_id") != "rtx-6000-ada-v1":
        raise ValueError(f"Unsupported hardware profile in {path}")
    return profile


class SshTransport:
    """Run constrained commands and copy artifacts over non-interactive direct SSH."""

    def __init__(self, host: str) -> None:
        endpoint = urlparse(host)
        if endpoint.scheme != "ssh" or endpoint.hostname is None:
            raise ValueError("GPU host must be an ssh:// endpoint")
        self.host = host
        self.hostname = endpoint.hostname
        self.port = endpoint.port
        self.username = endpoint.username or "root"

    def _base_command(self) -> list[str]:
        command = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
        ]
        if self.port is not None:
            command.extend(["-p", str(self.port)])
        command.append(f"{self.username}@{self.hostname}")
        return command

    def run(
        self,
        arguments: list[str],
        *,
        input_text: str | None = None,
        timeout: float | None = 600,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        remote_command = shlex.join(arguments)
        command = [*self._base_command(), remote_command]
        LOGGER.info(
            "Starting remote SSH command",
            extra={"host": self.host, "command": arguments, "has_stdin": input_text is not None},
        )
        try:
            completed = subprocess.run(
                command,
                input=input_text,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (subprocess.SubprocessError, FileNotFoundError) as error:
            raise RuntimeError(f"Unable to execute SSH command on {self.host}: {error}") from error
        if completed.stderr.strip():
            LOGGER.debug(
                "Remote SSH command stderr",
                extra={
                    "host": self.host,
                    "command": arguments,
                    "returncode": completed.returncode,
                    "stderr": completed.stderr,
                },
            )
        if check and completed.returncode != 0:
            LOGGER.error(
                "Remote SSH command failed",
                extra={
                    "host": self.host,
                    "command": arguments,
                    "returncode": completed.returncode,
                    "stderr": completed.stderr,
                },
            )
            raise RuntimeError(
                f"Remote command failed ({completed.returncode}): {completed.stderr.strip()}"
            )
        return completed

    def copy_directory(
        self, source: PurePosixPath, destination: Path, *, timeout: float = 600
    ) -> None:
        if not source.is_absolute() or source.name in {"", ".", ".."}:
            raise ValueError(f"Remote archive source must be a concrete absolute path: {source}")
        remote_command = shlex.join(
            ["tar", "-C", str(source.parent), "-cf", "-", "--", source.name]
        )
        command = [*self._base_command(), remote_command]
        destination.mkdir(parents=True, exist_ok=True)
        LOGGER.info(
            "Copying remote result archive",
            extra={"host": self.host, "source": str(source), "destination": str(destination)},
        )
        with tempfile.NamedTemporaryFile(suffix=".tar") as archive_file:
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    stdout=archive_file,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                )
            except (subprocess.SubprocessError, FileNotFoundError) as error:
                raise RuntimeError(
                    f"Unable to copy result archive from {self.host}: {error}"
                ) from error
            if completed.returncode != 0:
                stderr = completed.stderr.decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"Remote archive copy failed ({completed.returncode}): {stderr.strip()}"
                )
            archive_file.flush()
            try:
                with tarfile.open(archive_file.name) as archive:
                    archive.extractall(destination, filter="data")
            except (OSError, tarfile.TarError) as error:
                raise RuntimeError(
                    f"Remote result archive is unsafe or invalid: {error}"
                ) from error


class GpuOrchestrator:
    def __init__(self, host: str, root: Path = PROJECT_ROOT) -> None:
        self.host = host
        self.root = root
        self.transport = SshTransport(host)

    def _remote_json(self, arguments: list[str]) -> dict[str, Any]:
        completed = self.transport.run(arguments, timeout=60)
        try:
            return cast(dict[str, Any], json.loads(completed.stdout))
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"Remote diagnostic command returned invalid JSON: {error}"
            ) from error

    def doctor(self) -> dict[str, str | int | float | bool]:
        contract = load_hardware_profile(self.root / "hardware" / "rtx-6000-ada-v1.json")
        diagnostics = self._remote_json([REMOTE_CONTROLLER, "_doctor-probe"])
        gpu_rows = cast(list[str], diagnostics.get("gpu_rows", []))
        gpu_fields = [field.strip() for field in gpu_rows[0].split(",")] if gpu_rows else []
        problems: list[str] = []
        if len(gpu_rows) != 1 or len(gpu_fields) != 8:
            problems.append(
                f"Expected exactly one full physical GPU; diagnostics reported {gpu_rows}"
            )
            gpu_fields = ["unknown", "0", "unknown", "unknown", "0", "0", "unknown", "unknown"]
        (
            gpu_name,
            memory_mib_raw,
            driver_version,
            ecc_state,
            power_limit_raw,
            power_max_raw,
            gpu_uuid,
            throttle_state,
        ) = gpu_fields
        try:
            memory_mib = int(float(memory_mib_raw))
            power_limit = float(power_limit_raw)
            power_max = float(power_max_raw)
            driver_major = int(driver_version.split(".", maxsplit=1)[0])
        except ValueError:
            memory_mib = 0
            power_limit = 0.0
            power_max = 0.0
            driver_major = 0
            problems.append("GPU numeric diagnostics could not be parsed")

        architecture = str(diagnostics.get("architecture", ""))
        cpu_count = int(diagnostics.get("cpu_count", 0))
        memory_bytes = int(diagnostics.get("memory_bytes", 0))
        storage_bytes = int(diagnostics.get("storage_available_bytes", 0))
        cpu_model = str(diagnostics.get("cpu_model", "unknown"))
        gpu_processes = str(diagnostics.get("gpu_processes", ""))
        image_reference = str(diagnostics.get("image_reference", "unknown"))
        image_digest = str(diagnostics.get("image_digest", "unknown"))

        if architecture not in {"x86_64", "amd64"}:
            problems.append(f"Container architecture must be x86-64; got {architecture}")
        if gpu_name != contract["gpu_product_name"]:
            problems.append(f"GPU product must be {contract['gpu_product_name']}; got {gpu_name}")
        if memory_mib != int(contract["gpu_memory_mib"]):
            problems.append(
                f"GPU memory must be {contract['gpu_memory_mib']} MiB; got {memory_mib} MiB"
            )
        if driver_major < int(contract["minimum_driver_major"]):
            problems.append(
                f"Driver major must be at least {contract['minimum_driver_major']}; "
                f"got {driver_version}"
            )
        if ecc_state.casefold() != str(contract["ecc_state"]).casefold():
            problems.append(f"ECC must be {contract['ecc_state']}; got {ecc_state}")
        if not math.isclose(power_limit, float(contract["power_limit_watts"]), abs_tol=0.1):
            problems.append(
                f"Power limit must be {contract['power_limit_watts']} W; got {power_limit} W"
            )
        if not math.isclose(power_max, float(contract["power_max_limit_watts"]), abs_tol=0.1):
            problems.append(
                f"Board maximum must be {contract['power_max_limit_watts']} W; got {power_max} W"
            )
        if has_performance_clock_event(throttle_state):
            problems.append(
                f"A performance-limiting GPU clock event was detected: {throttle_state}"
            )
        if int(diagnostics.get("gpu_device_count", 0)) != 1:
            problems.append("Container must expose exactly one numbered NVIDIA GPU device")
        if cpu_count < int(contract["minimum_cpu_count"]):
            problems.append(f"At least {contract['minimum_cpu_count']} vCPUs are required")
        if memory_bytes < int(contract["minimum_ram_bytes"]):
            problems.append("At least 64 GiB container RAM is required")
        if storage_bytes < int(contract["minimum_storage_bytes"]):
            problems.append("At least 100 GiB free local storage is required")
        if gpu_processes:
            problems.append(f"Competing GPU processes: {gpu_processes}")
        if diagnostics.get("offline_guard_enforced") is not True:
            problems.append("Native offline socket guard is not enforced")
        if diagnostics.get("caches_read_only") is not True:
            problems.append("Benchmark caches are writable by the inference user")
        expected_repository = str(contract["carrier_image_repository"])
        if not image_reference.startswith(f"{expected_repository}@sha256:"):
            problems.append(f"Carrier image must be digest-pinned from {expected_repository}")
        if image_reference.rpartition("@")[2] != image_digest:
            problems.append("Carrier image reference and recorded digest do not match")
        source_revision = str(diagnostics.get("source_revision", "unknown"))
        if re.fullmatch(r"[0-9a-f]{40}", source_revision) is None:
            problems.append("Carrier image source revision must be a full Git commit hash")

        report: dict[str, str | int | float | bool] = {
            "profile_id": str(contract["profile_id"]),
            "profile": (
                "NVIDIA RTX 6000 Ada Generation 48 GB / 300 W / "
                f"driver >= {contract['minimum_driver_major']} / ECC {contract['ecc_state']}"
            ),
            "gpu_product_name": gpu_name,
            "gpu_memory_mib": memory_mib,
            "driver_version": driver_version,
            "ecc_state": ecc_state,
            "power_limit_watts": power_limit,
            "power_max_limit_watts": power_max,
            "throttle_state": throttle_state,
            "gpu_uuid": gpu_uuid,
            "cpu_count": cpu_count,
            "cpu_model": cpu_model,
            "memory_bytes": memory_bytes,
            "storage_available_bytes": storage_bytes,
            "container_architecture": architecture,
            "container_os": str(diagnostics.get("container_os", "unknown")),
            "host_kernel": str(diagnostics.get("host_kernel", "unknown")),
            "image_reference": image_reference,
            "image_digest": image_digest,
            "source_revision": source_revision,
            "offline_guard_enforced": bool(diagnostics.get("offline_guard_enforced")),
            "caches_read_only": bool(diagnostics.get("caches_read_only")),
            "competing_gpu_processes": gpu_processes,
        }
        provenance = str(diagnostics.get("cloud_provenance", "")) or os.environ.get(
            "PESTE_CLOUD_PROVENANCE"
        )
        if provenance:
            report["cloud_provenance"] = provenance
        if problems:
            LOGGER.error(
                "GPU doctor failed",
                extra={"host": self.host, "problems": problems, "profile": report},
            )
            raise RuntimeError("GPU doctor failed: " + "; ".join(problems))
        LOGGER.info("GPU doctor passed", extra={"host": self.host, "profile": report})
        return report

    def _run_remote_action(
        self,
        action: str,
        runtime: str,
        payload: dict[str, Any],
        *,
        check: bool = True,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return self.transport.run(
            [REMOTE_CONTROLLER, "_remote-action", "--action", action, "--runtime", runtime],
            input_text=json.dumps(payload, ensure_ascii=False, sort_keys=True),
            check=check,
            timeout=timeout,
        )

    def prepare_dataset(self, suite: SuiteSpec) -> None:
        environment = {}
        if token := os.environ.get("HF_TOKEN"):
            environment["HF_TOKEN"] = token
        self._run_remote_action(
            "dataset",
            "modern",
            {"suite": suite.suite_id, "environment": environment},
            timeout=None,
        )
        LOGGER.info(
            "Remote dataset is prepared",
            extra={"host": self.host, "suite": suite.suite_id},
        )

    def _prefetch(self, model: ModelSpec) -> None:
        environment = {}
        if token := os.environ.get("HF_TOKEN"):
            environment["HF_TOKEN"] = token
        self._run_remote_action(
            "prefetch",
            model.runtime.name,
            {"model": model.model_id, "environment": environment},
            timeout=None,
        )

    def smoke(self, suite: SuiteSpec, model: ModelSpec) -> None:
        profile = self.doctor()
        self._prefetch(model)
        completed = self._run_remote_action(
            "smoke",
            model.runtime.name,
            {
                "suite": suite.suite_id,
                "model": model.model_id,
                "hardware_profile_json": json.dumps(profile, sort_keys=True),
            },
            timeout=None,
        )
        LOGGER.info(
            "Remote real-adapter validation passed",
            extra={"host": self.host, "model": model.model_id, "remote_log": completed.stderr},
        )

    def profile_speed(self, suite: SuiteSpec, model: ModelSpec) -> dict[str, Any]:
        profile = self.doctor()
        self._prefetch(model)
        artifact = f"/results/profiles/profile-speed-{model.model_id}.json"
        self._run_remote_action(
            "profile-speed",
            model.runtime.name,
            {
                "suite": suite.suite_id,
                "model": model.model_id,
                "output": artifact,
                "hardware_profile_json": json.dumps(profile, sort_keys=True),
            },
            timeout=None,
        )
        completed = self.transport.run(["cat", artifact], timeout=60)
        try:
            return cast(dict[str, Any], json.loads(completed.stdout))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Remote speed profile returned invalid JSON: {error}") from error

    def _latest_resume(self, suite: SuiteSpec, model: ModelSpec) -> tuple[str, ResumeState] | None:
        results = self.root / "results" / suite.suite_id
        candidates: list[tuple[float, Path, RunBundle]] = []
        for path in results.glob("*/run.json"):
            bundle = RunBundle.model_validate_json(path.read_text(encoding="utf-8"))
            if bundle.model_id == model.model_id and bundle.status in {
                RunStatus.FAILED,
                RunStatus.KILLED,
            }:
                candidates.append((path.stat().st_mtime, path, bundle))
        if not candidates:
            return None
        _, path, bundle = max(candidates, key=lambda item: item[0])
        timing_path = path.parent / bundle.speed.timing_artifact
        completed_batches = 0
        completed_samples = 0
        if timing_path.exists():
            with timing_path.open(encoding="utf-8") as timing:
                for line in timing:
                    completed_batches += 1
                    completed_samples += len(json.loads(line)["records"])
        return bundle.run_id, ResumeState(
            completed_batches=completed_batches,
            completed_samples=completed_samples,
        )

    def run(self, suite: SuiteSpec, model: ModelSpec, *, resume: bool = False) -> RunBundle:
        profile = self.doctor()
        self._prefetch(model)
        resumed = self._latest_resume(suite, model) if resume else None
        if resume and resumed is None:
            raise ValueError(f"No resumable failed run exists for {model.model_id}")
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        run_id = resumed[0] if resumed else f"{suite.suite_id}-{model.model_id}-{timestamp}"
        request = RunRequest(
            schema_version=2,
            run_id=run_id,
            suite_id=suite.suite_id,
            suite_digest=spec_digest(suite),
            model_id=model.model_id,
            model_digest=spec_digest(model),
            seed=DEFAULT_SEED,
            dataset_cache=Path("/cache/dataset"),
            model_cache=Path("/cache/hf"),
            output_directory=Path("/results") / run_id,
            resume=resumed[1] if resumed else None,
        )
        completed = self._run_remote_action(
            "run",
            model.runtime.name,
            {
                "suite": suite.suite_id,
                "model": model.model_id,
                "run_request_json": request.model_dump_json(),
                "hardware_profile_json": json.dumps(profile, sort_keys=True),
            },
            check=False,
            timeout=None,
        )
        destination = self.root / "results" / suite.suite_id / run_id
        copy_error: RuntimeError | None = None
        try:
            self.transport.copy_directory(PurePosixPath("/results") / run_id, destination.parent)
        except RuntimeError as error:
            copy_error = error
            LOGGER.exception(
                "Failed to copy remote result bundle",
                extra={"host": self.host, "run": run_id},
            )
            destination.mkdir(parents=True, exist_ok=True)
        with (destination / "container.jsonl").open("a", encoding="utf-8") as container_log:
            container_log.write(completed.stderr)
        bundle_path = destination / "run.json"
        if not bundle_path.exists():
            status = RunStatus.KILLED if completed.returncode == 137 else RunStatus.FAILED
            (destination / "timing.jsonl").touch(exist_ok=True)
            (destination / "predictions.jsonl").touch(exist_ok=True)
            wait_result: dict[str, Any] = {
                "returncode": completed.returncode,
                "transport": "ssh",
                "copy_error": None if copy_error is None else str(copy_error),
            }
            (destination / "diagnostics.json").write_bytes(
                canonical_json({"remote_command": wait_result, "status": status.value})
            )
            bundle = self._external_failure(request, suite, model, profile, status, wait_result)
            bundle_path.write_bytes(canonical_json(bundle.model_dump(mode="json")))
        return RunBundle.model_validate_json(bundle_path.read_text(encoding="utf-8"))

    @staticmethod
    def _external_failure(
        request: RunRequest,
        suite: SuiteSpec,
        model: ModelSpec,
        profile: dict[str, str | int | float | bool],
        status: RunStatus,
        wait_result: dict[str, Any],
    ) -> RunBundle:
        return RunBundle(
            schema_version=2,
            run_id=request.run_id,
            suite_id=suite.suite_id,
            suite_digest=request.suite_digest,
            model_id=model.model_id,
            model_digest=request.model_digest,
            status=status,
            environment=EnvironmentFingerprint(
                peste_revision=str(profile.get("source_revision", "unknown-remote-failure")),
                image_reference=str(profile.get("image_reference", model.runtime.image)),
                image_digest=str(profile.get("image_digest", "unknown")),
                dependency_versions={},
                python_version="unknown",
                pytorch_version="unknown",
                cuda_version="unknown",
                hardware_profile=profile,
                gpu_product_name=str(profile.get("gpu_product_name", "unknown")),
                driver_version=str(profile.get("driver_version", "unknown")),
                ecc_state=str(profile.get("ecc_state", "unknown")),
                power_limit_watts=float(profile.get("power_limit_watts", 0)),
                cpu_model=str(profile.get("cpu_model", "unknown")),
                gpu_uuid=str(profile.get("gpu_uuid", "unknown")),
                cloud_provenance=(
                    str(profile["cloud_provenance"]) if "cloud_provenance" in profile else None
                ),
                seed=request.seed,
            ),
            speed=SpeedStatistics(
                valid=False,
                batch_size=model.speed_profile.batch_size,
                warmup_batches=2,
                measured_batches=0,
                total_audio_seconds=0,
                processing_seconds=0,
                audio_throughput_x=0,
                rtf=0,
                timing_artifact="timing.jsonl",
                invalidity_reason="Remote process exited before producing a result bundle",
            ),
            model_facts=ModelFacts(
                checkpoint_bytes=0,
                parameter_count=0,
                native_dtype=model.native_dtype,
            ),
            predictions_path="predictions.jsonl",
            aggregates=None,
            logs=LogReferences(
                runner="runner.jsonl", container="container.jsonl", diagnostics="diagnostics.json"
            ),
            error=f"Remote process exited without a bundle: {wait_result}",
        )
