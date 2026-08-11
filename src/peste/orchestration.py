"""Docker-over-SSH orchestration for doctor-approved discrete-GPU hosts."""

import json
import logging
import math
import os
import subprocess
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, cast
from urllib.parse import urlparse

import docker
from docker.models.containers import Container

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
BASE_IMAGE = (
    "nvcr.io/nvidia/pytorch@sha256:3cb18e2c438db8af2d3a659ca27fac5da328640261c38c48a34edcd223c38af9"
)
HF_VOLUME = "peste-huggingface-cache-v1"
OFFLINE_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
GPU_DEVICE_REQUESTS = [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])]


def load_hardware_profile(path: Path = HARDWARE_PROFILE_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as profile_file:
        profile = cast(dict[str, Any], json.load(profile_file))
    if profile.get("profile_id") != "rtx-6000-ada-v1":
        raise ValueError(f"Unsupported hardware profile in {path}")
    return profile


def _dataset_volume(suite: SuiteSpec) -> str:
    return f"peste-{suite.suite_id}-audio"


def _result_volume(suite: SuiteSpec) -> str:
    return f"peste-{suite.suite_id}-results"


def _marker(output: str, name: str) -> str:
    prefix = f"__{name}__="
    for line in output.splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix).strip()
    raise ValueError(f"Host diagnostics omitted {prefix}")


def _optional_marker(output: str, name: str) -> str | None:
    try:
        value = _marker(output, name)
    except ValueError:
        return None
    return value or None


def _gpu_container_names(containers: list[Any]) -> list[str]:
    names: list[str] = []
    for container in containers:
        host_config = container.attrs.get("HostConfig", {})
        if host_config.get("Runtime") == "nvidia" or host_config.get("DeviceRequests"):
            names.append(str(container.name))
    return names


class GpuOrchestrator:
    def __init__(self, host: str, root: Path = PROJECT_ROOT) -> None:
        if not host.startswith("ssh://"):
            raise ValueError("GPU host must use an ssh:// Docker endpoint")
        self.host = host
        self.root = root
        self.client = docker.DockerClient(base_url=host, use_ssh_client=True)

    def doctor(self) -> dict[str, str | int | float | bool]:
        contract = load_hardware_profile(self.root / "hardware" / "rtx-6000-ada-v1.json")
        info = self.client.info()
        host_diagnostics = self._host_diagnostics()
        running_gpu_containers = _gpu_container_names(self.client.containers.list())
        visible_gpu_output = self.client.containers.run(
            BASE_IMAGE,
            [
                "bash",
                "-lc",
                "nvidia-smi --query-gpu=name --format=csv,noheader,nounits",
            ],
            remove=True,
            device_requests=GPU_DEVICE_REQUESTS,
            network_disabled=True,
        ).decode("utf-8", errors="replace")

        gpu_rows = [row.strip() for row in _marker(host_diagnostics, "GPUS").split(";") if row]
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
        except ValueError:
            memory_mib = 0
            power_limit = 0.0
            power_max = 0.0
            problems.append("GPU numeric diagnostics could not be parsed")

        architecture = str(info.get("Architecture", ""))
        cpu_count = int(info.get("NCPU", _marker(host_diagnostics, "CPU_COUNT")))
        memory_bytes = int(info.get("MemTotal", _marker(host_diagnostics, "MEM_BYTES")))
        storage_bytes = int(_marker(host_diagnostics, "STORAGE_BYTES"))
        cpu_model = _marker(host_diagnostics, "CPU_MODEL")
        gpu_processes = _marker(host_diagnostics, "GPU_PROCESSES")
        visible_names = [line.strip() for line in visible_gpu_output.splitlines() if line.strip()]

        if architecture not in {"x86_64", "amd64"}:
            problems.append(f"Docker architecture must be x86-64; got {architecture}")
        if gpu_name != contract["gpu_product_name"]:
            problems.append(f"GPU product must be {contract['gpu_product_name']}; got {gpu_name}")
        if memory_mib != int(contract["gpu_memory_mib"]):
            problems.append(
                f"GPU memory must be {contract['gpu_memory_mib']} MiB; got {memory_mib} MiB"
            )
        if driver_version != contract["driver_version"]:
            problems.append(f"Driver must be {contract['driver_version']}; got {driver_version}")
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
        if throttle_state.casefold() not in {
            "0x0000000000000000",
            "0x00000000",
            "not active",
            "none",
        }:
            problems.append(f"Active GPU throttling was detected: {throttle_state}")
        if cpu_count < int(contract["minimum_cpu_count"]):
            problems.append(f"At least {contract['minimum_cpu_count']} vCPUs are required")
        if memory_bytes < int(contract["minimum_ram_bytes"]):
            problems.append("At least 64 GiB host RAM is required")
        if storage_bytes < int(contract["minimum_storage_bytes"]):
            problems.append("At least 100 GiB free local storage is required")
        if gpu_processes:
            problems.append(f"Competing host GPU processes: {gpu_processes}")
        if running_gpu_containers:
            problems.append(f"Competing GPU containers: {', '.join(running_gpu_containers)}")
        if visible_names != [str(contract["gpu_product_name"])]:
            problems.append(f"Container GPU visibility mismatch: {visible_names}")

        report: dict[str, str | int | float | bool] = {
            "profile_id": str(contract["profile_id"]),
            "profile": (
                "NVIDIA RTX 6000 Ada Generation 48 GB / 300 W / "
                f"driver {contract['driver_version']} / ECC {contract['ecc_state']}"
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
            "docker_architecture": architecture,
            "docker_os": str(info.get("OperatingSystem", "")),
            "competing_gpu_processes": gpu_processes,
            "competing_gpu_containers": ",".join(running_gpu_containers),
            "vm_os": _marker(host_diagnostics, "OS"),
        }
        provenance = os.environ.get("PESTE_CLOUD_PROVENANCE") or _optional_marker(
            host_diagnostics, "CLOUD_PROVENANCE"
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

    def _host_diagnostics(self) -> str:
        endpoint = urlparse(self.host)
        if endpoint.hostname is None:
            raise ValueError(f"Invalid Docker SSH endpoint: {self.host}")
        target = endpoint.hostname
        if endpoint.username:
            target = f"{endpoint.username}@{target}"
        command = ["ssh", "-o", "BatchMode=yes"]
        if endpoint.port:
            command.extend(["-p", str(endpoint.port)])
        remote_command = (
            "set -eu; "
            "printf '__GPUS__='; "
            "nvidia-smi --query-gpu=name,memory.total,driver_version,ecc.mode.current,"
            "power.limit,power.max_limit,uuid,clocks_event_reasons.active "
            "--format=csv,noheader,nounits | paste -sd ';' -; printf '\\n'; "
            "printf '__GPU_PROCESSES__='; "
            "nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits "
            "2>/dev/null | paste -sd ';' - || true; printf '\\n'; "
            "printf '__CPU_COUNT__='; getconf _NPROCESSORS_ONLN; "
            "printf '__CPU_MODEL__='; awk -F: '/model name/{sub(/^ /,\"\",$2); print $2; exit}' "
            "/proc/cpuinfo; "
            "printf '__MEM_BYTES__='; awk '/MemTotal/{print $2 * 1024}' /proc/meminfo; "
            "printf '__STORAGE_BYTES__='; df -PB1 / | awk 'END{print $4}'; "
            "printf '__OS__='; . /etc/os-release; printf '%s %s\\n' \"$ID\" \"$VERSION_ID\"; "
            "printf '__CLOUD_PROVENANCE__='; "
            "awk -F= '/^(CONTAINER_ID|VAST_CONTAINERLABEL|VAST_INSTANCE_ID)=/"
            '{printf "%s=%s ",$1,$2}\' '
            "/etc/environment 2>/dev/null || true; printf '\\n'"
        )
        command.extend([target, remote_command])
        try:
            return subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout
        except (subprocess.SubprocessError, FileNotFoundError) as error:
            raise RuntimeError(
                f"Unable to collect non-interactive host diagnostics: {error}"
            ) from error

    def _run_checked(
        self,
        image: str,
        command: list[str],
        *,
        network_disabled: bool,
        volumes: dict[str, dict[str, str]],
        environment: dict[str, str] | None = None,
    ) -> bytes:
        LOGGER.info(
            "Starting remote container",
            extra={"host": self.host, "image": image, "network_disabled": network_disabled},
        )
        return cast(
            bytes,
            self.client.containers.run(
                image,
                command,
                remove=True,
                device_requests=GPU_DEVICE_REQUESTS,
                network_disabled=network_disabled,
                volumes=volumes,
                environment=environment,
            ),
        )

    def build_images(self) -> None:
        for runtime in ("modern", "nemo"):
            tag = f"peste-{runtime}:2.0.0"
            LOGGER.info("Building remote runtime image", extra={"host": self.host, "image": tag})
            image, build_logs = self.client.images.build(
                path=str(self.root),
                dockerfile=f"runtimes/{runtime}/Dockerfile",
                tag=tag,
                rm=True,
            )
            for entry in build_logs:
                message = entry.get("stream") or entry.get("error")
                if message and message.strip():
                    LOGGER.debug(
                        "Remote image build output",
                        extra={"image": tag, "output": message.strip()},
                    )
            LOGGER.info(
                "Built remote runtime image",
                extra={"host": self.host, "image": tag, "digest": image.id},
            )

    def prepare_dataset(self, suite: SuiteSpec) -> None:
        environment = {}
        if token := os.environ.get("HF_TOKEN"):
            environment["HF_TOKEN"] = token
        self._run_checked(
            "peste-modern:2.0.0",
            ["peste", "_dataset-container", "--suite", suite.suite_id],
            network_disabled=False,
            volumes={_dataset_volume(suite): {"bind": "/cache/dataset", "mode": "rw"}},
            environment=environment,
        )

    def _prefetch(self, suite: SuiteSpec, model: ModelSpec) -> None:
        del suite
        environment = {}
        if token := os.environ.get("HF_TOKEN"):
            environment["HF_TOKEN"] = token
        self._run_checked(
            model.runtime.image,
            ["peste", "_prefetch-container", "--model", model.model_id],
            network_disabled=False,
            volumes={HF_VOLUME: {"bind": "/cache/hf", "mode": "rw"}},
            environment=environment,
        )

    def smoke(self, suite: SuiteSpec, model: ModelSpec) -> None:
        profile = self.doctor()
        self._prefetch(suite, model)
        output = self._run_checked(
            model.runtime.image,
            [
                "peste",
                "_smoke-container",
                "--suite",
                suite.suite_id,
                "--model",
                model.model_id,
            ],
            network_disabled=True,
            volumes={
                HF_VOLUME: {"bind": "/cache/hf", "mode": "ro"},
                _dataset_volume(suite): {"bind": "/cache/dataset", "mode": "ro"},
            },
            environment={
                **OFFLINE_ENVIRONMENT,
                "PESTE_HARDWARE_PROFILE_JSON": json.dumps(profile, sort_keys=True),
            },
        )
        LOGGER.info(
            "Remote real-adapter validation passed",
            extra={"host": self.host, "model": model.model_id, "container_log": output.decode()},
        )

    def profile_speed(self, suite: SuiteSpec, model: ModelSpec) -> dict[str, Any]:
        profile = self.doctor()
        self._prefetch(suite, model)
        artifact = f"profile-speed-{model.model_id}.json"
        self._run_checked(
            model.runtime.image,
            [
                "peste",
                "_profile-speed-container",
                "--suite",
                suite.suite_id,
                "--model",
                model.model_id,
                "--output",
                f"/results/{artifact}",
            ],
            network_disabled=True,
            volumes={
                HF_VOLUME: {"bind": "/cache/hf", "mode": "ro"},
                _dataset_volume(suite): {"bind": "/cache/dataset", "mode": "ro"},
                _result_volume(suite): {"bind": "/results", "mode": "rw"},
            },
            environment={
                **OFFLINE_ENVIRONMENT,
                "PESTE_HARDWARE_PROFILE_JSON": json.dumps(profile, sort_keys=True),
            },
        )
        helper: Container = self.client.containers.create(
            BASE_IMAGE,
            ["true"],
            volumes={_result_volume(suite): {"bind": "/results", "mode": "ro"}},
        )
        try:
            with tempfile.TemporaryDirectory() as temporary:
                destination = Path(temporary)
                self._copy_archive(helper, PurePosixPath("/results") / artifact, destination)
                return cast(
                    dict[str, Any],
                    json.loads((destination / artifact).read_text(encoding="utf-8")),
                )
        finally:
            helper.remove(force=True)

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
        self._prefetch(suite, model)
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
        image = self.client.images.get(model.runtime.image)
        image_digest = image.attrs.get("Id", "unknown")
        environment = {
            **OFFLINE_ENVIRONMENT,
            "PESTE_RUN_REQUEST_JSON": request.model_dump_json(),
            "PESTE_HARDWARE_PROFILE_JSON": json.dumps(profile, sort_keys=True),
            "PESTE_IMAGE_REFERENCE": model.runtime.image,
            "PESTE_IMAGE_DIGEST": str(image_digest),
            "PESTE_SOURCE_REVISION": self._local_source_revision(),
        }
        container: Container = self.client.containers.run(
            model.runtime.image,
            ["peste", "_run-container"],
            detach=True,
            remove=False,
            device_requests=GPU_DEVICE_REQUESTS,
            network_disabled=True,
            volumes={
                HF_VOLUME: {"bind": "/cache/hf", "mode": "ro"},
                _dataset_volume(suite): {"bind": "/cache/dataset", "mode": "ro"},
                _result_volume(suite): {"bind": "/results", "mode": "rw"},
            },
            environment=environment,
        )
        wait_result = container.wait()
        logs = container.logs(stdout=True, stderr=True)
        destination = self.root / "results" / suite.suite_id / run_id
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / "container.jsonl").open("ab") as container_log:
            container_log.write(logs)
        try:
            self._copy_archive(container, PurePosixPath("/results") / run_id, destination.parent)
        finally:
            container.remove(force=True)
        bundle_path = destination / "run.json"
        if not bundle_path.exists():
            status = RunStatus.KILLED if wait_result.get("StatusCode") == 137 else RunStatus.FAILED
            (destination / "timing.jsonl").touch(exist_ok=True)
            (destination / "predictions.jsonl").touch(exist_ok=True)
            (destination / "diagnostics.json").write_bytes(
                canonical_json({"container_wait": wait_result, "status": status.value})
            )
            bundle = self._external_failure(request, suite, model, profile, status, wait_result)
            bundle_path.write_bytes(canonical_json(bundle.model_dump(mode="json")))
        return RunBundle.model_validate_json(bundle_path.read_text(encoding="utf-8"))

    def _local_source_revision(self) -> str:
        try:
            source_status = subprocess.run(
                [
                    "git",
                    "status",
                    "--porcelain",
                    "--untracked-files=normal",
                    "--",
                    "src",
                    "models",
                    "suites",
                    "runtimes",
                    "hardware",
                    "pyproject.toml",
                ],
                cwd=self.root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if source_status.stdout.strip():
                LOGGER.warning("Benchmark source is uncommitted", extra={"root": str(self.root)})
                return "uncommitted"
            return subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            return "uncommitted"

    @staticmethod
    def _copy_archive(container: Container, source: PurePosixPath, destination: Path) -> None:
        stream, _ = container.get_archive(str(source))
        with tempfile.NamedTemporaryFile(suffix=".tar") as archive_file:
            for chunk in stream:
                archive_file.write(chunk)
            archive_file.flush()
            with tarfile.open(archive_file.name) as archive:
                archive.extractall(destination, filter="data")

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
                peste_revision="unknown-container-killed",
                image_reference=model.runtime.image,
                image_digest="unknown",
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
                invalidity_reason="Container exited before producing a result bundle",
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
            error=f"Container exited without a bundle: {wait_result}",
        )
