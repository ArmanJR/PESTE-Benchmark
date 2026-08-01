"""Docker-over-SSH orchestration for the official Jetson host."""

import json
import logging
import os
import re
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
    MemoryStatistics,
    ModelSpec,
    ResumeState,
    RunBundle,
    RunRequest,
    RunStatus,
    SuiteSpec,
)
from peste.specs import spec_digest

LOGGER = logging.getLogger(__name__)
BASE_IMAGE = (
    "nvcr.io/nvidia/pytorch@sha256:90f3c17838fde28d5c7ae2d5bfbc8a4c587d3797767ea96cdd48fe82e3613f3b"
)
HF_VOLUME = "peste-huggingface-cache-v1"
OFFLINE_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}


def _dataset_volume(suite: SuiteSpec) -> str:
    return f"peste-{suite.suite_id}-audio"


def _result_volume(suite: SuiteSpec) -> str:
    return f"peste-{suite.suite_id}-results"


class JetsonOrchestrator:
    def __init__(self, host: str, root: Path = PROJECT_ROOT) -> None:
        if not host.startswith("ssh://"):
            raise ValueError("Jetson host must use an ssh:// Docker endpoint")
        self.host = host
        self.root = root
        self.client = docker.DockerClient(base_url=host, use_ssh_client=True)

    def doctor(self) -> dict[str, str | int | float | bool]:
        info = self.client.info()
        host_diagnostics = self._host_diagnostics()
        running_gpu_containers = [
            container.name
            for container in self.client.containers.list()
            if container.attrs.get("HostConfig", {}).get("Runtime") == "nvidia"
        ]
        command = [
            "bash",
            "-lc",
            "set -euo pipefail; "
            "printf '__L4T__='; tr '\\n' ' ' < /host/etc/nv_tegra_release; echo; "
            "printf '__OS__='; "
            'awk -F= \'/^(DISTRIB_ID|DISTRIB_RELEASE)=/{gsub(/"/,""); '
            'printf "%s=%s ",$1,$2}\' /host/etc/lsb-release; echo; '
            "printf '__MODEL__='; tr -d '\\000' < /host/sys/firmware/devicetree/base/model; echo; "
            "printf '__MEM_KIB__='; awk '/MemTotal/{print $2}' /host/proc/meminfo; "
            "printf '__STORAGE_KIB__='; df -Pk /cache | awk 'END{print $4}'; "
            "printf '__IMAGE_TORCH_CUDA__='; "
            "python -c 'import sys,torch; "
            "sys.stdout.write(str(torch.version.cuda))'",
        ]
        output = self.client.containers.run(
            BASE_IMAGE,
            command,
            remove=True,
            runtime="nvidia",
            network_disabled=True,
            volumes={
                "/etc": {"bind": "/host/etc", "mode": "ro"},
                "/proc": {"bind": "/host/proc", "mode": "ro"},
                "/sys": {"bind": "/host/sys", "mode": "ro"},
                HF_VOLUME: {"bind": "/cache", "mode": "rw"},
            },
        ).decode("utf-8", errors="replace")
        report: dict[str, str | int | float | bool] = {
            "docker_architecture": str(info.get("Architecture", "")),
            "docker_os": str(info.get("OperatingSystem", "")),
            "docker_memory_bytes": int(info.get("MemTotal", 0)),
            "nvidia_runtime": "nvidia" in info.get("Runtimes", {}),
            "competing_gpu_containers": ",".join(running_gpu_containers),
            "raw_profile": output.strip(),
            "raw_host_diagnostics": host_diagnostics.strip(),
        }
        problems: list[str] = []
        if report["docker_architecture"] not in {"aarch64", "arm64"}:
            problems.append("Docker architecture is not ARM64")
        if not report["nvidia_runtime"]:
            problems.append("NVIDIA container runtime is unavailable")
        if "R36 (release), REVISION: 4.7" not in output:
            problems.append("L4T R36.4.7 was not detected")
        if "DISTRIB_RELEASE=22.04" not in output:
            problems.append("Ubuntu 22.04 was not detected")
        if not re.search(r"__HOST_CUDA__=12\.6(?:\.|\b)", host_diagnostics):
            problems.append("Host CUDA 12.6 was not detected")
        if "Jetson AGX Orin" not in output:
            problems.append("Jetson AGX Orin was not detected")
        memory_gib = int(report["docker_memory_bytes"]) / 1024**3
        if not 28 <= memory_gib <= 32:
            problems.append(f"Expected AGX Orin 32GB memory profile, got {memory_gib:.1f} GiB")
        if running_gpu_containers:
            problems.append(f"Competing NVIDIA containers: {', '.join(running_gpu_containers)}")
        maxn = bool(re.search(r"\bMAXN\b", host_diagnostics))
        report["maxn"] = maxn
        if not maxn:
            problems.append("MAXN power mode was not detected")
        gpu_process_match = re.search(r"__GPU_PIDS__=([^\n]*)", host_diagnostics)
        gpu_processes = gpu_process_match.group(1).strip() if gpu_process_match else "unknown"
        report["competing_gpu_processes"] = gpu_processes
        if gpu_processes == "unknown":
            problems.append("Could not verify competing host GPU processes")
        elif gpu_processes:
            problems.append(f"Competing host GPU processes: {gpu_processes}")
        storage_match = re.search(r"__STORAGE_KIB__=(\d+)", output)
        storage_available = int(storage_match.group(1)) * 1024 if storage_match else 0
        report["storage_available_bytes"] = storage_available
        minimum_storage = 60 * 1024**3
        if storage_available < minimum_storage:
            storage_gib = storage_available / 1024**3
            problems.append(
                f"At least 60 GiB free cache storage is required; found {storage_gib:.1f} GiB"
            )
        if problems:
            LOGGER.error(
                "Jetson doctor failed",
                extra={"host": self.host, "problems": problems, "profile": report},
            )
            raise RuntimeError("Jetson doctor failed: " + "; ".join(problems))
        report["profile"] = "Jetson AGX Orin 32GB / JetPack 6.2 / R36.4.7 / CUDA 12.6"
        LOGGER.info("Jetson doctor passed", extra={"host": self.host})
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
        command.extend(
            [
                target,
                "set -eu; command -v nvpmodel >/dev/null; command -v fuser >/dev/null; "
                "nvpmodel -q; printf '__GPU_PIDS__='; "
                "fuser /dev/nvhost-gpu 2>/dev/null || true; echo; "
                "printf '__HOST_CUDA__='; "
                "awk -F'\"' '/\"version\"/{print $4; exit}' /usr/local/cuda/version.json",
            ]
        )
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
                f"Unable to collect non-interactive Jetson host diagnostics: {error}"
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
                runtime="nvidia",
                network_disabled=network_disabled,
                volumes=volumes,
                environment=environment,
            ),
        )

    def prepare_dataset(self, suite: SuiteSpec) -> None:
        environment = {}
        if token := os.environ.get("HF_TOKEN"):
            environment["HF_TOKEN"] = token
        self._run_checked(
            "peste-modern:1.0.0",
            ["peste", "_dataset-container", "--suite", suite.suite_id],
            network_disabled=False,
            volumes={_dataset_volume(suite): {"bind": "/cache/dataset", "mode": "rw"}},
            environment=environment,
        )

    def _prefetch(self, suite: SuiteSpec, model: ModelSpec) -> None:
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
        prediction_path = path.parent / bundle.predictions_path
        completed = (
            sum(1 for _ in prediction_path.open(encoding="utf-8"))
            if prediction_path.exists()
            else 0
        )
        return (
            bundle.run_id,
            ResumeState(
                completed_samples=completed,
                peak_cuda_reserved_bytes=bundle.memory.peak_cuda_reserved_bytes,
                peak_cuda_allocated_bytes=bundle.memory.peak_cuda_allocated_bytes,
                peak_process_rss_bytes=bundle.memory.peak_process_rss_bytes,
            ),
        )

    def run(self, suite: SuiteSpec, model: ModelSpec, *, resume: bool = False) -> RunBundle:
        profile = self.doctor()
        self._prefetch(suite, model)
        resumed = self._latest_resume(suite, model) if resume else None
        if resume and resumed is None:
            raise ValueError(f"No resumable failed run exists for {model.model_id}")
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        run_id = resumed[0] if resumed else f"{suite.suite_id}-{model.model_id}-{timestamp}"
        resume_state = resumed[1] if resumed else None
        request = RunRequest(
            schema_version=1,
            run_id=run_id,
            suite_id=suite.suite_id,
            suite_digest=spec_digest(suite),
            model_id=model.model_id,
            model_digest=spec_digest(model),
            seed=DEFAULT_SEED,
            dataset_cache=Path("/cache/dataset"),
            model_cache=Path("/cache/hf"),
            output_directory=Path("/results") / run_id,
            resume=resume_state,
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
            runtime="nvidia",
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
            (destination / "diagnostics.json").write_bytes(
                canonical_json({"container_wait": wait_result, "status": status.value})
            )
            bundle = self._external_failure(request, suite, model, profile, status, wait_result)
            bundle_path.write_bytes(canonical_json(bundle.model_dump(mode="json")))
        return RunBundle.model_validate_json(bundle_path.read_text(encoding="utf-8"))

    def _local_source_revision(self) -> str:
        try:
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
            schema_version=1,
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
                seed=request.seed,
            ),
            memory=MemoryStatistics(
                peak_cuda_reserved_bytes=0,
                peak_cuda_allocated_bytes=0,
                peak_process_rss_bytes=0,
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
