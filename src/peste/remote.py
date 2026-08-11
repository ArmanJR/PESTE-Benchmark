"""Remote control and diagnostics inside the digest-pinned carrier container."""

import json
import logging
import os
import platform
import pwd
import shutil
import socket
import stat
import subprocess
from pathlib import Path
from typing import Any, Literal, cast

from peste.schemas import RunRequest

LOGGER = logging.getLogger(__name__)
RuntimeName = Literal["modern", "nemo"]
ActionName = Literal["dataset", "prefetch", "smoke", "profile-speed", "run"]
OFFLINE_GUARD = Path("/opt/peste/lib/libpeste_offline.so")
CACHE_ROOTS = (Path("/cache/dataset"), Path("/cache/hf"))
OFFLINE_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
ALLOWED_PAYLOAD_ENVIRONMENT = frozenset({"HF_TOKEN"})
SENSITIVE_INHERITED_ENVIRONMENT = frozenset(
    {"CONTAINER_API_KEY", "GITHUB_TOKEN", "HF_TOKEN", "VAST_API_KEY"}
)


def _child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for key in SENSITIVE_INHERITED_ENVIRONMENT:
        environment.pop(key, None)
    return environment


def _read_instance_environment() -> dict[str, str]:
    selected = {
        key: value
        for key, value in os.environ.items()
        if key
        in {
            "CONTAINER_ID",
            "PESTE_IMAGE_DIGEST",
            "PESTE_IMAGE_REFERENCE",
            "PESTE_SOURCE_REVISION",
            "VAST_CONTAINERLABEL",
        }
    }
    environment_path = Path("/etc/environment")
    if environment_path.is_file():
        for line in environment_path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and key in {
                "CONTAINER_ID",
                "PESTE_IMAGE_DIGEST",
                "PESTE_IMAGE_REFERENCE",
                "PESTE_SOURCE_REVISION",
                "VAST_CONTAINERLABEL",
            }:
                selected.setdefault(key, value.strip().strip('"'))
    return selected


def image_metadata() -> dict[str, str]:
    """Return non-secret image and Vast provenance values available to SSH sessions."""
    values = _read_instance_environment()
    return {
        "image_reference": values.get("PESTE_IMAGE_REFERENCE", "unknown"),
        "image_digest": values.get("PESTE_IMAGE_DIGEST", "unknown"),
        "source_revision": values.get("PESTE_SOURCE_REVISION", "unknown"),
        "container_id": values.get("CONTAINER_ID", "unknown"),
        "container_label": values.get("VAST_CONTAINERLABEL", "unknown"),
    }


def _runtime_executable(runtime: RuntimeName) -> Path:
    executable = Path("/opt/venvs") / runtime / "bin" / "peste"
    if not executable.is_file():
        raise RuntimeError(f"Runtime executable is missing: {executable}")
    return executable


def _walk_tree(root: Path) -> list[Path]:
    if not root.exists():
        raise RuntimeError(f"Required cache path does not exist: {root}")
    return [root, *root.rglob("*")]


def _seal_cache(root: Path) -> None:
    paths = _walk_tree(root)
    for path in paths:
        try:
            os.chown(path, 0, 0, follow_symlinks=False)
        except (NotImplementedError, PermissionError) as error:
            raise RuntimeError(f"Unable to assign cache ownership for {path}: {error}") from error
        if path.is_symlink():
            continue
        mode = stat.S_IMODE(path.stat().st_mode) & ~(stat.S_IWGRP | stat.S_IWOTH)
        path.chmod(mode)
    LOGGER.info("Sealed cache for unprivileged offline access", extra={"path": str(root)})


def _assert_cache_is_read_only(root: Path) -> None:
    for path in _walk_tree(root):
        metadata = path.lstat()
        if metadata.st_uid != 0:
            raise RuntimeError(f"Cache entry is not root-owned: {path}")
        if not path.is_symlink() and stat.S_IMODE(metadata.st_mode) & (stat.S_IWGRP | stat.S_IWOTH):
            raise RuntimeError(f"Cache entry is writable by the benchmark user: {path}")


def _prepare_result_path(path: Path) -> None:
    if not path.is_absolute() or not path.is_relative_to(Path("/results")):
        raise ValueError(f"Remote result path must be below /results: {path}")
    path.mkdir(parents=True, exist_ok=True)
    account = pwd.getpwnam("peste")
    for candidate in [path, *path.rglob("*")]:
        os.chown(candidate, account.pw_uid, account.pw_gid, follow_symlinks=False)


def _command_for_action(
    action: ActionName, payload: dict[str, Any]
) -> tuple[list[str], Path | None, Path | None]:
    suite = str(payload.get("suite", "fleurs-fa-ir-v1"))
    model = str(payload.get("model", ""))
    if action == "dataset":
        return ["_dataset-container", "--suite", suite], Path("/cache/dataset"), None
    if not model:
        raise ValueError(f"Remote action {action} requires a model id")
    if action == "prefetch":
        return ["_prefetch-container", "--model", model], Path("/cache/hf"), None
    if action == "smoke":
        return ["_smoke-container", "--suite", suite, "--model", model], None, None
    if action == "profile-speed":
        output = Path(str(payload["output"]))
        _prepare_result_path(output.parent)
        return (
            [
                "_profile-speed-container",
                "--suite",
                suite,
                "--model",
                model,
                "--output",
                str(output),
            ],
            None,
            output,
        )
    request = RunRequest.model_validate_json(str(payload["run_request_json"]))
    _prepare_result_path(request.output_directory)
    return ["_run-container"], None, request.output_directory


def execute_action(action: ActionName, runtime: RuntimeName, payload_text: str) -> int:
    """Execute one constrained orchestration action from a JSON stdin envelope."""
    if os.geteuid() != 0:
        raise RuntimeError("Remote control actions must start as root")
    try:
        payload = cast(dict[str, Any], json.loads(payload_text or "{}"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Remote action payload is not valid JSON: {error}") from error
    raw_environment = payload.get("environment", {})
    if not isinstance(raw_environment, dict):
        raise ValueError("Remote action environment must be a JSON object")
    unknown_environment = set(raw_environment) - ALLOWED_PAYLOAD_ENVIRONMENT
    if unknown_environment:
        raise ValueError(
            f"Remote action contains unsupported environment keys: {unknown_environment}"
        )

    arguments, cache_to_seal, result_path = _command_for_action(action, payload)
    environment = _child_environment()
    environment.update({str(key): str(value) for key, value in raw_environment.items()})
    offline = action in {"smoke", "profile-speed", "run"}
    if offline:
        if not OFFLINE_GUARD.is_file():
            raise RuntimeError(f"Native offline guard is missing: {OFFLINE_GUARD}")
        for root in CACHE_ROOTS:
            _assert_cache_is_read_only(root)
        environment.update(OFFLINE_ENVIRONMENT)
        environment["LD_PRELOAD"] = str(OFFLINE_GUARD)
        for key, value in _read_instance_environment().items():
            if key.startswith("PESTE_"):
                environment[key] = value
        environment["PESTE_HARDWARE_PROFILE_JSON"] = str(payload["hardware_profile_json"])
        if action == "run":
            environment["PESTE_RUN_REQUEST_JSON"] = str(payload["run_request_json"])

    executable = _runtime_executable(runtime)
    command = [str(executable), *arguments]
    LOGGER.info(
        "Starting constrained remote action",
        extra={"action": action, "runtime": runtime, "offline": offline},
    )
    run_options: dict[str, Any] = {
        "check": False,
        "env": environment,
    }
    if offline:
        account = pwd.getpwnam("peste")
        environment["HOME"] = account.pw_dir
        environment["USER"] = account.pw_name
        environment["LOGNAME"] = account.pw_name
        run_options.update(
            {
                "user": account.pw_uid,
                "group": account.pw_gid,
                "extra_groups": (),
                "umask": 0o022,
                "cwd": account.pw_dir,
            }
        )
    completed = subprocess.run(command, **run_options)
    if completed.returncode == 0 and cache_to_seal is not None:
        _seal_cache(cache_to_seal)
    LOGGER.info(
        "Remote action finished",
        extra={
            "action": action,
            "runtime": runtime,
            "returncode": completed.returncode,
            "result_path": None if result_path is None else str(result_path),
        },
    )
    return completed.returncode


def _memory_limit_bytes() -> int:
    values: list[int] = []
    for path in (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        if not path.is_file():
            continue
        raw = path.read_text(encoding="utf-8").strip()
        if raw.isdigit():
            value = int(raw)
            if value < 1 << 60:
                values.append(value)
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            values.append(int(line.split()[1]) * 1024)
            break
    if not values:
        raise RuntimeError("Unable to determine container memory allocation")
    return min(values)


def _os_release() -> str:
    values: dict[str, str] = {}
    for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value.strip().strip('"')
    return f"{values.get('ID', 'unknown')} {values.get('VERSION_ID', 'unknown')}"


def _allocated_cpu_count() -> int:
    affinity_reader = getattr(os, "sched_getaffinity", None)
    if affinity_reader is not None:
        return len(affinity_reader(0))
    return os.cpu_count() or 0


def _run_probe_as_benchmark_user() -> dict[str, bool]:
    account = pwd.getpwnam("peste")
    environment = _child_environment()
    environment["LD_PRELOAD"] = str(OFFLINE_GUARD)
    command = [str(_runtime_executable("modern")), "_offline-probe"]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        user=account.pw_uid,
        group=account.pw_gid,
        extra_groups=(),
        timeout=30,
    )
    return cast(dict[str, bool], json.loads(completed.stdout))


def collect_diagnostics() -> dict[str, Any]:
    """Collect host and container contract facts without requiring a Docker daemon."""
    gpu_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,ecc.mode.current,power.limit,"
            "power.max_limit,uuid,clocks_event_reasons.active",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    process_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    device_count = sum(
        1
        for path in Path("/dev").glob("nvidia[0-9]*")
        if path.name.removeprefix("nvidia").isdigit()
    )
    metadata = image_metadata()
    provenance = " ".join(
        value
        for value in (
            (
                f"CONTAINER_ID={metadata['container_id']}"
                if metadata["container_id"] != "unknown"
                else ""
            ),
            (
                f"VAST_CONTAINERLABEL={metadata['container_label']}"
                if metadata["container_label"] != "unknown"
                else ""
            ),
        )
        if value
    )
    offline = _run_probe_as_benchmark_user()
    return {
        "gpu_rows": [line.strip() for line in gpu_query.stdout.splitlines() if line.strip()],
        "gpu_processes": process_query.stdout.strip(),
        "gpu_device_count": device_count,
        "architecture": platform.machine(),
        "cpu_count": _allocated_cpu_count(),
        "cpu_model": next(
            (
                line.partition(":")[2].strip()
                for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
                if line.startswith("model name")
            ),
            "unknown",
        ),
        "memory_bytes": _memory_limit_bytes(),
        "storage_available_bytes": shutil.disk_usage("/").free,
        "container_os": _os_release(),
        "host_kernel": platform.release(),
        "cloud_provenance": provenance,
        "offline_guard_enforced": offline["network_denied"],
        "caches_read_only": offline["cache_writes_denied"],
        **metadata,
    }


def offline_probe() -> dict[str, bool]:
    """Prove that the benchmark user cannot open network sockets or write caches."""
    network_denied = False
    try:
        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except PermissionError:
        network_denied = True
    else:
        connection.close()

    cache_writes_denied = True
    for index, root in enumerate(CACHE_ROOTS):
        probe = root / f".peste-write-probe-{os.getpid()}-{index}"
        try:
            probe.write_text("probe", encoding="utf-8")
        except PermissionError:
            continue
        else:
            cache_writes_denied = False
            probe.unlink(missing_ok=True)
    return {"network_denied": network_denied, "cache_writes_denied": cache_writes_denied}
