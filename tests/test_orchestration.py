"""SSH transport and hardware-doctor tests without a remote GPU."""

import io
import json
import logging
import subprocess
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from peste.constants import PROJECT_ROOT
from peste.orchestration import GpuOrchestrator, SshTransport

IMAGE = f"ghcr.io/armanjr/peste-benchmark@sha256:{'a' * 64}"


def _diagnostics(
    *,
    gpu: str = "NVIDIA RTX 6000 Ada Generation",
    memory_mib: int = 49140,
    driver: str = "580.142",
    ecc: str = "Disabled",
    power: float = 300,
    power_max: float = 300,
    throttle: str = "0x0000000000000000",
    processes: str = "",
    gpu_rows: int = 1,
    device_count: int = 1,
    cpu_count: int = 8,
    memory_bytes: int = 64 * 1024**3,
    storage_bytes: int = 101 * 1024**3,
    offline: bool = True,
    caches_read_only: bool = True,
) -> dict[str, Any]:
    row = f"{gpu}, {memory_mib}, {driver}, {ecc}, {power}, {power_max}, GPU-id, {throttle}"
    return {
        "gpu_rows": [row] * gpu_rows,
        "gpu_processes": processes,
        "gpu_device_count": device_count,
        "architecture": "x86_64",
        "cpu_count": cpu_count,
        "cpu_model": "Test CPU",
        "memory_bytes": memory_bytes,
        "storage_available_bytes": storage_bytes,
        "container_os": "ubuntu 24.04",
        "host_kernel": "6.8.0",
        "cloud_provenance": "CONTAINER_ID=123",
        "offline_guard_enforced": offline,
        "caches_read_only": caches_read_only,
        "image_reference": IMAGE,
        "image_digest": f"sha256:{'a' * 64}",
        "source_revision": "b" * 40,
    }


class DiagnosticTransport:
    def __init__(self, diagnostics: dict[str, Any]) -> None:
        self.diagnostics = diagnostics

    def run(self, arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            arguments,
            0,
            stdout=json.dumps(self.diagnostics),
            stderr="",
        )


def _orchestrator(diagnostics: dict[str, Any]) -> GpuOrchestrator:
    orchestrator = object.__new__(GpuOrchestrator)
    orchestrator.host = "ssh://root@test:22"
    orchestrator.root = PROJECT_ROOT
    orchestrator.transport = DiagnosticTransport(diagnostics)  # type: ignore[assignment]
    return orchestrator


def test_doctor_accepts_exact_rtx_profile_and_digest_pinned_container() -> None:
    report = _orchestrator(_diagnostics()).doctor()
    assert report["profile_id"] == "rtx-6000-ada-v1"
    assert report["gpu_product_name"] == "NVIDIA RTX 6000 Ada Generation"
    assert report["driver_version"] == "580.142"
    assert report["offline_guard_enforced"] is True
    assert report["image_reference"] == IMAGE


def test_doctor_accepts_benign_gpu_idle_clock_event() -> None:
    report = _orchestrator(_diagnostics(throttle="0x0000000000000001")).doctor()
    assert report["throttle_state"] == "0x0000000000000001"


@pytest.mark.parametrize(
    ("diagnostics", "message"),
    [
        (_diagnostics(gpu="NVIDIA RTX A6000"), "GPU product"),
        (_diagnostics(gpu="NVIDIA RTX PRO 6000 Blackwell"), "GPU product"),
        (_diagnostics(memory_mib=24570), "GPU memory"),
        (_diagnostics(driver="579.99.01"), "Driver major"),
        (_diagnostics(ecc="Enabled"), "ECC"),
        (_diagnostics(power=250), "Power limit"),
        (_diagnostics(power_max=320), "Board maximum"),
        (_diagnostics(throttle="0x0000000000000004"), "clock event"),
        (_diagnostics(throttle="0x0000000000000005"), "clock event"),
        (_diagnostics(throttle="unparseable"), "clock event"),
        (_diagnostics(processes="1234, training"), "Competing GPU processes"),
        (_diagnostics(storage_bytes=99 * 1024**3), "100 GiB"),
        (_diagnostics(device_count=2), "exactly one numbered"),
        (_diagnostics(offline=False), "offline socket guard"),
        (_diagnostics(caches_read_only=False), "caches are writable"),
    ],
)
def test_doctor_rejects_contract_mismatches(diagnostics: dict[str, Any], message: str) -> None:
    with pytest.raises(RuntimeError, match=message):
        _orchestrator(diagnostics).doctor()


def test_doctor_rejects_multi_gpu_visibility() -> None:
    with pytest.raises(RuntimeError, match="exactly one full physical GPU"):
        _orchestrator(_diagnostics(gpu_rows=2)).doctor()


def test_doctor_rejects_insufficient_cpu_and_ram() -> None:
    with pytest.raises(RuntimeError) as error:
        _orchestrator(_diagnostics(cpu_count=4, memory_bytes=32 * 1024**3)).doctor()
    assert "vCPUs" in str(error.value)
    assert "64 GiB" in str(error.value)


def test_doctor_rejects_unpinned_or_mismatched_image() -> None:
    unpinned = _diagnostics()
    unpinned["image_reference"] = "ghcr.io/armanjr/peste-benchmark:2.0.0"
    with pytest.raises(RuntimeError, match="digest-pinned"):
        _orchestrator(unpinned).doctor()
    mismatch = _diagnostics()
    mismatch["image_digest"] = f"sha256:{'c' * 64}"
    with pytest.raises(RuntimeError, match="do not match"):
        _orchestrator(mismatch).doctor()
    unknown_source = _diagnostics()
    unknown_source["source_revision"] = "unknown"
    with pytest.raises(RuntimeError, match="full Git commit"):
        _orchestrator(unknown_source).doctor()


def test_ssh_transport_uses_direct_noninteractive_options_and_stdin(monkeypatch: Any) -> None:
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        assert kwargs["input"] == "payload"
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr("peste.orchestration.subprocess.run", run)
    completed = SshTransport("ssh://root@example.test:2222").run(
        ["peste", "doctor", "--value", "has space"], input_text="payload"
    )
    assert completed.stdout == "ok"
    command = commands[0]
    assert "BatchMode=yes" in command
    assert "StrictHostKeyChecking=accept-new" in command
    assert command[command.index("-p") + 1] == "2222"
    assert command[-2] == "root@example.test"
    assert "'has space'" in command[-1]


def test_ssh_transport_does_not_log_stdin_payload(
    monkeypatch: Any, caplog: pytest.LogCaptureFixture
) -> None:
    secret = "private-hub-token"

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("peste.orchestration.subprocess.run", run)
    caplog.set_level(logging.DEBUG)
    SshTransport("ssh://root@example.test:22").run(["true"], input_text=secret)
    assert secret not in caplog.text


def test_ssh_transport_reports_complete_remote_stderr(monkeypatch: Any) -> None:
    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 17, stdout="", stderr="full remote error")

    monkeypatch.setattr("peste.orchestration.subprocess.run", run)
    with pytest.raises(RuntimeError, match="full remote error"):
        SshTransport("ssh://root@example.test:22").run(["false"])


def test_ssh_archive_copy_extracts_only_remote_result_directory(
    monkeypatch: Any, tmp_path: Path
) -> None:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        content = b'{"status":"success"}'
        info = tarfile.TarInfo("run-id/run.json")
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        kwargs["stdout"].write(payload.getvalue())
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr("peste.orchestration.subprocess.run", run)
    SshTransport("ssh://root@example.test:22").copy_directory(
        PurePosixPath("/results/run-id"),
        tmp_path,
    )
    assert (tmp_path / "run-id" / "run.json").read_bytes() == b'{"status":"success"}'


def test_ssh_archive_copy_rejects_path_traversal(monkeypatch: Any, tmp_path: Path) -> None:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        content = b"unsafe"
        info = tarfile.TarInfo("../outside")
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        kwargs["stdout"].write(payload.getvalue())
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr("peste.orchestration.subprocess.run", run)
    with pytest.raises(RuntimeError, match="unsafe or invalid"):
        SshTransport("ssh://root@example.test:22").copy_directory(
            PurePosixPath("/results/run-id"), tmp_path
        )
    assert not (tmp_path.parent / "outside").exists()
