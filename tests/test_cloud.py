"""Recorded-JSON and lifecycle tests for the Vast.ai CLI wrapper."""

import json
import logging
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from peste.cloud import VastClient, bootstrap_vm, provision_official_vm


class RecordedRun:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = iter(responses)
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(response), stderr="")


def _configured_client(tmp_path: Path, run: RecordedRun, *, key: str | None = None) -> VastClient:
    config = tmp_path / "vast_api_key"
    config.write_text("configured", encoding="utf-8")
    return VastClient(executable="vastai", api_key=key, config_path=config, run=run)


def test_offer_query_filters_price_and_sorts_recorded_raw_json(tmp_path: Path) -> None:
    run = RecordedRun(
        [[{"id": 3, "dph_total": 0.9}, {"id": 2, "dph_total": 0.5}, {"id": 1, "dph_total": 0.5}]]
    )
    offers = _configured_client(tmp_path, run).search_offers(max_dph=0.6)
    assert [offer["id"] for offer in offers] == [1, 2]
    command = run.commands[0]
    assert command[:3] == ["vastai", "search", "offers"]
    assert "gpu_name=RTX_6000Ada" in command[3]
    assert "vms_enabled=true" in command[3]
    assert "cpu_ram>=64" in command[3]
    assert "driver_version" not in command[3]
    assert command[command.index("--storage") + 1] == "100"
    assert "--raw" in command


def test_create_uses_pinned_vm_image_direct_ssh_and_disk(tmp_path: Path) -> None:
    run = RecordedRun([{"success": True, "new_contract": 123}])
    assert _configured_client(tmp_path, run).create_instance(55) == 123
    command = run.commands[0]
    assert "docker.io/vastai/kvm:ubuntu_terminal" in command
    assert "--ssh" in command and "--direct" in command
    assert command[command.index("--disk") + 1] == "100"
    assert command[command.index("--label") + 1] == "peste-official"


@pytest.mark.parametrize(
    ("instance", "expected"),
    [
        ({"ssh_host": "1.2.3.4", "ssh_port": 2222}, "ssh://root@1.2.3.4:2222"),
        (
            {
                "public_ipaddr": "5.6.7.8",
                "ports": {"22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "2200"}]},
            },
            "ssh://root@5.6.7.8:2200",
        ),
        (
            {
                "ssh_host": "ssh5.vast.ai",
                "ssh_port": 29558,
                "public_ipaddr": "112.69.3.12",
                "ports": {"22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "43522"}]},
            },
            "ssh://root@112.69.3.12:43522",
        ),
    ],
)
def test_ssh_url_parses_direct_mappings(
    tmp_path: Path, instance: dict[str, Any], expected: str
) -> None:
    assert _configured_client(tmp_path, RecordedRun([])).ssh_url(instance) == expected


def test_missing_api_key_has_actionable_error(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="uv run vastai set api-key"):
        VastClient(config_path=tmp_path / "missing")


def test_wait_accepts_created_running_vm_when_direct_ssh_is_reachable(
    monkeypatch: Any, tmp_path: Path
) -> None:
    client = _configured_client(tmp_path, RecordedRun([]))
    instance = {
        "id": 123,
        "actual_status": "created",
        "intended_status": "running",
        "public_ipaddr": "112.69.3.12",
        "ports": {"22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "43522"}]},
    }
    monkeypatch.setattr(client, "show_instances", lambda: [instance])

    class SshSocket:
        def __enter__(self) -> "SshSocket":
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def settimeout(self, timeout: float) -> None:
            pass

        def recv(self, size: int) -> bytes:
            return b"SSH-"

    monkeypatch.setattr("peste.cloud.socket.create_connection", lambda *args, **kwargs: SshSocket())

    assert client.wait_until_running(123, clock=lambda: 0) == instance


def test_destroy_is_noninteractive(tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def empty_success(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    config = tmp_path / "vast_api_key"
    config.write_text("configured", encoding="utf-8")
    VastClient(executable="vastai", config_path=config, run=empty_success).destroy_instance(123)
    assert "--yes" in commands[0]


def test_api_key_is_redacted_from_failure_and_logs(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    secret = "secret-api-key"
    error = subprocess.CalledProcessError(
        1,
        ["vastai"],
        stderr=f"request rejected for {secret}",
    )
    client = _configured_client(tmp_path, RecordedRun([error]), key=secret)
    caplog.set_level(logging.DEBUG)
    with pytest.raises(RuntimeError, match="<redacted>"):
        client.show_instances()
    assert secret not in caplog.text


def test_bootstrap_is_noninteractive_and_configures_official_nvidia_repository(
    monkeypatch: Any,
) -> None:
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("peste.cloud.subprocess.run", run)
    bootstrap_vm("ssh://root@example.test:2222")

    assert len(commands) == 1
    command = commands[0]
    assert "StrictHostKeyChecking=accept-new" in command
    assert command[command.index("-p") + 1] == "2222"
    remote_command = command[-1]
    assert "DEBIAN_FRONTEND=noninteractive" in remote_command
    assert "nvidia.github.io/libnvidia-container/stable/deb" in remote_command
    assert "nvidia-ctk runtime configure --runtime=docker" in remote_command
    assert "docker run --rm --gpus all" in remote_command


class LifecycleClient:
    def __init__(self, *, wait_failure: bool = False) -> None:
        self.offers = [{"id": 1, "dph_total": 0.4}, {"id": 2, "dph_total": 0.5}]
        self.destroyed: list[int] = []
        self.created: list[int] = []
        self.wait_failure = wait_failure

    def search_offers(self, *, max_dph: float | None = None) -> list[dict[str, Any]]:
        return self.offers

    def show_instances(self) -> list[dict[str, Any]]:
        return []

    def create_instance(self, offer_id: int) -> int:
        self.created.append(offer_id)
        return offer_id + 100

    def wait_until_running(self, instance_id: int) -> dict[str, Any]:
        if self.wait_failure:
            raise TimeoutError("not reachable")
        return {"id": instance_id, "ssh_host": "host", "ssh_port": 22}

    def ssh_url(self, instance: dict[str, Any]) -> str:
        return "ssh://root@host:22"

    def destroy_instance(self, instance_id: int) -> None:
        self.destroyed.append(instance_id)


def test_doctor_rejection_destroys_instance_then_tries_next_offer() -> None:
    client = LifecycleClient()
    calls = 0

    class Doctor:
        def doctor(self) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("wrong ECC")

    result = provision_official_vm(
        client,  # type: ignore[arg-type]
        lambda host: Doctor(),
        bootstrap=lambda host: None,
    )
    assert result.offer_id == 2
    assert client.created == [1, 2]
    assert client.destroyed == [101]


def test_provisioning_failure_destroys_created_instance() -> None:
    client = LifecycleClient(wait_failure=True)
    with pytest.raises(RuntimeError, match="No doctor-approved"):
        provision_official_vm(
            client,  # type: ignore[arg-type]
            lambda host: SimpleNamespace(doctor=lambda: None),
            maximum_attempts=1,
            bootstrap=lambda host: None,
        )
    assert client.destroyed == [101]
