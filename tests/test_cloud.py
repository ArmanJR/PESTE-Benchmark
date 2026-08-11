"""Recorded-JSON and lifecycle tests for the Vast.ai CLI wrapper."""

import json
import logging
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from peste import cloud
from peste.cloud import (
    CONTAINER_ONSTART,
    VAST_IMAGE_REPOSITORY,
    VastClient,
    provision_official_container,
    validate_image_reference,
)

IMAGE = f"{VAST_IMAGE_REPOSITORY}@sha256:{'a' * 64}"


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
    assert "vms_enabled" not in command[3]
    assert "cpu_ram>=64" in command[3]
    assert "disk_space>=200" in command[3]
    assert "driver_version>=580.0.0" in command[3]
    assert command[command.index("--storage") + 1] == "200"
    assert "--raw" in command


def test_create_uses_digest_pinned_image_direct_ssh_and_container_policy(tmp_path: Path) -> None:
    run = RecordedRun([{"success": True, "new_contract": 123}])
    assert _configured_client(tmp_path, run).create_instance(55, IMAGE) == 123
    command = run.commands[0]
    assert command[command.index("--image") + 1] == IMAGE
    assert "--ssh" in command and "--direct" in command
    assert command[command.index("--disk") + 1] == "200"
    assert command[command.index("--label") + 1] == "peste-official"
    assert command[command.index("--onstart-cmd") + 1] == CONTAINER_ONSTART
    assert "PESTE_SOURCE_REVISION" in CONTAINER_ONSTART
    environment = command[command.index("--env") + 1]
    assert f"PESTE_IMAGE_REFERENCE={IMAGE}" in environment
    assert f"PESTE_IMAGE_DIGEST=sha256:{'a' * 64}" in environment


@pytest.mark.parametrize(
    "reference",
    [
        "ghcr.io/armanjr/peste-benchmark:2.0.0",
        f"docker.io/example/image@sha256:{'a' * 64}",
        f"{VAST_IMAGE_REPOSITORY}@sha256:{'A' * 64}",
        f"{VAST_IMAGE_REPOSITORY}@sha256:short",
    ],
)
def test_image_reference_must_be_public_project_image_by_digest(reference: str) -> None:
    with pytest.raises(ValueError, match="immutable PESTE GHCR"):
        validate_image_reference(reference)


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
    ],
)
def test_ssh_url_prefers_direct_mappings(
    tmp_path: Path, instance: dict[str, Any], expected: str
) -> None:
    assert _configured_client(tmp_path, RecordedRun([])).ssh_url(instance) == expected


def test_missing_api_key_has_actionable_error(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="uv run vastai set api-key"):
        VastClient(config_path=tmp_path / "missing")


def test_wait_accepts_running_container_when_direct_ssh_is_reachable(
    monkeypatch: Any, tmp_path: Path
) -> None:
    client = _configured_client(tmp_path, RecordedRun([]))
    instance = {
        "id": 123,
        "actual_status": "running",
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


def test_wait_fails_immediately_for_terminal_container_state(
    monkeypatch: Any, tmp_path: Path
) -> None:
    client = _configured_client(tmp_path, RecordedRun([]))
    monkeypatch.setattr(
        client,
        "show_instances",
        lambda: [{"id": 123, "actual_status": "exited", "status_msg": "image failed"}],
    )
    with pytest.raises(RuntimeError, match="image failed"):
        client.wait_until_running(123, clock=lambda: 0)


def test_wait_uses_one_hour_startup_timeout_by_default(monkeypatch: Any, tmp_path: Path) -> None:
    client = _configured_client(tmp_path, RecordedRun([]))
    calls = 0

    def show_instances() -> list[dict[str, Any]]:
        nonlocal calls
        calls += 1
        return [{"id": 123, "actual_status": "loading"}]

    clock_values = iter([0.0, 3599.0, 3600.0])
    monkeypatch.setattr(client, "show_instances", show_instances)

    with pytest.raises(TimeoutError, match="last state=loading"):
        client.wait_until_running(
            123,
            clock=lambda: next(clock_values),
            sleep=lambda seconds: None,
        )

    assert calls == 1


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
    error = subprocess.CalledProcessError(1, ["vastai"], stderr=f"rejected {secret}")
    client = _configured_client(tmp_path, RecordedRun([error]), key=secret)
    caplog.set_level(logging.DEBUG)
    with pytest.raises(RuntimeError, match="<redacted>"):
        client.show_instances()
    assert secret not in caplog.text


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

    def create_instance(self, offer_id: int, image_reference: str) -> int:
        assert image_reference == IMAGE
        self.created.append(offer_id)
        return offer_id + 100

    def wait_until_running(self, instance_id: int) -> dict[str, Any]:
        if self.wait_failure:
            raise TimeoutError("not reachable")
        return {
            "id": instance_id,
            "ssh_host": "host",
            "ssh_port": 22,
            "image_uuid": IMAGE,
        }

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

    result = provision_official_container(
        client,  # type: ignore[arg-type]
        lambda host: Doctor(),
        IMAGE,
    )
    assert result.offer_id == 2
    assert result.image_reference == IMAGE
    assert client.created == [1, 2]
    assert client.destroyed == [101]


def test_provisioning_failure_destroys_created_instance() -> None:
    client = LifecycleClient(wait_failure=True)
    with pytest.raises(RuntimeError, match="No doctor-approved"):
        provision_official_container(
            client,  # type: ignore[arg-type]
            lambda host: SimpleNamespace(doctor=lambda: None),
            IMAGE,
            maximum_attempts=1,
        )
    assert client.destroyed == [101]


def test_transient_ssh_auth_failure_is_retried_before_doctor_rejection() -> None:
    attempts = 0
    sleeps: list[float] = []

    class Doctor:
        def doctor(self) -> dict[str, bool]:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise RuntimeError("Remote command failed (255): Permission denied (publickey)")
            return {"accepted": True}

    result = cloud._doctor_when_ssh_ready(
        lambda host: Doctor(),
        "ssh://root@host:22",
        clock=lambda: 0,
        sleep=sleeps.append,
    )

    assert result == {"accepted": True}
    assert attempts == 3
    assert sleeps == [cloud.SSH_AUTH_POLL_SECONDS, cloud.SSH_AUTH_POLL_SECONDS]


def test_hardware_doctor_failure_is_not_retried() -> None:
    sleeps: list[float] = []

    with pytest.raises(RuntimeError, match="wrong ECC"):
        cloud._doctor_when_ssh_ready(
            lambda host: SimpleNamespace(
                doctor=lambda: (_ for _ in ()).throw(RuntimeError("wrong ECC"))
            ),
            "ssh://root@host:22",
            clock=lambda: 0,
            sleep=sleeps.append,
        )

    assert sleeps == []


def test_provisioning_skips_explicitly_excluded_offer() -> None:
    client = LifecycleClient()
    result = provision_official_container(
        client,  # type: ignore[arg-type]
        lambda host: SimpleNamespace(doctor=lambda: None),
        IMAGE,
        excluded_offer_ids=frozenset({1}),
    )
    assert result.offer_id == 2
    assert client.created == [2]
