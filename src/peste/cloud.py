"""Thin, redacting Vast.ai CLI integration for reference VM acquisition."""

import json
import logging
import os
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

LOGGER = logging.getLogger(__name__)
VAST_LABEL = "peste-official"
VAST_VM_IMAGE = "docker.io/vastai/kvm:ubuntu_terminal"
VAST_GPU_ENUM = "RTX_6000Ada"
PINNED_DRIVER = "580.142"
OFFER_QUERY = (
    f"gpu_name={VAST_GPU_ENUM} num_gpus=1 vms_enabled=true verified=true "
    "cpu_cores>=8 cpu_ram>=65536 disk_space>=100 reliability>0.98 "
    f"gpu_max_power>=300 driver_version={PINNED_DRIVER}"
)


def _instance_id(value: Mapping[str, Any]) -> int:
    raw_id = value.get("id", value.get("instance_id", value.get("contract_id")))
    if raw_id is None:
        raise ValueError(f"Vast.ai response omitted an instance id: {value}")
    return int(raw_id)


class VastClient:
    """Invoke the installed Vast.ai CLI and parse only its raw JSON output."""

    def __init__(
        self,
        *,
        executable: str | None = None,
        api_key: str | None = None,
        config_path: Path | None = None,
        run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        environment_cli = Path(sys.executable).with_name("vastai")
        self.executable = executable or str(environment_cli)
        if executable is None and not environment_cli.is_file():
            raise RuntimeError(
                "The Vast.ai CLI is missing from the PESTE environment; run `uv sync --frozen`"
            )
        self.api_key = api_key or os.environ.get("VAST_API_KEY")
        self.config_path = config_path or Path.home() / ".config" / "vastai" / "vast_api_key"
        self._run_subprocess = run
        if self.api_key is None and not self.config_path.is_file():
            raise RuntimeError(
                "Vast.ai API key is not configured; run `uv run vastai set api-key <key>` "
                "or set VAST_API_KEY"
            )

    def _invoke(self, arguments: Sequence[str]) -> Any:
        command = [self.executable, *arguments, "--raw"]
        if self.api_key is not None:
            command.extend(["--api-key", self.api_key])
        logged_command = ["<redacted>" if item == self.api_key else item for item in command]
        LOGGER.debug("Running Vast.ai CLI", extra={"command": logged_command})
        try:
            completed = self._run_subprocess(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.CalledProcessError as error:
            stderr = (error.stderr or "").replace(self.api_key or "\0", "<redacted>")
            LOGGER.error(
                "Vast.ai CLI command failed",
                extra={"command": logged_command, "returncode": error.returncode, "stderr": stderr},
            )
            raise RuntimeError(
                f"Vast.ai CLI failed ({error.returncode}): {stderr.strip()}"
            ) from error
        except (subprocess.SubprocessError, FileNotFoundError) as error:
            raise RuntimeError(f"Unable to execute Vast.ai CLI: {error}") from error
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Vast.ai CLI returned invalid JSON: {error}") from error

    def search_offers(self, *, max_dph: float | None = None) -> list[dict[str, Any]]:
        payload = self._invoke(
            ["search", "offers", OFFER_QUERY, "--order", "dph_total", "--storage", "100"]
        )
        if not isinstance(payload, list):
            raise RuntimeError("Vast.ai offer search did not return a JSON list")
        offers = [dict(offer) for offer in payload]
        if max_dph is not None:
            offers = [offer for offer in offers if float(offer["dph_total"]) <= max_dph]
        return sorted(offers, key=lambda offer: (float(offer["dph_total"]), int(offer["id"])))

    def create_instance(self, offer_id: int) -> int:
        payload = self._invoke(
            [
                "create",
                "instance",
                str(offer_id),
                "--image",
                VAST_VM_IMAGE,
                "--ssh",
                "--direct",
                "--disk",
                "100",
                "--label",
                VAST_LABEL,
                "--cancel-unavail",
            ]
        )
        if not isinstance(payload, dict) or not payload.get("success", False):
            raise RuntimeError(f"Vast.ai rejected instance creation: {payload}")
        instance_id = payload.get("new_contract", payload.get("id"))
        if instance_id is None:
            raise RuntimeError(f"Vast.ai creation response omitted the instance id: {payload}")
        return int(instance_id)

    def show_instances(self) -> list[dict[str, Any]]:
        payload = self._invoke(["show", "instances"])
        if not isinstance(payload, list):
            raise RuntimeError("Vast.ai instance listing did not return a JSON list")
        return [dict(instance) for instance in payload]

    def wait_until_running(
        self,
        instance_id: int,
        *,
        timeout_seconds: float = 900,
        poll_interval_seconds: float = 5,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> dict[str, Any]:
        deadline = clock() + timeout_seconds
        last_state = "unknown"
        while clock() < deadline:
            instance = next(
                (item for item in self.show_instances() if _instance_id(item) == instance_id),
                None,
            )
            if instance is None:
                raise RuntimeError(f"Vast.ai instance {instance_id} disappeared while provisioning")
            last_state = str(instance.get("actual_status", instance.get("status", "unknown")))
            LOGGER.info(
                "Waiting for Vast.ai VM",
                extra={"instance_id": instance_id, "state": last_state},
            )
            if last_state == "running":
                try:
                    host, port = self._ssh_address(instance)
                    with socket.create_connection((host, port), timeout=5):
                        return instance
                except (KeyError, ValueError, OSError):
                    pass
            sleep(poll_interval_seconds)
        raise TimeoutError(
            f"Vast.ai instance {instance_id} did not become SSH-reachable; last state={last_state}"
        )

    @staticmethod
    def _ssh_address(instance: Mapping[str, Any]) -> tuple[str, int]:
        host = (
            instance.get("ssh_host") or instance.get("public_ipaddr") or instance.get("public_ip")
        )
        port = instance.get("ssh_port")
        if port is None:
            mappings = instance.get("ports", {}).get("22/tcp", [])
            if mappings:
                port = mappings[0].get("HostPort")
                host = host or mappings[0].get("HostIp")
        if not host or not port:
            raise ValueError("Vast.ai instance has no direct SSH address")
        return str(host), int(port)

    def ssh_url(self, instance: Mapping[str, Any]) -> str:
        host, port = self._ssh_address(instance)
        return f"ssh://root@{host}:{port}"

    def destroy_instance(self, instance_id: int) -> None:
        payload = self._invoke(["destroy", "instance", str(instance_id)])
        if isinstance(payload, dict) and payload.get("success") is False:
            raise RuntimeError(f"Vast.ai rejected instance destruction: {payload}")
        LOGGER.info("Destroyed Vast.ai instance", extra={"instance_id": instance_id})


def bootstrap_vm(host: str) -> None:
    """Ensure systemd Docker and NVIDIA Container Toolkit work in the rented VM."""
    endpoint = urlparse(host)
    if endpoint.hostname is None:
        raise ValueError(f"Invalid SSH URL: {host}")
    target = f"{endpoint.username or 'root'}@{endpoint.hostname}"
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=10",
    ]
    if endpoint.port:
        command.extend(["-p", str(endpoint.port)])
    bootstrap = (
        "set -euo pipefail; export DEBIAN_FRONTEND=noninteractive; "
        "if ! command -v docker >/dev/null; then apt-get update; apt-get install -y docker.io; fi; "
        "if ! command -v nvidia-ctk >/dev/null; then "
        "apt-get update; apt-get install -y --no-install-recommends ca-certificates curl gnupg2; "
        "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey "
        "| gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg; "
        "curl -fsSL "
        "https://nvidia.github.io/libnvidia-container/stable/deb/"
        "nvidia-container-toolkit.list "
        "| sed 's#deb https://#deb "
        "[signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' "
        "> /etc/apt/sources.list.d/nvidia-container-toolkit.list; "
        "apt-get update; apt-get install -y nvidia-container-toolkit; fi; "
        "nvidia-ctk runtime configure --runtime=docker; "
        "systemctl enable --now docker; systemctl restart docker; "
        "docker info >/dev/null; "
        "docker run --rm --gpus all "
        "nvcr.io/nvidia/pytorch@sha256:"
        "3cb18e2c438db8af2d3a659ca27fac5da328640261c38c48a34edcd223c38af9 "
        "nvidia-smi --query-gpu=name --format=csv,noheader,nounits"
    )
    command.extend([target, bootstrap])
    LOGGER.info("Bootstrapping Vast.ai VM", extra={"host": host})
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=600)
    except subprocess.CalledProcessError as error:
        LOGGER.error(
            "Vast.ai VM bootstrap failed",
            extra={"host": host, "returncode": error.returncode, "stderr": error.stderr},
        )
        raise RuntimeError(f"Vast.ai VM bootstrap failed: {error.stderr.strip()}") from error
    except (subprocess.SubprocessError, FileNotFoundError) as error:
        raise RuntimeError(f"Unable to bootstrap Vast.ai VM: {error}") from error


@dataclass(frozen=True, slots=True)
class ProvisionedInstance:
    instance_id: int
    offer_id: int
    dph_total: float
    host: str


def provision_official_vm(
    client: VastClient,
    doctor_factory: Callable[[str], Any],
    *,
    max_dph: float | None = None,
    maximum_attempts: int = 3,
    bootstrap: Callable[[str], None] = bootstrap_vm,
) -> ProvisionedInstance:
    """Provision a doctor-approved VM, destroying every rejected attempt."""
    offers = client.search_offers(max_dph=max_dph)
    if not offers:
        raise RuntimeError("No Vast.ai offers satisfy the RTX 6000 Ada preselection policy")
    failures: list[str] = []
    for offer in offers[:maximum_attempts]:
        offer_id = int(offer["id"])
        dph_total = float(offer["dph_total"])
        instance_id: int | None = None
        LOGGER.info(
            "Trying Vast.ai offer",
            extra={"offer_id": offer_id, "dph_total": dph_total},
        )
        try:
            instance_id = client.create_instance(offer_id)
            instance = client.wait_until_running(instance_id)
            host = client.ssh_url(instance)
            bootstrap(host)
            doctor_factory(host).doctor()
            return ProvisionedInstance(instance_id, offer_id, dph_total, host)
        except Exception as error:
            failures.append(f"offer {offer_id}: {type(error).__name__}: {error}")
            LOGGER.exception(
                "Rejected Vast.ai offer",
                extra={"offer_id": offer_id, "instance_id": instance_id},
            )
            if instance_id is not None:
                try:
                    client.destroy_instance(instance_id)
                except Exception as destroy_error:
                    LOGGER.exception(
                        "Failed to destroy rejected Vast.ai instance",
                        extra={"instance_id": instance_id},
                    )
                    raise RuntimeError(
                        f"Failed to destroy rejected Vast.ai instance {instance_id}"
                    ) from destroy_error
    raise RuntimeError("No doctor-approved Vast.ai VM was provisioned: " + "; ".join(failures))


def labeled_instances(client: VastClient) -> list[dict[str, Any]]:
    return [instance for instance in client.show_instances() if instance.get("label") == VAST_LABEL]
