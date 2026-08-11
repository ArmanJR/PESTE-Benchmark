"""Thin, redacting Vast.ai CLI integration for ordinary container acquisition."""

import json
import logging
import os
import re
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
VAST_LABEL = "peste-official"
INSTANCE_STARTUP_TIMEOUT_SECONDS = 60 * 60
SSH_AUTH_TIMEOUT_SECONDS = 5 * 60
SSH_AUTH_POLL_SECONDS = 5
VAST_GPU_ENUM = "RTX_6000Ada"
VAST_DISK_GB = 200
VAST_IMAGE_REPOSITORY = "ghcr.io/armanjr/peste-benchmark"
IMAGE_REFERENCE_PATTERN = re.compile(rf"^{re.escape(VAST_IMAGE_REPOSITORY)}@sha256:[0-9a-f]{{64}}$")
OFFER_QUERY = (
    f"gpu_name={VAST_GPU_ENUM} num_gpus=1 verified=true "
    "cpu_cores>=8 cpu_ram>=64 disk_space>=200 reliability>0.98 "
    "gpu_max_power>=300 driver_version>=580.0.0"
)
CONTAINER_ONSTART = (
    "set -eu; install -d -m 0755 /cache/dataset /cache/hf /results; "
    "touch /etc/environment; "
    "for name in PESTE_IMAGE_REFERENCE PESTE_IMAGE_DIGEST PESTE_SOURCE_REVISION CONTAINER_ID "
    'VAST_CONTAINERLABEL; do value=$(printenv "$name" 2>/dev/null || true); '
    'if [ -n "$value" ]; then sed -i "/^${name}=/d" /etc/environment; '
    'printf \'%s=%s\\n\' "$name" "$value" >> /etc/environment; fi; done'
)


def validate_image_reference(image_reference: str) -> str:
    """Require the public PESTE GHCR image to be selected by immutable digest."""
    if IMAGE_REFERENCE_PATTERN.fullmatch(image_reference) is None:
        raise ValueError(
            "Vast container image must be an immutable PESTE GHCR reference: "
            f"{VAST_IMAGE_REPOSITORY}@sha256:<64 lowercase hex characters>"
        )
    return image_reference


def _instance_id(value: Mapping[str, Any]) -> int:
    raw_id = value.get("id", value.get("instance_id", value.get("contract_id")))
    if raw_id is None:
        raise ValueError(f"Vast.ai response omitted an instance id: {value}")
    return int(raw_id)


def instance_is_running(instance: Mapping[str, Any]) -> bool:
    """Return whether an ordinary Vast container has reached its running state."""
    return str(instance.get("actual_status", instance.get("status", "unknown"))) == "running"


def _require_instance_image(instance: Mapping[str, Any], image_reference: str) -> None:
    actual = str(instance.get("image_uuid", ""))
    if actual != image_reference:
        raise RuntimeError(
            f"Vast.ai instance image {actual or 'missing'} differs from {image_reference}"
        )


def _is_transient_ssh_error(error: RuntimeError) -> bool:
    message = str(error).casefold()
    return any(
        fragment in message
        for fragment in (
            "connection closed",
            "connection refused",
            "connection reset",
            "connection timed out",
            "no route to host",
            "operation timed out",
            "permission denied (publickey)",
        )
    )


def _doctor_when_ssh_ready(
    doctor_factory: Callable[[str], Any],
    host: str,
    *,
    timeout_seconds: float = SSH_AUTH_TIMEOUT_SECONDS,
    poll_interval_seconds: float = SSH_AUTH_POLL_SECONDS,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Retry only transient SSH readiness failures before running the hardware doctor."""
    deadline = clock() + timeout_seconds
    while True:
        try:
            return doctor_factory(host).doctor()
        except RuntimeError as error:
            if not _is_transient_ssh_error(error) or clock() >= deadline:
                raise
            LOGGER.warning(
                "SSH transport is not authenticated yet; retrying hardware doctor",
                extra={"host": host, "retry_seconds": poll_interval_seconds, "error": str(error)},
            )
            sleep(poll_interval_seconds)


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

    def _invoke(self, arguments: Sequence[str], *, allow_empty: bool = False) -> Any:
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
        if allow_empty and not completed.stdout.strip():
            return None
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Vast.ai CLI returned invalid JSON: {error}") from error

    def search_offers(self, *, max_dph: float | None = None) -> list[dict[str, Any]]:
        payload = self._invoke(
            [
                "search",
                "offers",
                OFFER_QUERY,
                "--order",
                "dph_total",
                "--storage",
                str(VAST_DISK_GB),
            ]
        )
        if not isinstance(payload, list):
            raise RuntimeError("Vast.ai offer search did not return a JSON list")
        offers = [dict(offer) for offer in payload]
        if max_dph is not None:
            offers = [offer for offer in offers if float(offer["dph_total"]) <= max_dph]
        return sorted(offers, key=lambda offer: (float(offer["dph_total"]), int(offer["id"])))

    def create_instance(self, offer_id: int, image_reference: str) -> int:
        image_reference = validate_image_reference(image_reference)
        digest = image_reference.rsplit("@", maxsplit=1)[1]
        payload = self._invoke(
            [
                "create",
                "instance",
                str(offer_id),
                "--image",
                image_reference,
                "--ssh",
                "--direct",
                "--disk",
                str(VAST_DISK_GB),
                "--label",
                VAST_LABEL,
                "--env",
                (f"-e PESTE_IMAGE_REFERENCE={image_reference} -e PESTE_IMAGE_DIGEST={digest}"),
                "--onstart-cmd",
                CONTAINER_ONSTART,
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
        timeout_seconds: float = INSTANCE_STARTUP_TIMEOUT_SECONDS,
        poll_interval_seconds: float = 5,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> dict[str, Any]:
        deadline = clock() + timeout_seconds
        last_state = "unknown"
        terminal_states = {"exited", "offline", "unknown"}
        while clock() < deadline:
            instance = next(
                (item for item in self.show_instances() if _instance_id(item) == instance_id),
                None,
            )
            if instance is None:
                raise RuntimeError(f"Vast.ai instance {instance_id} disappeared while provisioning")
            last_state = str(instance.get("actual_status", instance.get("status", "unknown")))
            LOGGER.info(
                "Waiting for Vast.ai container",
                extra={"instance_id": instance_id, "state": last_state},
            )
            if last_state in terminal_states:
                status_message = str(instance.get("status_msg", ""))
                raise RuntimeError(
                    f"Vast.ai instance {instance_id} entered {last_state}: {status_message}"
                )
            if instance_is_running(instance):
                try:
                    host, port = self._ssh_address(instance)
                    with socket.create_connection((host, port), timeout=5) as connection:
                        connection.settimeout(5)
                        if connection.recv(4) == b"SSH-":
                            return instance
                except (KeyError, ValueError, OSError):
                    pass
            sleep(poll_interval_seconds)
        raise TimeoutError(
            f"Vast.ai instance {instance_id} did not become SSH-reachable; last state={last_state}"
        )

    @staticmethod
    def _ssh_address(instance: Mapping[str, Any]) -> tuple[str, int]:
        direct_host = instance.get("public_ipaddr") or instance.get("public_ip")
        mappings = instance.get("ports", {}).get("22/tcp", [])
        if direct_host and mappings and mappings[0].get("HostPort"):
            return str(direct_host), int(mappings[0]["HostPort"])
        host = instance.get("ssh_host") or direct_host
        port = instance.get("ssh_port")
        if not host and mappings:
            host = mappings[0].get("HostIp")
            port = port or mappings[0].get("HostPort")
        if not host or not port:
            raise ValueError("Vast.ai instance has no SSH address")
        return str(host), int(port)

    def ssh_url(self, instance: Mapping[str, Any]) -> str:
        host, port = self._ssh_address(instance)
        return f"ssh://root@{host}:{port}"

    def destroy_instance(self, instance_id: int) -> None:
        payload = self._invoke(["destroy", "instance", str(instance_id), "--yes"], allow_empty=True)
        if isinstance(payload, dict) and payload.get("success") is False:
            raise RuntimeError(f"Vast.ai rejected instance destruction: {payload}")
        LOGGER.info("Destroyed Vast.ai instance", extra={"instance_id": instance_id})


@dataclass(frozen=True, slots=True)
class ProvisionedInstance:
    instance_id: int
    offer_id: int
    dph_total: float
    host: str
    image_reference: str


def provision_official_container(
    client: VastClient,
    doctor_factory: Callable[[str], Any],
    image_reference: str,
    *,
    max_dph: float | None = None,
    maximum_attempts: int = 3,
    excluded_offer_ids: frozenset[int] | None = None,
) -> ProvisionedInstance:
    """Provision a doctor-approved container, destroying every rejected attempt."""
    image_reference = validate_image_reference(image_reference)
    existing = labeled_instances(client)
    if len(existing) > 1:
        raise RuntimeError(f"Expected at most one {VAST_LABEL} instance; found {len(existing)}")
    if existing:
        existing_id = _instance_id(existing[0])
        existing_image = str(existing[0].get("image_uuid", ""))
        LOGGER.info(
            "Inspecting existing labeled Vast.ai container",
            extra={"instance_id": existing_id, "image": existing_image},
        )
        try:
            if existing_image and existing_image != image_reference:
                raise RuntimeError(
                    f"Existing instance image {existing_image} differs from {image_reference}"
                )
            instance = client.wait_until_running(existing_id)
            _require_instance_image(instance, image_reference)
            host = client.ssh_url(instance)
            _doctor_when_ssh_ready(doctor_factory, host)
            return ProvisionedInstance(
                instance_id=existing_id,
                offer_id=int(instance.get("ask_contract_id", existing_id)),
                dph_total=float(instance.get("dph_total", instance.get("dph_base", 0))),
                host=host,
                image_reference=image_reference,
            )
        except Exception:
            LOGGER.exception(
                "Rejected existing labeled Vast.ai container", extra={"instance_id": existing_id}
            )
            client.destroy_instance(existing_id)

    excluded = excluded_offer_ids or frozenset()
    offers = [
        offer for offer in client.search_offers(max_dph=max_dph) if int(offer["id"]) not in excluded
    ]
    if not offers:
        raise RuntimeError(
            "No non-excluded Vast.ai offers satisfy the RTX 6000 Ada container preselection policy"
        )
    failures: list[str] = []
    for offer in offers[:maximum_attempts]:
        offer_id = int(offer["id"])
        dph_total = float(offer["dph_total"])
        instance_id: int | None = None
        LOGGER.info("Trying Vast.ai offer", extra={"offer_id": offer_id, "dph_total": dph_total})
        try:
            instance_id = client.create_instance(offer_id, image_reference)
            instance = client.wait_until_running(instance_id)
            _require_instance_image(instance, image_reference)
            host = client.ssh_url(instance)
            _doctor_when_ssh_ready(doctor_factory, host)
            return ProvisionedInstance(instance_id, offer_id, dph_total, host, image_reference)
        except Exception as error:
            failures.append(f"offer {offer_id}: {type(error).__name__}: {error}")
            LOGGER.exception(
                "Rejected Vast.ai offer", extra={"offer_id": offer_id, "instance_id": instance_id}
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
    raise RuntimeError(
        "No doctor-approved Vast.ai container was provisioned: " + "; ".join(failures)
    )


def labeled_instances(client: VastClient) -> list[dict[str, Any]]:
    return [instance for instance in client.show_instances() if instance.get("label") == VAST_LABEL]
