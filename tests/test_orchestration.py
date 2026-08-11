"""Hardware-doctor acceptance and rejection tests without a Docker daemon."""

from types import SimpleNamespace
from typing import Any

import pytest

from peste.constants import PROJECT_ROOT
from peste.orchestration import GpuOrchestrator


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
    cpu_count: int = 8,
    memory_bytes: int = 64 * 1024**3,
    storage_bytes: int = 101 * 1024**3,
) -> str:
    return "\n".join(
        [
            (
                f"__GPUS__={gpu}, {memory_mib}, {driver}, {ecc}, {power}, {power_max}, "
                f"GPU-id, {throttle}"
            ),
            f"__GPU_PROCESSES__={processes}",
            f"__CPU_COUNT__={cpu_count}",
            "__CPU_MODEL__=Test CPU",
            f"__MEM_BYTES__={memory_bytes}",
            f"__STORAGE_BYTES__={storage_bytes}",
            "__OS__=ubuntu 22.04",
        ]
    )


class Containers:
    def __init__(self, visible_gpu: str, competing: list[Any] | None = None) -> None:
        self.visible_gpu = visible_gpu
        self.competing = competing or []
        self.run_kwargs: dict[str, Any] = {}

    def list(self) -> list[Any]:
        return self.competing

    def run(self, *args: Any, **kwargs: Any) -> bytes:
        self.run_kwargs = kwargs
        return (self.visible_gpu + "\n").encode()


def _orchestrator(
    diagnostics: str,
    *,
    architecture: str = "x86_64",
    cpus: int = 8,
    memory: int = 64 * 1024**3,
    visible_gpu: str = "NVIDIA RTX 6000 Ada Generation",
    competing: list[Any] | None = None,
) -> tuple[GpuOrchestrator, Containers]:
    containers = Containers(visible_gpu, competing)
    client = SimpleNamespace(
        info=lambda: {
            "Architecture": architecture,
            "NCPU": cpus,
            "MemTotal": memory,
            "OperatingSystem": "Ubuntu 22.04",
        },
        containers=containers,
    )
    orchestrator = object.__new__(GpuOrchestrator)
    orchestrator.host = "ssh://root@test:22"
    orchestrator.root = PROJECT_ROOT
    orchestrator.client = client
    orchestrator._host_diagnostics = lambda: diagnostics  # type: ignore[method-assign]
    return orchestrator, containers


def test_doctor_accepts_exact_rtx_profile_and_uses_device_requests() -> None:
    orchestrator, containers = _orchestrator(_diagnostics())
    report = orchestrator.doctor()
    assert report["profile_id"] == "rtx-6000-ada-v1"
    assert report["gpu_product_name"] == "NVIDIA RTX 6000 Ada Generation"
    assert report["ecc_state"] == "Disabled"
    assert containers.run_kwargs["device_requests"]
    assert "runtime" not in containers.run_kwargs


@pytest.mark.parametrize(
    ("diagnostics", "message"),
    [
        (_diagnostics(gpu="NVIDIA RTX A6000"), "GPU product"),
        (_diagnostics(gpu="NVIDIA RTX PRO 6000 Blackwell"), "GPU product"),
        (_diagnostics(memory_mib=24570), "GPU memory"),
        (_diagnostics(driver="580.141"), "Driver"),
        (_diagnostics(ecc="Enabled"), "ECC"),
        (_diagnostics(power=250), "Power limit"),
        (_diagnostics(power_max=320), "Board maximum"),
        (_diagnostics(throttle="0x0000000000000004"), "throttling"),
        (_diagnostics(processes="1234, training"), "Competing host GPU processes"),
        (_diagnostics(storage_bytes=99 * 1024**3), "100 GiB"),
    ],
)
def test_doctor_rejects_contract_mismatches(diagnostics: str, message: str) -> None:
    orchestrator, _ = _orchestrator(diagnostics)
    with pytest.raises(RuntimeError, match=message):
        orchestrator.doctor()


def test_doctor_rejects_multi_gpu_visibility() -> None:
    single = (
        "__GPUS__=NVIDIA RTX 6000 Ada Generation, 49140, 580.142, Disabled, 300, 300, "
        "GPU-id, 0x0000000000000000"
    )
    multiple = (
        "__GPUS__=NVIDIA RTX 6000 Ada Generation, 49140, 580.142, Disabled, 300, 300, "
        "GPU-a, 0x0000000000000000;NVIDIA RTX 6000 Ada Generation, 49140, 580.142, "
        "Disabled, 300, 300, GPU-b, 0x0000000000000000"
    )
    diagnostics = _diagnostics().replace(single, multiple)
    orchestrator, _ = _orchestrator(diagnostics)
    with pytest.raises(RuntimeError, match="exactly one full physical GPU"):
        orchestrator.doctor()


def test_doctor_rejects_insufficient_cpu_ram_and_competing_container() -> None:
    competing = [
        SimpleNamespace(
            name="training",
            attrs={"HostConfig": {"DeviceRequests": [{"Count": -1}]}},
        )
    ]
    orchestrator, _ = _orchestrator(
        _diagnostics(cpu_count=4, memory_bytes=32 * 1024**3),
        cpus=4,
        memory=32 * 1024**3,
        competing=competing,
    )
    with pytest.raises(RuntimeError) as error:
        orchestrator.doctor()
    assert "vCPUs" in str(error.value)
    assert "64 GiB" in str(error.value)
    assert "Competing GPU containers" in str(error.value)


def test_doctor_rejects_container_gpu_visibility_mismatch() -> None:
    orchestrator, _ = _orchestrator(_diagnostics(), visible_gpu="NVIDIA RTX A6000")
    with pytest.raises(RuntimeError, match="Container GPU visibility mismatch"):
        orchestrator.doctor()
