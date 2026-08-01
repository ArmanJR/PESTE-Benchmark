"""Narrow compatibility shims for single-device NeMo on Jetson PyTorch."""

import sys
from pathlib import Path
from types import ModuleType
from typing import Any, NoReturn

import torch


class _UnavailableGradBucket:
    """Type-only stand-in for an unavailable distributed-training API."""

    def buffer(self) -> NoReturn:
        raise RuntimeError(
            "torch.distributed.GradBucket is unavailable in the Jetson PyTorch build; "
            "PESTE does not use distributed training or DDP communication hooks"
        )


def _distributed_is_initialized() -> bool:
    return False


if not torch.distributed.is_available() and not hasattr(torch.distributed, "is_initialized"):
    torch.distributed.is_initialized = _distributed_is_initialized  # type: ignore[attr-defined]


if not torch.distributed.is_available() and not hasattr(torch.distributed, "GradBucket"):
    torch.distributed.GradBucket = _UnavailableGradBucket  # type: ignore[attr-defined]

    def _unavailable_noop_hook(*_args: Any, **_kwargs: Any) -> NoReturn:
        raise RuntimeError(
            "DDP communication hooks are unavailable in the single-device Jetson runtime"
        )

    hooks_name = "torch.distributed.algorithms.ddp_comm_hooks"
    hooks_module = ModuleType(hooks_name)
    hooks_module.__path__ = []  # type: ignore[attr-defined]
    debugging_module = ModuleType(f"{hooks_name}.debugging_hooks")
    debugging_module.noop_hook = _unavailable_noop_hook  # type: ignore[attr-defined]
    sys.modules[hooks_name] = hooks_module
    sys.modules[debugging_module.__name__] = debugging_module

# NeMo ASR imports ``nemo.lightning.callback_group`` for ModelPT bookkeeping. The
# package initializer eagerly imports unrelated FSDP/Megatron training strategies,
# which require a distributed PyTorch build. Expose the installed package as a
# namespace so the requested ASR-safe submodules load without executing that eager
# optional-training initializer.
for site_path in map(Path, sys.path):
    lightning_path = site_path / "nemo" / "lightning"
    if lightning_path.is_dir():
        lightning_module = ModuleType("nemo.lightning")
        lightning_module.__path__ = [str(lightning_path)]  # type: ignore[attr-defined]
        lightning_module.__package__ = "nemo.lightning"
        sys.modules[lightning_module.__name__] = lightning_module
        break
