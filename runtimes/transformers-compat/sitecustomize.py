"""Inference-only compatibility for NVIDIA's distributed-disabled Jetson PyTorch.

Transformers imports its FSDP helper while constructing generation utilities,
even when FSDP is not enabled. The NVIDIA Jetson wheel intentionally omits the
compiled distributed backend, so importing the upstream helper otherwise fails
before ordinary single-device inference can start.
"""

from __future__ import annotations

import logging
import sys
from types import ModuleType
from typing import Any

import torch

LOGGER = logging.getLogger(__name__)


def _is_fsdp_enabled() -> bool:
    return False


def _is_fsdp_managed_module(_module: Any) -> bool:
    return False


def _unavailable_fsdp_operation(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError(
        "FSDP is unavailable in the Jetson PyTorch build; PESTE runs single-device inference"
    )


class _UnavailableDTensor:
    """Sentinel type used only by non-distributed checkpoint-loading branches."""


class _UnavailableDtensorShardOperation:
    def __init__(self, _parameter: Any) -> None:
        raise RuntimeError(
            "DTensor checkpoint sharding is unavailable in the Jetson PyTorch build; "
            "PESTE runs ordinary single-device checkpoint loading"
        )


def _unavailable_dtensor_from_local_like(_local_tensor: Any, _reference: Any) -> Any:
    raise RuntimeError(
        "DTensor conversion is unavailable in the Jetson PyTorch build; "
        "PESTE runs ordinary single-device checkpoint loading"
    )


if not torch.distributed.is_available():
    fsdp_module_name = "transformers.distributed.fsdp"
    fsdp_compat = ModuleType(fsdp_module_name)
    fsdp_compat.get_fsdp_ckpt_kwargs = _unavailable_fsdp_operation  # type: ignore[attr-defined]
    fsdp_compat.is_fsdp_enabled = _is_fsdp_enabled  # type: ignore[attr-defined]
    fsdp_compat.is_fsdp_managed_module = _is_fsdp_managed_module  # type: ignore[attr-defined]
    fsdp_compat.update_fsdp_plugin_peft = _unavailable_fsdp_operation  # type: ignore[attr-defined]
    sys.modules[fsdp_module_name] = fsdp_compat

    sharding_module_name = "transformers.distributed.sharding_utils"
    sharding_compat = ModuleType(sharding_module_name)
    sharding_compat.DTensor = _UnavailableDTensor  # type: ignore[attr-defined]
    sharding_compat.DtensorShardOperation = (  # type: ignore[attr-defined]
        _UnavailableDtensorShardOperation
    )
    sharding_compat._dtensor_from_local_like = (  # type: ignore[attr-defined]
        _unavailable_dtensor_from_local_like
    )
    sys.modules[sharding_module_name] = sharding_compat
    LOGGER.debug(
        "Installed inference-only Transformers distributed compatibility modules",
        extra={"compat_modules": [fsdp_module_name, sharding_module_name]},
    )
