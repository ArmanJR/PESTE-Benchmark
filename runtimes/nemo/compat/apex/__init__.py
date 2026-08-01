"""Disable the base image's unsupported optional Apex/Megatron integration."""

raise ModuleNotFoundError(
    "Apex is intentionally disabled: Jetson PyTorch has distributed support disabled, "
    "and PESTE's NeMo ASR path does not use Apex or Megatron."
)
