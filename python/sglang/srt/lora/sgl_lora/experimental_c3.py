"""Compatibility wrapper for the former benchmark-only C3 entry point."""

from __future__ import annotations

from sglang.srt.lora.sgl_lora.bf16_execution import run_sgl_lora_moe_c3


def run_sgl_lora_moe_c3_experimental(*args, **kwargs):
    return run_sgl_lora_moe_c3(*args, **kwargs)


__all__ = [
    "run_sgl_lora_moe_c3_experimental",
]
