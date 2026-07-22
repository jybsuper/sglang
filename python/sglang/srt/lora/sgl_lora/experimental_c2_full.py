"""Compatibility wrapper for the former benchmark-only complete C2 path."""

from __future__ import annotations

from sglang.srt.lora.sgl_lora.bf16_execution import run_sgl_lora_moe_c2_full


def run_sgl_lora_moe_c2_full_experimental(*args, **kwargs):
    return run_sgl_lora_moe_c2_full(*args, **kwargs)


__all__ = ["run_sgl_lora_moe_c2_full_experimental"]
