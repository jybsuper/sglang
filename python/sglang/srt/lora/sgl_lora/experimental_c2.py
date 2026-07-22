"""Compatibility wrapper for the former benchmark-only C2 entry point.

Benchmarks now execute the same implementation as production dispatch.
"""

from __future__ import annotations

import torch


def _validate_token_lora_mapping(
    token_lora_mapping: torch.Tensor, num_tokens: int
) -> None:
    if token_lora_mapping.shape[0] != num_tokens:
        raise RuntimeError(
            "sgl_lora token/adapter assignment does not match the MoE token "
            f"domain: mapping has {token_lora_mapping.shape[0]} rows but the "
            f"runner received {num_tokens}. Gather/remap assignments before "
            "MoE-DP execution."
        )


def _scaled_down_lora_topk_weights(
    topk_weights: torch.Tensor, routed_scaling_factor: float | None
) -> torch.Tensor:
    if routed_scaling_factor is None or routed_scaling_factor == 1.0:
        return topk_weights
    return topk_weights * float(routed_scaling_factor)


def run_sgl_lora_moe_c2_experimental(*args, **kwargs):
    from sglang.srt.lora.sgl_lora.bf16_execution import (
        run_sgl_lora_moe_c2_partial,
    )

    return run_sgl_lora_moe_c2_partial(*args, **kwargs)


__all__ = ["run_sgl_lora_moe_c2_experimental"]
