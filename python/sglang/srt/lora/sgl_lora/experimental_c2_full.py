"""Benchmark-only complete BF16 C2 topology.

This composes the experimental fused gate/up consumer with the token-owned
down-B/base-finalize kernel.  It deliberately enters through the callback seam
of :mod:`experimental_c2`; production dispatch does not import this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.lora.sgl_lora.base_gemm import MoeLoraBaseGemm
    from sglang.srt.lora.sgl_lora.quant_info import SglLoraBf16QuantInfo


def run_sgl_lora_moe_c2_full_experimental(
    dispatch_output: StandardDispatchOutput,
    quant_info: SglLoraBf16QuantInfo,
    runner_config: MoeRunnerConfig,
    lora_info,
    base: MoeLoraBaseGemm,
    *,
    consumer_schedule: str = "pair",
    consumer_block_size_n: int = 32,
    consumer_num_warps: int = 4,
    finalize_block_size_h: int = 32,
    finalize_num_warps: int = 4,
    has_base_rows: bool = True,
) -> StandardCombineInput:
    """Run complete serial C2 without changing serving dispatch."""
    from sglang.srt.lora.sgl_lora.experimental_c2 import (
        run_sgl_lora_moe_c2_experimental,
    )
    from sglang.srt.lora.sgl_lora.triton_ops.fused_down_finalize import (
        fused_down_b_finalize,
    )

    def down_finalize(
        ws,
        down_out,
        down_intermediate,
        output,
        topk_ids,
        topk_weights,
        token_lora_mapping,
        callback_lora_info,
        callback_runner_config,
    ) -> None:
        fused_down_b_finalize(
            down_out,
            down_intermediate,
            callback_lora_info.down_lora_b_weights,
            output,
            ws.src2dst,
            topk_ids,
            topk_weights,
            token_lora_mapping,
            routed_scaling_factor=(
                callback_runner_config.routed_scaling_factor
                if callback_runner_config.routed_scaling_factor is not None
                else 1.0
            ),
            # The current BF16 C2P runner is an EP-local/offset-zero slice.
            # The raw finalizer itself is tested with nonzero global offsets.
            local_expert_offset=0,
            shared_outer=callback_lora_info.experts_shared_outer_loras,
            block_size_h=finalize_block_size_h,
            num_warps=finalize_num_warps,
        )

    return run_sgl_lora_moe_c2_experimental(
        dispatch_output,
        quant_info,
        runner_config,
        lora_info,
        base,
        consumer_schedule=consumer_schedule,
        block_size_n=consumer_block_size_n,
        num_warps=consumer_num_warps,
        down_finalize=down_finalize,
        has_base_rows=has_base_rows,
    )


__all__ = ["run_sgl_lora_moe_c2_full_experimental"]
