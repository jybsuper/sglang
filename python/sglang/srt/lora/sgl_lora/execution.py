"""Provider-neutral execution of a host-resolved SGL-LoRA MoE plan.

The execution planner is shared by every base provider.  Quantized providers
currently select the established C0 serial topology, whose base stages remain
fully provider-native.  The BF16 provider may additionally select the fused
C2/C3 implementations in :mod:`bf16_execution`.

Keeping this dispatch outside ``bf16_execution.py`` is important for the
production boundary: adding a quantized provider does not require pretending
its FP8/FP4 activation representation is a BF16 implementation detail.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.lora.sgl_lora.base_gemm import MoeLoraBaseGemm
    from sglang.srt.lora.sgl_lora.execution_plan import MoeLoraExecutionPlan
    from sglang.srt.lora.sgl_lora.quant_info import SglLoraQuantInfo


def run_sgl_lora_moe_plan(
    dispatch_output: StandardDispatchOutput,
    quant_info: SglLoraQuantInfo,
    runner_config: MoeRunnerConfig,
    lora_info,
    base: MoeLoraBaseGemm,
    plan: MoeLoraExecutionPlan,
    *,
    output_dtype: torch.dtype | None = None,
) -> StandardCombineInput:
    """Execute ``plan`` without making another topology decision.

    C0 is the common provider ABI and therefore works for BF16, FP8 W8A8,
    native NVFP4 W4A4, and Marlin W4A16.  C2/C3 own BF16-specific fused
    activation/finalize kernels and are imported only when such a plan was
    selected.
    """
    from sglang.srt.lora.sgl_lora.execution_plan import MoeLoraExecutionPath

    if plan.path is MoeLoraExecutionPath.C0_SERIAL:
        from sglang.srt.lora.sgl_lora.moe_lora_runner import run_sgl_lora_moe

        return run_sgl_lora_moe(
            dispatch_output,
            quant_info,
            runner_config,
            lora_info,
            base,
            two_stream_enabled=False,
            output_dtype=output_dtype,
            shared_outer_gate_a_plan=plan.shared_outer_gate_a_plan,
        )

    if base.contract.key != "deepgemm_bf16":
        raise RuntimeError(
            f"{plan.path.value} is a fused BF16 topology and cannot execute "
            f"provider {base.contract.key!r}"
        )

    from sglang.srt.lora.sgl_lora.bf16_execution import (
        run_sgl_lora_moe_c2_full,
        run_sgl_lora_moe_c2_partial,
        run_sgl_lora_moe_c3,
    )

    kwargs = dict(
        consumer_schedule=plan.consumer_schedule,
        consumer_block_size_n=plan.consumer_block_size_n,
        consumer_num_warps=plan.consumer_num_warps,
        has_base_rows=plan.has_base_rows,
        output_dtype=output_dtype,
        shared_outer_gate_a_plan=plan.shared_outer_gate_a_plan,
    )
    if plan.path is MoeLoraExecutionPath.C2_PARTIAL:
        return run_sgl_lora_moe_c2_partial(
            dispatch_output,
            quant_info,
            runner_config,
            lora_info,
            base,
            consumer_schedule=plan.consumer_schedule,
            block_size_n=plan.consumer_block_size_n,
            num_warps=plan.consumer_num_warps,
            has_base_rows=plan.has_base_rows,
            output_dtype=output_dtype,
            shared_outer_gate_a_plan=plan.shared_outer_gate_a_plan,
        )
    kwargs.update(
        finalize_block_size_h=plan.finalize_block_size_h,
        finalize_num_warps=plan.finalize_num_warps,
    )
    if plan.path is MoeLoraExecutionPath.C2_FULL:
        return run_sgl_lora_moe_c2_full(
            dispatch_output, quant_info, runner_config, lora_info, base, **kwargs
        )
    if plan.path is MoeLoraExecutionPath.C3_OVERLAP:
        return run_sgl_lora_moe_c3(
            dispatch_output, quant_info, runner_config, lora_info, base, **kwargs
        )
    raise AssertionError(f"unhandled SGL-LoRA execution path {plan.path!r}")


__all__ = ["run_sgl_lora_moe_plan"]
