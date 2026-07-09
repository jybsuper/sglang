"""Per-layer initialization and dispatch for SGL LoRA MoE execution.

Attach time (``init_sgl_lora_moe``) builds and caches on the wrapper layer:
  * ``_quant_info``             — standard-layout weight bundle (no prep)
  * ``_sgl_lora_base_gemm`` — the BF16 DeepGEMM provider
  * ``_sgl_lora_triton_qi`` — Triton quant info for the eager base-only strategy

Dispatch routes per batch:
  * eager batch with NO active adapter  -> stock triton fused base path
    (byte-identical no-LoRA outputs — parity the acc gates rely on)
  * otherwise -> the SGL LoRA pipeline.
"""

from __future__ import annotations

from sglang.srt.lora.sgl_lora.quant_info import SglLoraBf16QuantInfo


def init_sgl_lora_moe(layer, base_layer) -> None:
    """Build per-layer state at LoRA-attach time (standard layouts, no prep).

    Phase 1a supports BF16 ``UnquantizedFusedMoEMethod`` only.
    """
    from sglang.srt.lora.sgl_lora.base_gemm import resolve_base_gemm

    w13 = base_layer.w13_weight
    w2 = base_layer.w2_weight
    # Standard layouts: w13 [E_local, 2*inter, hidden], w2 [E_local, hidden, inter].
    num_local_experts, two_inter, hidden = w13.shape

    cfg = base_layer.moe_runner_config
    layer._lora_runner = None
    layer._quant_info = SglLoraBf16QuantInfo(
        w13_weight=w13,
        w2_weight=w2,
        num_local_experts=num_local_experts,
        intermediate_size=two_inter // 2,
        hidden_size=hidden,
    )
    layer._sgl_lora_base_gemm = resolve_base_gemm(layer._quant_info, cfg)

    # The no-active eager path runs the stock base-only Triton strategy with
    # the same standard-layout weight tensors and zero extra weight memory.
    layer._sgl_lora_triton_qi = base_layer.quant_method.get_triton_quant_info(
        base_layer
    )


def dispatch_sgl_lora_moe(dispatch_output, wrapper, lora_info):
    """Route one MoE-LoRA forward (see module docstring for the policy)."""
    from sglang.srt.lora.sgl_lora.moe_lora_runner import (
        run_sgl_lora_moe,
    )
    from sglang.srt.model_executor.runner_utils.capture_mode import (
        get_is_capture_mode,
    )

    base_layer = wrapper.base_layer

    if not get_is_capture_mode() and not lora_info.has_active_lora:
        # Byte-identical stock base path for no-adapter batches.
        from sglang.srt.layers.moe.moe_runner.triton import (
            fused_experts_none_to_triton,
        )

        return fused_experts_none_to_triton(
            dispatch_output,
            wrapper._sgl_lora_triton_qi,
            base_layer.moe_runner_config,
        )

    return run_sgl_lora_moe(
        dispatch_output,
        wrapper._quant_info,
        base_layer.moe_runner_config,
        lora_info,
        wrapper._sgl_lora_base_gemm,
        enable_two_stream=wrapper._sgl_lora_two_stream,
    )
