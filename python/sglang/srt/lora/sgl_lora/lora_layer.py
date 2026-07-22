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

import torch

from sglang.srt.lora.sgl_lora.quant_info import SglLoraBf16QuantInfo


def _phase1a_contract_violations(base_layer) -> list[str]:
    """Return the unsupported semantics that would otherwise be silent.

    This is deliberately one attach-time boundary rather than scattered
    forward-path assertions.  Later provider implementations can consume and
    retire individual entries as their contracts expand.
    """
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

    cfg = base_layer.moe_runner_config
    quant_method = base_layer.quant_method
    violations = []

    if not isinstance(quant_method, UnquantizedFusedMoEMethod):
        violations.append(
            "quant_method must be UnquantizedFusedMoEMethod "
            f"(got {type(quant_method).__name__})"
        )
    else:
        if quant_method.use_triton_kernels:
            violations.append("transposed Triton-kernel weights are unsupported")
        if quant_method.use_flashinfer_trtllm_moe:
            violations.append("BlockMajorK TRT-LLM weights are unsupported")
        if quant_method.with_bias:
            violations.append("expert bias is unsupported")

    w13 = base_layer.w13_weight
    w2 = base_layer.w2_weight
    if w13.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        violations.append(
            f"base weights must be BF16 (got w13={w13.dtype}, w2={w2.dtype})"
        )
    if w13.ndim != 3 or w2.ndim != 3:
        violations.append(
            f"base weights must use canonical 3-D [E,N,K] layout "
            f"(got w13={tuple(w13.shape)}, w2={tuple(w2.shape)})"
        )
    elif (
        w13.shape[0] != w2.shape[0]
        or w13.shape[1] % 2
        or w13.shape[2] != w2.shape[1]
        or w13.shape[1] // 2 != w2.shape[2]
    ):
        violations.append(
            f"base weight shapes must be w13=[E,2I,H], w2=[E,H,I] "
            f"(got w13={tuple(w13.shape)}, w2={tuple(w2.shape)})"
        )

    if cfg.activation != "silu" or not cfg.is_gated:
        violations.append(
            f"activation must be gated SiLU (got {cfg.activation!r}, "
            f"is_gated={cfg.is_gated})"
        )
    special_activation = {
        "gemm1_alpha": cfg.gemm1_alpha,
        "gemm1_clamp_limit": cfg.gemm1_clamp_limit,
        "swiglu_limit": cfg.swiglu_limit,
    }
    enabled_special = {k: v for k, v in special_activation.items() if v is not None}
    if enabled_special:
        violations.append(
            f"special gated-activation parameters are unsupported ({enabled_special})"
        )
    if cfg.apply_router_weight_on_input:
        violations.append("apply_router_weight_on_input=True is unsupported")
    if cfg.no_combine:
        violations.append("no_combine=True is unsupported")
    if cfg.num_fused_shared_experts:
        violations.append(
            f"fused shared experts are unsupported "
            f"(got {cfg.num_fused_shared_experts})"
        )
    if cfg.use_tp_all_gather_activation:
        violations.append("TP all-gather activation input is unsupported")

    return violations


def init_sgl_lora_moe(layer, base_layer) -> None:
    """Build per-layer state at LoRA-attach time (standard layouts, no prep).

    Phase 1a supports BF16 ``UnquantizedFusedMoEMethod`` only.
    """
    from sglang.srt.lora.sgl_lora.base_gemm import resolve_base_gemm
    from sglang.srt.lora.sgl_lora.workspace import MoeLoraWorkspacePlanner

    violations = _phase1a_contract_violations(base_layer)
    if violations:
        raise NotImplementedError(
            "sgl_lora Phase-1a cannot preserve this MoE layer's semantics: "
            + "; ".join(violations)
        )

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
    # Every MoE layer executes sequentially through one backend.  Sharing the
    # planner makes one admission decision per device/shape instead of issuing
    # a host memory query at every decoder layer.
    workspace_planner = getattr(layer.lora_backend, "_sgl_lora_workspace_planner", None)
    if workspace_planner is None:
        workspace_planner = MoeLoraWorkspacePlanner()
        layer.lora_backend._sgl_lora_workspace_planner = workspace_planner
    layer._sgl_lora_base_gemm = resolve_base_gemm(
        layer._quant_info, cfg, workspace_planner
    )

    # The no-active eager path runs the stock base-only Triton strategy with
    # the same standard-layout weight tensors and zero extra weight memory.
    layer._sgl_lora_triton_qi = base_layer.quant_method.get_triton_quant_info(
        base_layer
    )


def dispatch_sgl_lora_moe(dispatch_output, wrapper, lora_info):
    """Route one MoE-LoRA forward (see module docstring for the policy)."""
    from sglang.srt.lora.sgl_lora.moe_lora_runner import (
        resolve_lora_two_stream_auto,
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
        two_stream_enabled=resolve_lora_two_stream_auto(
            requested=wrapper._sgl_lora_two_stream,
            num_tokens=dispatch_output.hidden_states.shape[0],
        ),
    )
