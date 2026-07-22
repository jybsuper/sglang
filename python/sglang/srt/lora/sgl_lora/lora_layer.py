"""Per-layer initialization and dispatch for SGL LoRA MoE execution.

Attach time (``init_sgl_lora_moe``) builds and caches on the wrapper layer:
  * ``_quant_info``          — a typed, provider-resident weight bundle
  * ``_sgl_lora_base_gemm``  — the selected decomposed base provider

Dispatch routes per batch:
  * eager batch with NO active adapter  -> the layer's stock quant method
    (same backend/representation as a model without LoRA)
  * otherwise -> the SGL LoRA pipeline.
"""

from __future__ import annotations

import torch

from sglang.srt.lora.sgl_lora.quant_info import (
    SglLoraBf16QuantInfo,
    SglLoraFp8QuantInfo,
    SglLoraMarlinQuantInfo,
    SglLoraQuantInfo,
)


def _use_stock_base_path(
    *, has_active_lora: bool, capture_mode: bool, capture_variant: str | None
) -> bool:
    """Select the graph topology without reading mutable adapter metadata."""
    return not has_active_lora and (not capture_mode or capture_variant == "nolora")


def _phase1a_contract_violations(base_layer) -> list[str]:
    """Return the unsupported semantics that would otherwise be silent.

    This is deliberately one attach-time boundary rather than scattered
    forward-path assertions.  Later provider implementations can consume and
    retire individual entries as their contracts expand.
    """
    cfg = base_layer.moe_runner_config
    violations = []

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


def _canonical_weight_violations(
    w13: torch.Tensor,
    w2: torch.Tensor,
    *,
    expected_dtype: torch.dtype,
) -> list[str]:
    violations = []
    if w13.dtype != expected_dtype or w2.dtype != expected_dtype:
        violations.append(
            f"provider weights must be {expected_dtype} "
            f"(got w13={w13.dtype}, w2={w2.dtype})"
        )
    if w13.ndim != 3 or w2.ndim != 3:
        violations.append(
            "provider weights must use canonical 3-D [E,N,K] layout "
            f"(got w13={tuple(w13.shape)}, w2={tuple(w2.shape)})"
        )
    elif (
        w13.shape[0] != w2.shape[0]
        or w13.shape[1] % 2
        or w13.shape[2] != w2.shape[1]
        or w13.shape[1] // 2 != w2.shape[2]
    ):
        violations.append(
            "provider shapes must be w13=[E,2I,H], w2=[E,H,I] "
            f"(got w13={tuple(w13.shape)}, w2={tuple(w2.shape)})"
        )
    return violations


def _from_marlin_quant_info(base_layer, quant_info) -> SglLoraMarlinQuantInfo:
    """Translate the stock runner payload without copying packed weights."""
    expert_map = quant_info.expert_map
    global_num_experts = quant_info.global_num_experts
    if expert_map is None:
        # Some stock Marlin quant-info builders add EP mapping only at apply()
        # time. Preserve that semantic boundary here so the decomposed provider
        # can reject unsupported global IDs instead of indexing local weights
        # with them silently.
        expert_map = getattr(
            getattr(base_layer, "dispatcher", None),
            "local_expert_mapping",
            None,
        )
        if expert_map is not None:
            global_num_experts = base_layer.moe_runner_config.num_experts
    return SglLoraMarlinQuantInfo(
        w13_weight=quant_info.w13_qweight,
        w2_weight=quant_info.w2_qweight,
        w13_scales=quant_info.w13_scales,
        w2_scales=quant_info.w2_scales,
        weight_bits=quant_info.weight_bits,
        num_local_experts=base_layer.num_local_experts,
        intermediate_size=base_layer.intermediate_size_per_partition,
        hidden_size=base_layer.hidden_size,
        w13_g_idx_sort_indices=quant_info.w13_g_idx_sort_indices,
        w2_g_idx_sort_indices=quant_info.w2_g_idx_sort_indices,
        w13_g_idx=quant_info.w13_g_idx,
        w2_g_idx=quant_info.w2_g_idx,
        w13_qzeros=quant_info.w13_qzeros,
        w2_qzeros=quant_info.w2_qzeros,
        w13_global_scale=quant_info.w13_global_scale,
        w2_global_scale=quant_info.w2_global_scale,
        w13_bias=quant_info.w13_bias,
        w2_bias=quant_info.w2_bias,
        expert_map=expert_map,
        global_num_experts=global_num_experts,
        is_k_full=quant_info.is_k_full,
    )


def build_sgl_lora_quant_info(base_layer) -> SglLoraQuantInfo:
    """Build a typed provider payload from one already-processed MoE layer.

    This function never converts or reorders a base weight.  Resident layouts
    whose stock kernels hide the activation boundary are rejected with the
    representation named in the error.
    """
    from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
    from sglang.srt.layers.quantization.modelopt_quant import (
        ModelOptNvFp4FusedMoEMethod,
    )
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

    quant_method = base_layer.quant_method
    if isinstance(quant_method, UnquantizedFusedMoEMethod):
        if quant_method.use_triton_kernels:
            raise NotImplementedError(
                "transposed Triton-kernel BF16 weights do not expose canonical [E,N,K]"
            )
        if quant_method.use_flashinfer_trtllm_moe:
            raise NotImplementedError(
                "BlockMajorK TRT-LLM BF16 weights do not expose the LoRA activation seam"
            )
        if quant_method.with_bias:
            raise NotImplementedError("BF16 expert bias is not supported")
        w13, w2 = base_layer.w13_weight, base_layer.w2_weight
        violations = _canonical_weight_violations(
            w13, w2, expected_dtype=torch.bfloat16
        )
        if violations:
            raise NotImplementedError("; ".join(violations))
        return SglLoraBf16QuantInfo(
            w13_weight=w13,
            w2_weight=w2,
            num_local_experts=w13.shape[0],
            intermediate_size=w13.shape[1] // 2,
            hidden_size=w13.shape[2],
        )

    if isinstance(quant_method, Fp8MoEMethod):
        if quant_method.with_bias:
            raise NotImplementedError("DeepGEMM FP8 expert bias is unsupported")
        if quant_method.is_fp4_expert:
            raise NotImplementedError(
                "FP4-expert tensors carried by Fp8MoEMethod require a distinct recipe"
            )
        w13, w2 = base_layer.w13_weight, base_layer.w2_weight
        violations = _canonical_weight_violations(
            w13, w2, expected_dtype=torch.float8_e4m3fn
        )
        if violations:
            raise NotImplementedError("; ".join(violations))

        if quant_method.block_quant:
            block_shape = quant_method.quant_config.weight_block_size
            w13_scale = base_layer.w13_weight_scale_inv
            w2_scale = base_layer.w2_weight_scale_inv
        else:
            # DeepGEMM consumes block-scale tensors.  Materialize the same
            # scale views as the stock FP8 path once at attach time.
            block_shape = [128, 128]
            block_n, block_k = block_shape
            w13_scale = (
                base_layer.w13_weight_scale.unsqueeze(1)
                .repeat_interleave((w13.shape[1] + block_n - 1) // block_n, dim=1)
                .unsqueeze(2)
                .repeat_interleave((w13.shape[2] + block_k - 1) // block_k, dim=2)
            )
            w2_scale = (
                base_layer.w2_weight_scale.unsqueeze(1)
                .repeat_interleave((w2.shape[1] + block_n - 1) // block_n, dim=1)
                .unsqueeze(2)
                .repeat_interleave((w2.shape[2] + block_k - 1) // block_k, dim=2)
            )
        if block_shape is None:
            raise NotImplementedError("FP8 provider requires an explicit block shape")
        return SglLoraFp8QuantInfo(
            w13_weight=w13,
            w2_weight=w2,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
            block_shape=(int(block_shape[0]), int(block_shape[1])),
            num_local_experts=w13.shape[0],
            intermediate_size=w13.shape[1] // 2,
            hidden_size=w13.shape[2],
            use_mxfp8=quant_method.use_mxfp8,
        )

    if isinstance(quant_method, ModelOptNvFp4FusedMoEMethod):
        backend = getattr(quant_method, "_moe_runner_backend", None)
        if backend is not None and backend.is_marlin():
            from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo

            expert_map = getattr(
                getattr(base_layer, "dispatcher", None),
                "local_expert_mapping",
                None,
            )
            stock_qi = MarlinMoeQuantInfo(
                w13_qweight=base_layer.w13_weight,
                w2_qweight=base_layer.w2_weight,
                w13_scales=base_layer.w13_weight_scale,
                w2_scales=base_layer.w2_weight_scale,
                w13_g_idx_sort_indices=None,
                w2_g_idx_sort_indices=None,
                weight_bits=4,
                w13_global_scale=base_layer.w13_weight_scale_2,
                w2_global_scale=base_layer.w2_weight_scale_2,
                expert_map=expert_map,
                global_num_experts=(
                    base_layer.moe_runner_config.num_experts
                    if expert_map is not None
                    else -1
                ),
            )
            return _from_marlin_quant_info(base_layer, stock_qi)
        if getattr(quant_method, "_is_cutedsl_v2_standard", False):
            raise NotImplementedError(
                "CuteDSL-v2 NVFP4 stores [Up,Gate] interleaved weights and MMA "
                "scales in a fused kernel with no post-W13 injection point"
            )
        if quant_method.enable_flashinfer_trtllm_moe:
            raise NotImplementedError(
                "TRT-LLM NVFP4 owns activation/A2 quantization and cannot inject "
                "a BF16 gate/up LoRA delta"
            )
        raise NotImplementedError(
            "canonical CuteDSL-v1 NVFP4 is available only with DeepEP masked "
            "dispatch, which is not yet accepted by the unified Standard runner"
        )

    # The generic CompressedTensors method exposes a forwarding
    # ``get_marlin_quant_info`` for every scheme, including schemes that do not
    # implement Marlin. Inspect the resident scheme/runner rather than treating
    # a wrapper method (or a missing runner) as proof of a Marlin layout.
    resident_scheme = getattr(base_layer, "scheme", None)
    marlin_source = resident_scheme if resident_scheme is not None else quant_method
    get_marlin_quant_info = getattr(marlin_source, "get_marlin_quant_info", None)
    runner_backend = getattr(
        getattr(marlin_source, "runner", None), "runner_backend", None
    )
    if (
        callable(get_marlin_quant_info)
        and runner_backend is not None
        and runner_backend.is_marlin()
    ):
        return _from_marlin_quant_info(base_layer, get_marlin_quant_info(base_layer))

    raise NotImplementedError(
        "no injectable SGL LoRA provider for resident quant method "
        f"{type(quant_method).__name__}"
    )


def init_sgl_lora_moe(layer, base_layer) -> None:
    """Build per-layer provider state at LoRA-attach time without weight prep."""
    from sglang.srt.lora.sgl_lora.base_gemm import resolve_base_gemm
    from sglang.srt.lora.sgl_lora.workspace import MoeLoraWorkspacePlanner

    violations = _phase1a_contract_violations(base_layer)
    if violations:
        raise NotImplementedError(
            "sgl_lora cannot preserve this MoE layer's semantics: "
            + "; ".join(violations)
        )

    cfg = base_layer.moe_runner_config
    layer._lora_runner = None
    layer._quant_info = build_sgl_lora_quant_info(base_layer)
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
    # Kept for canonical BF16 no-adapter dispatch. Quant providers use
    # quant_method.apply so the stock path retains its selected physical
    # backend.
    get_triton_quant_info = (
        getattr(base_layer.quant_method, "get_triton_quant_info", None)
        if isinstance(layer._quant_info, SglLoraBf16QuantInfo)
        else None
    )
    layer._sgl_lora_triton_qi = (
        get_triton_quant_info(base_layer) if callable(get_triton_quant_info) else None
    )


def dispatch_sgl_lora_moe(
    dispatch_output,
    wrapper,
    lora_info,
    *,
    output_dtype: torch.dtype | None = None,
):
    """Route one MoE-LoRA forward (see module docstring for the policy)."""
    from sglang.srt.lora.sgl_lora.bf16_execution import (
        run_sgl_lora_moe_bf16_plan,
    )
    from sglang.srt.lora.sgl_lora.execution_plan import (
        build_moe_lora_execution_plan,
    )
    from sglang.srt.model_executor.runner_utils.capture_mode import (
        get_capture_lora_variant,
        get_is_capture_mode,
    )

    base_layer = wrapper.base_layer

    capture_mode = get_is_capture_mode()
    capture_variant = get_capture_lora_variant() if capture_mode else None
    use_stock_base = _use_stock_base_path(
        has_active_lora=lora_info.has_active_lora,
        capture_mode=capture_mode,
        capture_variant=capture_variant,
    )
    if use_stock_base:
        # Preserve the model's resident no-LoRA implementation. Canonical BF16
        # retains its established Triton control; packed quantized providers
        # must go through quant_method.apply rather than a fabricated Triton
        # payload or a provider-private layout conversion.
        if wrapper._sgl_lora_triton_qi is not None:
            from sglang.srt.layers.moe.moe_runner.triton import (
                fused_experts_none_to_triton,
            )

            return fused_experts_none_to_triton(
                dispatch_output,
                wrapper._sgl_lora_triton_qi,
                base_layer.moe_runner_config,
            )
        return base_layer.quant_method.apply(
            layer=base_layer,
            dispatch_output=dispatch_output,
        )

    plan = build_moe_lora_execution_plan(
        phase=lora_info.forward_phase,
        graph_mode=lora_info.use_cuda_graph,
        num_tokens=dispatch_output.hidden_states.shape[0],
        rank=lora_info.max_lora_rank,
        has_base_rows=lora_info.has_base_rows,
        two_stream_requested=wrapper._sgl_lora_two_stream,
        fused_supported=(
            wrapper._sgl_lora_base_gemm.contract.key == "deepgemm_bf16"
            and lora_info.lora_use_virtual_experts
            and not lora_info.fully_sharded
        ),
    )
    return run_sgl_lora_moe_bf16_plan(
        dispatch_output,
        wrapper._quant_info,
        base_layer.moe_runner_config,
        lora_info,
        wrapper._sgl_lora_base_gemm,
        plan,
        output_dtype=output_dtype,
    )
