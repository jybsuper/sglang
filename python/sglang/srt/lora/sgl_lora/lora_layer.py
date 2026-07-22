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

from dataclasses import replace

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

    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatcher

    if not isinstance(base_layer.dispatcher, StandardDispatcher):
        violations.append(
            "dispatcher must use the Standard dispatch/combine ABI "
            f"(got {type(base_layer.dispatcher).__name__})"
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


def _fp8_resident_scale_abi_violations(
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    requires_packed_ue8m0: bool,
) -> list[str]:
    """Validate the load-time weight-scale representation borrowed by SGL."""
    if not requires_packed_ue8m0:
        return []
    violations = []
    for name, scale in (("w13", w13_scale), ("w2", w2_scale)):
        if scale.dtype != torch.int32 or not getattr(scale, "format_ue8m0", False):
            violations.append(
                f"{name} scale is not resident packed UE8M0 "
                f"(dtype={scale.dtype}, format_ue8m0="
                f"{getattr(scale, 'format_ue8m0', False)})"
            )
    return violations


def _effective_sgl_lora_runner_config(base_layer):
    """Return the post-topk scaling contract consumed by SGL stages."""
    cfg = base_layer.moe_runner_config
    if not base_layer.should_fuse_routed_scaling_factor_in_topk:
        return cfg
    # TopK already multiplied its weights by the routed scale.  Base finalize
    # and down LoRA must therefore consume a neutral factor or they would apply
    # it a second time.
    return replace(cfg, routed_scaling_factor=1.0)


def validate_sgl_lora_factor_dtypes(contract, **factor_groups) -> None:
    """Validate persistent pool factors once when a layer binds its buffers."""
    for factor_name, factor_or_factors in factor_groups.items():
        factors = (
            factor_or_factors
            if isinstance(factor_or_factors, (tuple, list))
            else (factor_or_factors,)
        )
        for factor in factors:
            if factor.dtype != contract.lora_delta_dtype:
                raise TypeError(
                    f"sgl_lora requires {contract.lora_delta_dtype} "
                    f"{factor_name}, got {factor.dtype}"
                )


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
        if quant_method.load_up_proj_weight_first:
            raise NotImplementedError(
                "resident BF16 W13 stores [Up,Gate], but the current sgl_lora "
                "provider requires canonical [Gate,Up]"
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
        if quant_method.quant_config.activation_scheme != "dynamic":
            raise NotImplementedError(
                "static-activation FP8 checkpoints require resident input scales; "
                "the current sgl_lora FP8 provider supports dynamic activation "
                "quantization only"
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
        from sglang.srt.layers import deep_gemm_wrapper

        scale_violations = _fp8_resident_scale_abi_violations(
            w13_scale,
            w2_scale,
            requires_packed_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        )
        if scale_violations:
            raise NotImplementedError(
                "Blackwell DeepGEMM FP8 requires weights/scales prepared by a "
                "resident DeepGEMM runner; " + "; ".join(scale_violations)
            )
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
    layer._sgl_lora_runner_config = _effective_sgl_lora_runner_config(base_layer)
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
    if cfg.num_fused_shared_experts:
        # Contiguous global/local layouts already place shared slots after the
        # routed factors and need no lookup. DeepEP/MegaMOE global physical
        # layouts interleave shared slots per rank, so routed IDs contain gaps.
        # Build both factor-domain maps once: the memory pool may retain global
        # factors or pack only this rank's routed factors.
        from sglang.srt.layers.moe.utils import uses_per_rank_fused_shared_slots
        from sglang.srt.lora.sgl_lora.shared_experts import (
            MoeLoraExpertTopology,
            build_routed_expert_id_map,
        )

        if uses_per_rank_fused_shared_slots():
            num_routed = int(base_layer._num_global_routed)
            ep_size = int(base_layer.moe_ep_size)
            ep_rank = int(base_layer.moe_ep_rank)
            device = layer._quant_info.w13_weight.device
            for factor_domain in ("global", "local"):
                topology = MoeLoraExpertTopology(
                    num_routed_experts=num_routed,
                    num_fused_shared_experts=cfg.num_fused_shared_experts,
                    ep_size=ep_size,
                    ep_rank=ep_rank,
                    id_layout="global_per_rank_shared",
                    factor_domain=factor_domain,
                )
                layer._sgl_lora_base_gemm.lora_expert_id_maps[
                    topology.num_factor_experts
                ] = build_routed_expert_id_map(topology, device=device)


def dispatch_sgl_lora_moe(
    dispatch_output,
    wrapper,
    lora_info,
    *,
    output_dtype: torch.dtype | None = None,
):
    """Route one MoE-LoRA forward (see module docstring for the policy)."""
    from sglang.srt.lora.sgl_lora.execution import run_sgl_lora_moe_plan
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
        # Preserve the model's resident no-LoRA implementation and physical
        # weight representation.  This is also the fixed topology captured by
        # the ``nolora`` CUDA-graph family.
        result = base_layer.quant_method.apply(
            layer=base_layer,
            dispatch_output=dispatch_output,
        )
        if output_dtype is not None and result.hidden_states.dtype != output_dtype:
            # Keep the public LoRA layer output contract independent of whether
            # this replay selected the active-LoRA or resident no-LoRA graph.
            return type(result)(hidden_states=result.hidden_states.to(output_dtype))
        return result

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
        base_lora_expert_domains_match=not bool(
            base_layer.moe_runner_config.num_fused_shared_experts
        ),
        provider_key=wrapper._sgl_lora_base_gemm.contract.key,
    )
    return run_sgl_lora_moe_plan(
        dispatch_output,
        wrapper._quant_info,
        wrapper._sgl_lora_runner_config,
        lora_info,
        wrapper._sgl_lora_base_gemm,
        plan,
        output_dtype=output_dtype,
    )
