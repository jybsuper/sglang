"""Production BF16 fused execution paths for the SGL-LoRA MoE pipeline.

This module owns the C2 partial, C2 full, and C3 overlap implementations used
by serving dispatch. Benchmark modules are thin compatibility adapters over
these functions, so correctness fixes cannot diverge between measurement and
production. The caller supplies one immutable host-side execution plan.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.lora.sgl_lora.base_gemm import MoeLoraBaseGemm
    from sglang.srt.lora.sgl_lora.quant_info import (
        SglLoraBf16QuantInfo,
        SglLoraQuantInfo,
    )


def _admit_workspace(
    dispatch_output,
    lora_info,
    base,
    *,
    top_k: int,
    output_dtype: torch.dtype,
) -> None:
    """Preflight the provider workspace before any large runner allocation."""
    hidden_states = dispatch_output.hidden_states
    from sglang.srt.model_executor.runner_utils.capture_mode import (
        get_is_capture_mode,
    )

    inside_cuda_capture = (
        hidden_states.device.type == "cuda" and torch.cuda.is_current_stream_capturing()
    )
    base.admit_workspace(
        num_tokens=hidden_states.shape[0],
        top_k=top_k,
        rank=lora_info.max_lora_rank,
        max_loras=lora_info.gate_up_lora_a_weights.shape[0],
        # The current estimator scales all runner-owned storage by this item
        # size. Using the larger of the BF16 activation and requested output
        # keeps FP32 destinations conservatively admitted (at the cost of a
        # deliberate overestimate for BF16-only intermediates).
        dtype=output_dtype,
        device=hidden_states.device,
        capture=get_is_capture_mode() or inside_cuda_capture,
        memory_query_safe=not inside_cuda_capture,
    )


def _validate_token_lora_mapping(
    token_lora_mapping: torch.Tensor, num_tokens: int
) -> None:
    """Reject an adapter assignment expressed in a different token domain."""
    if token_lora_mapping.shape[0] != num_tokens:
        raise RuntimeError(
            "sgl_lora token/adapter assignment does not match the MoE token "
            f"domain: mapping has {token_lora_mapping.shape[0]} rows but the "
            f"runner received {num_tokens}. Gather/remap assignments before "
            "MoE-DP execution."
        )


def _resolve_output_dtype_and_validate(
    hidden_states: torch.Tensor,
    base,
    output_dtype: torch.dtype | None,
) -> torch.dtype:
    """Apply the provider boundary shared by C0 and fused BF16 plans."""
    output_dtype = hidden_states.dtype if output_dtype is None else output_dtype
    base.validate_runtime_inputs(hidden_states, output_dtype=output_dtype)
    return output_dtype


def _scaled_down_lora_topk_weights(
    topk_weights: torch.Tensor, routed_scaling_factor: float | None
) -> torch.Tensor:
    """Return the weights that apply the routed scale once to down LoRA-B."""
    if routed_scaling_factor is None or routed_scaling_factor == 1.0:
        return topk_weights
    return topk_weights * float(routed_scaling_factor)


def run_sgl_lora_moe_c2_partial(
    dispatch_output: StandardDispatchOutput,
    quant_info: SglLoraBf16QuantInfo,
    runner_config: MoeRunnerConfig,
    lora_info,
    base: MoeLoraBaseGemm,
    *,
    consumer_schedule: str = "pair",
    block_size_n: int = 32,
    num_warps: int = 4,
    down_finalize: Callable[..., None] | None = None,
    has_base_rows: bool = True,
    output_dtype: torch.dtype | None = None,
    shared_outer_gate_a_plan=None,
) -> StandardCombineInput:
    """Run the serial BF16 C2-consumer partial.

    The runner owns every destination allocation.  The fused consumer writes
    the provider-private masked BF16 W2 input and directly accumulates the
    canonical ``[token, top-k, rank]`` bridge consumed by down LoRA-B.

    ``down_finalize`` is the internal borrowed-buffer seam used by complete
    C2. When supplied, it replaces both base finalize and the decomposed
    down-B expand; this runner retains ownership of ``down_out`` and disposes
    it after the callback returns.

    ``has_base_rows`` is a static execution-plan property.  The planner may
    set it false for an all-active capture to omit the base-only activation
    fill; it must not be derived by scanning a device tensor during replay.
    """
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        use_symmetric_memory,
    )
    from sglang.srt.layers.dp_attention import is_allocation_symmetric
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.lora.sgl_lora.triton_ops.fused_c2 import (
        fused_gate_up_b_swiglu_down_a,
        fused_gate_up_b_swiglu_down_a_aligned,
    )
    from sglang.srt.lora.sgl_lora.triton_ops.virtual_experts import (
        merged_experts_fused_moe_lora_add,
    )
    from sglang.srt.utils import dispose_tensor

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    assert TopKOutputChecker.format_is_standard(topk_output)
    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights
    packed_topk_ids = getattr(topk_output, "packed_topk_ids", None)
    top_k = runner_config.top_k
    output_dtype = _resolve_output_dtype_and_validate(hidden_states, base, output_dtype)
    _admit_workspace(
        dispatch_output,
        lora_info,
        base,
        top_k=top_k,
        output_dtype=output_dtype,
    )
    num_tokens = hidden_states.shape[0]
    token_lora_mapping = lora_info.token_lora_mapping
    _validate_token_lora_mapping(token_lora_mapping, num_tokens)
    # Match C0's whole-forward route-plan reuse.  Prepare both A and B
    # schedules on the main stream before any stage kernel is launched.  This
    # keeps allocator/routing work off a future consumer side stream while the
    # following shrink and expand calls remain cache hits.
    fused_lora_routing_cache: dict = {}

    merged_experts_fused_moe_lora_add(
        output=None,
        hidden_states=hidden_states,
        lora_a=lora_info.gate_up_lora_a_weights,
        lora_b=lora_info.gate_up_lora_b_weights,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        token_lora_mapping=token_lora_mapping,
        mul_routed_weight=False,
        experts_shared_outer_loras_a=lora_info.experts_shared_outer_loras,
        experts_shared_outer_loras_b=False,
        routing_cache=fused_lora_routing_cache,
        fuse_add_to_output=False,
        use_direct_expand_add=False,
        num_output_slices=2,
        local_expert_offset=0,
        local_num_experts=quant_info.num_local_experts,
        stage="routing",
        shared_outer_gate_a_plan=shared_outer_gate_a_plan,
        segment_indptr=getattr(lora_info, "seg_indptr", None),
        segment_lora_ids=getattr(lora_info, "req_to_lora", None),
        max_segment_len=getattr(lora_info, "max_segment_len", 0),
    )

    # Gate/up A remains a standalone producer in C2.  Its 2R output is the
    # input boundary of the fused consumer; neither 2I delta nor I activation
    # is materialized in canonical pair order.
    gate_rank = lora_info.gate_up_lora_b_weights.shape[-1]
    gate_up_intermediate = torch.empty(
        (num_tokens, top_k, 2 * gate_rank),
        dtype=base.contract.lora_delta_dtype,
        device=hidden_states.device,
    )
    merged_experts_fused_moe_lora_add(
        output=gate_up_intermediate,
        hidden_states=hidden_states,
        lora_a=lora_info.gate_up_lora_a_weights,
        lora_b=lora_info.gate_up_lora_b_weights,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        token_lora_mapping=token_lora_mapping,
        mul_routed_weight=False,
        experts_shared_outer_loras_a=lora_info.experts_shared_outer_loras,
        experts_shared_outer_loras_b=False,
        routing_cache=fused_lora_routing_cache,
        fuse_add_to_output=False,
        use_direct_expand_add=False,
        num_output_slices=2,
        local_expert_offset=0,
        local_num_experts=quant_info.num_local_experts,
        stage="shrink",
        intermediate_buffer=gate_up_intermediate,
        shared_outer_gate_a_plan=shared_outer_gate_a_plan,
        segment_indptr=getattr(lora_info, "seg_indptr", None),
        segment_lora_ids=getattr(lora_info, "req_to_lora", None),
        max_segment_len=getattr(lora_info, "max_segment_len", 0),
    )

    ws = base.prepare(
        hidden_states,
        topk_ids,
        top_k,
        topk_weights=topk_weights,
        packed_topk_ids=packed_topk_ids,
    )
    gateup_out = torch.empty(
        base.gateup_out_shape(ws),
        dtype=base.contract.gate_up_output_dtype,
        device=hidden_states.device,
    )
    base.gateup(ws, gateup_out)
    if ws.hidden_permuted_owned:
        dispose_tensor(ws.hidden_permuted)

    act_out = torch.empty(
        base.act_out_shape(ws),
        dtype=base.contract.lora_activation_dtype,
        device=hidden_states.device,
    )
    down_rank = lora_info.down_lora_a_weights.shape[-2]
    # C2 accumulates one BF16 partial per I tile.  The explicit zero is part of
    # the measured plan and is safe under eager execution and graph replay.
    down_intermediate = torch.zeros(
        (num_tokens, top_k, down_rank),
        dtype=base.contract.lora_activation_dtype,
        device=hidden_states.device,
    )
    if consumer_schedule == "aligned":
        # A per-expert gate-A route and the gate-B route encode the same
        # virtual experts. Prefer the smallest cached block (normally BM16)
        # to limit the two simultaneous dot-product accumulator footprints.
        per_expert_routes = [
            (key, value)
            for key, value in fused_lora_routing_cache.items()
            if key[0] == lora_info.gate_up_lora_b_weights.shape[1] and not key[1]
        ]
        if not per_expert_routes:
            raise RuntimeError("aligned C2P requires a cached per-expert route")
        route_key, route = min(per_expert_routes, key=lambda item: item[0][2])
        sorted_pair_ids, virtual_expert_ids, num_pairs_post_padded, _ = route
        fused_gate_up_b_swiglu_down_a_aligned(
            gateup_out,
            gate_up_intermediate,
            lora_info.gate_up_lora_b_weights,
            lora_info.down_lora_a_weights,
            act_out,
            down_intermediate,
            ws.src2dst,
            topk_ids,
            sorted_pair_ids,
            virtual_expert_ids,
            num_pairs_post_padded,
            route_block_size_m=route_key[2],
            token_lora_mapping=(token_lora_mapping if has_base_rows else None),
            block_size_n=block_size_n,
            num_warps=num_warps,
        )
    elif consumer_schedule == "pair":
        fused_gate_up_b_swiglu_down_a(
            gateup_out,
            gate_up_intermediate,
            lora_info.gate_up_lora_b_weights,
            lora_info.down_lora_a_weights,
            act_out,
            down_intermediate,
            ws.src2dst,
            topk_ids,
            token_lora_mapping,
            local_expert_offset=0,
            block_size_n=block_size_n,
            num_warps=num_warps,
        )
    else:
        raise ValueError(f"unknown C2P consumer schedule {consumer_schedule!r}")
    dispose_tensor(gateup_out)
    dispose_tensor(gate_up_intermediate)

    down_out = torch.empty(
        base.down_out_shape(ws), dtype=torch.bfloat16, device=hidden_states.device
    )
    base.down(ws, act_out, down_out)
    dispose_tensor(act_out)

    with use_symmetric_memory(get_tp_group(), disabled=not is_allocation_symmetric()):
        output = torch.empty(
            (num_tokens, quant_info.hidden_size),
            dtype=output_dtype,
            device=hidden_states.device,
        )
    if down_finalize is None:
        base.finalize(
            ws,
            down_out,
            topk_ids,
            topk_weights,
            runner_config.routed_scaling_factor,
            output,
        )
        dispose_tensor(down_out)

        # The fused consumer has already produced down-A. Reuse the existing
        # routed down-B expansion from that explicit boundary.
        down_topk_weights = _scaled_down_lora_topk_weights(
            topk_weights, runner_config.routed_scaling_factor
        )
        merged_experts_fused_moe_lora_add(
            output=output,
            hidden_states=down_intermediate.view(-1, down_rank),
            lora_a=lora_info.down_lora_a_weights,
            lora_b=lora_info.down_lora_b_weights,
            topk_ids=topk_ids,
            topk_weights=down_topk_weights,
            token_lora_mapping=token_lora_mapping,
            mul_routed_weight=True,
            experts_shared_outer_loras_a=False,
            experts_shared_outer_loras_b=lora_info.experts_shared_outer_loras,
            routing_cache=fused_lora_routing_cache,
            fuse_add_to_output=False,
            fuse_sum_all_reduce=True,
            use_direct_expand_add=lora_info.max_lora_rank <= 64,
            num_output_slices=1,
            local_expert_offset=0,
            local_num_experts=quant_info.num_local_experts,
            stage="expand",
            intermediate_buffer=down_intermediate,
        )
    else:
        # Borrowed-buffer seam used by the complete C2 topology below.
        down_finalize(
            ws,
            down_out,
            down_intermediate,
            output,
            topk_ids,
            topk_weights,
            token_lora_mapping,
            lora_info,
            runner_config,
        )
        dispose_tensor(down_out)

    return StandardCombineInput(hidden_states=output)


def run_sgl_lora_moe_c2_full(
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
    output_dtype: torch.dtype | None = None,
    shared_outer_gate_a_plan=None,
) -> StandardCombineInput:
    """Run complete serial C2 with a fused down-B/base finalizer."""
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
            local_expert_offset=0,
            shared_outer=callback_lora_info.experts_shared_outer_loras,
            block_size_h=finalize_block_size_h,
            num_warps=finalize_num_warps,
        )

    return run_sgl_lora_moe_c2_partial(
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
        output_dtype=output_dtype,
        shared_outer_gate_a_plan=shared_outer_gate_a_plan,
    )


def run_sgl_lora_moe_c3(
    dispatch_output: StandardDispatchOutput,
    quant_info: SglLoraBf16QuantInfo,
    runner_config: MoeRunnerConfig,
    lora_info,
    base: MoeLoraBaseGemm,
    *,
    consumer_schedule: str = "aligned",
    consumer_block_size_n: int = 64,
    consumer_num_warps: int = 4,
    finalize_block_size_h: int = 32,
    finalize_num_warps: int = 4,
    has_base_rows: bool = True,
    output_dtype: torch.dtype | None = None,
    shared_outer_gate_a_plan=None,
) -> StandardCombineInput:
    """Run complete C2 while overlapping gate/up LoRA-A with base GEMM1."""
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        use_symmetric_memory,
    )
    from sglang.srt.layers.dp_attention import is_allocation_symmetric
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.lora.sgl_lora.runtime import get_lora_side_stream
    from sglang.srt.lora.sgl_lora.triton_ops.fused_c2 import (
        fused_gate_up_b_swiglu_down_a,
        fused_gate_up_b_swiglu_down_a_aligned,
    )
    from sglang.srt.lora.sgl_lora.triton_ops.fused_down_finalize import (
        fused_down_b_finalize,
    )
    from sglang.srt.lora.sgl_lora.triton_ops.virtual_experts import (
        merged_experts_fused_moe_lora_add,
    )
    from sglang.srt.utils import dispose_tensor

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    assert TopKOutputChecker.format_is_standard(topk_output)
    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights
    packed_topk_ids = getattr(topk_output, "packed_topk_ids", None)
    top_k = runner_config.top_k
    output_dtype = _resolve_output_dtype_and_validate(hidden_states, base, output_dtype)
    _admit_workspace(
        dispatch_output,
        lora_info,
        base,
        top_k=top_k,
        output_dtype=output_dtype,
    )
    num_tokens = hidden_states.shape[0]
    token_lora_mapping = lora_info.token_lora_mapping
    _validate_token_lora_mapping(token_lora_mapping, num_tokens)

    gate_rank = lora_info.gate_up_lora_b_weights.shape[-1]
    down_rank = lora_info.down_lora_a_weights.shape[-2]
    routing_cache: dict = {}
    gate_up_intermediate = torch.empty(
        (num_tokens, top_k, 2 * gate_rank),
        dtype=base.contract.lora_delta_dtype,
        device=hidden_states.device,
    )
    down_intermediate = torch.zeros(
        (num_tokens, top_k, down_rank),
        dtype=base.contract.lora_activation_dtype,
        device=hidden_states.device,
    )

    def _gate_a(stage: str) -> None:
        merged_experts_fused_moe_lora_add(
            output=None,
            hidden_states=hidden_states,
            lora_a=lora_info.gate_up_lora_a_weights,
            lora_b=lora_info.gate_up_lora_b_weights,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            token_lora_mapping=token_lora_mapping,
            mul_routed_weight=False,
            experts_shared_outer_loras_a=lora_info.experts_shared_outer_loras,
            experts_shared_outer_loras_b=False,
            routing_cache=routing_cache,
            fuse_add_to_output=False,
            use_direct_expand_add=False,
            num_output_slices=2,
            local_expert_offset=0,
            local_num_experts=quant_info.num_local_experts,
            stage=stage,
            intermediate_buffer=(gate_up_intermediate if stage == "shrink" else None),
            shared_outer_gate_a_plan=shared_outer_gate_a_plan,
            segment_indptr=getattr(lora_info, "seg_indptr", None),
            segment_lora_ids=getattr(lora_info, "req_to_lora", None),
            max_segment_len=getattr(lora_info, "max_segment_len", 0),
        )

    # Every allocation and route plan belongs to the main stream before fork.
    _gate_a("routing")
    side_stream = get_lora_side_stream()
    gate_a_done = torch.cuda.Event()
    main_stream = torch.cuda.current_stream()
    side_stream.wait_stream(main_stream)
    with torch.cuda.stream(side_stream):
        _gate_a("shrink")
        gate_a_done.record()
    if torch.cuda.is_current_stream_capturing():
        from sglang.srt.model_executor.runner_utils.capture_resources import (
            keep_cuda_graph_capture_resource,
        )

        keep_cuda_graph_capture_resource(gate_a_done)

    ws = base.prepare(
        hidden_states,
        topk_ids,
        top_k,
        topk_weights=topk_weights,
        packed_topk_ids=packed_topk_ids,
    )
    gateup_out = torch.empty(
        base.gateup_out_shape(ws),
        dtype=base.contract.gate_up_output_dtype,
        device=hidden_states.device,
    )
    base.gateup(ws, gateup_out)
    if ws.hidden_permuted_owned:
        dispose_tensor(ws.hidden_permuted)

    act_out = torch.empty(
        base.act_out_shape(ws),
        dtype=base.contract.lora_activation_dtype,
        device=hidden_states.device,
    )
    main_stream.wait_event(gate_a_done)
    if consumer_schedule == "aligned":
        per_expert_routes = [
            (key, value)
            for key, value in routing_cache.items()
            if key[0] == lora_info.gate_up_lora_b_weights.shape[1] and not key[1]
        ]
        if not per_expert_routes:
            raise RuntimeError("aligned C3 requires a cached per-expert route")
        route_key, route = min(per_expert_routes, key=lambda item: item[0][2])
        sorted_pair_ids, virtual_expert_ids, num_pairs_post_padded, _ = route
        fused_gate_up_b_swiglu_down_a_aligned(
            gateup_out,
            gate_up_intermediate,
            lora_info.gate_up_lora_b_weights,
            lora_info.down_lora_a_weights,
            act_out,
            down_intermediate,
            ws.src2dst,
            topk_ids,
            sorted_pair_ids,
            virtual_expert_ids,
            num_pairs_post_padded,
            route_block_size_m=route_key[2],
            token_lora_mapping=(token_lora_mapping if has_base_rows else None),
            block_size_n=consumer_block_size_n,
            num_warps=consumer_num_warps,
        )
    elif consumer_schedule == "pair":
        fused_gate_up_b_swiglu_down_a(
            gateup_out,
            gate_up_intermediate,
            lora_info.gate_up_lora_b_weights,
            lora_info.down_lora_a_weights,
            act_out,
            down_intermediate,
            ws.src2dst,
            topk_ids,
            token_lora_mapping,
            local_expert_offset=0,
            block_size_n=consumer_block_size_n,
            num_warps=consumer_num_warps,
        )
    else:
        raise ValueError(f"unknown C3 consumer schedule {consumer_schedule!r}")
    dispose_tensor(gateup_out)
    dispose_tensor(gate_up_intermediate)

    down_out = torch.empty(
        base.down_out_shape(ws), dtype=torch.bfloat16, device=hidden_states.device
    )
    base.down(ws, act_out, down_out)
    dispose_tensor(act_out)

    with use_symmetric_memory(get_tp_group(), disabled=not is_allocation_symmetric()):
        output = torch.empty(
            (num_tokens, quant_info.hidden_size),
            dtype=output_dtype,
            device=hidden_states.device,
        )
    fused_down_b_finalize(
        down_out,
        down_intermediate,
        lora_info.down_lora_b_weights,
        output,
        ws.src2dst,
        topk_ids,
        topk_weights,
        token_lora_mapping,
        routed_scaling_factor=(
            runner_config.routed_scaling_factor
            if runner_config.routed_scaling_factor is not None
            else 1.0
        ),
        local_expert_offset=0,
        shared_outer=lora_info.experts_shared_outer_loras,
        block_size_h=finalize_block_size_h,
        num_warps=finalize_num_warps,
    )
    dispose_tensor(down_out)
    return StandardCombineInput(hidden_states=output)


def run_sgl_lora_moe_bf16_plan(
    dispatch_output: StandardDispatchOutput,
    quant_info: SglLoraQuantInfo,
    runner_config: MoeRunnerConfig,
    lora_info,
    base: MoeLoraBaseGemm,
    plan,
    *,
    output_dtype: torch.dtype | None = None,
) -> StandardCombineInput:
    """Compatibility alias for callers predating provider-neutral dispatch."""
    from sglang.srt.lora.sgl_lora.execution import run_sgl_lora_moe_plan

    return run_sgl_lora_moe_plan(
        dispatch_output,
        quant_info,
        runner_config,
        lora_info,
        base,
        plan,
        output_dtype=output_dtype,
    )


__all__ = [
    "run_sgl_lora_moe_bf16_plan",
    "run_sgl_lora_moe_c2_full",
    "run_sgl_lora_moe_c2_partial",
    "run_sgl_lora_moe_c3",
]
