"""Benchmark-only C2-consumer partial for the BF16 SGL LoRA MoE pipeline.

Production dispatch continues to call :func:`run_sgl_lora_moe` (C0/C1).  This
module supplies the narrow fused-consumer vertical slice needed for K0/O0/M0
benchmarks without adding a flag or branch to serving code.  It is not the
complete planned C2 topology: down-B still runs after base finalize through the
existing expand/add path.  C2P applies
``topk_weights * routed_scaling_factor`` exactly once to down LoRA-B, matching
the base branch's finalize contract.  The current performance harness uses a
factor of 1, so this correction adds no benchmark-only scale launch.
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
    from sglang.srt.lora.sgl_lora.quant_info import SglLoraBf16QuantInfo


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


def _scaled_down_lora_topk_weights(
    topk_weights: torch.Tensor, routed_scaling_factor: float | None
) -> torch.Tensor:
    """Return the weights that apply the routed scale once to down LoRA-B."""
    if routed_scaling_factor is None or routed_scaling_factor == 1.0:
        return topk_weights
    return topk_weights * float(routed_scaling_factor)


def run_sgl_lora_moe_c2_experimental(
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
) -> StandardCombineInput:
    """Run the serial BF16 C2-consumer partial without changing dispatch.

    The runner owns every destination allocation.  The fused consumer writes
    the provider-private masked BF16 W2 input and directly accumulates the
    canonical ``[token, top-k, rank]`` bridge consumed by down LoRA-B.

    ``down_finalize`` is a benchmark-only borrowed-buffer callback for a full
    C2 finalizer candidate.  When supplied, it replaces both base finalize and
    the decomposed down-B expand; this runner retains ownership of ``down_out``
    and disposes it after the callback returns.
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
    top_k = runner_config.top_k
    num_tokens = hidden_states.shape[0]
    inter = quant_info.intermediate_size
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
    )

    # Gate/up A remains a standalone producer in C2.  Its 2R output is the
    # input boundary of the fused consumer; neither 2I delta nor I activation
    # is materialized in canonical pair order.
    gate_rank = lora_info.gate_up_lora_b_weights.shape[-1]
    gate_up_intermediate = hidden_states.new_empty((num_tokens, top_k, 2 * gate_rank))
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
    )

    ws = base.prepare(hidden_states, topk_ids, top_k)
    gateup_out = hidden_states.new_empty(base.gateup_out_shape(ws))
    base.gateup(ws, gateup_out)
    dispose_tensor(ws.hidden_permuted)

    act_out = hidden_states.new_empty(base.act_out_shape(ws))
    down_rank = lora_info.down_lora_a_weights.shape[-2]
    # C2 accumulates one BF16 partial per I tile.  The explicit zero is part of
    # the measured plan and is safe under eager execution and graph replay.
    down_intermediate = hidden_states.new_zeros((num_tokens, top_k, down_rank))
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

    down_out = hidden_states.new_empty(base.down_out_shape(ws))
    base.down(ws, act_out, down_out)
    dispose_tensor(act_out)

    with use_symmetric_memory(get_tp_group(), disabled=not is_allocation_symmetric()):
        output = torch.empty(
            (num_tokens, quant_info.hidden_size),
            dtype=hidden_states.dtype,
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

        # The fused consumer has already produced down-A.  Reuse the existing
        # down-B/finalize implementation from its explicit expand boundary.
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
        # Benchmark-only seam for evaluating a fused down-B/base-finalize
        # consumer.  Serving dispatch never supplies this hook.
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


__all__ = ["run_sgl_lora_moe_c2_experimental"]
