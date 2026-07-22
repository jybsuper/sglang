"""Benchmark-only BF16 C3 producer/consumer overlap topology.

C3 keeps the complete C2 fused consumers and changes only the gate/up LoRA-A
producer schedule::

    main stream                               LoRA side stream
    route metadata + buffers + event ----+   wait(main)
    base prepare + gate/up GEMM           |   gate/up LoRA-A shrink
    wait(event) <-------------------------+   record(event)
    fused gate-B + SwiGLU + down-A
    base down
    fused down-B + base finalize

The side stream never allocates.  All routing metadata, the shrink destination,
the downstream accumulation buffer, and the event are created on the main
stream before the fork.  Production dispatch does not import this module.
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
    from sglang.srt.lora.sgl_lora.quant_info import SglLoraBf16QuantInfo


def run_sgl_lora_moe_c3_experimental(
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
) -> StandardCombineInput:
    """Run complete BF16 C2 with gate/up LoRA-A overlapped as C3.

    This is a benchmark-only fixed-shape execution candidate.  It deliberately
    has no production flag or dispatch registration.
    """
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        use_symmetric_memory,
    )
    from sglang.srt.layers.dp_attention import is_allocation_symmetric
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.lora.sgl_lora.experimental_c2 import (
        _validate_token_lora_mapping,
    )
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
    top_k = runner_config.top_k
    num_tokens = hidden_states.shape[0]
    token_lora_mapping = lora_info.token_lora_mapping
    _validate_token_lora_mapping(token_lora_mapping, num_tokens)

    gate_rank = lora_info.gate_up_lora_b_weights.shape[-1]
    down_rank = lora_info.down_lora_a_weights.shape[-2]
    routing_cache: dict = {}

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
            intermediate_buffer=(
                gate_up_intermediate if stage == "shrink" else None
            ),
        )

    # Main-stream ownership boundary.  ``stage='routing'`` constructs both the
    # A route and the cached per-expert B route used by the aligned consumer.
    # The two destinations are also allocated before the side-stream fork.
    gate_up_intermediate = hidden_states.new_empty(
        (num_tokens, top_k, 2 * gate_rank)
    )
    down_intermediate = hidden_states.new_zeros((num_tokens, top_k, down_rank))
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

    # Only provider work independent of gate/up LoRA-A is placed before the
    # join.  The join is immediately before the first consumer of the shrink.
    ws = base.prepare(hidden_states, topk_ids, top_k)
    gateup_out = hidden_states.new_empty(base.gateup_out_shape(ws))
    base.gateup(ws, gateup_out)
    dispose_tensor(ws.hidden_permuted)

    act_out = hidden_states.new_empty(base.act_out_shape(ws))
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

    down_out = hidden_states.new_empty(base.down_out_shape(ws))
    base.down(ws, act_out, down_out)
    dispose_tensor(act_out)

    with use_symmetric_memory(get_tp_group(), disabled=not is_allocation_symmetric()):
        output = torch.empty(
            (num_tokens, quant_info.hidden_size),
            dtype=hidden_states.dtype,
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


__all__ = [
    "run_sgl_lora_moe_c3_experimental",
]
