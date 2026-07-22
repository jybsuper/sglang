"""Unified MoE-LoRA runner for the SGL LoRA execution engine.

ONE function for all quants: the per-quant base GEMM stages live behind
:class:`MoeLoraBaseGemm` (constructed once at LoRA-attach time); this runner
owns the pipeline, the two-stream overlap, and every buffer allocation.

The caller (``lora_layer.dispatch_sgl_lora_moe``) routes eager no-adapter
batches to the stock Triton strategy. The current vertical slice uses the
virtual-expert store and standard local expert IDs; the broader support matrix
is tracked in the refactor worklog.

Optional two-stream execution (explicitly enabled; off by default):

    main (alt) stream                              LoRA side stream
    routing pre-warm (BOTH A/B cache keys) ──┐
    (delta + shrink intermediate pre-alloc)  │ fork (side.wait_stream(main))
    S1 prepare (permute [+ quant])           │ gate_up LoRA shrink + expand
    S2 gateup grouped GEMM                   │ record lora_event
    wait lora_event  ◄───────────────────────┘
    S3 act: SwiGLU + delta + activation_lora_input
    S4 down grouped GEMM
    S5 finalize (+ routed scaling)
    serial down-LoRA

Serial (non-overlap) batches run the same pipeline with the gate_up LoRA
inline on the main stream before S1 — numerically equivalent (modulo the
pre-existing shrink split-K bf16-atomic nondeterminism).

``two_stream_enabled`` is an already-resolved execution decision. Production
dispatch applies the current ``num_tokens <= 256`` auto policy before entering
this runner; benchmark callers may therefore compare another resolved policy
without changing the production default or copying the pipeline.

Invariants carried over from the trtllm path (review-confirmed load-bearing):
  * NO device allocation inside the side-stream context — the stage="routing"
    pre-warm seeds both the A(shrink)- and B(expand)-stage routing-cache keys
    on the main stream, and the shrink intermediate is pre-allocated here.
  * lora_event keep-alive during cuda-graph capture (torch does NOT manage
    event lifetime under capture; see _LORA_OVERLAP_EVENTS in moe_overlap.py).
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
    from sglang.srt.lora.sgl_lora.quant_info import SglLoraQuantInfo

# Keep events recorded during cuda-graph capture alive until graph teardown.
_LORA_EVENTS_KEEPALIVE: list = []

# Production auto-policy boundary. Keep policy selection outside the runner so
# the execution function obeys one explicit decision under eager execution and
# CUDA graph capture alike.
LORA_TWO_STREAM_AUTO_MAX_TOKENS = 256


def resolve_lora_two_stream_auto(*, requested: bool, num_tokens: int) -> bool:
    """Resolve the production two-stream request for one fixed-shape forward."""
    return requested and num_tokens <= LORA_TWO_STREAM_AUTO_MAX_TOKENS


def run_sgl_lora_moe(
    dispatch_output: StandardDispatchOutput,
    quant_info: SglLoraQuantInfo,
    runner_config: MoeRunnerConfig,
    lora_info,
    base: MoeLoraBaseGemm,
    *,
    two_stream_enabled: bool,
) -> StandardCombineInput:
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        use_symmetric_memory,
    )
    from sglang.srt.layers.dp_attention import is_allocation_symmetric
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.lora.sgl_lora.runtime import get_lora_side_stream
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

    overlap = two_stream_enabled
    token_lora_mapping = lora_info.token_lora_mapping
    if token_lora_mapping.shape[0] != num_tokens:
        raise RuntimeError(
            "sgl_lora token/adapter assignment does not match the MoE token "
            f"domain: mapping has {token_lora_mapping.shape[0]} rows but the "
            f"runner received {num_tokens}. Gather/remap assignments before "
            "MoE-DP execution."
        )
    fused_lora_routing_cache: dict = {}

    gate_up_delta = hidden_states.new_empty((num_tokens, top_k, 2 * inter))
    gate_up_lora_intermediate = hidden_states.new_empty(
        (num_tokens, top_k, lora_info.gate_up_lora_a_weights.shape[2])
    )

    def _run_gate_up_lora(stage: str = "all") -> None:
        merged_experts_fused_moe_lora_add(
            output=gate_up_delta,
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
            stage=stage,
            fuse_add_to_output=False,
            use_direct_expand_add=lora_info.max_lora_rank <= 64,
            num_output_slices=2,
            local_expert_offset=0,
            local_num_experts=quant_info.num_local_experts,
            intermediate_buffer=(
                gate_up_lora_intermediate if stage != "routing" else None
            ),
        )

    lora_event = None
    if overlap:
        # O1 fork: gate_up shrink/expand on the side stream, concurrent with
        # the main-stream S1 permute + S2 grouped GEMM below. S3 (the only
        # consumer of gate_up_delta) waits on lora_event right before launch.
        # Pre-warm BOTH routing-cache keys on the main stream first so the
        # side-stream block launches kernels only (allocator safety under
        # capture).
        _run_gate_up_lora(stage="routing")
        side_stream = get_lora_side_stream()
        lora_event = torch.cuda.Event()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            _run_gate_up_lora()
            lora_event.record()
        if torch.cuda.is_current_stream_capturing():
            _LORA_EVENTS_KEEPALIVE.append(lora_event)
    else:
        # Serial: inline on the main stream.
        _run_gate_up_lora()

    # ---- base pipeline (main stream) ----
    ws = base.prepare(hidden_states, topk_ids, top_k)

    gateup_out = hidden_states.new_empty(base.gateup_out_shape(ws))
    base.gateup(ws, gateup_out)
    # The permuted hidden is dead after S2 — free it before the S3/S4 buffers
    # (memory parity with the stock deep_gemm runner's dispose_tensor calls).
    dispose_tensor(ws.hidden_permuted)

    act_out = hidden_states.new_empty(base.act_out_shape(ws))
    activation_lora_input = hidden_states.new_empty((num_tokens, top_k, inter))

    if lora_event is not None:
        torch.cuda.current_stream().wait_event(lora_event)
    base.act_with_delta(
        ws, gateup_out, gate_up_delta, topk_ids, act_out, activation_lora_input
    )
    dispose_tensor(gateup_out)
    dispose_tensor(gate_up_delta)

    down_out = hidden_states.new_empty(base.down_out_shape(ws))
    base.down(ws, act_out, down_out)
    dispose_tensor(act_out)

    with use_symmetric_memory(get_tp_group(), disabled=not is_allocation_symmetric()):
        output = torch.empty(
            (num_tokens, quant_info.hidden_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
    base.finalize(
        ws,
        down_out,
        topk_ids,
        topk_weights,
        runner_config.routed_scaling_factor,
        output,
    )
    dispose_tensor(down_out)

    # Base finalize applies routed_scaling_factor after its weighted top-k
    # reduction. Apply the same factor to the down-LoRA contribution exactly
    # once; otherwise non-unit models scale only the base branch.
    down_topk_weights = topk_weights
    routed_scale = runner_config.routed_scaling_factor
    if routed_scale is not None and routed_scale != 1.0:
        down_topk_weights = topk_weights * float(routed_scale)

    merged_experts_fused_moe_lora_add(
        output=output,
        hidden_states=activation_lora_input.view(-1, inter),
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
    )

    return StandardCombineInput(hidden_states=output)
