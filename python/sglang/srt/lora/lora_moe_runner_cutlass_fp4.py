"""NVFP4 MoE runner that exposes ``after_gate_up`` / ``after_down`` LoRA hooks
between the W13 and W2 group GEMMs of FlashInfer-CUTLASS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
)
from sglang.srt.environ import envs
from sglang.srt.lora.triton_ops import (
    cutlass_fp4_lora_shuffle_mul_sum,
    cutlass_fp4_lora_silu_and_mul,
)

# Persistent side CUDA stream(s) for the two-stream LoRA overlap, one per device.
_CUTLASS_LORA_SIDE_STREAM: dict[int, "torch.cuda.Stream"] = {}
# Keep overlap events recorded during cuda-graph capture alive so the captured cross-stream waits
# aren't torn down before graph instantiation (eager runs rely on CUDA's deferred destroy).
_CUTLASS_LORA_OVERLAP_EVENTS: list = []


def _cutlass_lora_side_stream(device: torch.device) -> "torch.cuda.Stream":
    key = device.index if device.index is not None else 0
    s = _CUTLASS_LORA_SIDE_STREAM.get(key)
    if s is None:
        s = torch.cuda.Stream(device=device)
        _CUTLASS_LORA_SIDE_STREAM[key] = s
    return s


def _keep_event_alive_if_capturing(event) -> None:
    from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

    if get_is_capture_mode():
        _CUTLASS_LORA_OVERLAP_EVENTS.append(event)


if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        StandardCombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.lora.lora_moe_runners import LoRAHooks


@dataclass
class CutlassFp4MoeQuantInfo(MoeQuantInfo):
    w13_weight: torch.Tensor  # [E, 2*N, K // 2] uint8
    w2_weight: torch.Tensor  # [E, K, N // 2]   uint8
    w13_blockscale_swizzled: torch.Tensor
    w2_blockscale_swizzled: torch.Tensor
    g1_alphas: torch.Tensor  # [E] fp32
    g2_alphas: torch.Tensor  # [E] fp32
    # [E]-expanded once at load time so the hot path doesn't broadcast.
    w13_input_scale_expanded: torch.Tensor  # [E] fp32
    w2_input_scale_expanded: torch.Tensor  # [E] fp32
    cutlass_moe_params: object
    num_local_experts: int
    hidden_size: int
    intermediate_size_per_partition: int
    moe_ep_rank: int
    # ``[Up|Gate]`` W13 layout (Kimi-K2.5): silu the second half, multiply the first.
    w13_swap_halves: bool


class CutlassFp4LoraRunnerCore:
    """LoRA-aware NVFP4 MoE forward on the FlashInfer-CUTLASS GEMM primitives."""

    def __init__(self, config: MoeRunnerConfig):
        # config is unused today; runner_config is passed per-call.
        pass

    def run_from_dispatch(
        self,
        dispatch_output: "StandardDispatchOutput",
        quant_info: CutlassFp4MoeQuantInfo,
        runner_config: MoeRunnerConfig,
        hooks: Optional["LoRAHooks"] = None,
        lora_info=None,
    ) -> "StandardCombineInput":
        from sgl_kernel import prepare_moe_input

        from sglang.srt.layers.moe.cutlass_moe import (
            cutlass_fp4_group_mm,
            scaled_fp4_experts_quant,
        )
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardCombineInput,
        )

        # FP4 all-gather dispatch (gated by should_use_flashinfer_cutlass_moe_fp4_allgather)
        # delivers pre-packed FP4 input; this runner re-quantizes bf16 only.
        if getattr(dispatch_output, "hidden_states_scale", None) is not None:
            raise NotImplementedError(
                "CutlassFp4LoraRunnerCore does not support the FP4 all-gather "
                "dispatch path; disable it for LoRA configurations."
            )

        hidden_states = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights = topk_output.topk_weights
        topk_ids = topk_output.topk_ids

        m_a = hidden_states.shape[0]
        num_topk = topk_ids.shape[1]
        out_dtype = hidden_states.dtype
        device = hidden_states.device
        E = quant_info.num_local_experts
        K = quant_info.hidden_size
        inter = quant_info.intermediate_size_per_partition
        # This LoRA runner injects between the gate/up projection and the
        # SwiGLU activation, so it only supports gated MoE weights.
        N = quant_info.w13_weight.shape[1]
        if N != inter * 2:
            raise NotImplementedError(
                "CutlassFp4LoraRunnerCore expects gated NVFP4 MoE weights with "
                f"w13 output dim {inter * 2}, but got {N}."
            )
        params = quant_info.cutlass_moe_params
        offsets = params.expert_offsets
        total_tokens = m_a * num_topk

        # StandardDispatcher hands flashinfer_cutlass global topk_ids; remap
        # to local. Non-local tokens go to local expert 0 with weight 0.
        local_offset = quant_info.moe_ep_rank * E
        local_ids = topk_ids.to(torch.int32) - local_offset
        non_local = (local_ids < 0) | (local_ids >= E)
        local_ids = local_ids.masked_fill(non_local, 0)
        local_weights = topk_weights.to(torch.float32).masked_fill(non_local, 0.0)

        a_map = torch.empty(total_tokens, dtype=torch.int32, device=device)
        c_map = torch.empty(total_tokens, dtype=torch.int32, device=device)
        prepare_moe_input(
            local_ids,
            offsets,
            params.problem_sizes1,
            params.problem_sizes2,
            a_map,
            c_map,
            E,
            inter,
            K,
            params.blockscale_offsets,
        )

        # Hand the LoRA hooks the c_map so their kernels read/write expert-sorted rows directly.
        # Set it BEFORE GEMM1 so the (optional) side-stream gate_up LoRA can use the sorted layout.
        if lora_info is not None:
            lora_info.c_map = c_map
            lora_info.sorted_layout = True

        # Two-stream: the gate_up LoRA shrink+expand depends only on hidden_states (+ c_map), so
        # compute its delta into a separate buffer on a side stream concurrent with GEMM1, then add
        # after a cross-stream join. Env-gated; default-off serial path is byte-identical.
        two_stream = (
            envs.SGLANG_OPT_CUTLASS_LORA_TWO_STREAM.get()
            and hooks is not None
            and hooks.after_gate_up is not None
        )
        gate_up_delta = None
        gu_event = None
        if two_stream:
            gate_up_delta = torch.zeros(total_tokens, N, dtype=out_dtype, device=device)
            side = _cutlass_lora_side_stream(device)
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                hooks.after_gate_up(
                    hidden_states, gate_up_delta.view(m_a, num_topk, N), topk_weights, topk_ids
                )
            # No record_stream: gate_up_delta's Python ref lives to function end and the
            # main-stream consumer (add_) waits on gu_event, so the side-stream write always
            # precedes any allocator reuse. record_stream is also unsafe under cuda-graph
            # capture; the validated lora-opti two-stream omits it for the same reason.
            gu_event = torch.cuda.Event()
            gu_event.record(side)
            _keep_event_alive_if_capturing(gu_event)

        # ---- GEMM 1 (w13)  [main stream; overlaps the side-stream gate_up LoRA when two_stream]
        rep_a_fp4, rep_a_blockscale = scaled_fp4_experts_quant(
            hidden_states,
            quant_info.w13_input_scale_expanded,
            offsets,
            params.blockscale_offsets,
            num_topk,
            expert_map=a_map,
        )
        gateup_flat = cutlass_fp4_group_mm(
            rep_a_fp4,
            quant_info.w13_weight,
            rep_a_blockscale,
            quant_info.w13_blockscale_swizzled,
            quant_info.g1_alphas,
            out_dtype,
            params.to_gemm1_args(),
        )

        # ---- LoRA w13 delta
        if two_stream:
            torch.cuda.current_stream().wait_event(gu_event)  # join GEMM1 (main) + gate_up (side)
            gateup_flat.add_(gate_up_delta)
        elif hooks is not None and hooks.after_gate_up is not None:
            gateup_3d = gateup_flat.view(m_a, num_topk, N)
            hooks.after_gate_up(hidden_states, gateup_3d, topk_weights, topk_ids)

        # ---- silu + mul
        # ``w13_swap_halves=True`` selects the ``[up | gate]`` convention
        # (silu(second) * first) for FlashInfer-CUTLASS NVFP4 W13 loaders.
        intermediate = torch.empty(total_tokens, N // 2, dtype=out_dtype, device=device)
        cutlass_fp4_lora_silu_and_mul(
            gateup_flat, intermediate, swap_halves=quant_info.w13_swap_halves
        )

        # Two-stream: the down LoRA depends on `intermediate` (just produced), so compute its delta
        # into a buffer on the side stream concurrent with GEMM2 below, then add after a join.
        down_delta = None
        dn_event = None
        if two_stream and hooks.after_down is not None:
            down_delta = torch.zeros(total_tokens, K, dtype=out_dtype, device=device)
            side = _cutlass_lora_side_stream(device)
            side.wait_stream(torch.cuda.current_stream())  # side waits for silu (intermediate)
            with torch.cuda.stream(side):
                hooks.after_down(
                    intermediate, down_delta.view(m_a, num_topk, K), local_weights, topk_ids
                )
            # No record_stream (see gate_up note): down_delta + intermediate are kept alive by
            # their Python refs through the dn_event join, and the main-stream consumers wait on it.
            dn_event = torch.cuda.Event()
            dn_event.record(side)
            _keep_event_alive_if_capturing(dn_event)

        # ---- GEMM 2 (w2)  [main stream; overlaps the side-stream down LoRA when two_stream]
        int_fp4, int_blockscale = scaled_fp4_experts_quant(
            intermediate,
            quant_info.w2_input_scale_expanded,
            offsets,
            params.blockscale_offsets,
            num_topk,
        )
        out_flat = cutlass_fp4_group_mm(
            int_fp4,
            quant_info.w2_weight,
            int_blockscale,
            quant_info.w2_blockscale_swizzled,
            quant_info.g2_alphas,
            out_dtype,
            params.to_gemm2_args(),
        )

        # ---- LoRA w2 delta. Sorted-layout: unweighted delta into out_flat; combine weights once.
        if two_stream and down_delta is not None:
            torch.cuda.current_stream().wait_event(dn_event)  # join GEMM2 (main) + down (side)
            out_flat.add_(down_delta)
        elif hooks is not None and hooks.after_down is not None:
            out_3d_sorted_view = out_flat.view(m_a, num_topk, K)
            hooks.after_down(intermediate, out_3d_sorted_view, local_weights, topk_ids)

        # ---- combine: un-sort, weight (base + delta), sum. Router weights stay
        # fp32 to match FlashInfer-CUTLASS fused MoE's final accumulation.
        output = torch.empty((m_a, K), dtype=out_dtype, device=device)
        cutlass_fp4_lora_shuffle_mul_sum(
            out_flat,
            output,
            c_map,
            (
                None
                if runner_config.apply_router_weight_on_input
                else local_weights.reshape(-1)
            ),
            num_topk,
        )
        return StandardCombineInput(hidden_states=output)
