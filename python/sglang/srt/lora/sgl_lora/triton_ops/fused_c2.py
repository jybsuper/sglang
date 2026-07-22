"""Experimental BF16 C2 consumer for gated MoE LoRA.

This kernel is deliberately separate from the production C0/C1 path.  It
fuses the part of the serial pipeline between gate/up LoRA-A and down LoRA-B::

    gate_up_A [T, K, 2R]
       + base W13 output [E, m_max, 2I]
       + gate_up_B [L, E, 2I, R]
       -> SwiGLU
       -> base-W2 input [E, m_max, I]
       -> down_A reduction [T, K, R]

The two outputs are the only values needed by the remaining base-W2 and
down-LoRA-B stages.  In particular, C2 does *not* materialize either the
``[T, K, 2I]`` gate/up delta or the ``[T, K, I]`` activation bridge.

Current BF16 contract (not a universal provider ABI):

* base W13 is the masked ``[E_local, m_max, 2I]`` layout;
* both base W13 and LoRA-B are contiguous ``[gate | up]`` (gate first);
* the activation is ordinary ``silu(gate) * up``;
* gate/up-B deltas and the activated W2 input round through BF16 at the same
  semantic boundaries as the decomposed C0 path;
* ``src2dst[pair]`` maps canonical ``(token, top-k)`` order to the masked base
  layout; negative expert IDs are invalid pairs and adapter ID ``-1`` means a
  base-only row.

Two benchmark schedules share this contract.  The pair-owned schedule maps one
program to one routed pair and is useful as the simplest low-density control.
The aligned schedule maps a block of pairs with the same virtual expert to
tensor-core dots, reusing gate/up-B and down-A weights across rows.  Both split
over the intermediate dimension: each program writes a disjoint W2-input tile
and atomically accumulates its partial down-A reduction.  The caller must
therefore zero ``down_intermediate`` before launch.  These remain experimental
implementation candidates; promotion requires matched K0/O0/M0 evidence.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_gate_up_b_swiglu_down_a_kernel(
    gateup_ptr,
    gate_intermediate_ptr,
    gate_b_ptr,
    down_a_ptr,
    act_out_ptr,
    down_intermediate_ptr,
    src2dst_ptr,
    topk_ids_ptr,
    token_lora_mapping_ptr,
    num_pairs,
    inter,
    num_local_experts,
    stride_gum,
    stride_gun,
    stride_gim,
    stride_gir,
    stride_gbl,
    stride_gbe,
    stride_gbn,
    stride_gbr,
    stride_dal,
    stride_dae,
    stride_dar,
    stride_dan,
    stride_aom,
    stride_aon,
    stride_dim,
    stride_dir,
    top_k: tl.constexpr,
    gate_rank: tl.constexpr,
    down_rank: tl.constexpr,
    max_loras: tl.constexpr,
    local_expert_offset: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_GATE_R: tl.constexpr,
    BLOCK_DOWN_R: tl.constexpr,
):
    pair_idx = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)

    token_idx = pair_idx // top_k
    global_expert = tl.load(topk_ids_ptr + pair_idx).to(tl.int64)
    local_expert = global_expert - local_expert_offset
    adapter = tl.load(token_lora_mapping_ptr + token_idx).to(tl.int64)

    valid_pair = (
        (pair_idx < num_pairs)
        & (local_expert >= 0)
        & (local_expert < num_local_experts)
    )
    has_lora = valid_pair & (adapter >= 0) & (adapter < max_loras)
    safe_expert = tl.maximum(0, tl.minimum(local_expert, num_local_experts - 1))
    safe_adapter = tl.maximum(0, tl.minimum(adapter, max_loras - 1))
    dst_row = tl.load(src2dst_ptr + pair_idx, mask=valid_pair, other=0).to(tl.int64)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    n_mask = offs_n < inter
    offs_gate_r = tl.arange(0, BLOCK_GATE_R).to(tl.int64)
    gate_r_mask = offs_gate_r < gate_rank

    gateup_row = gateup_ptr + dst_row * stride_gum
    gate = tl.load(
        gateup_row + offs_n * stride_gun,
        mask=valid_pair & n_mask,
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        gateup_row + (inter + offs_n) * stride_gun,
        mask=valid_pair & n_mask,
        other=0.0,
    ).to(tl.float32)

    gate_intermediate_row = gate_intermediate_ptr + pair_idx * stride_gim
    gate_a = tl.load(
        gate_intermediate_row + offs_gate_r * stride_gir,
        mask=has_lora & gate_r_mask,
        other=0.0,
    )
    up_a = tl.load(
        gate_intermediate_row + (gate_rank + offs_gate_r) * stride_gir,
        mask=has_lora & gate_r_mask,
        other=0.0,
    )

    gate_b_base = gate_b_ptr + safe_adapter * stride_gbl + safe_expert * stride_gbe
    gate_b = tl.load(
        gate_b_base + offs_n[:, None] * stride_gbn + offs_gate_r[None, :] * stride_gbr,
        mask=has_lora & n_mask[:, None] & gate_r_mask[None, :],
        other=0.0,
    )
    up_b = tl.load(
        gate_b_base
        + (inter + offs_n[:, None]) * stride_gbn
        + offs_gate_r[None, :] * stride_gbr,
        mask=has_lora & n_mask[:, None] & gate_r_mask[None, :],
        other=0.0,
    )

    gate_delta = tl.sum(gate_b.to(tl.float32) * gate_a[None, :], axis=1)
    up_delta = tl.sum(up_b.to(tl.float32) * up_a[None, :], axis=1)
    # Preserve C0's materialized BF16 gate_up_delta boundary.
    gate += gate_delta.to(gateup_ptr.dtype.element_ty).to(tl.float32)
    up += up_delta.to(gateup_ptr.dtype.element_ty).to(tl.float32)

    activated = gate * tl.sigmoid(gate) * up
    # This BF16 value is both the base-W2 input and the value down-A sees in C0.
    activated_bf16 = activated.to(act_out_ptr.dtype.element_ty)
    tl.store(
        act_out_ptr + dst_row * stride_aom + offs_n * stride_aon,
        activated_bf16,
        mask=valid_pair & n_mask,
    )

    offs_down_r = tl.arange(0, BLOCK_DOWN_R).to(tl.int64)
    down_r_mask = offs_down_r < down_rank
    down_a_base = down_a_ptr + safe_adapter * stride_dal + safe_expert * stride_dae
    down_a = tl.load(
        down_a_base + offs_down_r[:, None] * stride_dar + offs_n[None, :] * stride_dan,
        mask=has_lora & down_r_mask[:, None] & n_mask[None, :],
        other=0.0,
    )
    partial = tl.sum(
        down_a.to(tl.float32) * activated_bf16[None, :].to(tl.float32), axis=1
    )
    # Match the decomposed split-K shrink's BF16 partial-accumulation boundary.
    partial = partial.to(down_intermediate_ptr.dtype.element_ty)
    tl.atomic_add(
        down_intermediate_ptr + pair_idx * stride_dim + offs_down_r * stride_dir,
        partial,
        mask=has_lora & down_r_mask,
        sem="relaxed",
    )


@triton.jit
def _base_only_swiglu_activation_kernel(
    gateup_ptr,
    act_out_ptr,
    src2dst_ptr,
    topk_ids_ptr,
    token_lora_mapping_ptr,
    num_pairs,
    inter,
    num_local_experts,
    stride_gum,
    stride_gun,
    stride_aom,
    stride_aon,
    top_k: tl.constexpr,
    local_expert_offset: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fill W2 activations for base-only rows omitted by the LoRA route."""

    pair_idx = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    token_idx = pair_idx // top_k
    adapter = tl.load(token_lora_mapping_ptr + token_idx).to(tl.int64)
    global_expert = tl.load(topk_ids_ptr + pair_idx).to(tl.int64)
    local_expert = global_expert - local_expert_offset
    valid_base_pair = (
        (pair_idx < num_pairs)
        & (adapter < 0)
        & (local_expert >= 0)
        & (local_expert < num_local_experts)
    )
    dst_row = tl.load(src2dst_ptr + pair_idx, mask=valid_base_pair, other=0).to(
        tl.int64
    )
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    n_mask = offs_n < inter
    row = gateup_ptr + dst_row * stride_gum
    gate = tl.load(
        row + offs_n * stride_gun,
        mask=valid_base_pair & n_mask,
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        row + (inter + offs_n) * stride_gun,
        mask=valid_base_pair & n_mask,
        other=0.0,
    ).to(tl.float32)
    activated = (gate * tl.sigmoid(gate) * up).to(act_out_ptr.dtype.element_ty)
    tl.store(
        act_out_ptr + dst_row * stride_aom + offs_n * stride_aon,
        activated,
        mask=valid_base_pair & n_mask,
    )


@triton.jit
def _fused_gate_up_b_swiglu_down_a_aligned_kernel(
    gateup_ptr,
    gate_intermediate_ptr,
    gate_b_ptr,
    down_a_ptr,
    act_out_ptr,
    down_intermediate_ptr,
    src2dst_ptr,
    topk_ids_ptr,
    sorted_pair_ids_ptr,
    virtual_expert_ids_ptr,
    num_pairs_post_padded_ptr,
    num_pairs,
    inter,
    num_virtual_experts,
    num_local_experts,
    stride_gum,
    stride_gun,
    stride_gim,
    stride_gir,
    stride_gbe,
    stride_gbn,
    stride_gbr,
    stride_dae,
    stride_dar,
    stride_dan,
    stride_aom,
    stride_aon,
    stride_dim,
    stride_dir,
    gate_rank: tl.constexpr,
    down_rank: tl.constexpr,
    local_expert_offset: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_GATE_R: tl.constexpr,
    BLOCK_DOWN_R: tl.constexpr,
):
    """Aligned virtual-expert C2P schedule with shared LoRA weights per block."""
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    num_pairs_post_padded = tl.load(num_pairs_post_padded_ptr)
    if pid_m * BLOCK_M >= num_pairs_post_padded:
        return

    route_slots = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    pair_ids = tl.load(sorted_pair_ids_ptr + route_slots).to(tl.int64)
    pair_in_range = pair_ids < num_pairs
    # Clamp the route-padding sentinel before constructing any tensor pointer.
    # This also avoids relying on masked one-past-the-end vector addresses on
    # SM103. Expert validity is explicitly local: a positive global ID outside
    # this EP shard must not consume the provider-private ``src2dst`` value.
    safe_pair_ids = tl.where(pair_in_range, pair_ids, 0)
    base_expert = tl.load(
        topk_ids_ptr + safe_pair_ids, mask=pair_in_range, other=-1
    ).to(tl.int64)
    local_expert = base_expert - local_expert_offset
    valid_pair = (
        pair_in_range
        & (local_expert >= 0)
        & (local_expert < num_local_experts)
    )

    virtual_expert = tl.load(virtual_expert_ids_ptr + pid_m).to(tl.int64)
    valid_virtual_expert = (virtual_expert >= 0) & (
        virtual_expert < num_virtual_experts
    )
    safe_virtual_expert = tl.maximum(
        0, tl.minimum(virtual_expert, num_virtual_experts - 1)
    )
    has_lora = valid_pair & valid_virtual_expert

    dst_rows = tl.load(src2dst_ptr + safe_pair_ids, mask=valid_pair, other=0).to(
        tl.int64
    )
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    n_mask = offs_n < inter
    offs_gate_r = tl.arange(0, BLOCK_GATE_R).to(tl.int64)
    gate_r_mask = offs_gate_r < gate_rank

    gateup_rows = gateup_ptr + dst_rows[:, None] * stride_gum
    gate = tl.load(
        gateup_rows + offs_n[None, :] * stride_gun,
        mask=valid_pair[:, None] & n_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        gateup_rows + (inter + offs_n[None, :]) * stride_gun,
        mask=valid_pair[:, None] & n_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    gate_intermediate_rows = (
        gate_intermediate_ptr + safe_pair_ids[:, None] * stride_gim
    )
    gate_a = tl.load(
        gate_intermediate_rows + offs_gate_r[None, :] * stride_gir,
        mask=has_lora[:, None] & gate_r_mask[None, :],
        other=0.0,
    )
    up_a = tl.load(
        gate_intermediate_rows + (gate_rank + offs_gate_r[None, :]) * stride_gir,
        mask=has_lora[:, None] & gate_r_mask[None, :],
        other=0.0,
    )

    gate_b_base = gate_b_ptr + safe_virtual_expert * stride_gbe
    gate_b = tl.load(
        gate_b_base + offs_gate_r[:, None] * stride_gbr + offs_n[None, :] * stride_gbn,
        mask=(valid_virtual_expert & gate_r_mask[:, None] & n_mask[None, :]),
        other=0.0,
    )
    up_b = tl.load(
        gate_b_base
        + offs_gate_r[:, None] * stride_gbr
        + (inter + offs_n[None, :]) * stride_gbn,
        mask=(valid_virtual_expert & gate_r_mask[:, None] & n_mask[None, :]),
        other=0.0,
    )

    gate_delta = tl.dot(gate_a, gate_b, out_dtype=tl.float32)
    up_delta = tl.dot(up_a, up_b, out_dtype=tl.float32)
    gate += gate_delta.to(gateup_ptr.dtype.element_ty).to(tl.float32)
    up += up_delta.to(gateup_ptr.dtype.element_ty).to(tl.float32)

    activated = gate * tl.sigmoid(gate) * up
    activated_bf16 = activated.to(act_out_ptr.dtype.element_ty)
    tl.store(
        act_out_ptr + dst_rows[:, None] * stride_aom + offs_n[None, :] * stride_aon,
        activated_bf16,
        mask=valid_pair[:, None] & n_mask[None, :],
    )

    offs_down_r = tl.arange(0, BLOCK_DOWN_R).to(tl.int64)
    down_r_mask = offs_down_r < down_rank
    down_a_base = down_a_ptr + safe_virtual_expert * stride_dae
    # [I tile, R] so activated [M, I tile] @ down-A.T [I tile, R].
    down_a = tl.load(
        down_a_base + offs_n[:, None] * stride_dan + offs_down_r[None, :] * stride_dar,
        mask=(valid_virtual_expert & n_mask[:, None] & down_r_mask[None, :]),
        other=0.0,
    )
    partial = tl.dot(activated_bf16, down_a, out_dtype=tl.float32)
    partial = partial.to(down_intermediate_ptr.dtype.element_ty)
    tl.atomic_add(
        down_intermediate_ptr
        + safe_pair_ids[:, None] * stride_dim
        + offs_down_r[None, :] * stride_dir,
        partial,
        mask=has_lora[:, None] & down_r_mask[None, :],
        sem="relaxed",
    )


def fused_gate_up_b_swiglu_down_a(
    gateup_output: torch.Tensor,
    gate_up_intermediate: torch.Tensor,
    gate_up_lora_b: torch.Tensor,
    down_lora_a: torch.Tensor,
    act_out: torch.Tensor,
    down_intermediate: torch.Tensor,
    src2dst: torch.Tensor,
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    *,
    local_expert_offset: int = 0,
    block_size_n: int = 32,
    num_warps: int = 4,
) -> None:
    """Launch the benchmark-only BF16 C2 fused consumer.

    ``down_intermediate`` is an accumulation destination and must be zero on
    entry.  Keeping that operation caller-owned makes the buffer lifecycle and
    CUDA-graph cost visible to the execution-plan benchmark.
    """
    num_pairs = topk_ids.numel()
    inter = act_out.shape[-1]
    gate_rank = gate_up_lora_b.shape[-1]
    down_rank = down_lora_a.shape[-2]
    grid = (num_pairs, triton.cdiv(inter, block_size_n))
    _fused_gate_up_b_swiglu_down_a_kernel[grid](
        gateup_output.view(-1, 2 * inter),
        gate_up_intermediate.view(num_pairs, 2 * gate_rank),
        gate_up_lora_b,
        down_lora_a,
        act_out.view(-1, inter),
        down_intermediate.view(num_pairs, down_rank),
        src2dst,
        topk_ids,
        token_lora_mapping,
        num_pairs,
        inter,
        gate_up_lora_b.shape[1],
        gateup_output.stride(-2),
        gateup_output.stride(-1),
        gate_up_intermediate.stride(-2),
        gate_up_intermediate.stride(-1),
        gate_up_lora_b.stride(0),
        gate_up_lora_b.stride(1),
        gate_up_lora_b.stride(2),
        gate_up_lora_b.stride(3),
        down_lora_a.stride(0),
        down_lora_a.stride(1),
        down_lora_a.stride(2),
        down_lora_a.stride(3),
        act_out.stride(-2),
        act_out.stride(-1),
        down_intermediate.stride(-2),
        down_intermediate.stride(-1),
        top_k=topk_ids.shape[1],
        gate_rank=gate_rank,
        down_rank=down_rank,
        max_loras=gate_up_lora_b.shape[0],
        local_expert_offset=local_expert_offset,
        BLOCK_N=block_size_n,
        # Logical rank 8 is represented by a masked physical-16 tensor-core
        # tile.  Tail lanes read zero, so factor storage remains logical-rank
        # sized while Triton's dot-product K dimension stays compile-legal.
        BLOCK_GATE_R=max(16, triton.next_power_of_2(gate_rank)),
        BLOCK_DOWN_R=max(16, triton.next_power_of_2(down_rank)),
        num_warps=num_warps,
        num_stages=1,
    )


def fused_gate_up_b_swiglu_down_a_aligned(
    gateup_output: torch.Tensor,
    gate_up_intermediate: torch.Tensor,
    gate_up_lora_b: torch.Tensor,
    down_lora_a: torch.Tensor,
    act_out: torch.Tensor,
    down_intermediate: torch.Tensor,
    src2dst: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_pair_ids: torch.Tensor,
    virtual_expert_ids: torch.Tensor,
    num_pairs_post_padded: torch.Tensor,
    *,
    route_block_size_m: int,
    token_lora_mapping: torch.Tensor | None = None,
    local_expert_offset: int = 0,
    block_size_n: int = 64,
    num_warps: int = 4,
) -> None:
    """Launch the aligned virtual-expert C2P schedule.

    ``gate_up_lora_b`` and ``down_lora_a`` retain their logical
    ``[adapter, expert, ...]`` shapes. Their first two dimensions are folded by
    view only and indexed with the virtual expert IDs in the supplied route.
    As in the pair-owned schedule, ``down_intermediate`` must be zero on entry.
    When ``token_lora_mapping`` is supplied, a preceding scan fills base-only
    activations that the LoRA-aligned route is allowed to omit. Omitting that
    mapping is correct only when there are no base rows or the route includes
    them explicitly as ``virtual_expert == -1`` blocks.
    """
    num_pairs = topk_ids.numel()
    inter = act_out.shape[-1]
    gate_rank = gate_up_lora_b.shape[-1]
    down_rank = down_lora_a.shape[-2]
    gate_b_virtual = gate_up_lora_b.view(
        gate_up_lora_b.shape[0] * gate_up_lora_b.shape[1],
        gate_up_lora_b.shape[2],
        gate_rank,
    )
    down_a_virtual = down_lora_a.view(
        down_lora_a.shape[0] * down_lora_a.shape[1],
        down_rank,
        down_lora_a.shape[3],
    )
    grid = (
        triton.cdiv(sorted_pair_ids.shape[0], route_block_size_m),
        triton.cdiv(inter, block_size_n),
    )
    if token_lora_mapping is not None:
        base_grid = (num_pairs, triton.cdiv(inter, block_size_n))
        _base_only_swiglu_activation_kernel[base_grid](
            gateup_output.view(-1, 2 * inter),
            act_out.view(-1, inter),
            src2dst,
            topk_ids,
            token_lora_mapping,
            num_pairs,
            inter,
            gate_up_lora_b.shape[1],
            gateup_output.stride(-2),
            gateup_output.stride(-1),
            act_out.stride(-2),
            act_out.stride(-1),
            top_k=topk_ids.shape[1],
            local_expert_offset=local_expert_offset,
            BLOCK_N=block_size_n,
            num_warps=num_warps,
            num_stages=1,
        )
    _fused_gate_up_b_swiglu_down_a_aligned_kernel[grid](
        gateup_output.view(-1, 2 * inter),
        gate_up_intermediate.view(num_pairs, 2 * gate_rank),
        gate_b_virtual,
        down_a_virtual,
        act_out.view(-1, inter),
        down_intermediate.view(num_pairs, down_rank),
        src2dst,
        topk_ids,
        sorted_pair_ids,
        virtual_expert_ids,
        num_pairs_post_padded,
        num_pairs,
        inter,
        gate_b_virtual.shape[0],
        gate_up_lora_b.shape[1],
        gateup_output.stride(-2),
        gateup_output.stride(-1),
        gate_up_intermediate.stride(-2),
        gate_up_intermediate.stride(-1),
        gate_b_virtual.stride(0),
        gate_b_virtual.stride(1),
        gate_b_virtual.stride(2),
        down_a_virtual.stride(0),
        down_a_virtual.stride(1),
        down_a_virtual.stride(2),
        act_out.stride(-2),
        act_out.stride(-1),
        down_intermediate.stride(-2),
        down_intermediate.stride(-1),
        gate_rank=gate_rank,
        down_rank=down_rank,
        local_expert_offset=local_expert_offset,
        BLOCK_M=route_block_size_m,
        BLOCK_N=block_size_n,
        BLOCK_GATE_R=max(16, triton.next_power_of_2(gate_rank)),
        BLOCK_DOWN_R=max(16, triton.next_power_of_2(down_rank)),
        num_warps=num_warps,
        num_stages=1,
    )


__all__ = [
    "fused_gate_up_b_swiglu_down_a",
    "fused_gate_up_b_swiglu_down_a_aligned",
]
