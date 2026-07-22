"""Benchmark-only Triton primitives for the MoE-LoRA algorithm-family matrix.

Nothing in this module is imported by serving code.  The kernels deliberately
consume a small canonical contract so raw-indexed, padded aligned/grouped,
segment-block SGMV, and regular BMM schedules can be compared without changing
production dispatch:

* routed pair ids are canonical ``token * top_k + slot`` indices;
* a virtual group is ``adapter * E_local + local_expert``;
* segment-block metadata stores only nonempty BM-sized pieces of sorted groups;
* every compute kernel receives the same preallocated destination.

The segment-block schedule is a true segmented path: no output row padding is
materialized, and each program reads one descriptor ``(group, start, length)``
before applying that group's matrix to the corresponding sorted rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass(slots=True)
class SegmentBlockPlan:
    """GPU-resident route and segment-block descriptors.

    Arrays are allocated to static upper bounds so the same builder and
    consumers are CUDA-graph capturable.  ``num_blocks`` is a device scalar;
    consumers use it for an early return instead of synchronizing it to host.
    """

    raw_groups: torch.Tensor
    sorted_pair_ids: torch.Tensor
    sorted_groups: torch.Tensor
    group_counts: torch.Tensor
    group_block_counts: torch.Tensor
    segment_offsets: torch.Tensor
    block_offsets: torch.Tensor
    block_groups: torch.Tensor
    block_starts: torch.Tensor
    block_lengths: torch.Tensor
    num_blocks: torch.Tensor
    num_experts: int
    num_adapters: int
    num_groups: int
    block_m: int


@triton.jit
def _virtual_group_ids_kernel(
    topk_ids_ptr,
    token_lora_mapping_ptr,
    groups_ptr,
    num_pairs,
    num_experts,
    num_adapters,
    top_k: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < num_pairs
    token_ids = offsets // top_k
    experts = tl.load(topk_ids_ptr + offsets, mask=mask, other=-1).to(tl.int64)
    adapters = tl.load(token_lora_mapping_ptr + token_ids, mask=mask, other=-1).to(
        tl.int64
    )
    valid = (
        (experts >= 0)
        & (experts < num_experts)
        & (adapters >= 0)
        & (adapters < num_adapters)
    )
    sentinel = num_experts * num_adapters
    groups = tl.where(valid, adapters * num_experts + experts, sentinel)
    tl.store(groups_ptr + offsets, groups, mask=mask)


@triton.jit
def _segment_descriptors_kernel(
    sorted_groups_ptr,
    segment_offsets_ptr,
    block_offsets_ptr,
    block_groups_ptr,
    block_starts_ptr,
    block_lengths_ptr,
    num_pairs,
    num_groups,
    BLOCK_M: tl.constexpr,
    BLOCK: tl.constexpr,
):
    positions = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_range = positions < num_pairs
    groups = tl.load(sorted_groups_ptr + positions, mask=in_range, other=num_groups)
    valid = in_range & (groups >= 0) & (groups < num_groups)
    safe_groups = tl.minimum(tl.maximum(groups, 0), num_groups - 1)
    starts = tl.load(segment_offsets_ptr + safe_groups, mask=valid, other=0)
    ends = tl.load(segment_offsets_ptr + safe_groups + 1, mask=valid, other=0)
    local = positions - starts
    is_block_start = valid & (local % BLOCK_M == 0)
    destinations = (
        tl.load(block_offsets_ptr + safe_groups, mask=is_block_start, other=0)
        + local // BLOCK_M
    )
    lengths = tl.minimum(BLOCK_M, ends - positions)
    tl.store(block_groups_ptr + destinations, groups, mask=is_block_start)
    tl.store(block_starts_ptr + destinations, positions, mask=is_block_start)
    tl.store(block_lengths_ptr + destinations, lengths, mask=is_block_start)


def allocate_segment_plan(
    num_pairs: int,
    num_experts: int,
    num_adapters: int,
    block_m: int,
    device: torch.device,
) -> SegmentBlockPlan:
    num_groups = num_experts * num_adapters
    if num_pairs <= 0 or num_experts <= 0 or num_adapters <= 0 or block_m <= 0:
        raise ValueError(
            "num_pairs, num_experts, num_adapters, and block_m must be positive"
        )
    return SegmentBlockPlan(
        raw_groups=torch.empty(num_pairs, dtype=torch.int64, device=device),
        sorted_pair_ids=torch.empty(num_pairs, dtype=torch.int64, device=device),
        sorted_groups=torch.empty(num_pairs, dtype=torch.int64, device=device),
        group_counts=torch.empty(num_groups, dtype=torch.int64, device=device),
        group_block_counts=torch.empty(num_groups, dtype=torch.int64, device=device),
        segment_offsets=torch.empty(num_groups + 1, dtype=torch.int64, device=device),
        block_offsets=torch.empty(num_groups + 1, dtype=torch.int64, device=device),
        block_groups=torch.empty(num_pairs, dtype=torch.int64, device=device),
        block_starts=torch.empty(num_pairs, dtype=torch.int64, device=device),
        block_lengths=torch.empty(num_pairs, dtype=torch.int64, device=device),
        num_blocks=torch.empty(1, dtype=torch.int64, device=device),
        num_experts=num_experts,
        num_adapters=num_adapters,
        num_groups=num_groups,
        block_m=block_m,
    )


@triton.jit
def _histogram_groups_kernel(
    sorted_groups_ptr,
    counts_ptr,
    num_pairs,
    num_groups,
    BLOCK: tl.constexpr,
):
    positions = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = positions < num_pairs
    groups = tl.load(sorted_groups_ptr + positions, mask=mask, other=num_groups)
    valid = mask & (groups >= 0) & (groups < num_groups)
    safe_groups = tl.minimum(tl.maximum(groups, 0), num_groups - 1)
    tl.atomic_add(counts_ptr + safe_groups, 1, mask=valid)


def build_segment_plan_into(
    plan: SegmentBlockPlan,
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
) -> None:
    """Build sorting, segment pointers, and compact block descriptors on GPU.

    The caller owns all allocations.  This makes the full builder suitable for
    both route-inclusive eager timing and CUDA-graph capture.
    """
    num_pairs = topk_ids.numel()
    top_k = topk_ids.shape[1]
    _virtual_group_ids_kernel[(triton.cdiv(num_pairs, 256),)](
        topk_ids,
        token_lora_mapping,
        plan.raw_groups,
        num_pairs,
        plan.num_experts,
        plan.num_adapters,
        top_k=top_k,
        BLOCK=256,
        num_warps=4,
    )
    sorted_groups, sorted_pair_ids = torch.sort(plan.raw_groups, stable=True)
    plan.sorted_groups.copy_(sorted_groups)
    plan.sorted_pair_ids.copy_(sorted_pair_ids)
    plan.group_counts.zero_()
    _histogram_groups_kernel[(triton.cdiv(num_pairs, 256),)](
        plan.sorted_groups,
        plan.group_counts,
        num_pairs,
        plan.num_groups,
        BLOCK=256,
        num_warps=4,
    )
    plan.segment_offsets[0].zero_()
    torch.cumsum(plan.group_counts, dim=0, out=plan.segment_offsets[1:])
    torch.div(
        plan.group_counts + plan.block_m - 1,
        plan.block_m,
        rounding_mode="floor",
        out=plan.group_block_counts,
    )
    plan.block_offsets[0].zero_()
    torch.cumsum(plan.group_block_counts, dim=0, out=plan.block_offsets[1:])
    plan.num_blocks.copy_(plan.block_offsets[-1:])
    _segment_descriptors_kernel[(triton.cdiv(num_pairs, 256),)](
        plan.sorted_groups,
        plan.segment_offsets,
        plan.block_offsets,
        plan.block_groups,
        plan.block_starts,
        plan.block_lengths,
        num_pairs,
        plan.num_groups,
        BLOCK_M=plan.block_m,
        BLOCK=256,
        num_warps=4,
    )


@triton.jit
def _segmented_gemm_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    sorted_pair_ids_ptr,
    block_groups_ptr,
    block_starts_ptr,
    block_lengths_ptr,
    num_blocks_ptr,
    num_pairs,
    n_size,
    k_size,
    stride_xm,
    stride_xk,
    stride_wg,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    top_k: tl.constexpr,
    INPUT_PAIR_MAJOR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    block_id = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    num_blocks = tl.load(num_blocks_ptr)
    if block_id >= num_blocks:
        return

    group = tl.load(block_groups_ptr + block_id).to(tl.int64)
    start = tl.load(block_starts_ptr + block_id).to(tl.int64)
    length = tl.load(block_lengths_ptr + block_id).to(tl.int64)
    rows = tl.arange(0, BLOCK_M).to(tl.int64)
    pair_ids = tl.load(
        sorted_pair_ids_ptr + start + rows,
        mask=rows < length,
        other=num_pairs,
    ).to(tl.int64)
    row_mask = (rows < length) & (pair_ids < num_pairs)
    x_rows = pair_ids if INPUT_PAIR_MAJOR else pair_ids // top_k
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    n_mask = offs_n < n_size
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k0 in range(0, tl.cdiv(k_size, BLOCK_K)):
        offs_k = k0 * BLOCK_K + tl.arange(0, BLOCK_K).to(tl.int64)
        k_mask = offs_k < k_size
        x = tl.load(
            x_ptr + x_rows[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        w = tl.load(
            weight_ptr
            + group * stride_wg
            + offs_n[None, :] * stride_wn
            + offs_k[:, None] * stride_wk,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(x, w, out_dtype=tl.float32)

    tl.store(
        output_ptr + pair_ids[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=row_mask[:, None] & n_mask[None, :],
    )


def segmented_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    plan: SegmentBlockPlan,
    *,
    top_k: int,
    input_pair_major: bool,
    block_n: int = 32,
    block_k: int = 64,
    num_warps: int = 4,
) -> None:
    """Apply ``weight[group]`` to routed rows without padded output rows."""
    num_pairs, n_size = output.shape
    k_size = x.shape[1]
    if weight.shape != (plan.num_groups, n_size, k_size):
        raise ValueError(
            f"weight must be {(plan.num_groups, n_size, k_size)}, got {weight.shape}"
        )
    grid = (num_pairs, triton.cdiv(n_size, block_n))
    _segmented_gemm_kernel[grid](
        x,
        weight,
        output,
        plan.sorted_pair_ids,
        plan.block_groups,
        plan.block_starts,
        plan.block_lengths,
        plan.num_blocks,
        num_pairs,
        n_size,
        k_size,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        output.stride(0),
        output.stride(1),
        top_k=top_k,
        INPUT_PAIR_MAJOR=input_pair_major,
        BLOCK_M=plan.block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=3,
    )


@triton.jit
def _segmented_gate_b_consumer_kernel(
    gateup_base_ptr,
    gate_rank_ptr,
    gate_b_ptr,
    down_a_ptr,
    act_out_ptr,
    down_rank_ptr,
    sorted_pair_ids_ptr,
    block_groups_ptr,
    block_starts_ptr,
    block_lengths_ptr,
    num_blocks_ptr,
    num_pairs,
    inter_size,
    gate_rank_size,
    down_rank_size,
    stride_gum,
    stride_gun,
    stride_grm,
    stride_grr,
    stride_gbg,
    stride_gbn,
    stride_gbr,
    stride_dag,
    stride_dar,
    stride_dan,
    stride_aom,
    stride_aon,
    stride_drm,
    stride_drr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_GATE_R: tl.constexpr,
    BLOCK_DOWN_R: tl.constexpr,
    ROUND_GATE_DELTA_TO_BF16: tl.constexpr,
    ROUND_DOWN_PARTIAL_TO_BF16: tl.constexpr,
):
    block_id = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    num_blocks = tl.load(num_blocks_ptr)
    if block_id >= num_blocks:
        return
    group = tl.load(block_groups_ptr + block_id).to(tl.int64)
    start = tl.load(block_starts_ptr + block_id).to(tl.int64)
    length = tl.load(block_lengths_ptr + block_id).to(tl.int64)
    rows = tl.arange(0, BLOCK_M).to(tl.int64)
    pairs = tl.load(
        sorted_pair_ids_ptr + start + rows,
        mask=rows < length,
        other=num_pairs,
    ).to(tl.int64)
    row_mask = (rows < length) & (pairs < num_pairs)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    n_mask = offs_n < inter_size
    offs_gr = tl.arange(0, BLOCK_GATE_R).to(tl.int64)
    gr_mask = offs_gr < gate_rank_size

    rank_rows = gate_rank_ptr + pairs[:, None] * stride_grm
    gate_r = tl.load(
        rank_rows + offs_gr[None, :] * stride_grr,
        mask=row_mask[:, None] & gr_mask[None, :],
        other=0.0,
    )
    up_r = tl.load(
        rank_rows + (gate_rank_size + offs_gr[None, :]) * stride_grr,
        mask=row_mask[:, None] & gr_mask[None, :],
        other=0.0,
    )
    gate_b = tl.load(
        gate_b_ptr
        + group * stride_gbg
        + offs_gr[:, None] * stride_gbr
        + offs_n[None, :] * stride_gbn,
        mask=gr_mask[:, None] & n_mask[None, :],
        other=0.0,
    )
    up_b = tl.load(
        gate_b_ptr
        + group * stride_gbg
        + offs_gr[:, None] * stride_gbr
        + (inter_size + offs_n[None, :]) * stride_gbn,
        mask=gr_mask[:, None] & n_mask[None, :],
        other=0.0,
    )
    gate_delta = tl.dot(gate_r, gate_b, out_dtype=tl.float32)
    up_delta = tl.dot(up_r, up_b, out_dtype=tl.float32)
    if ROUND_GATE_DELTA_TO_BF16:
        gate_delta = gate_delta.to(tl.bfloat16).to(tl.float32)
        up_delta = up_delta.to(tl.bfloat16).to(tl.float32)
    base_rows = gateup_base_ptr + pairs[:, None] * stride_gum
    gate = tl.load(
        base_rows + offs_n[None, :] * stride_gun,
        mask=row_mask[:, None] & n_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        base_rows + (inter_size + offs_n[None, :]) * stride_gun,
        mask=row_mask[:, None] & n_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    gate += gate_delta
    up += up_delta
    activated = gate * tl.sigmoid(gate) * up
    activated_bf16 = activated.to(tl.bfloat16)
    tl.store(
        act_out_ptr + pairs[:, None] * stride_aom + offs_n[None, :] * stride_aon,
        activated_bf16,
        mask=row_mask[:, None] & n_mask[None, :],
    )

    offs_dr = tl.arange(0, BLOCK_DOWN_R).to(tl.int64)
    dr_mask = offs_dr < down_rank_size
    down_a = tl.load(
        down_a_ptr
        + group * stride_dag
        + offs_n[:, None] * stride_dan
        + offs_dr[None, :] * stride_dar,
        mask=n_mask[:, None] & dr_mask[None, :],
        other=0.0,
    )
    partial = tl.dot(activated_bf16, down_a, out_dtype=tl.float32)
    if ROUND_DOWN_PARTIAL_TO_BF16:
        partial = partial.to(tl.bfloat16).to(tl.float32)
    tl.atomic_add(
        down_rank_ptr + pairs[:, None] * stride_drm + offs_dr[None, :] * stride_drr,
        partial,
        mask=row_mask[:, None] & dr_mask[None, :],
        sem="relaxed",
    )


def segmented_gate_b_consumer(
    gateup_base: torch.Tensor,
    gate_rank: torch.Tensor,
    gate_b: torch.Tensor,
    down_a: torch.Tensor,
    act_out: torch.Tensor,
    down_rank: torch.Tensor,
    plan: SegmentBlockPlan,
    *,
    block_n: int = 32,
    num_warps: int = 4,
    round_gate_delta_to_bf16: bool = True,
    round_down_partial_to_bf16: bool = True,
) -> None:
    """Fused segmented gate/up-B + SwiGLU + down-A consumer."""
    num_pairs, two_i = gateup_base.shape
    inter_size = two_i // 2
    gate_rank_size = gate_b.shape[-1]
    down_rank_size = down_a.shape[-2]
    grid = (num_pairs, triton.cdiv(inter_size, block_n))
    _segmented_gate_b_consumer_kernel[grid](
        gateup_base,
        gate_rank,
        gate_b,
        down_a,
        act_out,
        down_rank,
        plan.sorted_pair_ids,
        plan.block_groups,
        plan.block_starts,
        plan.block_lengths,
        plan.num_blocks,
        num_pairs,
        inter_size,
        gate_rank_size,
        down_rank_size,
        gateup_base.stride(0),
        gateup_base.stride(1),
        gate_rank.stride(0),
        gate_rank.stride(1),
        gate_b.stride(0),
        gate_b.stride(1),
        gate_b.stride(2),
        down_a.stride(0),
        down_a.stride(1),
        down_a.stride(2),
        act_out.stride(0),
        act_out.stride(1),
        down_rank.stride(0),
        down_rank.stride(1),
        BLOCK_M=plan.block_m,
        BLOCK_N=block_n,
        # Triton tensor-core dots require K >= 16.  Keep rank 8 as a logical
        # shape, but execute it in a masked physical-16 tile; the inactive
        # lanes load zero and therefore do not require padded factor storage.
        BLOCK_GATE_R=max(16, triton.next_power_of_2(gate_rank_size)),
        BLOCK_DOWN_R=max(16, triton.next_power_of_2(down_rank_size)),
        ROUND_GATE_DELTA_TO_BF16=round_gate_delta_to_bf16,
        ROUND_DOWN_PARTIAL_TO_BF16=round_down_partial_to_bf16,
        num_warps=num_warps,
        num_stages=2,
    )


@triton.jit
def _base_only_swiglu_kernel(
    gateup_base_ptr,
    mapping_ptr,
    act_out_ptr,
    num_pairs,
    inter_size,
    stride_gm,
    stride_gn,
    stride_am,
    stride_an,
    top_k: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pair = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    token = pair // top_k
    adapter = tl.load(mapping_ptr + token).to(tl.int64)
    is_base = (pair < num_pairs) & (adapter < 0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    n_mask = offs_n < inter_size
    gate = tl.load(
        gateup_base_ptr + pair * stride_gm + offs_n * stride_gn,
        mask=is_base & n_mask,
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        gateup_base_ptr + pair * stride_gm + (inter_size + offs_n) * stride_gn,
        mask=is_base & n_mask,
        other=0.0,
    ).to(tl.float32)
    activated = gate * tl.sigmoid(gate) * up
    tl.store(
        act_out_ptr + pair * stride_am + offs_n * stride_an,
        activated,
        mask=is_base & n_mask,
    )


def base_only_swiglu(
    gateup_base: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    act_out: torch.Tensor,
    *,
    top_k: int,
    block_n: int = 64,
) -> None:
    num_pairs, two_i = gateup_base.shape
    inter_size = two_i // 2
    _base_only_swiglu_kernel[(num_pairs, triton.cdiv(inter_size, block_n))](
        gateup_base,
        token_lora_mapping,
        act_out,
        num_pairs,
        inter_size,
        gateup_base.stride(0),
        gateup_base.stride(1),
        act_out.stride(0),
        act_out.stride(1),
        top_k=top_k,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=1,
    )


@triton.jit
def _indexed_gemm_kernel(
    x_ptr,
    weight_ptr,
    topk_ids_ptr,
    mapping_ptr,
    output_ptr,
    num_pairs,
    num_experts,
    num_adapters,
    n_size,
    k_size,
    stride_xm,
    stride_xk,
    stride_wg,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    top_k: tl.constexpr,
    INPUT_PAIR_MAJOR: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pair = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    token = pair // top_k
    expert = tl.load(topk_ids_ptr + pair).to(tl.int64)
    adapter = tl.load(mapping_ptr + token).to(tl.int64)
    valid = (
        (pair < num_pairs)
        & (expert >= 0)
        & (expert < num_experts)
        & (adapter >= 0)
        & (adapter < num_adapters)
    )
    safe_expert = tl.minimum(tl.maximum(expert, 0), num_experts - 1)
    safe_adapter = tl.minimum(tl.maximum(adapter, 0), num_adapters - 1)
    group = safe_adapter * num_experts + safe_expert
    x_row = pair if INPUT_PAIR_MAJOR else token
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    n_mask = offs_n < n_size
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(0, tl.cdiv(k_size, BLOCK_K)):
        offs_k = k0 * BLOCK_K + tl.arange(0, BLOCK_K).to(tl.int64)
        k_mask = offs_k < k_size
        xv = tl.load(
            x_ptr + x_row * stride_xm + offs_k * stride_xk,
            mask=valid & k_mask,
            other=0.0,
        )
        w = tl.load(
            weight_ptr
            + group * stride_wg
            + offs_n[:, None] * stride_wn
            + offs_k[None, :] * stride_wk,
            mask=valid & n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        acc += tl.sum(w.to(tl.float32) * xv[None, :].to(tl.float32), axis=1)
    tl.store(
        output_ptr + pair * stride_om + offs_n * stride_on,
        acc,
        mask=valid & n_mask,
    )


def indexed_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    output: torch.Tensor,
    *,
    num_experts: int,
    num_adapters: int,
    input_pair_major: bool,
    block_n: int = 16,
    block_k: int = 64,
    num_warps: int = 4,
) -> None:
    num_pairs, n_size = output.shape
    num_adapters_times_experts, weight_n, k_size = weight.shape
    if num_adapters_times_experts != num_experts * num_adapters:
        raise ValueError("weight group dimension differs from E * L")
    if weight_n != n_size:
        raise ValueError("weight/output N dimensions differ")
    grid = (num_pairs, triton.cdiv(n_size, block_n))
    _indexed_gemm_kernel[grid](
        x,
        weight,
        topk_ids,
        token_lora_mapping,
        output,
        num_pairs,
        num_experts,
        num_adapters,
        n_size,
        k_size,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        output.stride(0),
        output.stride(1),
        top_k=topk_ids.shape[1],
        INPUT_PAIR_MAJOR=input_pair_major,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=2,
    )


@triton.jit
def _one_shot_down_ab_kernel(
    x_ptr,
    a_ptr,
    b_ptr,
    topk_ids_ptr,
    mapping_ptr,
    output_ptr,
    num_pairs,
    num_experts,
    num_adapters,
    i_size,
    h_size,
    rank,
    stride_xm,
    stride_xi,
    stride_ag,
    stride_ar,
    stride_ai,
    stride_bg,
    stride_bh,
    stride_br,
    stride_om,
    stride_oh,
    top_k: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_R: tl.constexpr,
    ROUND_RANK_TO_BF16: tl.constexpr,
):
    pair = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1).to(tl.int64)
    token = pair // top_k
    expert = tl.load(topk_ids_ptr + pair).to(tl.int64)
    adapter = tl.load(mapping_ptr + token).to(tl.int64)
    valid = (
        (pair < num_pairs)
        & (expert >= 0)
        & (expert < num_experts)
        & (adapter >= 0)
        & (adapter < num_adapters)
    )
    safe_expert = tl.minimum(tl.maximum(expert, 0), num_experts - 1)
    safe_adapter = tl.minimum(tl.maximum(adapter, 0), num_adapters - 1)
    group = safe_adapter * num_experts + safe_expert
    offs_r = tl.arange(0, BLOCK_R).to(tl.int64)
    r_mask = offs_r < rank
    rank_acc = tl.zeros((BLOCK_R,), tl.float32)
    for i0 in range(0, tl.cdiv(i_size, BLOCK_I)):
        offs_i = i0 * BLOCK_I + tl.arange(0, BLOCK_I).to(tl.int64)
        i_mask = offs_i < i_size
        xv = tl.load(
            x_ptr + pair * stride_xm + offs_i * stride_xi,
            mask=valid & i_mask,
            other=0.0,
        )
        av = tl.load(
            a_ptr
            + group * stride_ag
            + offs_r[:, None] * stride_ar
            + offs_i[None, :] * stride_ai,
            mask=valid & r_mask[:, None] & i_mask[None, :],
            other=0.0,
        )
        rank_acc += tl.sum(av.to(tl.float32) * xv[None, :].to(tl.float32), axis=1)
    if ROUND_RANK_TO_BF16:
        rank_acc = rank_acc.to(tl.bfloat16).to(tl.float32)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H).to(tl.int64)
    h_mask = offs_h < h_size
    bv = tl.load(
        b_ptr
        + group * stride_bg
        + offs_h[:, None] * stride_bh
        + offs_r[None, :] * stride_br,
        mask=valid & h_mask[:, None] & r_mask[None, :],
        other=0.0,
    )
    out = tl.sum(bv.to(tl.float32) * rank_acc[None, :], axis=1)
    tl.store(
        output_ptr + pair * stride_om + offs_h * stride_oh,
        out,
        mask=valid & h_mask,
    )


def one_shot_down_ab(
    activated_pairs: torch.Tensor,
    down_a: torch.Tensor,
    down_b: torch.Tensor,
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    output: torch.Tensor,
    *,
    num_experts: int,
    num_adapters: int,
    round_rank_to_bf16: bool,
    block_i: int = 64,
    block_h: int = 32,
    num_warps: int = 4,
) -> None:
    num_pairs, i_size = activated_pairs.shape
    num_groups, rank, a_i = down_a.shape
    b_groups, h_size, b_rank = down_b.shape
    if (a_i, b_groups, b_rank) != (i_size, num_groups, rank):
        raise ValueError("incompatible down A/B shapes")
    if num_groups != num_experts * num_adapters:
        raise ValueError("down factor group dimension differs from E * L")
    grid = (num_pairs, triton.cdiv(h_size, block_h))
    _one_shot_down_ab_kernel[grid](
        activated_pairs,
        down_a,
        down_b,
        topk_ids,
        token_lora_mapping,
        output,
        num_pairs,
        num_experts,
        num_adapters,
        i_size,
        h_size,
        rank,
        activated_pairs.stride(0),
        activated_pairs.stride(1),
        down_a.stride(0),
        down_a.stride(1),
        down_a.stride(2),
        down_b.stride(0),
        down_b.stride(1),
        down_b.stride(2),
        output.stride(0),
        output.stride(1),
        top_k=topk_ids.shape[1],
        BLOCK_I=block_i,
        BLOCK_H=block_h,
        BLOCK_R=triton.next_power_of_2(rank),
        ROUND_RANK_TO_BF16=round_rank_to_bf16,
        num_warps=num_warps,
        num_stages=2,
    )


@triton.jit
def _reduce_pairs_finalize_kernel(
    base_pairs_ptr,
    delta_pairs_ptr,
    topk_weights_ptr,
    output_ptr,
    h_size,
    stride_bm,
    stride_bh,
    stride_dm,
    stride_dh,
    stride_om,
    stride_oh,
    top_k: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROUND_DELTA_TO_BF16: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1).to(tl.int64)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H).to(tl.int64)
    h_mask = offs_h < h_size
    acc = tl.zeros((BLOCK_H,), tl.float32)
    for slot in tl.static_range(0, top_k):
        pair = token * top_k + slot
        routed = tl.load(topk_weights_ptr + pair).to(tl.float32)
        base = tl.load(
            base_pairs_ptr + pair * stride_bm + offs_h * stride_bh,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        delta = tl.load(
            delta_pairs_ptr + pair * stride_dm + offs_h * stride_dh,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        if ROUND_DELTA_TO_BF16:
            delta = delta.to(tl.bfloat16).to(tl.float32)
        acc += (base + delta) * routed
    tl.store(
        output_ptr + token * stride_om + offs_h * stride_oh,
        acc,
        mask=h_mask,
    )


def reduce_pairs_finalize(
    base_pairs: torch.Tensor,
    delta_pairs: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
    *,
    round_delta_to_bf16: bool,
    block_h: int = 64,
    num_warps: int = 4,
) -> None:
    h_size = output.shape[1]
    top_k = topk_weights.shape[1]
    grid = (output.shape[0], triton.cdiv(h_size, block_h))
    _reduce_pairs_finalize_kernel[grid](
        base_pairs,
        delta_pairs,
        topk_weights,
        output,
        h_size,
        base_pairs.stride(0),
        base_pairs.stride(1),
        delta_pairs.stride(0),
        delta_pairs.stride(1),
        output.stride(0),
        output.stride(1),
        top_k=top_k,
        BLOCK_H=block_h,
        ROUND_DELTA_TO_BF16=round_delta_to_bf16,
        num_warps=num_warps,
        num_stages=1,
    )


@triton.jit
def _direct_down_b_finalize_kernel(
    base_pairs_ptr,
    down_rank_ptr,
    down_b_ptr,
    topk_ids_ptr,
    mapping_ptr,
    topk_weights_ptr,
    output_ptr,
    num_experts,
    num_adapters,
    h_size,
    rank,
    stride_bpm,
    stride_bph,
    stride_drm,
    stride_drr,
    stride_dbg,
    stride_dbh,
    stride_dbr,
    stride_om,
    stride_oh,
    top_k: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_R: tl.constexpr,
    ROUND_DELTA_TO_BF16: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1).to(tl.int64)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H).to(tl.int64)
    h_mask = offs_h < h_size
    offs_r = tl.arange(0, BLOCK_R).to(tl.int64)
    r_mask = offs_r < rank
    adapter = tl.load(mapping_ptr + token).to(tl.int64)
    valid_adapter = (adapter >= 0) & (adapter < num_adapters)
    safe_adapter = tl.minimum(tl.maximum(adapter, 0), num_adapters - 1)
    acc = tl.zeros((BLOCK_H,), tl.float32)
    for slot in tl.static_range(0, top_k):
        pair = token * top_k + slot
        expert = tl.load(topk_ids_ptr + pair).to(tl.int64)
        valid_expert = (expert >= 0) & (expert < num_experts)
        safe_expert = tl.minimum(tl.maximum(expert, 0), num_experts - 1)
        routed = tl.load(topk_weights_ptr + pair).to(tl.float32)
        base = tl.load(
            base_pairs_ptr + pair * stride_bpm + offs_h * stride_bph,
            mask=valid_expert & h_mask,
            other=0.0,
        ).to(tl.float32)
        rank_row = tl.load(
            down_rank_ptr + pair * stride_drm + offs_r * stride_drr,
            mask=valid_expert & valid_adapter & r_mask,
            other=0.0,
        )
        group = safe_adapter * num_experts + safe_expert
        b = tl.load(
            down_b_ptr
            + group * stride_dbg
            + offs_h[:, None] * stride_dbh
            + offs_r[None, :] * stride_dbr,
            mask=(valid_expert & valid_adapter & h_mask[:, None] & r_mask[None, :]),
            other=0.0,
        )
        delta = tl.sum(b.to(tl.float32) * rank_row[None, :].to(tl.float32), axis=1)
        if ROUND_DELTA_TO_BF16:
            delta = delta.to(tl.bfloat16).to(tl.float32)
        acc += (base + delta) * routed
    tl.store(
        output_ptr + token * stride_om + offs_h * stride_oh,
        acc,
        mask=h_mask,
    )


def direct_down_b_finalize(
    base_pairs: torch.Tensor,
    down_rank: torch.Tensor,
    down_b: torch.Tensor,
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
    *,
    num_experts: int,
    num_adapters: int,
    round_delta_to_bf16: bool,
    block_h: int = 32,
    num_warps: int = 4,
) -> None:
    h_size = output.shape[1]
    rank = down_rank.shape[1]
    grid = (output.shape[0], triton.cdiv(h_size, block_h))
    _direct_down_b_finalize_kernel[grid](
        base_pairs,
        down_rank,
        down_b,
        topk_ids,
        token_lora_mapping,
        topk_weights,
        output,
        num_experts,
        num_adapters,
        h_size,
        rank,
        base_pairs.stride(0),
        base_pairs.stride(1),
        down_rank.stride(0),
        down_rank.stride(1),
        down_b.stride(0),
        down_b.stride(1),
        down_b.stride(2),
        output.stride(0),
        output.stride(1),
        top_k=topk_ids.shape[1],
        BLOCK_H=block_h,
        BLOCK_R=triton.next_power_of_2(rank),
        ROUND_DELTA_TO_BF16=round_delta_to_bf16,
        num_warps=num_warps,
        num_stages=1,
    )


__all__ = [
    "SegmentBlockPlan",
    "allocate_segment_plan",
    "base_only_swiglu",
    "build_segment_plan_into",
    "direct_down_b_finalize",
    "indexed_gemm",
    "one_shot_down_ab",
    "reduce_pairs_finalize",
    "segmented_gate_b_consumer",
    "segmented_gemm",
]
