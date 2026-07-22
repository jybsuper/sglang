"""Benchmark-only BF16 C2 consumer for non-gated ReLU-squared MoE.

Nemotron's non-gated experts expose one logical GEMM1 value slice instead of a
gate/up pair.  Keeping this kernel separate from ``fused_c2.py`` avoids adding
activation branches and dead gate accumulators to the two-slice tensor-core
schedule.  Both families consume the same routed-pair and aligned virtual-
expert metadata, and produce the same two C2 boundaries:

* provider-private activation output for base W2; and
* canonical ``[token, top-k, rank]`` down-A output for LoRA down-B.

The logical intermediate width is explicit and may be smaller than the
provider's physical per-slice width.  Valid destination rows have physical
padding zeroed; padding cannot leak into ReLU2 or down-A.  Production dispatch
does not import or select this module.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_value_b_relu2_down_a_pair_kernel(
    value_ptr,
    value_intermediate_ptr,
    value_b_ptr,
    down_a_ptr,
    act_out_ptr,
    down_intermediate_ptr,
    src2dst_ptr,
    topk_ids_ptr,
    token_lora_mapping_ptr,
    num_pairs,
    logical_inter,
    physical_inter,
    num_local_experts,
    stride_vm,
    stride_vn,
    stride_vim,
    stride_vir,
    stride_vbl,
    stride_vbe,
    stride_vbn,
    stride_vbr,
    stride_dal,
    stride_dae,
    stride_dar,
    stride_dan,
    stride_aom,
    stride_aon,
    stride_dim,
    stride_dir,
    top_k: tl.constexpr,
    value_rank: tl.constexpr,
    down_rank: tl.constexpr,
    max_loras: tl.constexpr,
    local_expert_offset: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_VALUE_R: tl.constexpr,
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
    physical_mask = offs_n < physical_inter
    logical_mask = offs_n < logical_inter
    offs_value_r = tl.arange(0, BLOCK_VALUE_R).to(tl.int64)
    value_r_mask = offs_value_r < value_rank

    value = tl.load(
        value_ptr + dst_row * stride_vm + offs_n * stride_vn,
        mask=valid_pair & logical_mask,
        other=0.0,
    ).to(tl.float32)
    value_a = tl.load(
        value_intermediate_ptr
        + pair_idx * stride_vim
        + offs_value_r * stride_vir,
        mask=has_lora & value_r_mask,
        other=0.0,
    )
    value_b_base = (
        value_b_ptr + safe_adapter * stride_vbl + safe_expert * stride_vbe
    )
    value_b = tl.load(
        value_b_base
        + offs_n[:, None] * stride_vbn
        + offs_value_r[None, :] * stride_vbr,
        mask=has_lora & logical_mask[:, None] & value_r_mask[None, :],
        other=0.0,
    )
    value_delta = tl.sum(value_b.to(tl.float32) * value_a[None, :], axis=1)
    # Preserve the decomposed BF16 value-B materialization boundary.
    value += value_delta.to(value_ptr.dtype.element_ty).to(tl.float32)
    activated = tl.maximum(value, 0.0)
    activated = activated * activated
    activated_dst = activated.to(act_out_ptr.dtype.element_ty)
    # Provider padding is part of the ABI: valid rows carry explicit zeros.
    tl.store(
        act_out_ptr + dst_row * stride_aom + offs_n * stride_aon,
        tl.where(logical_mask, activated_dst, 0.0),
        mask=valid_pair & physical_mask,
    )

    offs_down_r = tl.arange(0, BLOCK_DOWN_R).to(tl.int64)
    down_r_mask = offs_down_r < down_rank
    down_a_base = down_a_ptr + safe_adapter * stride_dal + safe_expert * stride_dae
    down_a = tl.load(
        down_a_base
        + offs_down_r[:, None] * stride_dar
        + offs_n[None, :] * stride_dan,
        mask=has_lora & down_r_mask[:, None] & logical_mask[None, :],
        other=0.0,
    )
    partial = tl.sum(
        down_a.to(tl.float32) * activated_dst[None, :].to(tl.float32), axis=1
    )
    partial = partial.to(down_intermediate_ptr.dtype.element_ty)
    tl.atomic_add(
        down_intermediate_ptr
        + pair_idx * stride_dim
        + offs_down_r * stride_dir,
        partial,
        mask=has_lora & down_r_mask,
        sem="relaxed",
    )


@triton.jit
def _base_only_relu2_activation_kernel(
    value_ptr,
    act_out_ptr,
    src2dst_ptr,
    topk_ids_ptr,
    token_lora_mapping_ptr,
    num_pairs,
    logical_inter,
    physical_inter,
    num_local_experts,
    stride_vm,
    stride_vn,
    stride_aom,
    stride_aon,
    top_k: tl.constexpr,
    local_expert_offset: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fill provider-padded ReLU2 values for base rows absent from LoRA route."""

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
    physical_mask = offs_n < physical_inter
    logical_mask = offs_n < logical_inter
    value = tl.load(
        value_ptr + dst_row * stride_vm + offs_n * stride_vn,
        mask=valid_base_pair & logical_mask,
        other=0.0,
    ).to(tl.float32)
    activated = tl.maximum(value, 0.0)
    activated = (activated * activated).to(act_out_ptr.dtype.element_ty)
    tl.store(
        act_out_ptr + dst_row * stride_aom + offs_n * stride_aon,
        tl.where(logical_mask, activated, 0.0),
        mask=valid_base_pair & physical_mask,
    )


@triton.jit
def _fused_value_b_relu2_down_a_aligned_kernel(
    value_ptr,
    value_intermediate_ptr,
    value_b_ptr,
    down_a_ptr,
    act_out_ptr,
    down_intermediate_ptr,
    src2dst_ptr,
    topk_ids_ptr,
    sorted_pair_ids_ptr,
    virtual_expert_ids_ptr,
    num_pairs_post_padded_ptr,
    num_pairs,
    logical_inter,
    physical_inter,
    num_virtual_experts,
    num_local_experts,
    stride_vm,
    stride_vn,
    stride_vim,
    stride_vir,
    stride_vbe,
    stride_vbn,
    stride_vbr,
    stride_dae,
    stride_dar,
    stride_dan,
    stride_aom,
    stride_aon,
    stride_dim,
    stride_dir,
    value_rank: tl.constexpr,
    down_rank: tl.constexpr,
    local_expert_offset: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_VALUE_R: tl.constexpr,
    BLOCK_DOWN_R: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    num_pairs_post_padded = tl.load(num_pairs_post_padded_ptr)
    if pid_m * BLOCK_M >= num_pairs_post_padded:
        return

    route_slots = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    pair_ids = tl.load(sorted_pair_ids_ptr + route_slots).to(tl.int64)
    pair_in_range = pair_ids < num_pairs
    # Do not rely on masked pointer arithmetic for the route-padding sentinel.
    # SM103 may still vectorize an address containing ``pair_id == num_pairs``;
    # clamp every padded lane before constructing any tensor pointer.
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
    physical_mask = offs_n < physical_inter
    logical_mask = offs_n < logical_inter
    offs_value_r = tl.arange(0, BLOCK_VALUE_R).to(tl.int64)
    value_r_mask = offs_value_r < value_rank

    value = tl.load(
        value_ptr
        + dst_rows[:, None] * stride_vm
        + offs_n[None, :] * stride_vn,
        mask=valid_pair[:, None] & logical_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    value_a = tl.load(
        value_intermediate_ptr
        + safe_pair_ids[:, None] * stride_vim
        + offs_value_r[None, :] * stride_vir,
        mask=has_lora[:, None] & value_r_mask[None, :],
        other=0.0,
    )
    value_b_base = value_b_ptr + safe_virtual_expert * stride_vbe
    value_b = tl.load(
        value_b_base
        + offs_value_r[:, None] * stride_vbr
        + offs_n[None, :] * stride_vbn,
        mask=(
            valid_virtual_expert
            & value_r_mask[:, None]
            & logical_mask[None, :]
        ),
        other=0.0,
    )
    value_delta = tl.dot(value_a, value_b, out_dtype=tl.float32)
    value += value_delta.to(value_ptr.dtype.element_ty).to(tl.float32)
    activated = tl.maximum(value, 0.0)
    activated = activated * activated
    activated_dst = activated.to(act_out_ptr.dtype.element_ty)
    tl.store(
        act_out_ptr
        + dst_rows[:, None] * stride_aom
        + offs_n[None, :] * stride_aon,
        tl.where(logical_mask[None, :], activated_dst, 0.0),
        mask=valid_pair[:, None] & physical_mask[None, :],
    )

    offs_down_r = tl.arange(0, BLOCK_DOWN_R).to(tl.int64)
    down_r_mask = offs_down_r < down_rank
    down_a_base = down_a_ptr + safe_virtual_expert * stride_dae
    down_a = tl.load(
        down_a_base
        + offs_n[:, None] * stride_dan
        + offs_down_r[None, :] * stride_dar,
        mask=(
            valid_virtual_expert
            & logical_mask[:, None]
            & down_r_mask[None, :]
        ),
        other=0.0,
    )
    partial = tl.dot(activated_dst, down_a, out_dtype=tl.float32)
    partial = partial.to(down_intermediate_ptr.dtype.element_ty)
    tl.atomic_add(
        down_intermediate_ptr
        + safe_pair_ids[:, None] * stride_dim
        + offs_down_r[None, :] * stride_dir,
        partial,
        mask=has_lora[:, None] & down_r_mask[None, :],
        sem="relaxed",
    )


def fused_value_b_relu2_down_a(
    value_output: torch.Tensor,
    value_intermediate: torch.Tensor,
    value_lora_b: torch.Tensor,
    down_lora_a: torch.Tensor,
    act_out: torch.Tensor,
    down_intermediate: torch.Tensor,
    src2dst: torch.Tensor,
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    *,
    logical_intermediate_size: int,
    local_expert_offset: int = 0,
    block_size_n: int = 32,
    num_warps: int = 4,
) -> None:
    """Launch the pair-owned non-gated ReLU2 C2 consumer."""

    num_pairs = topk_ids.numel()
    physical_inter = act_out.shape[-1]
    value_rank = value_lora_b.shape[-1]
    down_rank = down_lora_a.shape[-2]
    grid = (num_pairs, triton.cdiv(physical_inter, block_size_n))
    _fused_value_b_relu2_down_a_pair_kernel[grid](
        value_output.view(-1, value_output.shape[-1]),
        value_intermediate.view(num_pairs, value_rank),
        value_lora_b,
        down_lora_a,
        act_out.view(-1, physical_inter),
        down_intermediate.view(num_pairs, down_rank),
        src2dst,
        topk_ids,
        token_lora_mapping,
        num_pairs,
        logical_intermediate_size,
        physical_inter,
        value_lora_b.shape[1],
        value_output.stride(-2),
        value_output.stride(-1),
        value_intermediate.stride(-2),
        value_intermediate.stride(-1),
        value_lora_b.stride(0),
        value_lora_b.stride(1),
        value_lora_b.stride(2),
        value_lora_b.stride(3),
        down_lora_a.stride(0),
        down_lora_a.stride(1),
        down_lora_a.stride(2),
        down_lora_a.stride(3),
        act_out.stride(-2),
        act_out.stride(-1),
        down_intermediate.stride(-2),
        down_intermediate.stride(-1),
        top_k=topk_ids.shape[1],
        value_rank=value_rank,
        down_rank=down_rank,
        max_loras=value_lora_b.shape[0],
        local_expert_offset=local_expert_offset,
        BLOCK_N=block_size_n,
        BLOCK_VALUE_R=triton.next_power_of_2(value_rank),
        BLOCK_DOWN_R=triton.next_power_of_2(down_rank),
        num_warps=num_warps,
        num_stages=1,
    )


def fused_value_b_relu2_down_a_aligned(
    value_output: torch.Tensor,
    value_intermediate: torch.Tensor,
    value_lora_b: torch.Tensor,
    down_lora_a: torch.Tensor,
    act_out: torch.Tensor,
    down_intermediate: torch.Tensor,
    src2dst: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_pair_ids: torch.Tensor,
    virtual_expert_ids: torch.Tensor,
    num_pairs_post_padded: torch.Tensor,
    *,
    logical_intermediate_size: int,
    route_block_size_m: int,
    token_lora_mapping: torch.Tensor | None = None,
    local_expert_offset: int = 0,
    block_size_n: int = 64,
    num_warps: int = 4,
) -> None:
    """Launch the aligned virtual-expert non-gated ReLU2 C2 consumer.

    Supplying ``token_lora_mapping`` fills provider-padded activations for
    base-only rows that are absent from the LoRA route. It may be omitted only
    when no base rows exist or the supplied route contains them explicitly.
    """

    num_pairs = topk_ids.numel()
    physical_inter = act_out.shape[-1]
    value_rank = value_lora_b.shape[-1]
    down_rank = down_lora_a.shape[-2]
    value_b_virtual = value_lora_b.view(
        value_lora_b.shape[0] * value_lora_b.shape[1], physical_inter, value_rank
    )
    down_a_virtual = down_lora_a.view(
        down_lora_a.shape[0] * down_lora_a.shape[1], down_rank, physical_inter
    )
    grid = (
        triton.cdiv(sorted_pair_ids.shape[0], route_block_size_m),
        triton.cdiv(physical_inter, block_size_n),
    )
    if token_lora_mapping is not None:
        base_grid = (num_pairs, triton.cdiv(physical_inter, block_size_n))
        _base_only_relu2_activation_kernel[base_grid](
            value_output.view(-1, value_output.shape[-1]),
            act_out.view(-1, physical_inter),
            src2dst,
            topk_ids,
            token_lora_mapping,
            num_pairs,
            logical_intermediate_size,
            physical_inter,
            value_lora_b.shape[1],
            value_output.stride(-2),
            value_output.stride(-1),
            act_out.stride(-2),
            act_out.stride(-1),
            top_k=topk_ids.shape[1],
            local_expert_offset=local_expert_offset,
            BLOCK_N=block_size_n,
            num_warps=num_warps,
            num_stages=1,
        )
    _fused_value_b_relu2_down_a_aligned_kernel[grid](
        value_output.view(-1, value_output.shape[-1]),
        value_intermediate.view(num_pairs, value_rank),
        value_b_virtual,
        down_a_virtual,
        act_out.view(-1, physical_inter),
        down_intermediate.view(num_pairs, down_rank),
        src2dst,
        topk_ids,
        sorted_pair_ids,
        virtual_expert_ids,
        num_pairs_post_padded,
        num_pairs,
        logical_intermediate_size,
        physical_inter,
        value_b_virtual.shape[0],
        value_lora_b.shape[1],
        value_output.stride(-2),
        value_output.stride(-1),
        value_intermediate.stride(-2),
        value_intermediate.stride(-1),
        value_b_virtual.stride(0),
        value_b_virtual.stride(1),
        value_b_virtual.stride(2),
        down_a_virtual.stride(0),
        down_a_virtual.stride(1),
        down_a_virtual.stride(2),
        act_out.stride(-2),
        act_out.stride(-1),
        down_intermediate.stride(-2),
        down_intermediate.stride(-1),
        value_rank=value_rank,
        down_rank=down_rank,
        local_expert_offset=local_expert_offset,
        BLOCK_M=route_block_size_m,
        BLOCK_N=block_size_n,
        BLOCK_VALUE_R=triton.next_power_of_2(value_rank),
        BLOCK_DOWN_R=triton.next_power_of_2(down_rank),
        num_warps=num_warps,
        num_stages=1,
    )


__all__ = [
    "fused_value_b_relu2_down_a",
    "fused_value_b_relu2_down_a_aligned",
]
