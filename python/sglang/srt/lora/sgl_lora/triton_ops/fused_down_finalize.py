"""Benchmark-only BF16 down-LoRA-B plus base-MoE finalizers.

The production C0/C1 path first reorders and reduces the masked base-W2
output, then launches a routed LoRA-B expansion into that token-owned output.
The C2 candidate below consumes both branches at once::

    masked base W2 [E_local, m_max, H] --src2dst--+
                                                     +--> [T, H]
    canonical down-A [T, K, R] --down-B[L,E,H,R]----+

Each program owns one ``(token, H tile)`` and loops over top-k.  Both the base
and LoRA terms are accumulated in FP32 and multiplied by
``topk_weight * routed_scaling_factor`` exactly once before the requested
output dtype conversion.  Invalid/non-local experts contribute nothing;
adapter ``-1`` retains the base branch but suppresses LoRA.

Shared-outer down-B has a separate specialization.  It first performs the
weighted top-k reduction in rank space and loads the shared B matrix once per
token/H tile, rather than reloading it for every routed expert.  These kernels
remain an experimental benchmark implementation until matched K0/O0/M0 and
profiler evidence justify a serving dispatch change.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_down_b_finalize_per_expert_kernel(
    down_out_ptr,
    down_intermediate_ptr,
    down_b_ptr,
    output_ptr,
    src2dst_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    token_lora_mapping_ptr,
    hidden_size,
    num_local_experts,
    routed_scaling_factor,
    stride_dom,
    stride_doh,
    stride_dim,
    stride_dir,
    stride_dbl,
    stride_dbe,
    stride_dbh,
    stride_dbr,
    stride_om,
    stride_oh,
    top_k: tl.constexpr,
    rank: tl.constexpr,
    max_loras: tl.constexpr,
    local_expert_offset: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1).to(tl.int64)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H).to(tl.int64)
    h_mask = offs_h < hidden_size
    offs_r = tl.arange(0, BLOCK_R).to(tl.int64)
    r_mask = offs_r < rank

    adapter = tl.load(token_lora_mapping_ptr + token_idx).to(tl.int64)
    valid_adapter = (adapter >= 0) & (adapter < max_loras)
    safe_adapter = tl.maximum(0, tl.minimum(adapter, max_loras - 1))
    output_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    for k_idx in tl.static_range(0, top_k):
        pair_idx = token_idx * top_k + k_idx
        global_expert = tl.load(topk_ids_ptr + pair_idx).to(tl.int64)
        local_expert = global_expert - local_expert_offset
        valid_pair = (local_expert >= 0) & (local_expert < num_local_experts)
        safe_expert = tl.maximum(0, tl.minimum(local_expert, num_local_experts - 1))
        dst_row = tl.load(src2dst_ptr + pair_idx, mask=valid_pair, other=0).to(tl.int64)
        routed_weight = tl.load(
            topk_weights_ptr + pair_idx, mask=valid_pair, other=0.0
        ).to(tl.float32)
        routed_weight *= routed_scaling_factor

        base = tl.load(
            down_out_ptr + dst_row * stride_dom + offs_h * stride_doh,
            mask=valid_pair & h_mask,
            other=0.0,
        ).to(tl.float32)
        output_acc += base * routed_weight

        has_lora = valid_pair & valid_adapter
        down_rank = tl.load(
            down_intermediate_ptr + pair_idx * stride_dim + offs_r * stride_dir,
            mask=has_lora & r_mask,
            other=0.0,
        )
        b_base = down_b_ptr + safe_adapter * stride_dbl + safe_expert * stride_dbe
        down_b = tl.load(
            b_base + offs_h[:, None] * stride_dbh + offs_r[None, :] * stride_dbr,
            mask=has_lora & h_mask[:, None] & r_mask[None, :],
            other=0.0,
        )
        delta = tl.sum(
            down_b.to(tl.float32) * down_rank[None, :].to(tl.float32), axis=1
        )
        output_acc += delta * routed_weight

    tl.store(
        output_ptr + token_idx * stride_om + offs_h * stride_oh,
        output_acc,
        mask=h_mask,
    )


@triton.jit
def _fused_down_b_finalize_shared_outer_kernel(
    down_out_ptr,
    down_intermediate_ptr,
    down_b_ptr,
    output_ptr,
    src2dst_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    token_lora_mapping_ptr,
    hidden_size,
    num_local_experts,
    routed_scaling_factor,
    stride_dom,
    stride_doh,
    stride_dim,
    stride_dir,
    stride_dbl,
    stride_dbh,
    stride_dbr,
    stride_om,
    stride_oh,
    top_k: tl.constexpr,
    rank: tl.constexpr,
    max_loras: tl.constexpr,
    local_expert_offset: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1).to(tl.int64)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H).to(tl.int64)
    h_mask = offs_h < hidden_size
    offs_r = tl.arange(0, BLOCK_R).to(tl.int64)
    r_mask = offs_r < rank

    adapter = tl.load(token_lora_mapping_ptr + token_idx).to(tl.int64)
    valid_adapter = (adapter >= 0) & (adapter < max_loras)
    safe_adapter = tl.maximum(0, tl.minimum(adapter, max_loras - 1))
    output_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    weighted_rank = tl.zeros((BLOCK_R,), dtype=tl.float32)

    for k_idx in tl.static_range(0, top_k):
        pair_idx = token_idx * top_k + k_idx
        global_expert = tl.load(topk_ids_ptr + pair_idx).to(tl.int64)
        local_expert = global_expert - local_expert_offset
        valid_pair = (local_expert >= 0) & (local_expert < num_local_experts)
        dst_row = tl.load(src2dst_ptr + pair_idx, mask=valid_pair, other=0).to(tl.int64)
        routed_weight = tl.load(
            topk_weights_ptr + pair_idx, mask=valid_pair, other=0.0
        ).to(tl.float32)
        routed_weight *= routed_scaling_factor

        base = tl.load(
            down_out_ptr + dst_row * stride_dom + offs_h * stride_doh,
            mask=valid_pair & h_mask,
            other=0.0,
        ).to(tl.float32)
        output_acc += base * routed_weight

        has_lora = valid_pair & valid_adapter
        down_rank = tl.load(
            down_intermediate_ptr + pair_idx * stride_dim + offs_r * stride_dir,
            mask=has_lora & r_mask,
            other=0.0,
        ).to(tl.float32)
        weighted_rank += down_rank * routed_weight

    # Expert dimension is one for a shared-outer B; adapter remains token-local.
    b_base = down_b_ptr + safe_adapter * stride_dbl
    down_b = tl.load(
        b_base + offs_h[:, None] * stride_dbh + offs_r[None, :] * stride_dbr,
        mask=valid_adapter & h_mask[:, None] & r_mask[None, :],
        other=0.0,
    )
    output_acc += tl.sum(down_b.to(tl.float32) * weighted_rank[None, :], axis=1)

    tl.store(
        output_ptr + token_idx * stride_om + offs_h * stride_oh,
        output_acc,
        mask=h_mask,
    )


@triton.jit
def _indexed_down_b_add_kernel(
    down_intermediate_ptr,
    down_b_ptr,
    output_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    token_lora_mapping_ptr,
    num_pairs,
    hidden_size,
    num_local_experts,
    routed_scaling_factor,
    stride_dim,
    stride_dir,
    stride_dbl,
    stride_dbe,
    stride_dbh,
    stride_dbr,
    stride_om,
    stride_oh,
    top_k: tl.constexpr,
    rank: tl.constexpr,
    max_loras: tl.constexpr,
    local_expert_offset: tl.constexpr,
    shared_outer: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """Route-free pair-owned B arm used to isolate fusion from route removal."""
    pair_idx = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1).to(tl.int64)
    token_idx = pair_idx // top_k
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H).to(tl.int64)
    h_mask = offs_h < hidden_size
    offs_r = tl.arange(0, BLOCK_R).to(tl.int64)
    r_mask = offs_r < rank

    global_expert = tl.load(topk_ids_ptr + pair_idx).to(tl.int64)
    local_expert = global_expert - local_expert_offset
    valid_pair = (
        (pair_idx < num_pairs)
        & (local_expert >= 0)
        & (local_expert < num_local_experts)
    )
    adapter = tl.load(token_lora_mapping_ptr + token_idx).to(tl.int64)
    has_lora = valid_pair & (adapter >= 0) & (adapter < max_loras)
    safe_adapter = tl.maximum(0, tl.minimum(adapter, max_loras - 1))
    if shared_outer:
        safe_expert = 0
    else:
        safe_expert = tl.maximum(0, tl.minimum(local_expert, num_local_experts - 1))

    down_rank = tl.load(
        down_intermediate_ptr + pair_idx * stride_dim + offs_r * stride_dir,
        mask=has_lora & r_mask,
        other=0.0,
    )
    b_base = down_b_ptr + safe_adapter * stride_dbl + safe_expert * stride_dbe
    down_b = tl.load(
        b_base + offs_h[:, None] * stride_dbh + offs_r[None, :] * stride_dbr,
        mask=has_lora & h_mask[:, None] & r_mask[None, :],
        other=0.0,
    )
    delta = tl.sum(down_b.to(tl.float32) * down_rank[None, :].to(tl.float32), axis=1)
    routed_weight = tl.load(topk_weights_ptr + pair_idx, mask=valid_pair, other=0.0).to(
        tl.float32
    )
    delta *= routed_weight * routed_scaling_factor
    tl.atomic_add(
        output_ptr + token_idx * stride_om + offs_h * stride_oh,
        delta,
        mask=has_lora & h_mask,
        sem="relaxed",
    )


def fused_down_b_finalize(
    down_output: torch.Tensor,
    down_intermediate: torch.Tensor,
    down_lora_b: torch.Tensor,
    output: torch.Tensor,
    src2dst: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    *,
    routed_scaling_factor: float = 1.0,
    local_expert_offset: int = 0,
    shared_outer: bool = False,
    block_size_h: int = 32,
    num_warps: int = 4,
) -> None:
    """Launch the token-owned fused down-B/base-finalize candidate.

    Args follow the C2 runner's two input domains: ``down_output`` is the
    provider-private masked base layout and ``down_intermediate`` is canonical
    ``[T,K,R]``.  ``output`` determines the requested destination dtype.
    """
    if down_intermediate.ndim != 3:
        raise ValueError("down_intermediate must have shape [T, K, R]")
    if topk_ids.shape != topk_weights.shape:
        raise ValueError("topk_ids and topk_weights must have the same shape")
    if tuple(down_intermediate.shape[:2]) != tuple(topk_ids.shape):
        raise ValueError("down_intermediate and top-k token/pair domains differ")
    if output.shape[0] != topk_ids.shape[0]:
        raise ValueError("output and top-k token domains differ")
    if token_lora_mapping.numel() != topk_ids.shape[0]:
        raise ValueError("token_lora_mapping and top-k token domains differ")
    if down_lora_b.shape[-2] != output.shape[-1]:
        raise ValueError("down_lora_b and output hidden dimensions differ")
    if down_lora_b.shape[-1] != down_intermediate.shape[-1]:
        raise ValueError("down_lora_b and down_intermediate ranks differ")
    if shared_outer and down_lora_b.shape[1] != 1:
        raise ValueError("shared-outer down_lora_b must have one expert row")
    if not shared_outer and down_lora_b.shape[1] != down_output.shape[0]:
        raise ValueError("per-expert down_lora_b must match local base experts")

    num_tokens, top_k = topk_ids.shape
    hidden_size = output.shape[-1]
    rank = down_intermediate.shape[-1]
    block_r = triton.next_power_of_2(rank)
    grid = (num_tokens, triton.cdiv(hidden_size, block_size_h))
    common = (
        down_output.view(-1, hidden_size),
        down_intermediate,
        down_lora_b,
        output,
        src2dst,
        topk_ids,
        topk_weights,
        token_lora_mapping,
        hidden_size,
        down_output.shape[0],
        float(routed_scaling_factor),
    )
    if shared_outer:
        _fused_down_b_finalize_shared_outer_kernel[grid](
            *common,
            down_output.stride(-2),
            down_output.stride(-1),
            down_intermediate.stride(-2),
            down_intermediate.stride(-1),
            down_lora_b.stride(0),
            down_lora_b.stride(2),
            down_lora_b.stride(3),
            output.stride(0),
            output.stride(1),
            top_k=top_k,
            rank=rank,
            max_loras=down_lora_b.shape[0],
            local_expert_offset=local_expert_offset,
            BLOCK_H=block_size_h,
            BLOCK_R=block_r,
            num_warps=num_warps,
            num_stages=1,
        )
    else:
        _fused_down_b_finalize_per_expert_kernel[grid](
            *common,
            down_output.stride(-2),
            down_output.stride(-1),
            down_intermediate.stride(-2),
            down_intermediate.stride(-1),
            down_lora_b.stride(0),
            down_lora_b.stride(1),
            down_lora_b.stride(2),
            down_lora_b.stride(3),
            output.stride(0),
            output.stride(1),
            top_k=top_k,
            rank=rank,
            max_loras=down_lora_b.shape[0],
            local_expert_offset=local_expert_offset,
            BLOCK_H=block_size_h,
            BLOCK_R=block_r,
            num_warps=num_warps,
            num_stages=1,
        )


def indexed_down_b_add(
    down_intermediate: torch.Tensor,
    down_lora_b: torch.Tensor,
    output: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    *,
    routed_scaling_factor: float = 1.0,
    local_expert_offset: int = 0,
    num_local_experts: int | None = None,
    shared_outer: bool = False,
    block_size_h: int = 32,
    num_warps: int = 4,
) -> None:
    """Add route-free pair-indexed down-B into an initialized output.

    This benchmark control keeps base finalization separate, but removes the
    virtual-expert sort/align plan. It lets K0/O0/M0 attribute how much of C2F
    comes from route elimination versus token-owned fusion.
    """
    if down_intermediate.ndim != 3:
        raise ValueError("down_intermediate must have shape [T, K, R]")
    if tuple(down_intermediate.shape[:2]) != tuple(topk_ids.shape):
        raise ValueError("down_intermediate and top-k token/pair domains differ")
    if topk_ids.shape != topk_weights.shape:
        raise ValueError("topk_ids and topk_weights must have the same shape")
    if token_lora_mapping.numel() != topk_ids.shape[0]:
        raise ValueError("token_lora_mapping and top-k token domains differ")
    if down_lora_b.shape[-2] != output.shape[-1]:
        raise ValueError("down_lora_b and output hidden dimensions differ")
    if down_lora_b.shape[-1] != down_intermediate.shape[-1]:
        raise ValueError("down_lora_b and down_intermediate ranks differ")
    if shared_outer and down_lora_b.shape[1] != 1:
        raise ValueError("shared-outer down_lora_b must have one expert row")

    num_pairs = topk_ids.numel()
    hidden_size = output.shape[-1]
    rank = down_intermediate.shape[-1]
    if num_local_experts is None:
        if shared_outer:
            raise ValueError("shared-outer indexed B requires num_local_experts")
        num_local_experts = down_lora_b.shape[1]
    grid = (num_pairs, triton.cdiv(hidden_size, block_size_h))
    _indexed_down_b_add_kernel[grid](
        down_intermediate,
        down_lora_b,
        output,
        topk_ids,
        topk_weights,
        token_lora_mapping,
        num_pairs,
        hidden_size,
        num_local_experts,
        float(routed_scaling_factor),
        down_intermediate.stride(-2),
        down_intermediate.stride(-1),
        down_lora_b.stride(0),
        down_lora_b.stride(1),
        down_lora_b.stride(2),
        down_lora_b.stride(3),
        output.stride(0),
        output.stride(1),
        top_k=topk_ids.shape[1],
        rank=rank,
        max_loras=down_lora_b.shape[0],
        local_expert_offset=local_expert_offset,
        shared_outer=shared_outer,
        BLOCK_H=block_size_h,
        BLOCK_R=triton.next_power_of_2(rank),
        num_warps=num_warps,
        num_stages=1,
    )


__all__ = ["fused_down_b_finalize", "indexed_down_b_add"]
