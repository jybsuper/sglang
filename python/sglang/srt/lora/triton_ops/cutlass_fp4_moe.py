from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _cutlass_fp4_lora_silu_and_mul_kernel(
    input_ptr,
    output_ptr,
    delta_ptr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SWAP_HALVES: tl.constexpr,
    HAS_DELTA: tl.constexpr,
):
    row = tl.program_id(0)
    n_block = tl.program_id(1)
    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    row_base = row * (2 * N)
    first = tl.load(input_ptr + row_base + offs_n, mask=mask_n, other=0.0).to(
        tl.float32
    )
    second = tl.load(input_ptr + row_base + N + offs_n, mask=mask_n, other=0.0).to(
        tl.float32
    )
    # Two-stream: fold the side-stream gate_up LoRA delta in here (in fp32) rather than a
    # standalone gateup.add_(delta) pass. The silu already loads gateup, so the merge costs
    # one extra load and no extra HBM write of gateup -- the same in-kernel merge the
    # fp8/trtllm fused MoE does right before activation (which is why it has no slowdown).
    if HAS_DELTA:
        first += tl.load(delta_ptr + row_base + offs_n, mask=mask_n, other=0.0).to(
            tl.float32
        )
        second += tl.load(delta_ptr + row_base + N + offs_n, mask=mask_n, other=0.0).to(
            tl.float32
        )

    if SWAP_HALVES:
        gate = second
        up = first
    else:
        gate = first
        up = second

    out = gate * tl.sigmoid(gate) * up
    tl.store(output_ptr + row * N + offs_n, out, mask=mask_n)


def cutlass_fp4_lora_silu_and_mul(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    swap_halves: bool,
    gate_up_delta: Optional[torch.Tensor] = None,
) -> None:
    N = output_tensor.shape[1]
    block_n = 1024
    grid = (output_tensor.shape[0], triton.cdiv(N, block_n))
    _cutlass_fp4_lora_silu_and_mul_kernel[grid](
        input_tensor,
        output_tensor,
        gate_up_delta,
        N,
        BLOCK_N=block_n,
        SWAP_HALVES=swap_halves,
        HAS_DELTA=gate_up_delta is not None,
        num_warps=8,
    )


@triton.jit
def _cutlass_fp4_lora_shuffle_mul_sum_kernel(
    input_ptr,
    output_ptr,
    c_map_ptr,
    factors_ptr,
    delta_ptr,
    K: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_FACTORS: tl.constexpr,
    HAS_DELTA: tl.constexpr,
):
    m_idx = tl.program_id(0)
    k_block = tl.program_id(1)
    offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K

    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for j in tl.static_range(0, TOPK):
        pair_idx = m_idx * TOPK + j
        src_row = tl.load(c_map_ptr + pair_idx).to(tl.int64)
        vals = tl.load(input_ptr + src_row * K + offs_k, mask=mask_k, other=0.0).to(
            tl.float32
        )
        # Two-stream: fold the side-stream down LoRA delta (same expert-sorted layout as
        # out_flat) in here instead of a standalone out_flat.add_(down_delta) pass.
        if HAS_DELTA:
            vals += tl.load(
                delta_ptr + src_row * K + offs_k, mask=mask_k, other=0.0
            ).to(tl.float32)
        if HAS_FACTORS:
            factor = tl.load(factors_ptr + pair_idx).to(tl.float32)
            vals *= factor
        acc += vals

    tl.store(output_ptr + m_idx * K + offs_k, acc, mask=mask_k)


def cutlass_fp4_lora_shuffle_mul_sum(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    c_map: torch.Tensor,
    factors: Optional[torch.Tensor],
    topk: int,
    down_delta: Optional[torch.Tensor] = None,
) -> None:
    K = output_tensor.shape[1]
    block_k = 1024
    grid = (output_tensor.shape[0], triton.cdiv(K, block_k))
    _cutlass_fp4_lora_shuffle_mul_sum_kernel[grid](
        input_tensor,
        output_tensor,
        c_map,
        factors,
        down_delta,
        K,
        topk,
        BLOCK_K=block_k,
        HAS_FACTORS=factors is not None,
        HAS_DELTA=down_delta is not None,
        num_warps=8,
    )
