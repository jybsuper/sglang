"""Rank-specialized LoRA-B expand kernels owned by ``sgl_lora``.

Gate and up are independent output slices.  The fast path keeps the existing
flat grid when a tensor-core-sized tile can end exactly at their midpoint.  A
two-slice grid handles other widths: gate and up receive independent ``pid_n``
ranges, so each final tile can be masked without reading the wrong LoRA-A half.

The non-gated down projection uses the flat kernel with one input/output slice.
"""

from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def _moe_lora_expand_add_flat_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    R: tl.constexpr,
    num_valid_tokens,
    stride_am,
    stride_ar,
    stride_be,
    stride_bn,
    stride_br,
    stride_cm,
    stride_cn,
    router_topk: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    FUSE_SUM_ALL_REDUCE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    GATED_A_HALF: tl.constexpr,
    WAIT_FOR_SHRINK_PDL: tl.constexpr = False,
):
    """Flat-N reference and non-gated expand kernel.

    ``GATED_A_HALF`` enables the midpoint-safe gate/up schedule.  Its launcher
    guarantees that no tile crosses the midpoint.
    """
    if WAIT_FOR_SHRINK_PDL:
        # Paired with the shrink kernel's gdc_launch_dependents().  Waiting at
        # the consumer boundary permits early launch without reading a
        # partially accumulated LoRA-A intermediate.
        tl.extra.cuda.gdc_wait()

    pid = tl.program_id(0)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)

    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_expert == -1:
        if not FUSE_SUM_ALL_REDUCE:
            c_ptrs = (
                c_ptr + offs_token[:, None] * stride_cm + offs_n[None, :] * stride_cn
            )
            c_mask = token_mask[:, None] & (offs_n[None, :] < N)
            zeros = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=c_ptr.dtype.element_ty)
            tl.store(c_ptrs, zeros, mask=c_mask)
        return

    offs_r = tl.arange(0, BLOCK_SIZE_R).to(tl.int64)
    rank_mask = offs_r < R
    a_col = offs_r
    if GATED_A_HALF > 0:
        a_col = offs_r + tl.where(pid_n * BLOCK_SIZE_N >= GATED_A_HALF, R, 0)

    a = tl.load(
        a_ptr + offs_token[:, None] * stride_am + a_col[None, :] * stride_ar,
        mask=token_mask[:, None] & rank_mask[None, :],
        other=0.0,
    )
    b = tl.load(
        b_ptr
        + off_expert * stride_be
        + offs_n[None, :] * stride_bn
        + offs_r[:, None] * stride_br,
        mask=(offs_n[None, :] < N) & rank_mask[:, None],
        other=0.0,
    )

    accumulator = tl.dot(a, b, out_dtype=tl.float32)
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator *= moe_weight[:, None]

    if FUSE_SUM_ALL_REDUCE:
        offs_token_out = offs_token // router_topk
    else:
        offs_token_out = offs_token
    c_ptrs = c_ptr + offs_token_out[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = token_mask[:, None] & (offs_n[None, :] < N)
    if FUSE_SUM_ALL_REDUCE:
        tl.atomic_add(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=c_mask)
    else:
        tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=c_mask)


@triton.jit
def _moe_lora_expand_add_two_slice_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    SLICE_N,
    R: tl.constexpr,
    num_valid_tokens,
    stride_am,
    stride_ar,
    stride_be,
    stride_bn,
    stride_br,
    stride_cm,
    stride_cn,
    router_topk: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    FUSE_SUM_ALL_REDUCE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    WAIT_FOR_SHRINK_PDL: tl.constexpr = False,
):
    """Gate/up expand with an independent N grid for each output slice."""
    if WAIT_FOR_SHRINK_PDL:
        tl.extra.cuda.gdc_wait()

    pid = tl.program_id(0)
    slice_id = tl.program_id(1)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(SLICE_N, BLOCK_SIZE_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    local_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_n = slice_id * SLICE_N + local_n
    output_mask = local_n < SLICE_N

    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_expert == -1:
        if not FUSE_SUM_ALL_REDUCE:
            c_ptrs = (
                c_ptr + offs_token[:, None] * stride_cm + offs_n[None, :] * stride_cn
            )
            c_mask = token_mask[:, None] & output_mask[None, :]
            zeros = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=c_ptr.dtype.element_ty)
            tl.store(c_ptrs, zeros, mask=c_mask)
        return

    offs_r = tl.arange(0, BLOCK_SIZE_R).to(tl.int64)
    rank_mask = offs_r < R
    a_col = slice_id * R + offs_r
    a = tl.load(
        a_ptr + offs_token[:, None] * stride_am + a_col[None, :] * stride_ar,
        mask=token_mask[:, None] & rank_mask[None, :],
        other=0.0,
    )
    b = tl.load(
        b_ptr
        + off_expert * stride_be
        + offs_n[None, :] * stride_bn
        + offs_r[:, None] * stride_br,
        mask=output_mask[None, :] & rank_mask[:, None],
        other=0.0,
    )

    accumulator = tl.dot(a, b, out_dtype=tl.float32)
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator *= moe_weight[:, None]

    if FUSE_SUM_ALL_REDUCE:
        offs_token_out = offs_token // router_topk
    else:
        offs_token_out = offs_token
    c_ptrs = c_ptr + offs_token_out[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = token_mask[:, None] & output_mask[None, :]
    if FUSE_SUM_ALL_REDUCE:
        tl.atomic_add(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=c_mask)
    else:
        tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=c_mask)


def _common_launch_args(
    intermediate: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
) -> tuple:
    return (
        intermediate,
        weight,
        output,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        weight.shape[1],
        weight.shape[2],
        topk_ids.numel(),
        intermediate.stride(0),
        intermediate.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        output.stride(-2),
        output.stride(-1),
    )


def _invoke_flat(
    intermediate: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    config: dict[str, Any],
    mul_routed_weight: bool,
    fuse_sum_all_reduce: bool,
    *,
    gated_midpoint: bool,
    force_block_size_n: int | None = None,
    wait_for_shrink_pdl: bool = False,
) -> None:
    n = weight.shape[1]
    rank = weight.shape[2]
    block_m = config["BLOCK_SIZE_M"]
    block_n = (
        force_block_size_n
        if force_block_size_n is not None
        else (128 if n % 128 == 0 else config["BLOCK_SIZE_N"])
    )
    if gated_midpoint:
        half = n // 2
        while block_n > 16 and half % block_n != 0:
            block_n //= 2
        if half % block_n != 0:
            raise ValueError(
                f"flat midpoint schedule cannot tile output half {half} with BLOCK_SIZE_N={block_n}"
            )

    grid = (triton.cdiv(sorted_token_ids.shape[0], block_m) * triton.cdiv(n, block_n),)
    _moe_lora_expand_add_flat_kernel[grid](
        *_common_launch_args(
            intermediate,
            weight,
            output,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
        ),
        router_topk=topk_ids.shape[1],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        FUSE_SUM_ALL_REDUCE=fuse_sum_all_reduce,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        # Logical rank 8 executes in a masked physical-16 dot tile.  Loads
        # outside the logical rank are already masked, so no padded storage or
        # zero materialization is needed.
        BLOCK_SIZE_R=max(16, triton.next_power_of_2(rank)),
        GROUP_SIZE_M=config.get("GROUP_SIZE_M", 1),
        GATED_A_HALF=n // 2 if gated_midpoint else 0,
        WAIT_FOR_SHRINK_PDL=wait_for_shrink_pdl,
        num_warps=config.get("num_warps", 4),
        num_stages=1,
        **({"launch_pdl": True} if wait_for_shrink_pdl else {}),
    )


def _invoke_two_slice(
    intermediate: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    config: dict[str, Any],
    mul_routed_weight: bool,
    fuse_sum_all_reduce: bool,
    *,
    force_block_size_n: int | None = None,
    wait_for_shrink_pdl: bool = False,
) -> None:
    n = weight.shape[1]
    rank = weight.shape[2]
    slice_n = n // 2
    block_m = config["BLOCK_SIZE_M"]
    block_n = (
        force_block_size_n
        if force_block_size_n is not None
        else min(64, max(16, triton.next_power_of_2(slice_n)))
    )
    grid = (
        triton.cdiv(sorted_token_ids.shape[0], block_m) * triton.cdiv(slice_n, block_n),
        2,
    )
    common = _common_launch_args(
        intermediate,
        weight,
        output,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    )
    _moe_lora_expand_add_two_slice_kernel[grid](
        *common[:8],
        slice_n,
        *common[8:],
        router_topk=topk_ids.shape[1],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        FUSE_SUM_ALL_REDUCE=fuse_sum_all_reduce,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_R=max(16, triton.next_power_of_2(rank)),
        GROUP_SIZE_M=config.get("GROUP_SIZE_M", 1),
        WAIT_FOR_SHRINK_PDL=wait_for_shrink_pdl,
        num_warps=config.get("num_warps", 4),
        num_stages=1,
        **({"launch_pdl": True} if wait_for_shrink_pdl else {}),
    )


def invoke_moe_lora_expand_add(
    intermediate: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    config: dict[str, Any],
    mul_routed_weight: bool,
    fuse_sum_all_reduce: bool,
    *,
    num_output_slices: int,
    wait_for_shrink_pdl: bool = False,
) -> None:
    """Launch the direct expand used by ``sgl_lora`` for ranks up to 64.

    Across the tested GB300 rank range, the flat midpoint schedule is the safer
    static policy when each half is divisible by a 16-column tensor-core tile;
    the two-slice grid helps only some rank-16 shapes and regresses larger
    ranks.  It remains the layout-general fallback when the flat schedule
    cannot satisfy the boundary without a sub-tensor-core tile.
    """
    if num_output_slices == 2:
        if weight.shape[1] % 2 == 0 and (weight.shape[1] // 2) % 16 == 0:
            _invoke_flat(
                intermediate,
                weight,
                output,
                topk_weights,
                topk_ids,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                config,
                mul_routed_weight,
                fuse_sum_all_reduce,
                gated_midpoint=True,
                wait_for_shrink_pdl=wait_for_shrink_pdl,
            )
        else:
            _invoke_two_slice(
                intermediate,
                weight,
                output,
                topk_weights,
                topk_ids,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                config,
                mul_routed_weight,
                fuse_sum_all_reduce,
                wait_for_shrink_pdl=wait_for_shrink_pdl,
            )
        return
    _invoke_flat(
        intermediate,
        weight,
        output,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        config,
        mul_routed_weight,
        fuse_sum_all_reduce,
        gated_midpoint=False,
        wait_for_shrink_pdl=wait_for_shrink_pdl,
    )


def invoke_moe_lora_expand_add_flat_for_benchmark(
    intermediate: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    config: dict[str, Any],
    mul_routed_weight: bool = False,
    fuse_sum_all_reduce: bool = False,
    *,
    force_block_size_n: int | None = None,
) -> None:
    """Launch the old midpoint/divisor schedule for controlled A/B tests."""
    _invoke_flat(
        intermediate,
        weight,
        output,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        config,
        mul_routed_weight,
        fuse_sum_all_reduce,
        gated_midpoint=True,
        force_block_size_n=force_block_size_n,
    )


def invoke_moe_lora_expand_add_sliced_for_benchmark(
    intermediate: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    config: dict[str, Any],
    mul_routed_weight: bool = False,
    fuse_sum_all_reduce: bool = False,
    *,
    force_block_size_n: int | None = None,
) -> None:
    """Launch the two-slice schedule for controlled A/B tests."""
    _invoke_two_slice(
        intermediate,
        weight,
        output,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        config,
        mul_routed_weight,
        fuse_sum_all_reduce,
        force_block_size_n=force_block_size_n,
    )


__all__ = [
    "invoke_moe_lora_expand_add",
    "invoke_moe_lora_expand_add_flat_for_benchmark",
    "invoke_moe_lora_expand_add_sliced_for_benchmark",
]
