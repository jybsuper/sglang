"""Sliced routed LoRA-B expand/add kernels owned by ``sgl_lora``.

One semantic operation covers ordinary projections, complete stacked
projections, and arbitrary active subsets of a stacked projection::

    C[:, output_i] <epilogue>= A[:, a_i] @ B[:, b_i].T

The logical slice layout is deliberately independent from routed-row metadata.
In particular, ``b_offsets`` and ``output_offsets`` need not match: a compact B
buffer may contain only Q and V while those slices still write to their original
destinations in a Q/K/V/Z base output.

The source compiles into three slice schedules:

* ``ALIGNED_FLAT`` for regular contiguous slices whose boundaries align to the
  chosen N tile (or the single-slice case);
* ``UNIFORM_SLICED`` for regular equal-width slices with independent tail masks;
* ``DESCRIPTOR_RAGGED`` for unequal widths, compact B, or output holes.

The current MoE runner uses only the first two allocation-free schedules. A
descriptor plan is built and cached by setup code before the ragged schedule is
launched, keeping device addresses stable across CUDA-graph replay.

Rank scheduling is orthogonal. ``WHOLE_RANK`` issues one padded rank dot, while
``LOOPED_RANK`` lets one CTA accumulate bounded rank chunks in FP32 and store
once. ``AUTO`` chooses a bounded tile until per-device tuning tables replace the
fallback policy; this is not split-K and introduces no reduction atomics.

The caller-owned destination tensor defines the final LoRA delta/result dtype.
The dot accumulates in FP32, then the epilogue converts once to the destination
dtype for store, add-to-base, or routed pair-to-token reduction. This policy is
independent from the LoRA-A intermediate and split-K accumulation dtype.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import torch
import triton
import triton.language as tl


class SlicedLoraBSchedule(IntEnum):
    ALIGNED_FLAT = 0
    UNIFORM_SLICED = 1
    DESCRIPTOR_RAGGED = 2


class SlicedLoraBRankSchedule(IntEnum):
    WHOLE_RANK = 0
    LOOPED_RANK = 1
    AUTO = 2


@dataclass(frozen=True)
class SlicedLoraBLayout:
    """Host description of the active LoRA-B output slices.

    Version one uses one common rank, matching SGLang's adapter-level rank and
    current packed weight arena. Multiple slices may reference the same
    ``a_offset`` so a later A plan can compute a shared factor only once.

    Setup code owns bounds validation against the packed A, B, and output
    arenas. Store destinations must not overlap within one layout.
    """

    a_offsets: tuple[int, ...]
    b_offsets: tuple[int, ...]
    output_offsets: tuple[int, ...]
    widths: tuple[int, ...]
    rank: int

    def __post_init__(self) -> None:
        num_slices = len(self.widths)
        if num_slices == 0 or any(
            len(offsets) != num_slices
            for offsets in (self.a_offsets, self.b_offsets, self.output_offsets)
        ):
            raise ValueError("sliced LoRA-B layout fields must have one active entry")
        if self.rank <= 0 or any(width <= 0 for width in self.widths):
            raise ValueError("sliced LoRA-B rank and widths must be positive")

    @classmethod
    def contiguous(
        cls,
        *,
        widths: tuple[int, ...],
        rank: int,
    ) -> SlicedLoraBLayout:
        output_offsets = []
        offset = 0
        for width in widths:
            output_offsets.append(offset)
            offset += width
        return cls(
            a_offsets=tuple(slice_id * rank for slice_id in range(len(widths))),
            b_offsets=tuple(output_offsets),
            output_offsets=tuple(output_offsets),
            widths=widths,
            rank=rank,
        )

    @property
    def num_slices(self) -> int:
        return len(self.widths)

    @property
    def active_width(self) -> int:
        return sum(self.widths)

    @property
    def is_uniform_contiguous(self) -> bool:
        if len(set(self.widths)) != 1:
            return False
        expected = SlicedLoraBLayout.contiguous(widths=self.widths, rank=self.rank)
        return (
            self.a_offsets == expected.a_offsets
            and self.b_offsets == expected.b_offsets
            and self.output_offsets == expected.output_offsets
        )


@dataclass(frozen=True)
class SlicedLoraBDescriptors:
    """Graph-stable device metadata for ``DESCRIPTOR_RAGGED``.

    Construct this object during adapter/layout setup, not inside a captured
    forward. ``tile_prefix`` is specific to ``block_size_n``.
    """

    layout: SlicedLoraBLayout
    block_size_n: int
    total_tiles: int
    a_offsets: torch.Tensor
    b_offsets: torch.Tensor
    output_offsets: torch.Tensor
    widths: torch.Tensor
    tile_prefix: torch.Tensor


def build_sliced_lora_b_descriptors(
    layout: SlicedLoraBLayout,
    *,
    block_size_n: int,
    device: torch.device | str,
) -> SlicedLoraBDescriptors:
    """Build descriptor tensors outside the graph-captured execution path."""
    tile_prefix = [0]
    for width in layout.widths:
        tile_prefix.append(tile_prefix[-1] + triton.cdiv(width, block_size_n))
    tensor_kwargs = {"dtype": torch.int32, "device": device}
    return SlicedLoraBDescriptors(
        layout=layout,
        block_size_n=block_size_n,
        total_tiles=tile_prefix[-1],
        a_offsets=torch.tensor(layout.a_offsets, **tensor_kwargs),
        b_offsets=torch.tensor(layout.b_offsets, **tensor_kwargs),
        output_offsets=torch.tensor(layout.output_offsets, **tensor_kwargs),
        widths=torch.tensor(layout.widths, **tensor_kwargs),
        tile_prefix=torch.tensor(tile_prefix, **tensor_kwargs),
    )


def select_sliced_lora_b_rank_schedule(
    rank: int,
    config: dict[str, Any],
    requested: SlicedLoraBRankSchedule = SlicedLoraBRankSchedule.AUTO,
    *,
    force_block_size_r: int | None = None,
) -> tuple[SlicedLoraBRankSchedule, int]:
    """Resolve a resource-safe compiled rank schedule.

    ``LORA_BLOCK_SIZE_R`` is the provider/autotune seam for the rank tile. Until
    per-device tables are generated, 256 is a bounded fallback that avoids the
    large shared-memory footprint of arbitrary whole-rank tiles. Explicit
    whole-rank and looped requests remain provider/benchmark-controlled. A
    configured tile is clamped to at least 16 and rounded down to a power of two.
    """
    whole_block_r = max(16, triton.next_power_of_2(rank))
    configured_block_r = force_block_size_r or config.get("LORA_BLOCK_SIZE_R", 256)
    configured_block_r = max(16, configured_block_r)
    configured_block_r = 1 << (configured_block_r.bit_length() - 1)

    if requested == SlicedLoraBRankSchedule.WHOLE_RANK:
        return requested, whole_block_r
    if requested == SlicedLoraBRankSchedule.LOOPED_RANK:
        return requested, min(configured_block_r, whole_block_r)
    if whole_block_r <= configured_block_r:
        return SlicedLoraBRankSchedule.WHOLE_RANK, whole_block_r
    return SlicedLoraBRankSchedule.LOOPED_RANK, configured_block_r


@triton.jit
def _sliced_lora_b_expand_add_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    a_offsets_ptr,
    b_offsets_ptr,
    output_offsets_ptr,
    widths_ptr,
    tile_prefix_ptr,
    R: tl.constexpr,
    num_valid_tokens,
    total_slice_tiles,
    stride_am,
    stride_ar,
    stride_be,
    stride_bn,
    stride_br,
    stride_cm,
    stride_cn,
    router_topk: tl.constexpr,
    SCHEDULE: tl.constexpr,
    RANK_SCHEDULE: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    UNIFORM_SLICE_N: tl.constexpr,
    TOTAL_ACTIVE_N: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    FUSE_ADD_TO_OUTPUT: tl.constexpr,
    FUSE_SUM_ALL_REDUCE: tl.constexpr,
    NUM_M_BLOCKS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """One source specialized into flat, uniform-sliced, or ragged schedules."""
    pid = tl.program_id(0)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if SCHEDULE == 0:  # ALIGNED_FLAT
        num_pid_n = tl.cdiv(TOTAL_ACTIVE_N, BLOCK_SIZE_N)
    elif SCHEDULE == 1:  # UNIFORM_SLICED
        num_pid_n = tl.cdiv(UNIFORM_SLICE_N, BLOCK_SIZE_N)
    else:  # DESCRIPTOR_RAGGED
        num_pid_n = total_slice_tiles

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(NUM_M_BLOCKS - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    if SCHEDULE == 0:
        local_tile_id = pid_n
        slice_id = (pid_n * BLOCK_SIZE_N) // UNIFORM_SLICE_N
        a_begin = slice_id * R
        b_begin = 0
        output_begin = 0
        slice_width = TOTAL_ACTIVE_N
    elif SCHEDULE == 1:
        local_tile_id = pid_n
        slice_id = tl.program_id(1)
        a_begin = slice_id * R
        b_begin = slice_id * UNIFORM_SLICE_N
        output_begin = slice_id * UNIFORM_SLICE_N
        slice_width = UNIFORM_SLICE_N
    else:
        global_tile_id = pid_n
        slice_id = global_tile_id * 0
        for slice_idx in tl.static_range(1, NUM_SLICES):
            slice_tile_begin = tl.load(tile_prefix_ptr + slice_idx)
            slice_id += (global_tile_id >= slice_tile_begin).to(tl.int32)

        tile_begin = tl.load(tile_prefix_ptr + slice_id).to(tl.int64)
        local_tile_id = global_tile_id - tile_begin
        a_begin = tl.load(a_offsets_ptr + slice_id).to(tl.int64)
        b_begin = tl.load(b_offsets_ptr + slice_id).to(tl.int64)
        output_begin = tl.load(output_offsets_ptr + slice_id).to(tl.int64)
        slice_width = tl.load(widths_ptr + slice_id).to(tl.int64)

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    local_n = local_tile_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    output_mask = local_n < slice_width
    offs_b_n = b_begin + local_n
    offs_output_n = output_begin + local_n

    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_expert == -1:
        if not FUSE_ADD_TO_OUTPUT and not FUSE_SUM_ALL_REDUCE:
            c_ptrs = (
                c_ptr
                + offs_token[:, None] * stride_cm
                + offs_output_n[None, :] * stride_cn
            )
            c_mask = token_mask[:, None] & output_mask[None, :]
            zeros = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=c_ptr.dtype.element_ty)
            tl.store(c_ptrs, zeros, mask=c_mask)
        return

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if RANK_SCHEDULE == 0:  # WHOLE_RANK
        offs_r = tl.arange(0, BLOCK_SIZE_R).to(tl.int64)
        rank_mask = offs_r < R
        a = tl.load(
            a_ptr
            + offs_token[:, None] * stride_am
            + (a_begin + offs_r)[None, :] * stride_ar,
            mask=token_mask[:, None] & rank_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptr
            + off_expert * stride_be
            + offs_b_n[None, :] * stride_bn
            + offs_r[:, None] * stride_br,
            mask=output_mask[None, :] & rank_mask[:, None],
            other=0.0,
        )
        a = a.to(b.dtype)
        accumulator = tl.dot(a, b, out_dtype=tl.float32)
    else:  # LOOPED_RANK
        for rank_begin in range(0, R, BLOCK_SIZE_R):
            offs_r = rank_begin + tl.arange(0, BLOCK_SIZE_R).to(tl.int64)
            rank_mask = offs_r < R
            a = tl.load(
                a_ptr
                + offs_token[:, None] * stride_am
                + (a_begin + offs_r)[None, :] * stride_ar,
                mask=token_mask[:, None] & rank_mask[None, :],
                other=0.0,
            )
            b = tl.load(
                b_ptr
                + off_expert * stride_be
                + offs_b_n[None, :] * stride_bn
                + offs_r[:, None] * stride_br,
                mask=output_mask[None, :] & rank_mask[:, None],
                other=0.0,
            )
            a = a.to(b.dtype)
            accumulator += tl.dot(a, b, out_dtype=tl.float32)
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator *= moe_weight[:, None]

    if FUSE_SUM_ALL_REDUCE:
        offs_token_out = offs_token // router_topk
    else:
        offs_token_out = offs_token
    c_ptrs = (
        c_ptr + offs_token_out[:, None] * stride_cm + offs_output_n[None, :] * stride_cn
    )
    c_mask = token_mask[:, None] & output_mask[None, :]
    if FUSE_SUM_ALL_REDUCE:
        tl.atomic_add(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=c_mask)
    elif FUSE_ADD_TO_OUTPUT:
        base = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
        tl.store(
            c_ptrs,
            (accumulator + base).to(c_ptr.dtype.element_ty),
            mask=c_mask,
        )
    else:
        tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=c_mask)


def invoke_sliced_lora_b_expand_add(
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
    fuse_add_to_output: bool = False,
    layout: SlicedLoraBLayout,
    schedule: SlicedLoraBSchedule,
    rank_schedule: SlicedLoraBRankSchedule = SlicedLoraBRankSchedule.AUTO,
    descriptors: SlicedLoraBDescriptors | None = None,
    force_block_size_n: int | None = None,
    force_block_size_r: int | None = None,
) -> None:
    """Launch one compiled slice schedule for a common-rank LoRA-B layout."""
    block_m = config["BLOCK_SIZE_M"]
    block_n = force_block_size_n or config["BLOCK_SIZE_N"]
    num_m_blocks = triton.cdiv(sorted_token_ids.shape[0], block_m)

    if schedule == SlicedLoraBSchedule.ALIGNED_FLAT:
        if not layout.is_uniform_contiguous or (
            layout.num_slices > 1 and layout.widths[0] % block_n != 0
        ):
            raise ValueError("flat sliced LoRA-B requires aligned contiguous slices")
        grid = (num_m_blocks * triton.cdiv(layout.active_width, block_n),)
        total_slice_tiles = 0
        descriptor_ptrs = (output, output, output, output, output)
    elif schedule == SlicedLoraBSchedule.UNIFORM_SLICED:
        if not layout.is_uniform_contiguous:
            raise ValueError("uniform sliced LoRA-B requires contiguous equal slices")
        slice_width = layout.widths[0]
        grid = (
            num_m_blocks * triton.cdiv(slice_width, block_n),
            layout.num_slices,
        )
        total_slice_tiles = 0
        descriptor_ptrs = (output, output, output, output, output)
    else:
        if descriptors is None or descriptors.layout != layout:
            raise ValueError("ragged sliced LoRA-B requires matching descriptors")
        if descriptors.block_size_n != block_n:
            raise ValueError("ragged descriptors must match BLOCK_SIZE_N")
        total_slice_tiles = descriptors.total_tiles
        grid = (num_m_blocks * total_slice_tiles,)
        descriptor_ptrs = (
            descriptors.a_offsets,
            descriptors.b_offsets,
            descriptors.output_offsets,
            descriptors.widths,
            descriptors.tile_prefix,
        )

    compiled_rank_schedule, block_r = select_sliced_lora_b_rank_schedule(
        layout.rank,
        config,
        rank_schedule,
        force_block_size_r=force_block_size_r,
    )

    _sliced_lora_b_expand_add_kernel[grid](
        intermediate,
        weight,
        output,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        *descriptor_ptrs,
        layout.rank,
        topk_ids.numel(),
        total_slice_tiles,
        intermediate.stride(0),
        intermediate.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        output.stride(-2),
        output.stride(-1),
        router_topk=topk_ids.shape[1],
        SCHEDULE=int(schedule),
        RANK_SCHEDULE=int(compiled_rank_schedule),
        NUM_SLICES=layout.num_slices,
        UNIFORM_SLICE_N=layout.widths[0],
        TOTAL_ACTIVE_N=layout.active_width,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        FUSE_ADD_TO_OUTPUT=fuse_add_to_output,
        FUSE_SUM_ALL_REDUCE=fuse_sum_all_reduce,
        NUM_M_BLOCKS=num_m_blocks,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_R=block_r,
        GROUP_SIZE_M=config.get("GROUP_SIZE_M", 1),
        num_warps=config.get("num_warps", 4),
        num_stages=1,
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
    fuse_add_to_output: bool = False,
    force_block_size_n: int | None = None,
) -> None:
    n = weight.shape[1]
    rank = weight.shape[2]
    widths = (n // 2, n // 2) if gated_midpoint else (n,)
    layout = SlicedLoraBLayout.contiguous(widths=widths, rank=rank)
    block_n = (
        force_block_size_n
        if force_block_size_n is not None
        else (128 if n % 128 == 0 else config["BLOCK_SIZE_N"])
    )
    if gated_midpoint:
        while block_n > 16 and widths[0] % block_n != 0:
            block_n //= 2
        if widths[0] % block_n != 0:
            raise ValueError(
                f"flat midpoint schedule cannot tile output half {widths[0]} "
                f"with BLOCK_SIZE_N={block_n}"
            )
    invoke_sliced_lora_b_expand_add(
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
        fuse_add_to_output=fuse_add_to_output,
        layout=layout,
        schedule=SlicedLoraBSchedule.ALIGNED_FLAT,
        force_block_size_n=block_n,
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
    fuse_add_to_output: bool = False,
    force_block_size_n: int | None = None,
) -> None:
    slice_n = weight.shape[1] // 2
    layout = SlicedLoraBLayout.contiguous(
        widths=(slice_n, slice_n), rank=weight.shape[2]
    )
    block_n = (
        force_block_size_n
        if force_block_size_n is not None
        else min(64, max(16, triton.next_power_of_2(slice_n)))
    )
    invoke_sliced_lora_b_expand_add(
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
        fuse_add_to_output=fuse_add_to_output,
        layout=layout,
        schedule=SlicedLoraBSchedule.UNIFORM_SLICED,
        force_block_size_n=block_n,
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
    fuse_add_to_output: bool = False,
) -> None:
    """Launch complete gate/up or down layouts into a caller-owned destination."""
    if num_output_slices == 2:
        if weight.shape[1] % 2 != 0:
            raise ValueError("complete gate/up LoRA-B requires two equal-width slices")
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
                fuse_add_to_output=fuse_add_to_output,
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
                fuse_add_to_output=fuse_add_to_output,
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
        fuse_add_to_output=fuse_add_to_output,
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
    """Launch the midpoint-compatible flat schedule for controlled A/B tests."""
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
    """Launch the uniform two-slice schedule for controlled A/B tests."""
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
    "SlicedLoraBDescriptors",
    "SlicedLoraBLayout",
    "SlicedLoraBRankSchedule",
    "SlicedLoraBSchedule",
    "build_sliced_lora_b_descriptors",
    "invoke_moe_lora_expand_add",
    "invoke_moe_lora_expand_add_flat_for_benchmark",
    "invoke_moe_lora_expand_add_sliced_for_benchmark",
    "invoke_sliced_lora_b_expand_add",
    "select_sliced_lora_b_rank_schedule",
]
