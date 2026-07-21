"""Sliced routed LoRA-B expand/add kernels owned by ``sgl_lora``.

One semantic operation covers ordinary projections, complete stacked
projections, and arbitrary active subsets of a stacked projection::

    C[:, output_i] <epilogue>= A[:, a_i] @ B[:, b_i].T

The logical slice layout is deliberately independent from routed-row metadata.
In particular, ``b_offsets`` and ``output_offsets`` need not match: a compact B
buffer may contain only Q and V while those slices still write to their original
destinations in a Q/K/V/Z base output.

The source compiles into four slice schedules:

* ``ALIGNED_FLAT`` for regular contiguous slices whose boundaries align to the
  chosen N tile (or the single-slice case);
* ``UNIFORM_SLICED`` for regular equal-width slices with independent tail masks;
* ``COMPILED_RAGGED`` for a stable irregular layout baked into one binary; and
* ``DESCRIPTOR_RAGGED`` for runtime-selected layouts, including mixed adapters
  whose virtual experts target different stacked slices.

The current MoE runner uses only the first two common schedules. Stable irregular
layouts can be compiled directly, while mixed/runtime layouts use a descriptor
plan built and cached by setup code before launch, keeping device addresses stable
across CUDA-graph replay.

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
    COMPILED_RAGGED = 2
    DESCRIPTOR_RAGGED = 3


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

    ``layouts`` are host-side semantic descriptions used for validation and
    launch planning. ``metadata`` is their packed device execution table; the
    Triton kernel cannot dynamically index Python tuples. Keeping both avoids a
    device-to-host synchronization while making the host/device boundary
    explicit. The table is built once and reused, so forward performs no
    allocation or host-to-device copy.

    With ``group_layout_ids=None``, every routed group uses layout row zero and
    the launcher uses its exact tile grid. Otherwise the graph-stable mapping is
    indexed by the same virtual-expert/group id used to select LoRA-B weights;
    ``-1`` means that group does not target this layer. The launcher uses
    ``max_slice_tiles`` capacity and inactive CTAs return.

    Ragged direct-store launches update only descriptor-selected destinations;
    callers therefore initialize any untouched rows or slices they consume.
    Both a ``-1`` group layout and a routed ``expert_id == -1`` leave the
    destination unchanged. Add-to-output and routed reduction already have the
    same no-delta behavior.

    Construct this object during adapter/layout setup, not inside a captured
    forward. Each metadata row starts with a four-int aligned header containing
    its tile count, followed by aligned records
    ``(a_begin, b_tile_begin, output_tile_begin, valid_columns)``. This direct
    per-tile form avoids a runtime prefix search and is specific to
    ``block_size_n``.
    """

    layouts: tuple[SlicedLoraBLayout, ...]
    block_size_n: int
    layout_tile_counts: tuple[int, ...]
    max_slice_tiles: int
    metadata: torch.Tensor
    group_layout_ids: torch.Tensor | None

    @property
    def rank(self) -> int:
        return self.layouts[0].rank

    @property
    def is_group_indexed(self) -> bool:
        return self.group_layout_ids is not None


def build_sliced_lora_b_descriptors(
    layout: SlicedLoraBLayout | tuple[SlicedLoraBLayout, ...],
    *,
    block_size_n: int,
    device: torch.device | str,
    group_layout_ids: tuple[int, ...] | None = None,
) -> SlicedLoraBDescriptors:
    """Build a packed descriptor table outside graph-captured execution."""
    layouts = (layout,) if isinstance(layout, SlicedLoraBLayout) else layout
    if not layouts:
        raise ValueError("sliced LoRA-B descriptors require at least one layout")
    if len({item.rank for item in layouts}) != 1:
        raise ValueError("descriptor layouts must use one common packed rank")
    if len(layouts) > 1 and group_layout_ids is None:
        raise ValueError("multiple descriptor layouts require group_layout_ids")
    if group_layout_ids is not None and (
        not group_layout_ids
        or min(group_layout_ids) < -1
        or max(group_layout_ids) >= len(layouts)
    ):
        raise ValueError("group_layout_ids must be -1 or index the supplied layouts")

    layout_records: list[list[int]] = []
    layout_tile_counts: list[int] = []
    for item in layouts:
        records: list[int] = []
        for a_begin, b_begin, output_begin, width in zip(
            item.a_offsets,
            item.b_offsets,
            item.output_offsets,
            item.widths,
            strict=True,
        ):
            for tile_offset in range(0, width, block_size_n):
                records.extend(
                    (
                        a_begin,
                        b_begin + tile_offset,
                        output_begin + tile_offset,
                        min(block_size_n, width - tile_offset),
                    )
                )
        layout_records.append(records)
        layout_tile_counts.append(len(records) // 4)

    max_slice_tiles = max(layout_tile_counts)
    rows = [
        [
            tile_count,
            0,
            0,
            0,
            *records,
            *([0] * (4 * (max_slice_tiles - tile_count))),
        ]
        for records, tile_count in zip(layout_records, layout_tile_counts, strict=True)
    ]

    return SlicedLoraBDescriptors(
        layouts=layouts,
        block_size_n=block_size_n,
        layout_tile_counts=tuple(layout_tile_counts),
        max_slice_tiles=max_slice_tiles,
        metadata=torch.tensor(rows, dtype=torch.int32, device=device),
        group_layout_ids=(
            torch.tensor(group_layout_ids, dtype=torch.int32, device=device)
            if group_layout_ids is not None
            else None
        ),
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
    descriptor_metadata_ptr,
    group_layout_ids_ptr,
    R: tl.constexpr,
    num_valid_tokens,
    total_slice_tiles,
    descriptor_row_stride,
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
    COMPILED_A_OFFSETS: tl.constexpr,
    COMPILED_B_OFFSETS: tl.constexpr,
    COMPILED_OUTPUT_OFFSETS: tl.constexpr,
    COMPILED_WIDTHS: tl.constexpr,
    COMPILED_TILE_PREFIX: tl.constexpr,
    PER_GROUP_LAYOUT: tl.constexpr,
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
    else:  # COMPILED_RAGGED or DESCRIPTOR_RAGGED
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
    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if SCHEDULE >= 2 and off_expert == -1:
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
    elif SCHEDULE == 2:  # COMPILED_RAGGED
        global_tile_id = pid_n
        tile_begin = global_tile_id * 0
        a_begin = COMPILED_A_OFFSETS[0]
        b_begin = COMPILED_B_OFFSETS[0]
        output_begin = COMPILED_OUTPUT_OFFSETS[0]
        slice_width = COMPILED_WIDTHS[0]
        for slice_idx in tl.static_range(1, NUM_SLICES):
            slice_tile_begin = COMPILED_TILE_PREFIX[slice_idx]
            is_later_slice = global_tile_id >= slice_tile_begin
            tile_begin = tl.where(is_later_slice, slice_tile_begin, tile_begin)
            a_begin = tl.where(is_later_slice, COMPILED_A_OFFSETS[slice_idx], a_begin)
            b_begin = tl.where(is_later_slice, COMPILED_B_OFFSETS[slice_idx], b_begin)
            output_begin = tl.where(
                is_later_slice, COMPILED_OUTPUT_OFFSETS[slice_idx], output_begin
            )
            slice_width = tl.where(
                is_later_slice, COMPILED_WIDTHS[slice_idx], slice_width
            )
        local_tile_id = global_tile_id - tile_begin
    else:  # DESCRIPTOR_RAGGED
        global_tile_id = pid_n
        if PER_GROUP_LAYOUT:
            safe_expert = tl.maximum(off_expert, 0)
            layout_id = tl.load(group_layout_ids_ptr + safe_expert).to(tl.int64)
            if layout_id < 0:
                return
        else:
            layout_id = 0

        descriptor_begin = layout_id * descriptor_row_stride
        if PER_GROUP_LAYOUT:
            group_total_tiles = tl.load(descriptor_metadata_ptr + descriptor_begin)
            if global_tile_id >= group_total_tiles:
                return

        tile_descriptor = descriptor_begin + 4 + 4 * global_tile_id
        local_tile_id = global_tile_id * 0
        a_begin = tl.load(descriptor_metadata_ptr + tile_descriptor).to(tl.int64)
        b_begin = tl.load(descriptor_metadata_ptr + tile_descriptor + 1).to(tl.int64)
        output_begin = tl.load(descriptor_metadata_ptr + tile_descriptor + 2).to(
            tl.int64
        )
        slice_width = tl.load(descriptor_metadata_ptr + tile_descriptor + 3).to(
            tl.int64
        )

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    local_n = local_tile_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    output_mask = local_n < slice_width
    offs_b_n = b_begin + local_n
    offs_output_n = output_begin + local_n

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
    slice_plan: SlicedLoraBLayout | SlicedLoraBDescriptors,
    schedule: SlicedLoraBSchedule,
    rank_schedule: SlicedLoraBRankSchedule = SlicedLoraBRankSchedule.AUTO,
    force_block_size_n: int | None = None,
    force_block_size_r: int | None = None,
) -> None:
    """Launch one compiled schedule from a logical layout or descriptor plan."""
    block_m = config["BLOCK_SIZE_M"]
    block_n = force_block_size_n or config["BLOCK_SIZE_N"]
    num_m_blocks = triton.cdiv(sorted_token_ids.shape[0], block_m)

    compiled_a_offsets = (0,)
    compiled_b_offsets = (0,)
    compiled_output_offsets = (0,)
    compiled_widths = (0,)
    compiled_tile_prefix = (0,)
    per_group_layout = False
    descriptor_ptrs = (output, output)
    descriptor_row_stride = 0

    if schedule == SlicedLoraBSchedule.DESCRIPTOR_RAGGED:
        if not isinstance(slice_plan, SlicedLoraBDescriptors):
            raise ValueError("descriptor-ragged LoRA-B requires a descriptor plan")
        descriptors = slice_plan
        if descriptors.block_size_n != block_n:
            raise ValueError("ragged descriptors must match BLOCK_SIZE_N")
        rank = descriptors.rank
        # Runtime descriptors are already expanded per tile, so their compiled
        # binary does not specialize on the logical slice count.
        num_slices = 1
        per_group_layout = descriptors.is_group_indexed
        total_slice_tiles = (
            descriptors.max_slice_tiles
            if per_group_layout
            else descriptors.layout_tile_counts[0]
        )
        grid = (num_m_blocks * total_slice_tiles,)
        descriptor_ptrs = (
            descriptors.metadata,
            (
                descriptors.group_layout_ids
                if descriptors.group_layout_ids is not None
                else output
            ),
        )
        descriptor_row_stride = descriptors.metadata.stride(0)
        compiled_uniform_slice_n = 0
        compiled_total_active_n = 0
    elif not isinstance(slice_plan, SlicedLoraBLayout):
        raise ValueError("compiled slice schedules require a logical layout")
    elif schedule == SlicedLoraBSchedule.ALIGNED_FLAT:
        layout = slice_plan
        rank = layout.rank
        num_slices = layout.num_slices
        if not layout.is_uniform_contiguous or (
            layout.num_slices > 1 and layout.widths[0] % block_n != 0
        ):
            raise ValueError("flat sliced LoRA-B requires aligned contiguous slices")
        grid = (num_m_blocks * triton.cdiv(layout.active_width, block_n),)
        total_slice_tiles = 0
        compiled_uniform_slice_n = layout.widths[0]
        compiled_total_active_n = layout.active_width
    elif schedule == SlicedLoraBSchedule.UNIFORM_SLICED:
        layout = slice_plan
        rank = layout.rank
        num_slices = layout.num_slices
        if not layout.is_uniform_contiguous:
            raise ValueError("uniform sliced LoRA-B requires contiguous equal slices")
        slice_width = layout.widths[0]
        grid = (
            num_m_blocks * triton.cdiv(slice_width, block_n),
            layout.num_slices,
        )
        total_slice_tiles = 0
        compiled_uniform_slice_n = layout.widths[0]
        compiled_total_active_n = 0
    elif schedule == SlicedLoraBSchedule.COMPILED_RAGGED:
        layout = slice_plan
        rank = layout.rank
        num_slices = layout.num_slices
        tile_prefix = [0]
        for width in layout.widths:
            tile_prefix.append(tile_prefix[-1] + triton.cdiv(width, block_n))
        total_slice_tiles = tile_prefix[-1]
        grid = (num_m_blocks * total_slice_tiles,)
        compiled_a_offsets = layout.a_offsets
        compiled_b_offsets = layout.b_offsets
        compiled_output_offsets = layout.output_offsets
        compiled_widths = layout.widths
        compiled_tile_prefix = tuple(tile_prefix)
        compiled_uniform_slice_n = 0
        compiled_total_active_n = 0
    else:
        raise ValueError(f"unsupported sliced LoRA-B schedule: {schedule}")

    compiled_rank_schedule, block_r = select_sliced_lora_b_rank_schedule(
        rank,
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
        rank,
        topk_ids.numel(),
        total_slice_tiles,
        descriptor_row_stride,
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
        NUM_SLICES=num_slices,
        UNIFORM_SLICE_N=compiled_uniform_slice_n,
        TOTAL_ACTIVE_N=compiled_total_active_n,
        COMPILED_A_OFFSETS=compiled_a_offsets,
        COMPILED_B_OFFSETS=compiled_b_offsets,
        COMPILED_OUTPUT_OFFSETS=compiled_output_offsets,
        COMPILED_WIDTHS=compiled_widths,
        COMPILED_TILE_PREFIX=compiled_tile_prefix,
        PER_GROUP_LAYOUT=per_group_layout,
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
        slice_plan=layout,
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
        slice_plan=layout,
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


__all__ = [
    "SlicedLoraBDescriptors",
    "SlicedLoraBLayout",
    "SlicedLoraBRankSchedule",
    "SlicedLoraBSchedule",
    "build_sliced_lora_b_descriptors",
    "invoke_moe_lora_expand_add",
    "invoke_sliced_lora_b_expand_add",
    "select_sliced_lora_b_rank_schedule",
]
