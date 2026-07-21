"""Packed-factor LoRA-A shrink primitives owned by ``sgl_lora``.

LoRA-A is factor-centric rather than output-slice-centric. For one homogeneous
factor-plan signature, setup code packs each unique active A factor contiguously
and this kernel computes the flat packed rank dimension::

    intermediate[pair_row, :packed_rank] = input[input_row] @ A[group].T

Tiles may cross factor boundaries because every packed factor consumes the same
input row. The B slice plan alone maps packed factor offsets to logical output
slices. Adapters with different factor-plan signatures are grouped into separate
launches rather than represented by synthetic zero factors.

This first production policy handles indexed rows grouped by virtual expert. A
compile-time input-row domain distinguishes token-domain gate/up inputs from
already-routed pair-domain down inputs. Reduction over the large hidden K is
either single-owner looped K or split-K. Split-K accumulation precision is an
explicit schedule policy: FP32 avoids low-precision reduction, while
``OUTPUT_DTYPE`` preserves the current backend's lower-traffic BF16/FP16 option.
The caller-owned output dtype must agree with that policy, and the launcher clears
the output before every split-K launch.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import torch
import triton
import triton.language as tl

from sglang.jit_kernel.utils import is_arch_support_pdl


class LoraAInputRowDomain(IntEnum):
    TOKEN = 0
    ROUTED_PAIR = 1


class LoraASplitKAccumulation(IntEnum):
    FP32 = 0
    OUTPUT_DTYPE = 1


@dataclass(frozen=True)
class IndexedLoraARowPlan:
    """Graph-stable indexed rows, bound to the block size used to align them."""

    sorted_pair_ids: torch.Tensor
    block_group_ids: torch.Tensor
    num_pairs_post_padded: torch.Tensor
    block_m: int

    @property
    def capacity_m_blocks(self) -> int:
        return triton.cdiv(self.sorted_pair_ids.shape[0], self.block_m)


@dataclass(frozen=True)
class IndexedLoraAKernelConfig:
    block_n: int
    block_k: int
    split_k: int
    num_warps: int
    num_stages: int = 1
    split_k_accumulation: LoraASplitKAccumulation = LoraASplitKAccumulation.FP32

    @property
    def uses_split_k(self) -> bool:
        return self.split_k > 1

    @property
    def requires_fp32_output(self) -> bool:
        return self.uses_split_k and (
            self.split_k_accumulation == LoraASplitKAccumulation.FP32
        )


def select_indexed_lora_a_kernel_config(
    weight: torch.Tensor,
    row_plan: IndexedLoraARowPlan,
    config: dict[str, Any],
    *,
    split_k: int | None = None,
    split_k_accumulation: LoraASplitKAccumulation = LoraASplitKAccumulation.FP32,
) -> IndexedLoraAKernelConfig:
    """Resolve a bounded indexed-row A schedule without allocating workspace."""
    packed_rank = weight.shape[1]
    hidden_size = weight.shape[2]
    configured_block_n = config.get(
        "LORA_BLOCK_SIZE_N",
        min(64, max(16, triton.next_power_of_2(packed_rank))),
    )
    configured_block_n = max(16, configured_block_n)
    block_n = 1 << (configured_block_n.bit_length() - 1)
    block_k = config.get("LORA_BLOCK_SIZE_K", min(256, hidden_size))
    block_k = max(16, block_k)
    block_k = 1 << (block_k.bit_length() - 1)

    max_split_k = max(1, triton.cdiv(hidden_size, block_k))
    if split_k is None:
        split_k = config.get("LORA_SPLIT_K")
    if split_k is None:
        num_m_blocks = row_plan.capacity_m_blocks
        num_n_blocks = triton.cdiv(packed_rank, block_n)
        base_grid = num_m_blocks * num_n_blocks
        if base_grid == 0:
            split_k = 1
        else:
            # Provisional occupancy fallback only. Device/provider tuning tables
            # should override LORA_SPLIT_K for production Hopper/Blackwell policy.
            target_grid = (
                512 if packed_rank <= 32 else 384 if packed_rank <= 64 else 256
            )
            split_k = min(triton.cdiv(target_grid, base_grid), max_split_k, 8)
    split_k = max(1, min(split_k, max_split_k))

    return IndexedLoraAKernelConfig(
        block_n=block_n,
        block_k=block_k,
        split_k=split_k,
        num_warps=config.get("LORA_NUM_WARPS", config.get("num_warps", 4)),
        num_stages=config.get("LORA_NUM_STAGES", config.get("num_stages", 1)),
        split_k_accumulation=split_k_accumulation,
    )


@triton.jit
def _indexed_lora_a_shrink_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    sorted_pair_ids_ptr,
    group_ids_ptr,
    num_pairs_post_padded_ptr,
    stride_input_m,
    stride_input_k,
    stride_weight_g,
    stride_weight_n,
    stride_weight_k,
    stride_output_m,
    stride_output_n,
    PACKED_RANK: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    NUM_VALID_PAIRS,
    ROUTER_TOPK: tl.constexpr,
    INPUT_ROW_DOMAIN: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    ENABLE_PDL: tl.constexpr = False,
):
    pid = tl.program_id(0)
    pid_split_k = pid % SPLIT_K
    pid_mn = pid // SPLIT_K

    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    num_pairs_post_padded = tl.load(num_pairs_post_padded_ptr)
    num_pid_n = tl.cdiv(PACKED_RANK, BLOCK_SIZE_N)
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    if pid_m * BLOCK_SIZE_M >= num_pairs_post_padded:
        return

    offs_pair_slot = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_pair = tl.load(sorted_pair_ids_ptr + offs_pair_slot).to(tl.int64)
    pair_mask = offs_pair < NUM_VALID_PAIRS
    if INPUT_ROW_DOMAIN == 0:
        offs_input_m = offs_pair // ROUTER_TOPK
    else:
        offs_input_m = offs_pair

    off_group = tl.load(group_ids_ptr + pid_m).to(tl.int64)
    if off_group == -1:
        return

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    n_mask = offs_n < PACKED_RANK
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    num_k_tiles = tl.cdiv(HIDDEN_SIZE, BLOCK_SIZE_K * SPLIT_K)
    for k_tile in range(0, num_k_tiles):
        offs_k = (
            pid_split_k * BLOCK_SIZE_K
            + k_tile * BLOCK_SIZE_K * SPLIT_K
            + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
        )
        k_mask = offs_k < HIDDEN_SIZE
        input_tile = tl.load(
            input_ptr
            + offs_input_m[:, None] * stride_input_m
            + offs_k[None, :] * stride_input_k,
            mask=pair_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        weight_tile = tl.load(
            weight_ptr
            + off_group * stride_weight_g
            + offs_n[None, :] * stride_weight_n
            + offs_k[:, None] * stride_weight_k,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        accumulator += tl.dot(input_tile, weight_tile, out_dtype=tl.float32)

    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    output_ptrs = (
        output_ptr
        + offs_pair[:, None] * stride_output_m
        + offs_n[None, :] * stride_output_n
    )
    output_mask = pair_mask[:, None] & n_mask[None, :]
    if SPLIT_K == 1:
        tl.store(
            output_ptrs,
            accumulator.to(output_ptr.dtype.element_ty),
            mask=output_mask,
        )
    else:
        tl.atomic_add(
            output_ptrs,
            accumulator.to(output_ptr.dtype.element_ty),
            mask=output_mask,
            sem="relaxed",
        )


def invoke_indexed_lora_a_shrink(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    row_plan: IndexedLoraARowPlan,
    *,
    num_valid_pairs: int,
    router_topk: int,
    input_row_domain: LoraAInputRowDomain,
    kernel_config: IndexedLoraAKernelConfig,
    clear_output: bool = False,
) -> None:
    """Launch packed-factor LoRA-A for precomputed indexed group rows.

    Output is always pair-domain; ``TOKEN`` changes only which input row each
    pair reads and does not yet deduplicate shared-A work across top-k slots.

    Split-K owns and clears its accumulation buffer. Its explicit accumulation
    policy must agree with the caller-owned output dtype. ``clear_output`` also
    clears the single-owner output when a later consumer has a broader routing
    domain and may read rows skipped by this A stage (for example EP-local A
    followed by shared-outer B).
    """
    if kernel_config.requires_fp32_output and output.dtype != torch.float32:
        raise ValueError("FP32 split-K LoRA-A requires an FP32 output")
    if kernel_config.uses_split_k or clear_output:
        output.zero_()

    if row_plan.sorted_pair_ids.numel() == 0 or weight.shape[1] == 0:
        return

    num_m_blocks = row_plan.capacity_m_blocks
    num_n_blocks = triton.cdiv(weight.shape[1], kernel_config.block_n)
    grid = (num_m_blocks * num_n_blocks * kernel_config.split_k,)
    enable_pdl = is_arch_support_pdl()
    pdl_kwargs = {"launch_pdl": True} if enable_pdl else {}
    _indexed_lora_a_shrink_kernel[grid](
        hidden_states,
        weight,
        output,
        row_plan.sorted_pair_ids,
        row_plan.block_group_ids,
        row_plan.num_pairs_post_padded,
        hidden_states.stride(0),
        hidden_states.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        output.stride(0),
        output.stride(1),
        PACKED_RANK=weight.shape[1],
        HIDDEN_SIZE=weight.shape[2],
        NUM_VALID_PAIRS=num_valid_pairs,
        ROUTER_TOPK=router_topk,
        INPUT_ROW_DOMAIN=int(input_row_domain),
        BLOCK_SIZE_M=row_plan.block_m,
        BLOCK_SIZE_N=kernel_config.block_n,
        BLOCK_SIZE_K=kernel_config.block_k,
        SPLIT_K=kernel_config.split_k,
        ENABLE_PDL=enable_pdl,
        num_warps=kernel_config.num_warps,
        num_stages=kernel_config.num_stages,
        **pdl_kwargs,
    )


__all__ = [
    "IndexedLoraAKernelConfig",
    "IndexedLoraARowPlan",
    "LoraAInputRowDomain",
    "LoraASplitKAccumulation",
    "invoke_indexed_lora_a_shrink",
    "select_indexed_lora_a_kernel_config",
]
