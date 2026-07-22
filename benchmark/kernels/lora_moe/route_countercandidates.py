"""Provider-neutral route-plan counter-candidates for SGL LoRA MoE.

This module is benchmark-owned.  Serving code must not import it.  It exposes a
small route-plan contract shared by the current SGL producer and a snapshot of
the legacy fused merged-align algorithm.  Reuse and memoization policies live
above this interface, so the benchmark does not couple them to a GEMM provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class RoutePlan:
    """Aligned pair routing consumed by grouped/segmented LoRA kernels."""

    sorted_pair_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_pairs_post_padded: torch.Tensor
    token_lora_mask: torch.Tensor
    virtual_num_experts: int
    block_m: int
    producer: str


@dataclass(frozen=True)
class RoutePlanKey:
    """Identity key for a plan whose inputs are stable for one route epoch.

    Pointer identity alone is unsafe under CUDA graph replay because graph input
    buffers keep their addresses while a router overwrites their contents.  The
    caller-owned ``route_epoch`` is therefore mandatory: it must advance after
    every router write/graph input refresh.
    """

    topk_ptr: int
    topk_version: int
    topk_shape: tuple[int, ...]
    topk_stride: tuple[int, ...]
    mapping_ptr: int
    mapping_version: int
    mapping_shape: tuple[int, ...]
    mapping_stride: tuple[int, ...]
    device: str
    num_experts: int
    max_loras: int
    local_expert_offset: int
    local_num_experts: int
    block_m: int
    route_epoch: int
    producer: str


def make_route_plan_key(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    *,
    num_experts: int,
    max_loras: int,
    local_expert_offset: int,
    local_num_experts: int,
    block_m: int,
    route_epoch: int,
    producer: str,
) -> RoutePlanKey:
    return RoutePlanKey(
        topk_ptr=topk_ids.data_ptr(),
        topk_version=topk_ids._version,
        topk_shape=tuple(topk_ids.shape),
        topk_stride=tuple(topk_ids.stride()),
        mapping_ptr=token_lora_mapping.data_ptr(),
        mapping_version=token_lora_mapping._version,
        mapping_shape=tuple(token_lora_mapping.shape),
        mapping_stride=tuple(token_lora_mapping.stride()),
        device=str(topk_ids.device),
        num_experts=num_experts,
        max_loras=max_loras,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
        block_m=block_m,
        route_epoch=route_epoch,
        producer=producer,
    )


class RoutePlanMemo:
    """Explicit-lifetime memo; ownership remains with one forward/graph epoch."""

    def __init__(self) -> None:
        self._plans: dict[RoutePlanKey, RoutePlan] = {}
        self.hits = 0
        self.misses = 0

    def get_or_build(
        self, key: RoutePlanKey, build: Callable[[], RoutePlan]
    ) -> RoutePlan:
        plan = self._plans.get(key)
        if plan is not None:
            self.hits += 1
            return plan
        self.misses += 1
        plan = build()
        self._plans[key] = plan
        return plan

    def clear(self) -> None:
        self._plans.clear()


def _max_route_storage(num_pairs: int, num_buckets: int, block_m: int) -> int:
    if num_pairs < num_buckets:
        return num_pairs * block_m
    return num_pairs + num_buckets * (block_m - 1)


def _current_align(
    virtual_topk_ids: torch.Tensor,
    *,
    block_m: int,
    virtual_num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if virtual_num_experts >= 1024:
        from sglang.srt.lora.sgl_lora.triton_ops.virtual_experts import (
            _align_block_size_large,
        )

        return _align_block_size_large(virtual_topk_ids, block_m, virtual_num_experts)

    from sglang.kernels.ops.moe import moe_align_block_size

    num_pairs = virtual_topk_ids.numel()
    max_padded = _max_route_storage(num_pairs, virtual_num_experts + 1, block_m)
    sorted_pair_ids = torch.empty(
        max_padded, dtype=torch.int32, device=virtual_topk_ids.device
    )
    expert_ids = torch.empty(
        triton.cdiv(max_padded, block_m),
        dtype=torch.int32,
        device=virtual_topk_ids.device,
    )
    num_post = torch.empty(1, dtype=torch.int32, device=virtual_topk_ids.device)
    cumsum = torch.empty(
        virtual_num_experts + 2,
        dtype=torch.int32,
        device=virtual_topk_ids.device,
    )
    moe_align_block_size(
        virtual_topk_ids,
        virtual_num_experts + 1,
        block_m,
        sorted_pair_ids,
        expert_ids,
        num_post,
        cumsum,
        True,
    )
    return sorted_pair_ids, expert_ids, num_post


def build_current_sgl_plan(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    *,
    num_experts: int,
    max_loras: int,
    local_expert_offset: int,
    local_num_experts: int,
    block_m: int,
) -> RoutePlan:
    """Current SGL virtual-id + align (+sanitize for multi-adapter) path."""
    from sglang.srt.lora.sgl_lora.triton_ops.virtual_experts import (
        _fused_virtual_topk_ids,
        fused_sanitize_expert_ids,
    )

    virtual_ids, token_mask, virtual_num_experts = _fused_virtual_topk_ids(
        topk_ids,
        token_lora_mapping,
        num_experts,
        False,
        max_loras,
        local_expert_offset,
        local_num_experts,
    )
    sorted_pair_ids, expert_ids, num_post = _current_align(
        virtual_ids,
        block_m=block_m,
        virtual_num_experts=virtual_num_experts,
    )
    # Match production's post-align view tightening.  The align producer keeps
    # its conservative allocation, but downstream grids must not launch over
    # buckets that EP guarantees are empty.
    num_pairs = topk_ids.numel()
    ep_local = local_num_experts < num_experts
    populated_buckets = (
        local_num_experts * max_loras + 1 if ep_local else virtual_num_experts
    )
    max_nonempty = min(num_pairs, populated_buckets)
    tight_padded = (
        triton.cdiv(num_pairs + max_nonempty * (block_m - 1), block_m) * block_m
    )
    sorted_pair_ids = sorted_pair_ids[:tight_padded]
    expert_ids = expert_ids[: tight_padded // block_m]
    if max_loras != 1:
        expert_ids = fused_sanitize_expert_ids(expert_ids, virtual_num_experts)
    return RoutePlan(
        sorted_pair_ids=sorted_pair_ids,
        expert_ids=expert_ids,
        num_pairs_post_padded=num_post,
        token_lora_mask=token_mask,
        virtual_num_experts=virtual_num_experts,
        block_m=block_m,
        producer="current_sgl",
    )


@lru_cache(maxsize=None)
def _merged_align_module(dtype: torch.dtype):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(dtype)
    return load_jit(
        "benchmark_route_countercandidate_merged_align",
        *args,
        cuda_files=[
            "../../../../benchmark/kernels/lora_moe/"
            "route_countercandidate_merged_align.cu"
        ],
        cuda_wrappers=[
            (
                "benchmark_moe_lora_merged_align",
                f"MoeLoraMergedAlignKernel<{args}>::run",
            )
        ],
    )


def merged_align_supported(
    *,
    num_experts: int,
    max_loras: int,
    local_num_experts: int,
) -> tuple[bool, str]:
    ep_local = local_num_experts < num_experts
    # The inspected legacy compact mapping is valid only for a single adapter.
    bucket_experts = (
        local_num_experts if ep_local and max_loras == 1 else num_experts * max_loras
    )
    if bucket_experts + 1 > 1024:
        return False, "legacy bucket-count limit (>1024 including sentinel)"
    return True, ""


def build_legacy_merged_plan(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    *,
    num_experts: int,
    max_loras: int,
    local_expert_offset: int,
    local_num_experts: int,
    block_m: int,
) -> RoutePlan:
    """Benchmark-owned snapshot of legacy fused merged-align.

    It computes virtual IDs inside histogram/scatter and never materializes the
    virtual-ID matrix.  The copied algorithm retains its <=1024 bucket limit and
    single-adapter-only EP compaction; unsupported cells are reported rather
    than silently falling back to the current SGL path.
    """
    supported, reason = merged_align_supported(
        num_experts=num_experts,
        max_loras=max_loras,
        local_num_experts=local_num_experts,
    )
    if not supported:
        raise NotImplementedError(reason)

    flat_topk = topk_ids.reshape(-1)
    if flat_topk.dtype == torch.int64:
        flat_topk = flat_topk.to(torch.int32)
    num_tokens, top_k = topk_ids.shape
    num_pairs = topk_ids.numel()
    ep_local = local_num_experts < num_experts
    compact = ep_local and max_loras == 1
    bucket_experts = local_num_experts if compact else num_experts * max_loras
    max_padded = _max_route_storage(num_pairs, bucket_experts + 1, block_m)
    sorted_alloc = (max_padded + 3) & ~3
    sorted_pair_ids = torch.empty(
        sorted_alloc, dtype=torch.int32, device=topk_ids.device
    )
    expert_ids = torch.empty(
        triton.cdiv(max_padded, block_m),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    num_post = torch.empty(1, dtype=torch.int32, device=topk_ids.device)
    cumsum = torch.empty(bucket_experts + 2, dtype=torch.int32, device=topk_ids.device)
    token_mask = torch.empty(num_tokens, dtype=torch.bool, device=topk_ids.device)

    # Match the legacy launch policy: one-block scatter for decode-sized pair
    # sets when its dynamic shared-memory footprint remains below 47 KiB.
    fuse_scatter = num_pairs <= 2048
    if fuse_scatter:
        num_buckets = bucket_experts + 1
        scan_size = 1 << (num_buckets - 1).bit_length()
        shmem = (
            num_buckets + (num_buckets + 1) + scan_size + 32 + num_buckets + num_pairs
        ) * 4
        fuse_scatter = shmem <= 47 * 1024

    module = _merged_align_module(flat_topk.dtype)
    module.benchmark_moe_lora_merged_align(
        flat_topk,
        token_lora_mapping,
        token_mask,
        bucket_experts + 1,
        block_m,
        sorted_pair_ids,
        expert_ids,
        num_post,
        cumsum,
        True,
        top_k,
        num_experts,
        local_expert_offset,
        local_num_experts,
        ep_local,
        False,
        True,
        compact,
        fuse_scatter,
    )
    return RoutePlan(
        sorted_pair_ids=sorted_pair_ids,
        expert_ids=expert_ids,
        num_pairs_post_padded=num_post,
        token_lora_mask=token_mask,
        virtual_num_experts=num_experts * max_loras,
        block_m=block_m,
        producer="legacy_merged",
    )


@triton.jit
def _consume_route_plan_kernel(
    sorted_pair_ids,
    expert_ids,
    num_post_ptr,
    pair_values,
    output,
    num_pairs,
    BLOCK_M: tl.constexpr,
):
    block = tl.program_id(0)
    offsets = block * BLOCK_M + tl.arange(0, BLOCK_M)
    num_post = tl.load(num_post_ptr)
    active = offsets < num_post
    pair = tl.load(sorted_pair_ids + offsets, mask=active, other=num_pairs)
    expert = tl.load(expert_ids + block)
    valid = active & (pair < num_pairs) & (expert >= 0)
    value = tl.load(pair_values + pair, mask=valid, other=0.0)
    tl.store(output + pair, value * (expert + 1), mask=valid)


def consume_route_plan(
    plan: RoutePlan,
    pair_values: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Common synthetic delta consumer used for strict semantic checks."""
    output.zero_()
    grid = (plan.expert_ids.numel(),)
    _consume_route_plan_kernel[grid](
        plan.sorted_pair_ids,
        plan.expert_ids,
        plan.num_pairs_post_padded,
        pair_values,
        output,
        pair_values.numel(),
        BLOCK_M=plan.block_m,
        num_warps=4,
    )
    return output
