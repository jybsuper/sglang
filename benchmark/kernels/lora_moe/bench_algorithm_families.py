#!/usr/bin/env python3
"""Matched BF16 MoE-LoRA algorithm-family benchmark.

This driver is intentionally independent of serving dispatch.  It compares
four implementation families behind canonical routed-pair boundaries:

``indexed``
    Direct address resolution from top-k ids and token adapter ids.
``aligned``
    Current padded virtual-expert grouped/Tensor-Core schedule.
``segmented``
    Compact segment pointers plus BM-sized block descriptors, with no padded
    output rows.
``bmm``
    Only admitted when every recorded nonempty virtual group has exactly the
    same M_g.  Packing, scatter, and all consumers remain in the timed region.

Sites are gate/up A, down A, the fused gate/up-B -> SwiGLU -> down-A consumer,
and down-B/finalize.  The final site also includes one-shot down A+B and
BF16-vs-FP32 semantic accumulation arms.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import triton

from benchmark.kernels.lora_moe.algorithm_family_kernels import (
    SegmentBlockPlan,
    allocate_segment_plan,
    base_only_swiglu,
    build_segment_plan_into,
    direct_down_b_finalize,
    indexed_gemm,
    one_shot_down_ab,
    reduce_pairs_finalize,
    segmented_gate_b_consumer,
    segmented_gemm,
)
from benchmark.kernels.lora_moe.bench_shrink_schedules import (
    _align_virtual_experts,
    _CacheControl,
    _make_cache_control,
)
from benchmark.kernels.lora_moe.profiling import make_batch, time_cuda_events

FAMILIES = ("indexed", "aligned", "segmented", "bmm")
SITES = ("gate_a", "gate_consumer", "down_a", "down_finalize")


@dataclass(slots=True)
class KernelTuning:
    indexed_bn: int = 32
    indexed_bk: int = 128
    indexed_warps: int = 4
    aligned_bn: int = 64
    aligned_bk: int = 64
    aligned_warps: int = 4
    segmented_bn: int = 64
    segmented_bk: int = 64
    segmented_warps: int = 4
    pair_consumer_bn: int = 64
    aligned_consumer_bn: int = 64
    segmented_consumer_bn: int = 64
    consumer_warps: int = 4
    direct_finalize_bh: int = 64
    reduce_finalize_bh: int = 128
    one_shot_bi: int = 64
    one_shot_bh: int = 32


_TUNING = KernelTuning()


@dataclass(frozen=True, slots=True)
class FamilyCase:
    tokens: int
    rank: int
    hidden: int = 2048
    intermediate: int = 512
    experts: int = 32
    top_k: int = 8
    adapters: int = 4
    adapter_mode: str = "mixed"
    route_pattern: str = "regular"
    seed: int = 19

    @property
    def pairs(self) -> int:
        return self.tokens * self.top_k

    @property
    def groups(self) -> int:
        return self.experts * self.adapters


@dataclass(slots=True)
class AlignedPlan:
    sorted_pair_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_pairs_post_padded: torch.Tensor
    block_m: int
    num_groups: int


@dataclass(slots=True)
class RegularBmmPlan:
    eligible: bool
    reason: str
    sorted_pair_ids: torch.Tensor | None
    active_groups: torch.Tensor | None
    valid_pairs: int
    m_per_group: int | None
    metrics: dict[str, object]


@dataclass(slots=True)
class Fixture:
    case: FamilyCase
    device: torch.device
    hidden_states: torch.Tensor
    activated_pairs: torch.Tensor
    gateup_base: torch.Tensor
    base_down_pairs: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    mapping: torch.Tensor
    gate_a: torch.Tensor
    gate_b: torch.Tensor
    down_a: torch.Tensor
    down_b: torch.Tensor
    gate_rank_input: torch.Tensor
    down_rank_input: torch.Tensor
    aligned_plan: AlignedPlan
    segment_plan: SegmentBlockPlan
    bmm_plan: RegularBmmPlan


def _make_mapping(case: FamilyCase, device: torch.device) -> torch.Tensor:
    mapping = torch.arange(case.tokens, device=device, dtype=torch.int64)
    mapping = mapping.remainder(case.adapters).to(torch.int32)
    if case.adapter_mode == "mixed":
        # Retain multiple active adapters while introducing base-only rows.
        mapping = mapping.clone()
        mapping[::7] = -1
    elif case.adapter_mode != "multi":
        raise ValueError(f"unknown adapter mode {case.adapter_mode!r}")
    return mapping


def _make_topk(case: FamilyCase, device: torch.device) -> torch.Tensor:
    pairs = torch.arange(case.pairs, device=device, dtype=torch.int64)
    if case.route_pattern == "regular":
        ids = pairs.remainder(case.experts)
    elif case.route_pattern == "iid":
        generator = torch.Generator(device=device).manual_seed(case.seed)
        # Independent random permutations per token avoid duplicate experts.
        scores = torch.rand(
            (case.tokens, case.experts), device=device, generator=generator
        )
        ids = scores.topk(case.top_k, dim=-1).indices.reshape(-1)
    elif case.route_pattern == "skewed":
        generator = torch.Generator(device=device).manual_seed(case.seed)
        probs = torch.arange(1, case.experts + 1, device=device).float().pow(-1.2)
        probs /= probs.sum()
        ids = torch.multinomial(
            probs.expand(case.tokens, -1),
            case.top_k,
            replacement=False,
            generator=generator,
        ).reshape(-1)
    else:
        raise ValueError(f"unknown route pattern {case.route_pattern!r}")
    return ids.reshape(case.tokens, case.top_k).to(torch.int32)


def _virtual_groups(
    topk_ids: torch.Tensor,
    mapping: torch.Tensor,
    *,
    experts: int,
    adapters: int,
) -> torch.Tensor:
    groups = mapping[:, None].to(torch.int64) * experts + topk_ids.to(torch.int64)
    valid = (mapping[:, None] >= 0) & (mapping[:, None] < adapters)
    return torch.where(valid, groups, torch.full_like(groups, adapters * experts))


def _build_aligned_plan(
    topk_ids: torch.Tensor,
    mapping: torch.Tensor,
    *,
    experts: int,
    adapters: int,
    block_m: int,
) -> AlignedPlan:
    groups = _virtual_groups(topk_ids, mapping, experts=experts, adapters=adapters)
    sorted_ids, expert_ids, post = _align_virtual_experts(
        groups, block_m=block_m, virtual_num_experts=experts * adapters
    )
    num_groups = experts * adapters
    expert_ids = torch.where(
        (expert_ids >= 0) & (expert_ids < num_groups),
        expert_ids,
        torch.full_like(expert_ids, -1),
    )
    return AlignedPlan(sorted_ids, expert_ids, post, block_m, num_groups)


def _route_metrics(groups: torch.Tensor, num_groups: int) -> dict[str, object]:
    counts = torch.bincount(groups.reshape(-1), minlength=num_groups + 1)[:num_groups]
    active = counts[counts > 0].float()
    if active.numel() == 0:
        return {
            "active_groups": 0,
            "min_m_g": 0,
            "max_m_g": 0,
            "mean_m_g": 0.0,
            "cv_m_g": None,
            "equal_m_g": False,
        }
    mean = active.mean()
    return {
        "active_groups": int(active.numel()),
        "min_m_g": int(active.min()),
        "max_m_g": int(active.max()),
        "mean_m_g": float(mean),
        "cv_m_g": float(active.std(unbiased=False) / mean),
        "equal_m_g": bool(active.min() == active.max()),
    }


def _build_bmm_plan(
    groups: torch.Tensor,
    *,
    num_groups: int,
) -> RegularBmmPlan:
    flat = groups.reshape(-1)
    metrics = _route_metrics(groups, num_groups)
    equal = bool(metrics["equal_m_g"])
    active_count = int(metrics["active_groups"])
    if not equal or active_count < 2:
        reason = "nonuniform_M_g" if not equal else "fewer_than_two_nonempty_groups"
        return RegularBmmPlan(False, reason, None, None, 0, None, metrics)
    sorted_groups, order = torch.sort(flat, stable=True)
    valid = int((sorted_groups < num_groups).sum().item())
    active_groups = torch.unique_consecutive(sorted_groups[:valid])
    m_per_group = int(metrics["min_m_g"])
    if valid != active_groups.numel() * m_per_group:
        return RegularBmmPlan(
            False, "shape_product_mismatch", None, None, valid, None, metrics
        )
    return RegularBmmPlan(
        True,
        "all_nonempty_groups_have_equal_M_g",
        order[:valid],
        active_groups,
        valid,
        m_per_group,
        metrics,
    )


def _randn(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    generator: torch.Generator,
    scale: float,
) -> torch.Tensor:
    return (
        torch.randn(shape, device=device, generator=generator, dtype=torch.bfloat16)
        * scale
    )


def _build_fixture(case: FamilyCase, device: torch.device) -> Fixture:
    generator = torch.Generator(device=device).manual_seed(case.seed + 1000)
    mapping = _make_mapping(case, device)
    topk_ids = _make_topk(case, device)
    weight_gen = torch.Generator(device=device).manual_seed(1701)
    topk_weights = torch.rand(
        (case.tokens, case.top_k), device=device, generator=weight_gen
    )
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    groups = _virtual_groups(
        topk_ids, mapping, experts=case.experts, adapters=case.adapters
    )
    segment_plan = allocate_segment_plan(
        case.pairs, case.experts, case.adapters, 16, device
    )
    build_segment_plan_into(segment_plan, topk_ids, mapping)
    aligned_plan = _build_aligned_plan(
        topk_ids,
        mapping,
        experts=case.experts,
        adapters=case.adapters,
        block_m=16,
    )
    bmm_plan = _build_bmm_plan(groups, num_groups=case.groups)
    torch.cuda.synchronize()
    return Fixture(
        case=case,
        device=device,
        hidden_states=_randn(
            (case.tokens, case.hidden),
            device=device,
            generator=generator,
            scale=0.1,
        ),
        activated_pairs=_randn(
            (case.pairs, case.intermediate),
            device=device,
            generator=generator,
            scale=0.1,
        ),
        gateup_base=_randn(
            (case.pairs, 2 * case.intermediate),
            device=device,
            generator=generator,
            scale=0.1,
        ),
        base_down_pairs=_randn(
            (case.pairs, case.hidden),
            device=device,
            generator=generator,
            scale=0.1,
        ),
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        mapping=mapping,
        gate_a=_randn(
            (case.groups, 2 * case.rank, case.hidden),
            device=device,
            generator=generator,
            scale=0.01,
        ),
        gate_b=_randn(
            (case.groups, 2 * case.intermediate, case.rank),
            device=device,
            generator=generator,
            scale=0.01,
        ),
        down_a=_randn(
            (case.groups, case.rank, case.intermediate),
            device=device,
            generator=generator,
            scale=0.01,
        ),
        down_b=_randn(
            (case.groups, case.hidden, case.rank),
            device=device,
            generator=generator,
            scale=0.01,
        ),
        gate_rank_input=_randn(
            (case.pairs, 2 * case.rank),
            device=device,
            generator=generator,
            scale=0.05,
        ),
        down_rank_input=_randn(
            (case.pairs, case.rank),
            device=device,
            generator=generator,
            scale=0.05,
        ),
        aligned_plan=aligned_plan,
        segment_plan=segment_plan,
        bmm_plan=bmm_plan,
    )


def _rebuild_segment(fixture: Fixture) -> None:
    build_segment_plan_into(fixture.segment_plan, fixture.topk_ids, fixture.mapping)


def _fresh_aligned(fixture: Fixture) -> AlignedPlan:
    case = fixture.case
    return _build_aligned_plan(
        fixture.topk_ids,
        fixture.mapping,
        experts=case.experts,
        adapters=case.adapters,
        block_m=fixture.aligned_plan.block_m,
    )


def _invoke_aligned_a(
    fixture: Fixture,
    x: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    *,
    input_pair_major: bool,
    plan: AlignedPlan,
    split_k: int,
) -> None:
    from sglang.srt.lora.sgl_lora.triton_ops.virtual_experts import (
        _moe_lora_shrink_splitk_kernel,
    )

    n_size, k_size = weight.shape[1:]
    block_m = plan.block_m
    block_n = min(_TUNING.aligned_bn, triton.next_power_of_2(n_size))
    block_k = _TUNING.aligned_bk
    # The aligned route has no writer for base-only/sentinel rows.  Clearing is
    # part of the operator boundary for every split count, not just split-K.
    output.zero_()
    grid = (
        split_k
        * triton.cdiv(plan.sorted_pair_ids.shape[0], block_m)
        * triton.cdiv(n_size, block_n),
    )
    _moe_lora_shrink_splitk_kernel[grid](
        x,
        weight,
        output,
        plan.sorted_pair_ids,
        plan.expert_ids,
        plan.num_pairs_post_padded,
        n_size,
        k_size,
        fixture.case.pairs,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        output.stride(0),
        output.stride(1),
        top_k=1 if input_pair_major else fixture.case.top_k,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        GROUP_SIZE_M=1,
        SPLIT_K=split_k,
        ENABLE_PDL=False,
        num_warps=_TUNING.aligned_warps,
        num_stages=3,
    )


def _invoke_aligned_b(
    fixture: Fixture,
    intermediate: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    *,
    plan: AlignedPlan,
) -> None:
    from sglang.srt.lora.sgl_lora.triton_ops.expand import (
        invoke_moe_lora_expand_add,
    )

    invoke_moe_lora_expand_add(
        intermediate,
        weight,
        output,
        fixture.topk_weights,
        fixture.topk_ids,
        plan.sorted_pair_ids,
        plan.expert_ids,
        plan.num_pairs_post_padded,
        {
            "BLOCK_SIZE_M": plan.block_m,
            "BLOCK_SIZE_N": _TUNING.aligned_bn,
            "GROUP_SIZE_M": 1,
            "num_warps": _TUNING.aligned_warps,
        },
        False,
        False,
        num_output_slices=1,
    )


def _bmm_gemm(
    fixture: Fixture,
    x: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    *,
    input_pair_major: bool,
) -> None:
    plan = fixture.bmm_plan
    if not plan.eligible:
        raise RuntimeError(plan.reason)
    assert plan.sorted_pair_ids is not None
    assert plan.active_groups is not None
    assert plan.m_per_group is not None
    order = plan.sorted_pair_ids
    input_rows = order if input_pair_major else order // fixture.case.top_k
    packed_x = x[input_rows].reshape(
        plan.active_groups.numel(), plan.m_per_group, x.shape[1]
    )
    packed_w = weight[plan.active_groups].transpose(1, 2)
    packed_out = torch.bmm(packed_x, packed_w).reshape(plan.valid_pairs, -1)
    output.zero_()
    output.index_copy_(0, order, packed_out)


def _runtime_bmm_order(fixture: Fixture) -> torch.Tensor:
    """Rebuild the valid equal-M_g order inside an O0 timed region."""
    case = fixture.case
    groups = _virtual_groups(
        fixture.topk_ids,
        fixture.mapping,
        experts=case.experts,
        adapters=case.adapters,
    ).reshape(-1)
    _, order = torch.sort(groups, stable=True)
    return order[: fixture.bmm_plan.valid_pairs]


def _reference_gemm(
    fixture: Fixture,
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    input_pair_major: bool,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    case = fixture.case
    out = torch.zeros(
        (case.pairs, weight.shape[1]), dtype=output_dtype, device=fixture.device
    )
    groups = _virtual_groups(
        fixture.topk_ids,
        fixture.mapping,
        experts=case.experts,
        adapters=case.adapters,
    ).reshape(-1)
    valid_pairs = torch.nonzero(groups < case.groups).flatten()
    # Bound gathered FP32 factors to roughly 64 MiB.
    bytes_per_pair = weight.shape[1] * weight.shape[2] * 4
    chunk = max(1, min(64, (64 * 1024 * 1024) // bytes_per_pair))
    for start in range(0, valid_pairs.numel(), chunk):
        pairs = valid_pairs[start : start + chunk]
        rows = pairs if input_pair_major else pairs // case.top_k
        xv = x[rows].float().unsqueeze(1)
        wv = weight[groups[pairs]].float()
        values = torch.bmm(xv, wv.transpose(1, 2)).squeeze(1)
        out[pairs] = values.to(output_dtype)
    return out


def _strict_check(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    name: str,
    atol: float = 3e-2,
    rtol: float = 3e-2,
) -> dict[str, float]:
    torch.cuda.synchronize()
    torch.testing.assert_close(
        actual, expected, atol=atol, rtol=rtol, check_dtype=False
    )
    error = (actual.float() - expected.float()).abs()
    signal = float(expected.float().abs().max().item())
    max_error = float(error.max().item())
    return {
        "max_abs_error": max_error,
        "mean_abs_error": float(error.mean().item()),
        "reference_max_abs": signal,
        "error_over_signal": max_error / signal if signal else 0.0,
        "atol": atol,
        "rtol": rtol,
    }


def _site_gate_or_down_a(
    fixture: Fixture,
    site: str,
    family: str,
    *,
    scope: str,
) -> tuple[Callable[[], None], torch.Tensor, dict[str, object]]:
    case = fixture.case
    if site == "gate_a":
        x, weight, n_size, pair_major = (
            fixture.hidden_states,
            fixture.gate_a,
            2 * case.rank,
            False,
        )
    else:
        x, weight, n_size, pair_major = (
            fixture.activated_pairs,
            fixture.down_a,
            case.rank,
            True,
        )
    output = torch.empty(
        (case.pairs, n_size), dtype=torch.bfloat16, device=fixture.device
    )
    metadata: dict[str, object] = {}
    if family == "indexed":

        def invoke() -> None:
            output.zero_()
            indexed_gemm(
                x,
                weight,
                fixture.topk_ids,
                fixture.mapping,
                output,
                num_experts=case.experts,
                num_adapters=case.adapters,
                input_pair_major=pair_major,
                block_n=_TUNING.indexed_bn,
                block_k=_TUNING.indexed_bk,
                num_warps=_TUNING.indexed_warps,
            )

    elif family == "cutedsl":
        from benchmark.kernels.lora_moe.cutedsl_indexed import (
            capability,
            cutedsl_indexed_gemm,
        )

        cap = capability()
        if not cap["available"]:
            raise RuntimeError(
                f"UNSUPPORTED:cutedsl_toolchain:{cap['error_type']}:{cap['error']}"
            )

        def invoke() -> None:
            cutedsl_indexed_gemm(
                x,
                weight,
                fixture.topk_ids,
                fixture.mapping,
                output,
                experts=case.experts,
                adapters=case.adapters,
                input_pair_major=pair_major,
            )

        metadata.update(
            route_representation="inline_raw",
            implementation="cutedsl_scalar_indexed_capability_probe",
            capability=cap,
        )
    elif family == "segmented":

        def invoke() -> None:
            if scope == "O0":
                _rebuild_segment(fixture)
            output.zero_()
            segmented_gemm(
                x,
                weight,
                output,
                fixture.segment_plan,
                top_k=case.top_k,
                input_pair_major=pair_major,
                block_n=_TUNING.segmented_bn,
                block_k=_TUNING.segmented_bk,
                num_warps=_TUNING.segmented_warps,
            )

        metadata["route_representation"] = "compact_segment_block_descriptors"
    elif family == "aligned":
        # Split-K candidates were pre-screened on these skinny-N A shapes.  The
        # rank tier keeps enough blocks resident without changing the route.
        split_k = 4 if case.tokens <= 32 else 2 if case.tokens <= 256 else 1

        def invoke() -> None:
            plan = _fresh_aligned(fixture) if scope == "O0" else fixture.aligned_plan
            _invoke_aligned_a(
                fixture,
                x,
                weight,
                output,
                input_pair_major=pair_major,
                plan=plan,
                split_k=split_k,
            )

        metadata.update(
            route_representation="padded_aligned_virtual_expert",
            split_k=split_k,
        )
    elif family == "bmm":
        if not fixture.bmm_plan.eligible:
            raise RuntimeError(f"BMM_DISQUALIFIED:{fixture.bmm_plan.reason}")

        def invoke() -> None:
            if scope == "O0":
                # Charge raw virtual ids + stable sort.  Equal group shapes are
                # static, so the same fixed slicing is graph-capturable.
                order = _runtime_bmm_order(fixture)
                original = fixture.bmm_plan.sorted_pair_ids
                fixture.bmm_plan.sorted_pair_ids = order
                _bmm_gemm(fixture, x, weight, output, input_pair_major=pair_major)
                fixture.bmm_plan.sorted_pair_ids = original
            else:
                _bmm_gemm(fixture, x, weight, output, input_pair_major=pair_major)

        metadata["route_representation"] = "equal_M_g_dense_batch"
    else:
        raise ValueError(family)
    reference = _reference_gemm(
        fixture, x, weight, input_pair_major=pair_major, output_dtype=output.dtype
    )
    return invoke, reference, {"output": output, **metadata}


def _reference_gate_consumer(
    fixture: Fixture,
) -> tuple[torch.Tensor, torch.Tensor]:
    case = fixture.case
    gate_delta = torch.zeros_like(fixture.gateup_base)
    groups = _virtual_groups(
        fixture.topk_ids,
        fixture.mapping,
        experts=case.experts,
        adapters=case.adapters,
    ).reshape(-1)
    valid_pairs = torch.nonzero(groups < case.groups).flatten()
    # Gate and up are two independent R-wide inputs applied to two I-wide
    # slices of the stacked B factor.
    for start in range(0, valid_pairs.numel(), 64):
        pairs = valid_pairs[start : start + 64]
        ranks = fixture.gate_rank_input[pairs].float()
        weights = fixture.gate_b[groups[pairs]].float()
        gate = torch.bmm(
            ranks[:, None, : case.rank],
            weights[:, : case.intermediate].transpose(1, 2),
        ).squeeze(1)
        up = torch.bmm(
            ranks[:, None, case.rank :],
            weights[:, case.intermediate :].transpose(1, 2),
        ).squeeze(1)
        gate_delta[pairs] = torch.cat((gate, up), dim=-1).to(torch.bfloat16)
    combined = fixture.gateup_base.float() + gate_delta.float()
    gate, up = combined.chunk(2, dim=-1)
    act = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
    down_rank = _reference_gemm(
        fixture,
        act,
        fixture.down_a,
        input_pair_major=True,
        output_dtype=torch.bfloat16,
    )
    # Base-only rows run activation but suppress down LoRA-A.
    base_rows = fixture.mapping < 0
    if bool(base_rows.any()):
        down_rank.view(case.tokens, case.top_k, case.rank)[base_rows].zero_()
    return act, down_rank


def _site_gate_consumer(
    fixture: Fixture,
    family: str,
    *,
    scope: str,
    accumulation: str,
) -> tuple[Callable[[], None], tuple[torch.Tensor, torch.Tensor], dict[str, object]]:
    from sglang.srt.lora.sgl_lora.triton_ops.fused_c2 import (
        fused_gate_up_b_swiglu_down_a,
        fused_gate_up_b_swiglu_down_a_aligned,
    )

    case = fixture.case
    act = torch.empty_like(fixture.activated_pairs)
    rank_dtype = torch.float32 if accumulation == "fp32" else torch.bfloat16
    down_rank = torch.empty(
        (case.pairs, case.rank), dtype=rank_dtype, device=fixture.device
    )
    gate_b_4d = fixture.gate_b.view(
        case.adapters, case.experts, 2 * case.intermediate, case.rank
    )
    down_a_4d = fixture.down_a.view(
        case.adapters, case.experts, case.rank, case.intermediate
    )
    src2dst = torch.arange(case.pairs, device=fixture.device, dtype=torch.int32)
    metadata: dict[str, object] = {}
    if family == "indexed":
        if accumulation == "fp32":
            raise RuntimeError("UNSUPPORTED:current_pair_consumer_bf16_destination")

        def invoke() -> None:
            down_rank.zero_()
            fused_gate_up_b_swiglu_down_a(
                fixture.gateup_base,
                fixture.gate_rank_input,
                gate_b_4d,
                down_a_4d,
                act,
                down_rank,
                src2dst,
                fixture.topk_ids,
                fixture.mapping,
                block_size_n=_TUNING.pair_consumer_bn,
                num_warps=_TUNING.consumer_warps,
            )

        metadata["consumer"] = "current_pair_owned_fused_C2P"
    elif family == "aligned":
        if accumulation == "fp32":
            raise RuntimeError("UNSUPPORTED:current_aligned_consumer_bf16_destination")

        def invoke() -> None:
            down_rank.zero_()
            plan = _fresh_aligned(fixture) if scope == "O0" else fixture.aligned_plan
            # The native aligned route omits its sentinel bucket from the
            # post-padded work count.  Preserve base-only activation rows with
            # an explicit prepass; this cost is part of the mixed-row contract.
            if case.adapter_mode == "mixed":
                base_only_swiglu(
                    fixture.gateup_base,
                    fixture.mapping,
                    act,
                    top_k=case.top_k,
                )
            fused_gate_up_b_swiglu_down_a_aligned(
                fixture.gateup_base,
                fixture.gate_rank_input,
                gate_b_4d,
                down_a_4d,
                act,
                down_rank,
                src2dst,
                fixture.topk_ids,
                plan.sorted_pair_ids,
                plan.expert_ids,
                plan.num_pairs_post_padded,
                route_block_size_m=plan.block_m,
                block_size_n=_TUNING.aligned_consumer_bn,
                num_warps=_TUNING.consumer_warps,
            )

        metadata["consumer"] = "current_padded_aligned_fused_C2P"
    elif family == "segmented":

        def invoke() -> None:
            if scope == "O0":
                _rebuild_segment(fixture)
            down_rank.zero_()
            if case.adapter_mode == "mixed":
                base_only_swiglu(
                    fixture.gateup_base,
                    fixture.mapping,
                    act,
                    top_k=case.top_k,
                )
            segmented_gate_b_consumer(
                fixture.gateup_base,
                fixture.gate_rank_input,
                fixture.gate_b,
                fixture.down_a,
                act,
                down_rank,
                fixture.segment_plan,
                block_n=_TUNING.segmented_consumer_bn,
                num_warps=_TUNING.consumer_warps,
                round_gate_delta_to_bf16=True,
                round_down_partial_to_bf16=accumulation == "bf16",
            )

        metadata["consumer"] = "compact_segment_block_fused_C2P"
    elif family == "bmm":
        if not fixture.bmm_plan.eligible:
            raise RuntimeError(f"BMM_DISQUALIFIED:{fixture.bmm_plan.reason}")
        plan = fixture.bmm_plan
        assert plan.sorted_pair_ids is not None
        assert plan.active_groups is not None
        assert plan.m_per_group is not None

        def invoke() -> None:
            order = (
                _runtime_bmm_order(fixture) if scope == "O0" else plan.sorted_pair_ids
            )
            packed_gate_rank = fixture.gate_rank_input[order].reshape(
                plan.active_groups.numel(), plan.m_per_group, 2 * case.rank
            )
            packed_gate = packed_gate_rank[..., : case.rank]
            packed_up = packed_gate_rank[..., case.rank :]
            weights = fixture.gate_b[plan.active_groups]
            gate_delta = torch.bmm(
                packed_gate, weights[:, : case.intermediate].transpose(1, 2)
            )
            up_delta = torch.bmm(
                packed_up, weights[:, case.intermediate :].transpose(1, 2)
            )
            base = fixture.gateup_base[order].reshape(
                plan.active_groups.numel(), plan.m_per_group, 2 * case.intermediate
            )
            gate = base[..., : case.intermediate].float() + gate_delta.float()
            up = base[..., case.intermediate :].float() + up_delta.float()
            packed_act = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
            packed_down = torch.bmm(
                packed_act, fixture.down_a[plan.active_groups].transpose(1, 2)
            )
            act.zero_()
            down_rank.zero_()
            if case.adapter_mode == "mixed":
                base_only_swiglu(
                    fixture.gateup_base,
                    fixture.mapping,
                    act,
                    top_k=case.top_k,
                )
            act.index_copy_(0, order, packed_act.reshape(plan.valid_pairs, -1))
            down_rank.index_copy_(
                0, order, packed_down.reshape(plan.valid_pairs, -1).to(rank_dtype)
            )

        metadata["consumer"] = "equal_M_g_bmm_pack_consume_scatter"
    else:
        raise ValueError(family)
    return (
        invoke,
        _reference_gate_consumer(fixture),
        {
            "act": act,
            "down_rank": down_rank,
            **metadata,
        },
    )


def _reference_down_finalize(
    fixture: Fixture,
    *,
    rank_input: torch.Tensor,
    round_delta_to_bf16: bool,
) -> torch.Tensor:
    case = fixture.case
    delta = _reference_gemm(
        fixture,
        rank_input,
        fixture.down_b,
        input_pair_major=True,
        output_dtype=torch.float32,
    )
    if round_delta_to_bf16:
        delta = delta.to(torch.bfloat16).float()
    pair_values = fixture.base_down_pairs.float() + delta
    return (
        pair_values.view(case.tokens, case.top_k, case.hidden)
        * fixture.topk_weights[..., None]
    ).sum(dim=1)


def _site_down_finalize(
    fixture: Fixture,
    family: str,
    *,
    scope: str,
    accumulation: str,
) -> tuple[Callable[[], None], torch.Tensor, dict[str, object]]:
    case = fixture.case
    output = torch.empty(
        (case.tokens, case.hidden), dtype=torch.float32, device=fixture.device
    )
    delta_dtype = torch.float32 if accumulation == "fp32" else torch.bfloat16
    delta_pairs = torch.empty(
        (case.pairs, case.hidden), dtype=delta_dtype, device=fixture.device
    )
    round_bf16 = accumulation == "bf16"
    rank_input = fixture.down_rank_input
    metadata: dict[str, object] = {}
    if family == "indexed":

        def invoke() -> None:
            direct_down_b_finalize(
                fixture.base_down_pairs,
                rank_input,
                fixture.down_b,
                fixture.topk_ids,
                fixture.mapping,
                fixture.topk_weights,
                output,
                num_experts=case.experts,
                num_adapters=case.adapters,
                round_delta_to_bf16=round_bf16,
                block_h=_TUNING.direct_finalize_bh,
                num_warps=_TUNING.consumer_warps,
            )

        metadata["consumer"] = "token_owned_direct_fused_down_B_finalize"
    elif family == "segmented":

        def invoke() -> None:
            if scope == "O0":
                _rebuild_segment(fixture)
            delta_pairs.zero_()
            segmented_gemm(
                rank_input,
                fixture.down_b,
                delta_pairs,
                fixture.segment_plan,
                top_k=case.top_k,
                input_pair_major=True,
                block_n=_TUNING.segmented_bn,
                block_k=_TUNING.segmented_bk,
                num_warps=_TUNING.segmented_warps,
            )
            reduce_pairs_finalize(
                fixture.base_down_pairs,
                delta_pairs,
                fixture.topk_weights,
                output,
                round_delta_to_bf16=False,
                block_h=_TUNING.reduce_finalize_bh,
            )

        metadata["consumer"] = "segmented_down_B_materialize_then_finalize"
    elif family == "aligned":

        def invoke() -> None:
            plan = _fresh_aligned(fixture) if scope == "O0" else fixture.aligned_plan
            delta_pairs.zero_()
            _invoke_aligned_b(
                fixture, rank_input, fixture.down_b, delta_pairs, plan=plan
            )
            reduce_pairs_finalize(
                fixture.base_down_pairs,
                delta_pairs,
                fixture.topk_weights,
                output,
                round_delta_to_bf16=False,
                block_h=_TUNING.reduce_finalize_bh,
            )

        metadata["consumer"] = "current_aligned_down_B_materialize_then_finalize"
    elif family == "bmm":
        if not fixture.bmm_plan.eligible:
            raise RuntimeError(f"BMM_DISQUALIFIED:{fixture.bmm_plan.reason}")
        plan = fixture.bmm_plan
        assert plan.sorted_pair_ids is not None
        assert plan.active_groups is not None
        assert plan.m_per_group is not None

        def invoke() -> None:
            order = (
                _runtime_bmm_order(fixture) if scope == "O0" else plan.sorted_pair_ids
            )
            packed_rank = rank_input[order].reshape(
                plan.active_groups.numel(), plan.m_per_group, case.rank
            )
            if accumulation == "fp32":
                packed_delta = torch.bmm(
                    packed_rank.float(),
                    fixture.down_b[plan.active_groups].float().transpose(1, 2),
                )
            else:
                packed_delta = torch.bmm(
                    packed_rank,
                    fixture.down_b[plan.active_groups].transpose(1, 2),
                )
            delta_pairs.zero_()
            delta_pairs.index_copy_(
                0, order, packed_delta.reshape(plan.valid_pairs, -1).to(delta_dtype)
            )
            reduce_pairs_finalize(
                fixture.base_down_pairs,
                delta_pairs,
                fixture.topk_weights,
                output,
                round_delta_to_bf16=False,
                block_h=_TUNING.reduce_finalize_bh,
            )

        metadata["consumer"] = "equal_M_g_bmm_pack_down_B_scatter_finalize"
    elif family in ("one_shot_bf16", "one_shot_fp32"):
        pair_delta = torch.empty(
            (case.pairs, case.hidden), dtype=torch.float32, device=fixture.device
        )
        round_rank = family.endswith("bf16")

        def invoke() -> None:
            pair_delta.zero_()
            one_shot_down_ab(
                fixture.activated_pairs,
                fixture.down_a,
                fixture.down_b,
                fixture.topk_ids,
                fixture.mapping,
                pair_delta,
                num_experts=case.experts,
                num_adapters=case.adapters,
                round_rank_to_bf16=round_rank,
                block_i=_TUNING.one_shot_bi,
                block_h=_TUNING.one_shot_bh,
            )
            reduce_pairs_finalize(
                fixture.base_down_pairs,
                pair_delta,
                fixture.topk_weights,
                output,
                round_delta_to_bf16=False,
                block_h=_TUNING.reduce_finalize_bh,
            )

        rank_input = _reference_gemm(
            fixture,
            fixture.activated_pairs,
            fixture.down_a,
            input_pair_major=True,
            output_dtype=torch.float32,
        )
        if round_rank:
            rank_input = rank_input.to(torch.bfloat16)
        metadata.update(
            consumer="one_shot_down_A_B_then_finalize",
            intermediate_rank_materialized=False,
            rank_accumulation="bf16_boundary" if round_rank else "fp32",
        )
    else:
        raise ValueError(family)
    reference = _reference_down_finalize(
        fixture,
        rank_input=rank_input,
        round_delta_to_bf16=round_bf16,
    )
    return invoke, reference, {"output": output, **metadata}


def _prepare_site(
    fixture: Fixture,
    site: str,
    family: str,
    *,
    scope: str,
    accumulation: str,
):
    if site in ("gate_a", "down_a"):
        return _site_gate_or_down_a(fixture, site, family, scope=scope)
    if site == "gate_consumer":
        return _site_gate_consumer(
            fixture, family, scope=scope, accumulation=accumulation
        )
    if site == "down_finalize":
        return _site_down_finalize(
            fixture, family, scope=scope, accumulation=accumulation
        )
    raise ValueError(site)


def _check_site(
    site: str,
    invoke: Callable[[], None],
    reference,
    outputs: dict[str, object],
) -> dict[str, object]:
    invoke()
    torch.cuda.synchronize()
    if site == "gate_consumer":
        act_ref, rank_ref = reference
        return {
            "act": _strict_check(outputs["act"], act_ref, name="act"),
            "down_rank": _strict_check(
                outputs["down_rank"], rank_ref, name="down_rank", atol=5e-2
            ),
        }
    return _strict_check(outputs["output"], reference, name=site, atol=5e-2)


def _time_one(
    invoke: Callable[[], None],
    *,
    site: str,
    reference,
    outputs: dict[str, object],
    execution: str,
    cache: _CacheControl,
    warmup: int,
    samples: int,
) -> tuple[dict[str, object], dict[str, float] | None]:
    batch = make_batch(invoke, execution=execution, inner_iterations=1)
    graph_check = None
    if execution == "cuda_graph":
        # Replay is checked independently rather than only compared to eager;
        # this catches a graph that faithfully replays an already-wrong result.
        graph_check = {
            "replay_completed": True,
            "independent_oracle": _check_site(site, batch.run, reference, outputs),
        }
    timing = time_cuda_events(
        batch.run,
        launches_per_batch=1,
        warmup=warmup,
        samples=samples,
        before_sample=cache.before_sample(),
    )
    return asdict(timing), graph_check


def _environment(device: torch.device) -> dict[str, object]:
    props = torch.cuda.get_device_properties(device)
    return {
        "device_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "torch": torch.__version__,
        "triton": getattr(triton, "__version__", "unknown"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(","))
    if not result or any(item <= 0 for item in result):
        raise ValueError("expected comma-separated positive integers")
    return result


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", default="1,32,256,2048")
    parser.add_argument("--ranks", default="16,32,64,128")
    parser.add_argument("--sites", default=",".join(SITES))
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--scope", choices=("K0", "O0"), default="K0")
    parser.add_argument(
        "--execution", choices=("eager", "cuda_graph", "both"), default="both"
    )
    parser.add_argument(
        "--cache-state", choices=("hot", "cold", "both"), default="both"
    )
    parser.add_argument("--adapter-mode", choices=("multi", "mixed"), default="mixed")
    parser.add_argument(
        "--route-pattern", choices=("regular", "iid", "skewed"), default="regular"
    )
    parser.add_argument(
        "--accumulation", choices=("bf16", "fp32", "both"), default="bf16"
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward")
    for field_name, field in KernelTuning.__dataclass_fields__.items():
        parser.add_argument(
            f"--{field_name.replace('_', '-')}",
            type=int,
            default=field.default,
        )
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    for field_name in KernelTuning.__dataclass_fields__:
        value = getattr(args, field_name)
        if value <= 0:
            raise ValueError(f"{field_name} must be positive")
        setattr(_TUNING, field_name, value)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    tokens = _parse_csv_ints(args.tokens)
    ranks = _parse_csv_ints(args.ranks)
    sites = tuple(args.sites.split(","))
    if any(site not in SITES for site in sites):
        raise ValueError(f"sites must be selected from {SITES}")
    families = tuple(args.families.split(","))
    allowed_families = (*FAMILIES, "cutedsl", "one_shot_bf16", "one_shot_fp32")
    if any(family not in allowed_families for family in families):
        raise ValueError(f"families must be selected from {allowed_families}")
    executions = (
        ("eager", "cuda_graph") if args.execution == "both" else (args.execution,)
    )
    cache_states = (
        ("hot", "cold") if args.cache_state == "both" else (args.cache_state,)
    )
    accumulations = (
        ("bf16", "fp32") if args.accumulation == "both" else (args.accumulation,)
    )
    if args.order == "reverse":
        families = tuple(reversed(families))
        executions = tuple(reversed(executions))
        cache_states = tuple(reversed(cache_states))
        accumulations = tuple(reversed(accumulations))

    report: dict[str, object] = {
        "schema_version": 1,
        "scope": "benchmark_only_bf16_moe_lora_algorithm_family_matrix",
        "environment": _environment(device),
        "kernel_tuning": asdict(_TUNING),
        "measurement_order": args.order,
        "contract": {
            "route_domain": "canonical_token_topk_pairs",
            "virtual_group": "adapter_times_E_local_plus_local_expert",
            "route_build_charged_in": "O0",
            "route_prebuilt_in": "K0",
            "clears_casts_pack_scatter_consumers_charged": True,
            "production_dispatch_changed": False,
        },
        "runs": [],
        "disqualifications": [],
    }
    runs: list[dict[str, object]] = report["runs"]  # type: ignore[assignment]
    disqualifications: list[dict[str, object]] = report["disqualifications"]  # type: ignore[assignment]
    for token_count in tokens:
        for rank in ranks:
            case = FamilyCase(
                tokens=token_count,
                rank=rank,
                adapter_mode=args.adapter_mode,
                route_pattern=args.route_pattern,
            )
            fixture = _build_fixture(case, device)
            for site in sites:
                site_families = families
                if site == "down_finalize" and set(families) == set(FAMILIES):
                    site_families = (*families, "one_shot_bf16", "one_shot_fp32")
                for family in site_families:
                    for accumulation in accumulations:
                        if site in ("gate_a", "down_a") and accumulation == "fp32":
                            continue
                        try:
                            invoke, reference, outputs = _prepare_site(
                                fixture,
                                site,
                                family,
                                scope=args.scope,
                                accumulation=accumulation,
                            )
                            correctness = (
                                None
                                if args.skip_check
                                else _check_site(site, invoke, reference, outputs)
                            )
                        except RuntimeError as exc:
                            reason = str(exc)
                            if not (
                                reason.startswith("BMM_DISQUALIFIED:")
                                or reason.startswith("UNSUPPORTED:")
                            ):
                                raise
                            row = {
                                "case": asdict(case),
                                "site": site,
                                "family": family,
                                "accumulation": accumulation,
                                "scope": args.scope,
                                "status": "disqualified",
                                "reason": reason,
                                "route_metrics": fixture.bmm_plan.metrics,
                            }
                            disqualifications.append(row)
                            print(
                                f"T={token_count} R={rank} {site}/{family}/{accumulation}: {reason}"
                            )
                            continue
                        for execution in executions:
                            for cache_state in cache_states:
                                cache = _make_cache_control(cache_state, device)
                                try:
                                    timing, graph_check = _time_one(
                                        invoke,
                                        site=site,
                                        reference=reference,
                                        outputs=outputs,
                                        execution=execution,
                                        cache=cache,
                                        warmup=args.warmup,
                                        samples=args.samples,
                                    )
                                    status = "ok"
                                    error = None
                                except Exception as exc:
                                    if execution != "cuda_graph":
                                        raise
                                    timing = None
                                    graph_check = None
                                    status = "unsupported"
                                    error = f"{type(exc).__name__}: {exc}"
                                row = {
                                    "case": asdict(case),
                                    "site": site,
                                    "family": family,
                                    "accumulation": accumulation,
                                    "scope": args.scope,
                                    "execution": execution,
                                    "cache_state": cache_state,
                                    "status": status,
                                    "error": error,
                                    "correctness": correctness,
                                    "route_metrics": fixture.bmm_plan.metrics,
                                    "bmm_admission": {
                                        "eligible": fixture.bmm_plan.eligible,
                                        "reason": fixture.bmm_plan.reason,
                                    },
                                    "implementation": {
                                        key: value
                                        for key, value in outputs.items()
                                        if not isinstance(value, torch.Tensor)
                                    },
                                    "timing": timing,
                                    "graph_correctness": graph_check,
                                    "cache_control": cache.metadata(),
                                }
                                runs.append(row)
                                timing_text = (
                                    f"p50={timing['p50_us']:.3f}us" if timing else error
                                )
                                print(
                                    f"T={token_count} R={rank} {site}/{family}/{accumulation} "
                                    f"{args.scope}/{execution}/{cache_state}: {timing_text}"
                                )
            del fixture
            torch.cuda.empty_cache()

    winners: list[dict[str, object]] = []
    keys = {
        (
            row["case"]["tokens"],  # type: ignore[index]
            row["case"]["rank"],  # type: ignore[index]
            row["site"],
            row["accumulation"],
            row["execution"],
            row["cache_state"],
        )
        for row in runs
        if row["status"] == "ok"
    }
    for key in sorted(keys, key=str):
        matched = [
            row
            for row in runs
            if row["status"] == "ok"
            and (
                row["case"]["tokens"],  # type: ignore[index]
                row["case"]["rank"],  # type: ignore[index]
                row["site"],
                row["accumulation"],
                row["execution"],
                row["cache_state"],
            )
            == key
        ]
        winner = min(matched, key=lambda row: row["timing"]["p50_us"])  # type: ignore[index]
        winners.append(
            {
                "tokens": key[0],
                "rank": key[1],
                "site": key[2],
                "accumulation": key[3],
                "execution": key[4],
                "cache_state": key[5],
                "winner": winner["family"],
                "p50_us": winner["timing"]["p50_us"],  # type: ignore[index]
                "runner_up_pct": (
                    (
                        sorted(row["timing"]["p50_us"] for row in matched)[1]  # type: ignore[index]
                        / winner["timing"]["p50_us"]  # type: ignore[index]
                        - 1.0
                    )
                    * 100.0
                    if len(matched) > 1
                    else None
                ),
            }
        )
    report["winners"] = winners
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
