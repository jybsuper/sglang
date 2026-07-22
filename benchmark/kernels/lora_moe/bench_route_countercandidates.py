#!/usr/bin/env python3
"""Benchmark current and counter-candidate SGL LoRA MoE route plans.

The matrix measures device time.  O0 builds every plan used by one logical
layer; M0 charges all producer work over ``--macro-layers``; K0 is explicitly
prebuilt and isolates the common synthetic plan consumer.  Candidate order is
rotated and reversed across rounds to counter warm/cold and drift bias.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch

from route_countercandidates import (
    RoutePlan,
    RoutePlanMemo,
    build_current_sgl_plan,
    build_legacy_merged_plan,
    consume_route_plan,
    make_route_plan_key,
    merged_align_supported,
)

VARIANTS = (
    "current_sgl",
    "legacy_merged",
    "prefill_reuse",
    "cross_layer_memo",
)


@dataclass(frozen=True)
class BenchCase:
    tokens: int
    max_loras: int
    num_experts: int
    local_num_experts: int
    distribution: str
    top_k: int = 8

    @property
    def block_a(self) -> int:
        return 16 if self.tokens < 512 else 32

    @property
    def block_b(self) -> int:
        return 64

    @property
    def common_block(self) -> int:
        return self.block_b

    @property
    def case_id(self) -> str:
        return (
            f"T{self.tokens}_L{self.max_loras}_E{self.num_experts}_"
            f"EL{self.local_num_experts}_{self.distribution}"
        )


def all_cases() -> list[BenchCase]:
    return [
        BenchCase(t, l, e, el, dist)
        for t in (1, 32, 256, 2048)
        for l in (1, 8)
        for e, el in ((32, 32), (256, 32))
        for dist in ("iid", "skew", "base_rows")
    ]


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _stats(samples: list[float], *, divisor: int = 1) -> dict:
    normalized = [x / divisor for x in samples]
    return {
        "samples_us": samples,
        "logical_layer_divisor": divisor,
        "median_us_per_layer": statistics.median(normalized),
        "mean_us_per_layer": statistics.fmean(normalized),
        "p10_us_per_layer": _percentile(normalized, 0.10),
        "p90_us_per_layer": _percentile(normalized, 0.90),
        "min_us_per_layer": min(normalized),
        "max_us_per_layer": max(normalized),
    }


def _counterbalanced_order(names: list[str], round_index: int) -> list[str]:
    offset = round_index % len(names)
    order = names[offset:] + names[:offset]
    if (round_index // len(names)) % 2:
        order.reverse()
    return order


def _measure_eager(
    callables: dict[str, Callable[[], object]], *, warmup: int, reps: int
) -> dict[str, list[float]]:
    for name in callables:
        for _ in range(warmup):
            callables[name]()
    torch.cuda.synchronize()
    samples = {name: [] for name in callables}
    names = list(callables)
    for round_index in range(reps):
        for name in _counterbalanced_order(names, round_index):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            hold = callables[name]()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000.0)
            # Keep results alive through the end event and synchronization.
            del hold
    return samples


def _capture(callable_: Callable[[], object]) -> tuple[torch.cuda.CUDAGraph, dict]:
    # Warm allocator/JIT state outside capture.
    warm = callable_()
    torch.cuda.synchronize()
    del warm
    graph = torch.cuda.CUDAGraph()
    holder: dict[str, object] = {}
    with torch.cuda.graph(graph):
        holder["result"] = callable_()
    return graph, holder


def _measure_graph(
    callables: dict[str, Callable[[], object]], *, warmup: int, reps: int
) -> tuple[dict[str, list[float]], dict[str, dict]]:
    captured = {name: _capture(fn) for name, fn in callables.items()}
    for name, (graph, _holder) in captured.items():
        for _ in range(warmup):
            graph.replay()
    torch.cuda.synchronize()
    samples = {name: [] for name in callables}
    names = list(callables)
    for round_index in range(reps):
        for name in _counterbalanced_order(names, round_index):
            graph, _holder = captured[name]
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000.0)
    holders = {name: holder for name, (_graph, holder) in captured.items()}
    return samples, holders


def _generate_case(case: BenchCase, device: torch.device, seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    scores = torch.rand(
        (case.tokens, case.num_experts), generator=generator, dtype=torch.float32
    )
    if case.distribution == "skew":
        hot = min(4, case.num_experts)
        scores[:, :hot] += 4.0
    topk_ids = scores.topk(case.top_k, dim=1).indices.to(torch.int32)

    if case.max_loras == 1:
        mapping = torch.zeros(case.tokens, dtype=torch.int32)
    else:
        mapping = torch.randint(
            0,
            case.max_loras,
            (case.tokens,),
            generator=generator,
            dtype=torch.int32,
        )
    if case.distribution == "base_rows":
        # Deterministic 25% base-only rows; T=1 deliberately exercises all-base.
        mapping[::4] = -1

    pair_values = torch.arange(case.tokens * case.top_k, dtype=torch.float32) % 17 + 1
    return (
        topk_ids.to(device),
        mapping.to(device),
        pair_values.to(device),
    )


def _expected_virtual_ids(
    case: BenchCase, topk_ids: torch.Tensor, mapping: torch.Tensor
) -> torch.Tensor:
    topk = topk_ids.cpu().to(torch.int64)
    adapters = mapping.cpu().to(torch.int64).view(-1, 1).expand_as(topk)
    owned = (topk >= 0) & (topk < case.local_num_experts)
    valid = (adapters >= 0) & owned
    virtual = topk + adapters.clamp_min(0) * case.num_experts
    return torch.where(valid, virtual, torch.full_like(virtual, -1)).reshape(-1)


def _validate_plan(
    case: BenchCase,
    plan: RoutePlan,
    topk_ids: torch.Tensor,
    mapping: torch.Tensor,
    pair_values: torch.Tensor,
) -> dict:
    expected_ids = _expected_virtual_ids(case, topk_ids, mapping)
    expected_mask = mapping.cpu() >= 0
    actual_mask = plan.token_lora_mask.cpu()
    if not torch.equal(actual_mask, expected_mask):
        raise AssertionError(f"{plan.producer}: token LoRA mask mismatch")

    num_post = int(plan.num_pairs_post_padded.cpu().item())
    if num_post % plan.block_m:
        raise AssertionError(f"{plan.producer}: non-aligned post-pad count")
    num_blocks = num_post // plan.block_m
    sorted_ids = plan.sorted_pair_ids[:num_post].cpu().to(torch.int64)
    experts = plan.expert_ids[:num_blocks].cpu().to(torch.int64)
    slot_experts = experts.repeat_interleave(plan.block_m)
    num_pairs = expected_ids.numel()
    real_slot = sorted_ids < num_pairs
    routed_slot = real_slot & (slot_experts >= 0)
    routed_pairs = sorted_ids[routed_slot]
    routed_experts = slot_experts[routed_slot]
    if routed_pairs.numel() and not torch.equal(
        expected_ids[routed_pairs], routed_experts
    ):
        raise AssertionError(f"{plan.producer}: pair/expert assignment mismatch")
    counts = torch.bincount(routed_pairs, minlength=num_pairs)
    expected_counts = (expected_ids >= 0).to(torch.int64)
    if not torch.equal(counts, expected_counts):
        raise AssertionError(f"{plan.producer}: dropped or duplicate routed pair")

    output = torch.empty_like(pair_values)
    consume_route_plan(plan, pair_values, output)
    torch.cuda.synchronize()
    expected_delta = pair_values.cpu() * (expected_ids + 1).clamp_min(0)
    actual_delta = output.cpu()
    if not torch.equal(actual_delta, expected_delta):
        max_err = (actual_delta - expected_delta).abs().max().item()
        raise AssertionError(f"{plan.producer}: delta mismatch, max error {max_err}")
    return {
        "num_pairs": num_pairs,
        "num_valid_pairs": int((expected_ids >= 0).sum().item()),
        "num_pairs_post_padded": num_post,
        "num_blocks": num_blocks,
        "route_digest": int(((routed_pairs + 1) * (routed_experts + 3)).sum().item()),
        "delta_exact": True,
    }


def _build_plan(
    producer: str,
    case: BenchCase,
    topk_ids: torch.Tensor,
    mapping: torch.Tensor,
    block_m: int,
) -> RoutePlan:
    builder = (
        build_current_sgl_plan
        if producer == "current_sgl"
        else build_legacy_merged_plan
    )
    return builder(
        topk_ids,
        mapping,
        num_experts=case.num_experts,
        max_loras=case.max_loras,
        local_expert_offset=0,
        local_num_experts=case.local_num_experts,
        block_m=block_m,
    )


def _single_layer_callable(
    variant: str,
    case: BenchCase,
    topk_ids: torch.Tensor,
    mapping: torch.Tensor,
    pair_values: torch.Tensor,
    *,
    prebuilt: bool,
) -> Callable[[], object]:
    if variant in ("current_sgl", "legacy_merged"):
        producer = "current_sgl" if variant == "current_sgl" else "legacy_merged"
        cached = (
            [
                _build_plan(producer, case, topk_ids, mapping, case.block_a),
                _build_plan(producer, case, topk_ids, mapping, case.block_b),
            ]
            if prebuilt
            else None
        )

        def run():
            plans = cached or [
                _build_plan(producer, case, topk_ids, mapping, case.block_a),
                _build_plan(producer, case, topk_ids, mapping, case.block_b),
            ]
            outputs = []
            for plan in plans:
                out = torch.empty_like(pair_values)
                consume_route_plan(plan, pair_values, out)
                outputs.append(out)
            return plans, outputs

        return run

    cached_plan = (
        _build_plan("current_sgl", case, topk_ids, mapping, case.common_block)
        if prebuilt
        else None
    )

    def run_reuse():
        plan = cached_plan or _build_plan(
            "current_sgl", case, topk_ids, mapping, case.common_block
        )
        outputs = []
        for _ in range(2):
            out = torch.empty_like(pair_values)
            consume_route_plan(plan, pair_values, out)
            outputs.append(out)
        return plan, outputs

    return run_reuse


def _macro_callable(
    variant: str,
    case: BenchCase,
    topk_ids: torch.Tensor,
    mapping: torch.Tensor,
    pair_values: torch.Tensor,
    *,
    layers: int,
) -> Callable[[], object]:
    if variant != "cross_layer_memo":
        single = _single_layer_callable(
            variant, case, topk_ids, mapping, pair_values, prebuilt=False
        )

        def run_layers():
            return [single() for _ in range(layers)]

        return run_layers

    def run_memo():
        memo = RoutePlanMemo()
        key = make_route_plan_key(
            topk_ids,
            mapping,
            num_experts=case.num_experts,
            max_loras=case.max_loras,
            local_expert_offset=0,
            local_num_experts=case.local_num_experts,
            block_m=case.common_block,
            route_epoch=0,
            producer="current_sgl",
        )
        results = []
        for _ in range(layers):
            plan = memo.get_or_build(
                key,
                lambda: _build_plan(
                    "current_sgl",
                    case,
                    topk_ids,
                    mapping,
                    case.common_block,
                ),
            )
            outputs = []
            for _ in range(2):
                out = torch.empty_like(pair_values)
                consume_route_plan(plan, pair_values, out)
                outputs.append(out)
            results.append((plan, outputs))
        return results, memo.hits, memo.misses

    return run_memo


def _memo_host_cost_ns(
    case: BenchCase, topk_ids: torch.Tensor, mapping: torch.Tensor, reps: int = 10000
) -> dict:
    plan = _build_plan("current_sgl", case, topk_ids, mapping, case.common_block)
    key = make_route_plan_key(
        topk_ids,
        mapping,
        num_experts=case.num_experts,
        max_loras=case.max_loras,
        local_expert_offset=0,
        local_num_experts=case.local_num_experts,
        block_m=case.common_block,
        route_epoch=7,
        producer="current_sgl",
    )
    memo = RoutePlanMemo()
    memo.get_or_build(key, lambda: plan)
    start = time.perf_counter_ns()
    for _ in range(reps):
        memo.get_or_build(key, lambda: plan)
    elapsed = time.perf_counter_ns() - start
    return {
        "lookup_reps": reps,
        "mean_hit_ns": elapsed / reps,
        "hits": memo.hits,
        "misses": memo.misses,
    }


def run_case(
    case: BenchCase,
    *,
    device: torch.device,
    seed: int,
    warmup: int,
    reps: int,
    macro_layers: int,
) -> dict:
    topk_ids, mapping, pair_values = _generate_case(case, device, seed)
    supported, unsupported_reason = merged_align_supported(
        num_experts=case.num_experts,
        max_loras=case.max_loras,
        local_num_experts=case.local_num_experts,
    )
    active_variants = [v for v in VARIANTS if v != "legacy_merged" or supported]

    validation: dict[str, dict] = {}
    for producer in ("current_sgl", "legacy_merged"):
        if producer == "legacy_merged" and not supported:
            continue
        for block_m in sorted({case.block_a, case.block_b}):
            plan = _build_plan(producer, case, topk_ids, mapping, block_m)
            validation[f"{producer}_BM{block_m}"] = _validate_plan(
                case, plan, topk_ids, mapping, pair_values
            )

    results: dict[str, dict] = {"eager": {}, "cuda_graph": {}}
    for scope in ("K0_prebuilt", "O0_one_layer", "M0_macro"):
        if scope == "M0_macro":
            callables = {
                variant: _macro_callable(
                    variant,
                    case,
                    topk_ids,
                    mapping,
                    pair_values,
                    layers=macro_layers,
                )
                for variant in active_variants
            }
            divisor = macro_layers
        else:
            prebuilt = scope == "K0_prebuilt"
            callables = {
                variant: _single_layer_callable(
                    variant,
                    case,
                    topk_ids,
                    mapping,
                    pair_values,
                    prebuilt=prebuilt,
                )
                for variant in active_variants
            }
            divisor = 1

        eager = _measure_eager(callables, warmup=warmup, reps=reps)
        graph, graph_holders = _measure_graph(callables, warmup=warmup, reps=reps)
        results["eager"][scope] = {
            name: _stats(samples, divisor=divisor) for name, samples in eager.items()
        }
        results["cuda_graph"][scope] = {
            name: _stats(samples, divisor=divisor) for name, samples in graph.items()
        }
        # Captured outputs remain alive until after replay.  Verify all device
        # work has completed before holders leave scope.
        torch.cuda.synchronize()
        del graph_holders

    return {
        "case": asdict(case),
        "case_id": case.case_id,
        "seed": seed,
        "blocks": {
            "gate_down_a": case.block_a,
            "expand_b": case.block_b,
            "reuse_common": case.common_block,
        },
        "prefill_reuse_policy_eligible": case.tokens >= 512,
        "legacy_merged_supported": supported,
        "legacy_merged_unsupported_reason": unsupported_reason,
        "validation": validation,
        "memo_host": _memo_host_cost_ns(case, topk_ids, mapping),
        "timing": results,
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-label", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=15)
    parser.add_argument("--macro-layers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    cases = all_cases()
    if args.case:
        wanted = set(args.case)
        cases = [case for case in cases if case.case_id in wanted]
    cases = [
        case
        for index, case in enumerate(cases)
        if index % args.num_shards == args.shard_index
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": 1,
        "benchmark": "sgl_lora_route_countercandidates",
        "device_label": args.device_label,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "git_commit": _git_commit(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "warmup": args.warmup,
        "reps": args.reps,
        "macro_layers": args.macro_layers,
        "cases": [],
    }
    for index, case in enumerate(cases):
        print(
            f"[{args.device_label} shard {args.shard_index}] "
            f"{index + 1}/{len(cases)} {case.case_id}",
            flush=True,
        )
        payload["cases"].append(
            run_case(
                case,
                device=device,
                # Stable across device labels and shard layouts, so H200 and
                # GB300 consume bit-identical route inputs.
                seed=args.seed + zlib.crc32(case.case_id.encode("utf-8")),
                warmup=args.warmup,
                reps=args.reps,
                macro_layers=args.macro_layers,
            )
        )
        tmp = args.output.with_suffix(args.output.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        tmp.replace(args.output)


if __name__ == "__main__":
    main()
