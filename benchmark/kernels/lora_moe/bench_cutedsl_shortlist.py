#!/usr/bin/env python3
"""Interleaved final shortlist and profiler entry point for CuTe gate-A.

The fixed anchor is the only cell where an untuned baseline initially made the
CuTe grouped path look dispatch-worthy: Qwen3.5-35B local MoE, T=2048, R=128.
This script compares the independently selected CuTe ``m128n128`` tactic with
the independently selected Triton aligned ``BN128/BK64/4 warps`` schedule in
both measurement orders and hot/cold cache states.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from benchmark.kernels.lora_moe.bench_algorithm_families import (
    _TUNING,
    FamilyCase,
    _build_fixture,
    _check_site,
    _prepare_site,
    _reference_gemm,
    _strict_check,
)
from benchmark.kernels.lora_moe.bench_shrink_schedules import _make_cache_control
from benchmark.kernels.lora_moe.cutedsl_grouped_tensorcore import GroupedTactic
from benchmark.kernels.lora_moe.cutedsl_moe_boundary import (
    GroupedGemmBoundary,
    build_compact_grouped_route,
)
from benchmark.kernels.lora_moe.profiling import make_batch, time_cuda_events


def _prepare():
    case = FamilyCase(
        tokens=2048,
        rank=128,
        hidden=2048,
        intermediate=512,
        experts=32,
        top_k=8,
        adapters=4,
        adapter_mode="multi",
        route_pattern="iid",
    )
    fixture = _build_fixture(case, torch.device("cuda"))
    _TUNING.aligned_bn = 128
    _TUNING.aligned_bk = 64
    _TUNING.aligned_warps = 4
    triton_invoke, triton_reference, triton_outputs = _prepare_site(
        fixture,
        "gate_a",
        "aligned",
        scope="K0",
        accumulation="bf16",
    )
    triton_check = _check_site(
        "gate_a", triton_invoke, triton_reference, triton_outputs
    )

    route = build_compact_grouped_route(
        fixture.topk_ids,
        fixture.mapping,
        num_experts=case.experts,
        num_adapters=case.adapters,
    )
    cute_output = torch.empty(
        (case.pairs, 2 * case.rank),
        dtype=torch.bfloat16,
        device=fixture.device,
    )
    cute = GroupedGemmBoundary(
        fixture.hidden_states,
        fixture.gate_a,
        cute_output,
        route,
        input_pair_major=False,
        top_k=case.top_k,
        tactic=GroupedTactic(mma_m=128, mma_n=128),
    )
    reference = _reference_gemm(
        fixture,
        fixture.hidden_states,
        fixture.gate_a,
        input_pair_major=False,
        output_dtype=torch.bfloat16,
    )
    cute.invoke_boundary()
    cute_check = _strict_check(cute_output, reference, name="gate_a", atol=5e-2)
    return case, fixture, triton_invoke, triton_check, cute, cute_check


def _time(
    invoke: Callable[[], None],
    *,
    execution: str,
    cache_state: str,
    warmup: int,
    samples: int,
) -> dict[str, object]:
    batch = make_batch(invoke, execution=execution, inner_iterations=1)
    cache = _make_cache_control(cache_state, torch.device("cuda"))
    return asdict(
        time_cuda_events(
            batch.run,
            launches_per_batch=1,
            warmup=warmup,
            samples=samples,
            before_sample=cache.before_sample(),
        )
    )


def _route_mutation_audit(fixture, cute: GroupedGemmBoundary) -> dict[str, object]:
    """Show why the fixed grouped descriptors are not a serving graph ABI.

    A real serving graph reuses tensor addresses while expert IDs and adapter
    assignments change.  The grouped plan below owns the original sort order,
    group offsets, and per-group M dimensions.  Mutating the route in-place
    therefore must *not* be treated as a correctness-supported replay.
    """

    original_topk = fixture.topk_ids.clone()
    original_mapping = fixture.mapping.clone()
    original_reference = _reference_gemm(
        fixture,
        fixture.hidden_states,
        fixture.gate_a,
        input_pair_major=False,
        output_dtype=torch.bfloat16,
    )
    graph = make_batch(cute.invoke_boundary, execution="cuda_graph", inner_iterations=1)
    graph.run()
    torch.cuda.synchronize()
    original_output = cute.output.clone()
    try:
        fixture.topk_ids.add_(1).remainder_(fixture.case.experts)
        fixture.mapping.add_(1).remainder_(fixture.case.adapters)
        mutated_reference = _reference_gemm(
            fixture,
            fixture.hidden_states,
            fixture.gate_a,
            input_pair_major=False,
            output_dtype=torch.bfloat16,
        )
        graph.run()
        torch.cuda.synchronize()
        replay_output = cute.output.clone()
        stale_error = float(
            (replay_output.float() - original_reference.float()).abs().max().item()
        )
        mutated_error = float(
            (replay_output.float() - mutated_reference.float()).abs().max().item()
        )
        route_reference_change = float(
            (mutated_reference.float() - original_reference.float()).abs().max().item()
        )
        replay_change = float(
            (replay_output.float() - original_output.float()).abs().max().item()
        )
    finally:
        fixture.topk_ids.copy_(original_topk)
        fixture.mapping.copy_(original_mapping)
    return {
        "mutation": "expert_ids_plus_1_mod_E_and_adapter_ids_plus_1_mod_L",
        "tensor_addresses_unchanged": True,
        "captured_route_metadata_rebuilt": False,
        "original_replay_max_abs_error": stale_error,
        "mutated_replay_max_abs_error": mutated_error,
        "route_reference_max_abs_change": route_reference_change,
        "replay_output_max_abs_change": replay_change,
        "supports_dynamic_route_graph_replay": False,
        "dispatch_eligible": False,
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executions", default="eager,cuda_graph")
    parser.add_argument("--cache-states", default="hot,cold")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument(
        "--profile", choices=("none", "cutedsl", "triton"), default="none"
    )
    parser.add_argument("--profile-iterations", type=int, default=1)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    case, fixture, triton_invoke, triton_check, cute, cute_check = _prepare()
    if args.profile != "none":
        invoke = cute.invoke_boundary if args.profile == "cutedsl" else triton_invoke
        for _ in range(5):
            invoke()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(f"{args.profile}_shortlist")
        for _ in range(args.profile_iterations):
            invoke()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        print(
            json.dumps(
                {
                    "profile": args.profile,
                    "iterations": args.profile_iterations,
                    "case": asdict(case),
                    "cutedsl": cute.metadata(),
                },
                sort_keys=True,
            )
        )
        return 0

    executions = tuple(item for item in args.executions.split(",") if item)
    cache_states = tuple(item for item in args.cache_states.split(",") if item)
    providers = {
        "cutedsl_m128n128": cute.invoke_boundary,
        "triton_aligned_bn128_bk64_w4": triton_invoke,
    }
    props = torch.cuda.get_device_properties(0)
    report: dict[str, object] = {
        "schema_version": 1,
        "scope": "interleaved_final_gate_a_shortlist",
        "environment": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "device": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "case": asdict(case),
        "correctness": {"cutedsl": cute_check, "triton": triton_check},
        "route_mutation_audit": _route_mutation_audit(fixture, cute),
        "cutedsl_implementation": cute.metadata(),
        "triton_tuning": {"block_n": 128, "block_k": 64, "num_warps": 4},
        "runs": [],
    }
    runs: list[dict[str, object]] = report["runs"]  # type: ignore[assignment]
    for order in (tuple(providers), tuple(reversed(tuple(providers)))):
        for execution in executions:
            for cache_state in cache_states:
                for provider_name in order:
                    timing = _time(
                        providers[provider_name],
                        execution=execution,
                        cache_state=cache_state,
                        warmup=args.warmup,
                        samples=args.samples,
                    )
                    row = {
                        "order": list(order),
                        "provider": provider_name,
                        "execution": execution,
                        "cache_state": cache_state,
                        "timing": timing,
                    }
                    runs.append(row)
                    print(
                        f"{provider_name} {execution}/{cache_state} "
                        f"order={order}: {timing['p50_us']:.3f} us"
                    )
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    del fixture
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
