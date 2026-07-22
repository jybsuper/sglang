#!/usr/bin/env python3
"""Matched optimized CuTe DSL versus Triton MoE-LoRA benchmark.

The optimized candidate is NVIDIA's persistent Blackwell grouped GEMM (TMA,
``tcgen05``, TMEM), adapted by :mod:`cutedsl_grouped_tensorcore`.  This driver
reports two distinct scopes:

``compute``
    Prepacked grouped GEMM ceiling.  It is diagnostic and never sufficient for
    a dispatch decision.
``boundary``
    The exact routed-pair contract, including compact gather, zero/base-row
    behavior, activation/finalize consumers, and unpack where required.

Triton baselines always use the same boundary scope.  Both eager launches and
CUDA-graph replay are independently correctness checked.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import socket
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import triton

from benchmark.kernels.lora_moe.bench_algorithm_families import (
    FamilyCase,
    _build_fixture,
    _check_site,
    _prepare_site,
    _reference_down_finalize,
    _reference_gate_consumer,
    _reference_gemm,
    _strict_check,
)
from benchmark.kernels.lora_moe.bench_shrink_schedules import (
    _make_cache_control,
)
from benchmark.kernels.lora_moe.cutedsl_grouped_tensorcore import (
    GroupedTactic,
    capability,
    clear_compile_cache,
)
from benchmark.kernels.lora_moe.cutedsl_moe_boundary import (
    GroupedC2Boundary,
    GroupedDownFinalizeBoundary,
    GroupedGemmBoundary,
    build_compact_grouped_route,
)
from benchmark.kernels.lora_moe.profiling import make_batch, time_cuda_events


@dataclass(frozen=True, slots=True)
class CoreModel:
    key: str
    hidden: int
    intermediate: int
    experts: int
    top_k: int


CORE_MODELS = {
    "qwen35": CoreModel("qwen3.5-35b-a3b-local", 2048, 512, 32, 8),
    "qwen397": CoreModel("qwen3.5-397b-a17b-local", 4096, 1024, 32, 10),
    "kimi": CoreModel("kimi-k2.5-local-wide", 7168, 2048, 32, 8),
}

SITES = ("gate_a", "down_a", "gate_consumer", "down_finalize")
BASELINES = ("indexed", "segmented", "aligned")

TACTICS: dict[str, GroupedTactic] = {
    "m64n32": GroupedTactic(mma_m=64, mma_n=32),
    "m64n64": GroupedTactic(mma_m=64, mma_n=64),
    "m64n128": GroupedTactic(mma_m=64, mma_n=128),
    "m128n32": GroupedTactic(mma_m=128, mma_n=32),
    "m128n64": GroupedTactic(mma_m=128, mma_n=64),
    "m128n128": GroupedTactic(mma_m=128, mma_n=128),
    "m128n64-gmem": GroupedTactic(
        mma_m=128, mma_n=64, tensormap_update="GMEM"
    ),
    "m128n64-host": GroupedTactic(
        mma_m=128, mma_n=64, host_problem_shapes=True
    ),
    "2cta-m128n32": GroupedTactic(
        mma_m=128,
        mma_n=32,
        cluster_m=2,
        cluster_n=1,
        use_2cta=True,
    ),
    "2cta-m128n64": GroupedTactic(
        mma_m=128,
        mma_n=64,
        cluster_m=2,
        cluster_n=1,
        use_2cta=True,
    ),
    "2cta-m256n64": GroupedTactic(
        mma_m=256,
        mma_n=64,
        cluster_m=2,
        cluster_n=1,
        use_2cta=True,
    ),
}


def _csv(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("expected a nonempty comma-separated list")
    return items


def _csv_ints(value: str) -> tuple[int, ...]:
    items = tuple(int(item) for item in _csv(value))
    if any(item <= 0 for item in items):
        raise ValueError("integer lists must be positive")
    return items


def _environment() -> dict[str, object]:
    props = torch.cuda.get_device_properties(0)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "device": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "triton": getattr(triton, "__version__", "unknown"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _provider(
    fixture,
    site: str,
    tactic: GroupedTactic,
):
    case = fixture.case
    route = build_compact_grouped_route(
        fixture.topk_ids,
        fixture.mapping,
        num_experts=case.experts,
        num_adapters=case.adapters,
    )
    if site == "gate_a":
        output = torch.empty(
            (case.pairs, 2 * case.rank),
            dtype=torch.bfloat16,
            device=fixture.device,
        )
        provider = GroupedGemmBoundary(
            fixture.hidden_states,
            fixture.gate_a,
            output,
            route,
            input_pair_major=False,
            top_k=case.top_k,
            tactic=tactic,
        )
        reference = _reference_gemm(
            fixture,
            fixture.hidden_states,
            fixture.gate_a,
            input_pair_major=False,
            output_dtype=torch.bfloat16,
        )

        def check() -> dict[str, object]:
            provider.invoke_boundary()
            return _strict_check(output, reference, name=site, atol=5e-2)

    elif site == "down_a":
        output = torch.empty(
            (case.pairs, case.rank),
            dtype=torch.bfloat16,
            device=fixture.device,
        )
        provider = GroupedGemmBoundary(
            fixture.activated_pairs,
            fixture.down_a,
            output,
            route,
            input_pair_major=True,
            top_k=case.top_k,
            tactic=tactic,
        )
        reference = _reference_gemm(
            fixture,
            fixture.activated_pairs,
            fixture.down_a,
            input_pair_major=True,
            output_dtype=torch.bfloat16,
        )

        def check() -> dict[str, object]:
            provider.invoke_boundary()
            return _strict_check(output, reference, name=site, atol=5e-2)

    elif site == "gate_consumer":
        act = torch.empty_like(fixture.activated_pairs)
        down_rank = torch.empty(
            (case.pairs, case.rank),
            dtype=torch.bfloat16,
            device=fixture.device,
        )
        src2dst = torch.arange(
            case.pairs, dtype=torch.int32, device=fixture.device
        )
        provider = GroupedC2Boundary(
            value=fixture.gateup_base,
            value_a=fixture.gate_rank_input,
            value_b=fixture.gate_b,
            down_a=fixture.down_a,
            act_out=act,
            down_rank=down_rank,
            src2dst=src2dst,
            topk_ids=fixture.topk_ids,
            mapping=fixture.mapping,
            route=route,
            logical_i=case.intermediate,
            physical_i=case.intermediate,
            activation="swiglu",
            gate_tactic=tactic,
            down_tactic=tactic,
        )
        act_ref, rank_ref = _reference_gate_consumer(fixture)

        def check() -> dict[str, object]:
            provider.invoke_boundary()
            return {
                "act": _strict_check(act, act_ref, name="act", atol=5e-2),
                "down_rank": _strict_check(
                    down_rank, rank_ref, name="down_rank", atol=5e-2
                ),
            }

    elif site == "down_finalize":
        output = torch.empty(
            (case.tokens, case.hidden),
            dtype=torch.float32,
            device=fixture.device,
        )
        provider = GroupedDownFinalizeBoundary(
            rank_input=fixture.down_rank_input,
            down_b=fixture.down_b,
            base_pairs=fixture.base_down_pairs,
            topk_weights=fixture.topk_weights,
            output=output,
            route=route,
            tactic=tactic,
        )
        reference = _reference_down_finalize(
            fixture,
            rank_input=fixture.down_rank_input,
            round_delta_to_bf16=True,
        )

        def check() -> dict[str, object]:
            provider.invoke_boundary()
            return _strict_check(output, reference, name=site, atol=5e-2)

    else:
        raise ValueError(site)
    return provider, check


def _measure(
    invoke: Callable[[], None],
    *,
    execution: str,
    cache_state: str,
    warmup: int,
    samples: int,
) -> tuple[dict[str, object], dict[str, object] | None]:
    batch = make_batch(invoke, execution=execution, inner_iterations=1)
    graph_status = None
    if execution == "cuda_graph":
        batch.run()
        torch.cuda.synchronize()
        graph_status = {"captured": True, "replayed": True}
    cache = _make_cache_control(cache_state, torch.device("cuda"))
    timing = time_cuda_events(
        batch.run,
        launches_per_batch=1,
        warmup=warmup,
        samples=samples,
        before_sample=cache.before_sample(),
    )
    return asdict(timing), graph_status


def _write_report(path: Path | None, report: dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="qwen35")
    parser.add_argument("--tokens", default="1,32,256,2048")
    parser.add_argument("--ranks", default="16,32,64,128")
    parser.add_argument("--sites", default=",".join(SITES))
    parser.add_argument("--tactics", default="m64n64")
    parser.add_argument(
        "--baselines", default=",".join(BASELINES), help="none or comma-separated"
    )
    parser.add_argument("--scopes", default="compute,boundary")
    parser.add_argument("--executions", default="eager,cuda_graph")
    parser.add_argument("--cache-states", default="hot")
    parser.add_argument(
        "--adapter-mode", choices=("multi", "mixed"), default="multi"
    )
    parser.add_argument(
        "--route-pattern", choices=("regular", "iid", "skewed"), default="iid"
    )
    parser.add_argument("--adapters", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    models = _csv(args.models)
    unknown_models = set(models).difference(CORE_MODELS)
    if unknown_models:
        raise ValueError(f"unknown models: {sorted(unknown_models)}")
    tokens = _csv_ints(args.tokens)
    ranks = _csv_ints(args.ranks)
    sites = _csv(args.sites)
    if set(sites).difference(SITES):
        raise ValueError(f"sites must be selected from {SITES}")
    tactic_names = _csv(args.tactics)
    if set(tactic_names).difference(TACTICS):
        raise ValueError(f"unknown tactics: {set(tactic_names).difference(TACTICS)}")
    baselines = () if args.baselines == "none" else _csv(args.baselines)
    if set(baselines).difference(BASELINES):
        raise ValueError(f"baselines must be selected from {BASELINES}")
    scopes = _csv(args.scopes)
    if set(scopes).difference(("compute", "boundary")):
        raise ValueError("scopes must be compute and/or boundary")
    executions = _csv(args.executions)
    if set(executions).difference(("eager", "cuda_graph")):
        raise ValueError("executions must be eager and/or cuda_graph")
    cache_states = _csv(args.cache_states)
    if set(cache_states).difference(("hot", "cold")):
        raise ValueError("cache states must be hot and/or cold")
    if args.adapters <= 0:
        raise ValueError("adapters must be positive")

    cap = capability()
    if not cap["available"]:
        raise RuntimeError(f"CuTe provider unavailable: {cap}")
    clear_compile_cache()
    report: dict[str, object] = {
        "schema_version": 1,
        "scope": "optimized_blackwell_cutedsl_grouped_moe_lora_comparison",
        "environment": _environment(),
        "capability": cap,
        "contract": {
            "route_domain": "canonical_token_topk_pairs",
            "virtual_group": "adapter_times_E_local_plus_local_expert",
            "route_scope": "K0_prebuilt",
            "compute_scope_dispatch_eligible": False,
            "boundary_scope_dispatch_eligible": True,
            "clears_gathers_unpacks_activation_finalize_charged": True,
            "production_dispatch_changed": False,
        },
        "arguments": vars(args) | {"json_output": str(args.json_output)},
        "runs": [],
        "errors": [],
    }
    runs: list[dict[str, object]] = report["runs"]  # type: ignore[assignment]
    errors: list[dict[str, object]] = report["errors"]  # type: ignore[assignment]

    for model_name in models:
        model = CORE_MODELS[model_name]
        for token_count in tokens:
            for rank in ranks:
                case = FamilyCase(
                    tokens=token_count,
                    rank=rank,
                    hidden=model.hidden,
                    intermediate=model.intermediate,
                    experts=model.experts,
                    top_k=model.top_k,
                    adapters=args.adapters,
                    adapter_mode=args.adapter_mode,
                    route_pattern=args.route_pattern,
                )
                fixture = _build_fixture(case, torch.device("cuda"))
                for site in sites:
                    for baseline in baselines:
                        try:
                            invoke, reference, outputs = _prepare_site(
                                fixture,
                                site,
                                baseline,
                                scope="K0",
                                accumulation="bf16",
                            )
                            correctness = (
                                None
                                if args.skip_check
                                else _check_site(
                                    site, invoke, reference, outputs
                                )
                            )
                            for execution in executions:
                                for cache_state in cache_states:
                                    timing, graph_status = _measure(
                                        invoke,
                                        execution=execution,
                                        cache_state=cache_state,
                                        warmup=args.warmup,
                                        samples=args.samples,
                                    )
                                    row = {
                                        "status": "ok",
                                        "model": asdict(model),
                                        "case": asdict(case),
                                        "site": site,
                                        "provider": f"triton_{baseline}",
                                        "scope": "boundary",
                                        "execution": execution,
                                        "cache_state": cache_state,
                                        "correctness": correctness,
                                        "graph_status": graph_status,
                                        "timing": timing,
                                        "implementation": {
                                            key: value
                                            for key, value in outputs.items()
                                            if not isinstance(value, torch.Tensor)
                                        },
                                    }
                                    runs.append(row)
                                    print(
                                        f"{model_name} T{token_count} R{rank} {site} "
                                        f"triton-{baseline} boundary {execution}/{cache_state}: "
                                        f"{timing['p50_us']:.3f} us"
                                    )
                                    _write_report(args.json_output, report)
                        except Exception as exc:
                            row = {
                                "model": model_name,
                                "case": asdict(case),
                                "site": site,
                                "provider": f"triton_{baseline}",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                            errors.append(row)
                            print(f"ERROR {row}")
                            _write_report(args.json_output, report)

                    for tactic_name in tactic_names:
                        tactic = TACTICS[tactic_name]
                        try:
                            provider, check = _provider(fixture, site, tactic)
                            correctness = None if args.skip_check else check()
                            for scope in scopes:
                                invoke = (
                                    provider.invoke_compute_only
                                    if scope == "compute"
                                    else provider.invoke_boundary
                                )
                                for execution in executions:
                                    for cache_state in cache_states:
                                        timing, graph_status = _measure(
                                            invoke,
                                            execution=execution,
                                            cache_state=cache_state,
                                            warmup=args.warmup,
                                            samples=args.samples,
                                        )
                                        if execution == "cuda_graph" and not args.skip_check:
                                            replay_check = check()
                                        else:
                                            replay_check = None
                                        row = {
                                            "status": "ok",
                                            "model": asdict(model),
                                            "case": asdict(case),
                                            "site": site,
                                            "provider": "cutedsl_grouped_tensorcore",
                                            "tactic_name": tactic_name,
                                            "scope": scope,
                                            "execution": execution,
                                            "cache_state": cache_state,
                                            "correctness": correctness,
                                            "graph_replay_correctness": replay_check,
                                            "graph_status": graph_status,
                                            "timing": timing,
                                            "implementation": provider.metadata(),
                                        }
                                        runs.append(row)
                                        print(
                                            f"{model_name} T{token_count} R{rank} {site} "
                                            f"cute-{tactic_name} {scope} "
                                            f"{execution}/{cache_state}: "
                                            f"{timing['p50_us']:.3f} us"
                                        )
                                        _write_report(args.json_output, report)
                            del provider
                            gc.collect()
                        except Exception as exc:
                            row = {
                                "model": model_name,
                                "case": asdict(case),
                                "site": site,
                                "provider": "cutedsl_grouped_tensorcore",
                                "tactic_name": tactic_name,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                            errors.append(row)
                            print(f"ERROR {row}")
                            _write_report(args.json_output, report)
                del fixture
                gc.collect()
    _write_report(args.json_output, report)
    print(json.dumps({"runs": len(runs), "errors": len(errors)}, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
