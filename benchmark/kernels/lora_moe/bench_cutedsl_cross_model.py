#!/usr/bin/env python3
"""Cross-model C2 comparison for the optimized Blackwell CuTe grouped path.

This extends the tensor-core comparison beyond gated Qwen shapes.  It reuses
the existing provider-destination guardrail fixture so global/local expert IDs,
base rows, invalid pairs, logical versus physical intermediate widths, SwiGLU,
and one-slice ReLU2 are all part of the measured boundary.
"""

from __future__ import annotations

import argparse
import gc
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
import triton

from benchmark.kernels.lora_moe.bench_c2_cross_model_contracts import (
    GuardrailCase,
    MODEL_KEYS,
    _build_fixture,
    _invoke,
    _model_dims,
    _strict_check,
)
from benchmark.kernels.lora_moe.bench_cutedsl_tensorcore import TACTICS
from benchmark.kernels.lora_moe.c2_semantic_contracts import reference_c2_consumer
from benchmark.kernels.lora_moe.cutedsl_grouped_tensorcore import (
    capability,
    clear_compile_cache,
)
from benchmark.kernels.lora_moe.cutedsl_moe_boundary import (
    GroupedC2Boundary,
    build_compact_grouped_route,
)
from benchmark.kernels.lora_moe.matrix import MODEL_PRESETS
from benchmark.kernels.lora_moe.profiling import make_batch, time_cuda_events


def _csv(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise ValueError("expected a nonempty comma-separated list")
    return result


def _csv_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in _csv(value))
    if any(item <= 0 for item in result):
        raise ValueError("integer lists must be positive")
    return result


def _case(model: str, tokens: int, rank: int) -> GuardrailCase:
    activation, logical_i, physical_i, experts = _model_dims(model)
    top_k = 5 if model == "odd-provider-padding" else MODEL_PRESETS[model].top_k
    return GuardrailCase(
        model=model,
        activation=activation,
        logical_i=logical_i,
        physical_i=physical_i,
        experts=experts,
        top_k=top_k,
        tokens=tokens,
        rank=rank,
    )


def _measure(
    invoke: Callable[[], None],
    *,
    execution: str,
    warmup: int,
    samples: int,
) -> tuple[dict[str, object], dict[str, float] | None]:
    batch = make_batch(invoke, execution=execution, inner_iterations=1)
    graph_error = None
    if execution == "cuda_graph":
        invoke()
        torch.cuda.synchronize()
        batch.run()
        torch.cuda.synchronize()
        graph_error = {"replay_completed": 1.0}
    timing = time_cuda_events(
        batch.run,
        launches_per_batch=1,
        warmup=warmup,
        samples=samples,
    )
    return asdict(timing), graph_error


def _provider_check(fixture, provider: GroupedC2Boundary) -> dict[str, object]:
    fixture.act_out.fill_(-123.0)
    fixture.down_rank.zero_()
    provider.invoke_boundary()
    torch.cuda.synchronize()
    reference = reference_c2_consumer(
        fixture.contract,
        fixture.value,
        fixture.value_a,
        fixture.value_b,
        fixture.down_a,
        fixture.src2dst,
        fixture.topk_ids,
        fixture.mapping,
        local_expert_offset=fixture.local_expert_offset,
        # Grouped down-A reduces the full logical I in one problem and rounds
        # once at its BF16 output boundary.
        block_size_n=fixture.case.logical_i,
    )
    act_error = (fixture.act_out.float() - reference.activation_output.float()).abs()
    down_error = (
        fixture.down_rank.float() - reference.down_rank_input.float()
    ).abs()
    act_signal = float(reference.activation_output.float().abs().max().item())
    down_signal = float(reference.down_rank_input.float().abs().max().item())
    act_max = float(act_error.max().item())
    down_max = float(down_error.max().item())
    if act_max > max(2e-4, 0.03 * act_signal):
        raise AssertionError(f"CuTe activation error {act_max} over {act_signal}")
    if down_max > max(5e-4, 0.05 * down_signal):
        raise AssertionError(f"CuTe down rank error {down_max} over {down_signal}")
    return {
        "activation_max_abs_error": act_max,
        "activation_signal_max_abs": act_signal,
        "down_rank_max_abs_error": down_max,
        "down_rank_signal_max_abs": down_signal,
        "physical_padding_zero": bool(
            fixture.case.logical_i == fixture.case.physical_i
            or (
                fixture.act_out[:, fixture.case.logical_i :] == 0
            ).logical_or(fixture.act_out[:, fixture.case.logical_i :] == -123).all()
        ),
        "oracle_down_rounding": "one_full_logical_I_grouped_output",
    }


def _write(path: Path | None, report: dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default="qwen3.5-397b-a17b,kimi-k2.5,nemotron-3-super,nemotron-3-nano,odd-provider-padding",
    )
    parser.add_argument("--tokens", default="1,32,256")
    parser.add_argument("--ranks", default="16,32,64,128")
    parser.add_argument("--tactics", default="m64n64")
    parser.add_argument("--schedules", default="pair,aligned")
    parser.add_argument("--block-n", default="16,32,64,128")
    parser.add_argument("--scopes", default="compute,boundary")
    parser.add_argument("--executions", default="eager,cuda_graph")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    models = _csv(args.models)
    if set(models).difference(MODEL_KEYS):
        raise ValueError(f"unknown models: {set(models).difference(MODEL_KEYS)}")
    tokens = _csv_ints(args.tokens)
    ranks = _csv_ints(args.ranks)
    tactics = _csv(args.tactics)
    if set(tactics).difference(TACTICS):
        raise ValueError(f"unknown tactics: {set(tactics).difference(TACTICS)}")
    schedules = _csv(args.schedules)
    if set(schedules).difference(("pair", "aligned")):
        raise ValueError("schedules must be pair and/or aligned")
    block_ns = _csv_ints(args.block_n)
    scopes = _csv(args.scopes)
    if set(scopes).difference(("compute", "boundary")):
        raise ValueError("scopes must be compute and/or boundary")
    executions = _csv(args.executions)
    if set(executions).difference(("eager", "cuda_graph")):
        raise ValueError("executions must be eager and/or cuda_graph")
    cap = capability()
    if not cap["available"]:
        raise RuntimeError(f"CuTe provider unavailable: {cap}")
    clear_compile_cache()
    props = torch.cuda.get_device_properties(0)
    report: dict[str, object] = {
        "schema_version": 1,
        "scope": "optimized_cutedsl_cross_model_C2_contract",
        "environment": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "device": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "triton": getattr(triton, "__version__", "unknown"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "capability": cap,
        "arguments": vars(args) | {"json_output": str(args.json_output)},
        "contract": {
            "provider_destinations": True,
            "global_to_local_expert_offset": True,
            "invalid_expert_rows": True,
            "base_only_rows": True,
            "logical_physical_intermediate_widths": True,
            "activations": ["swiglu", "relu2"],
            "compute_scope_dispatch_eligible": False,
            "boundary_scope_dispatch_eligible": True,
        },
        "runs": [],
        "errors": [],
    }
    runs: list[dict[str, object]] = report["runs"]  # type: ignore[assignment]
    errors: list[dict[str, object]] = report["errors"]  # type: ignore[assignment]

    for model in models:
        model_tokens = (32,) if model == "odd-provider-padding" else tokens
        model_ranks = (32,) if model == "odd-provider-padding" else ranks
        for token_count in model_tokens:
            for rank in model_ranks:
                case = _case(model, token_count, rank)
                fixture = _build_fixture(case)
                for schedule in schedules:
                    for block_n in block_ns:
                        try:
                            correctness = (
                                None
                                if args.skip_check
                                else _strict_check(fixture, schedule, block_n)
                            )
                            invoke = lambda schedule=schedule, block_n=block_n: _invoke(
                                fixture, schedule, block_n
                            )
                            for execution in executions:
                                timing, graph = _measure(
                                    invoke,
                                    execution=execution,
                                    warmup=args.warmup,
                                    samples=args.samples,
                                )
                                row = {
                                    "status": "ok",
                                    "case": asdict(case),
                                    "provider": f"triton_{schedule}",
                                    "block_n": block_n,
                                    "scope": "boundary",
                                    "execution": execution,
                                    "correctness": correctness,
                                    "graph_status": graph,
                                    "timing": timing,
                                }
                                runs.append(row)
                                print(
                                    f"{case.case_id} triton-{schedule}/BN{block_n} "
                                    f"{execution}: {timing['p50_us']:.3f} us"
                                )
                                _write(args.json_output, report)
                        except Exception as exc:
                            row = {
                                "case": asdict(case),
                                "provider": f"triton_{schedule}",
                                "block_n": block_n,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                            errors.append(row)
                            print(f"ERROR {row}")
                            _write(args.json_output, report)

                route = build_compact_grouped_route(
                    fixture.topk_ids,
                    fixture.mapping,
                    num_experts=case.experts,
                    num_adapters=fixture.value_b.shape[0],
                    local_expert_offset=fixture.local_expert_offset,
                )
                for tactic_name in tactics:
                    try:
                        tactic = TACTICS[tactic_name]
                        provider = GroupedC2Boundary(
                            value=fixture.value,
                            value_a=fixture.value_a,
                            value_b=fixture.value_b,
                            down_a=fixture.down_a,
                            act_out=fixture.act_out,
                            down_rank=fixture.down_rank,
                            src2dst=fixture.src2dst,
                            topk_ids=fixture.topk_ids,
                            mapping=fixture.mapping,
                            route=route,
                            logical_i=case.logical_i,
                            physical_i=case.physical_i,
                            activation=case.activation,
                            gate_tactic=tactic,
                            down_tactic=tactic,
                        )
                        correctness = (
                            None
                            if args.skip_check
                            else _provider_check(fixture, provider)
                        )
                        for scope in scopes:
                            invoke = (
                                provider.invoke_compute_only
                                if scope == "compute"
                                else provider.invoke_boundary
                            )
                            for execution in executions:
                                timing, graph = _measure(
                                    invoke,
                                    execution=execution,
                                    warmup=args.warmup,
                                    samples=args.samples,
                                )
                                replay_check = (
                                    _provider_check(fixture, provider)
                                    if execution == "cuda_graph"
                                    and not args.skip_check
                                    else None
                                )
                                row = {
                                    "status": "ok",
                                    "case": asdict(case),
                                    "provider": "cutedsl_grouped_tensorcore",
                                    "tactic_name": tactic_name,
                                    "scope": scope,
                                    "execution": execution,
                                    "correctness": correctness,
                                    "graph_replay_correctness": replay_check,
                                    "graph_status": graph,
                                    "timing": timing,
                                    "implementation": provider.metadata(),
                                }
                                runs.append(row)
                                print(
                                    f"{case.case_id} cute-{tactic_name} {scope} "
                                    f"{execution}: {timing['p50_us']:.3f} us"
                                )
                                _write(args.json_output, report)
                        del provider
                        gc.collect()
                    except Exception as exc:
                        row = {
                            "case": asdict(case),
                            "provider": "cutedsl_grouped_tensorcore",
                            "tactic_name": tactic_name,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        errors.append(row)
                        print(f"ERROR {row}")
                        _write(args.json_output, report)
                del fixture
                gc.collect()
    _write(args.json_output, report)
    print(json.dumps({"runs": len(runs), "errors": len(errors)}, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
