#!/usr/bin/env python3
"""Benchmark the experimental BF16 C2-consumer partial against production C0.

This is a full local-MoE pipeline benchmark, not a production dispatch path.
It holds weights, routing, the DeepGEMM provider, and down-B constant while
varying only the C2-consumer boundary and its ``BLOCK_N``/warp schedule.

Examples::

    python benchmark/kernels/lora_moe/bench_c2_fused_consumer.py --list-cases
    python benchmark/kernels/lora_moe/bench_c2_fused_consumer.py \
      --case-id p0-qwen3.5-35b-a3b-cap1-h200 --execution cuda_graph \
      --block-n 16,32,64 --json-output c2.json
    ncu --nvtx --nvtx-include 'sgl_lora_moe::C2P.*' --target-processes all \
      python benchmark/kernels/lora_moe/bench_c2_fused_consumer.py \
      --case-id p0-qwen3.5-35b-a3b-cap1-h200 --variant C2P \
      --block-n 32 --mode ncu
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from benchmark.kernels.lora_moe.bench_moe_pipeline import (
    PipelineFixture,
    _build_fixture,
    _case_summary,
    _detect_device,
    _environment,
    _list_cases,
    _max_abs_diff,
    _select_case,
    _single_rank_runtime,
    _strict_delta_atol,
)
from benchmark.kernels.lora_moe.profiling import (
    cuda_profile_range,
    make_batch,
    time_cuda_events,
)

VARIANTS = ("C0", "C2P")
CONSUMER_SCHEDULES = ("pair", "aligned")


def _invoke_c2(
    fixture: PipelineFixture,
    *,
    consumer_schedule: str,
    block_n: int,
    num_warps: int,
) -> None:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.lora.sgl_lora.experimental_c2 import (
        run_sgl_lora_moe_c2_experimental,
    )

    if fixture.lora_info is None or fixture.sgl_quant_info is None:
        raise RuntimeError("C2P requires an active-LoRA fixture")
    dispatch_output = StandardDispatchOutput(
        hidden_states=fixture.hidden_work,
        hidden_states_scale=None,
        topk_output=fixture.topk_output,
    )
    result = run_sgl_lora_moe_c2_experimental(
        dispatch_output,
        fixture.sgl_quant_info,
        fixture.runner_config,
        fixture.lora_info,
        fixture.sgl_base,
        consumer_schedule=consumer_schedule,
        block_size_n=block_n,
        num_warps=num_warps,
    )
    fixture.last_output = result.hidden_states


def _invoke(
    fixture: PipelineFixture,
    variant: str,
    *,
    consumer_schedule: str,
    block_n: int,
    num_warps: int,
) -> None:
    if variant == "C0":
        fixture.invoke("C0")
    else:
        _invoke_c2(
            fixture,
            consumer_schedule=consumer_schedule,
            block_n=block_n,
            num_warps=num_warps,
        )


def _checked_output(
    fixture: PipelineFixture,
    variant: str,
    *,
    consumer_schedule: str,
    block_n: int,
    num_warps: int,
) -> torch.Tensor:
    fixture.reset_hidden()
    _invoke(
        fixture,
        variant,
        consumer_schedule=consumer_schedule,
        block_n=block_n,
        num_warps=num_warps,
    )
    torch.cuda.synchronize()
    if fixture.last_output is None or not bool(
        torch.isfinite(fixture.last_output).all()
    ):
        raise AssertionError(f"{variant} produced no finite output")
    return fixture.last_output.clone()


def _correctness(
    fixture: PipelineFixture,
    *,
    consumer_schedule: str,
    block_n: int,
    num_warps: int,
) -> dict[str, object]:
    base = _checked_output(
        fixture,
        "C0",
        consumer_schedule=consumer_schedule,
        block_n=block_n,
        num_warps=num_warps,
    )
    # The active C0 result above is the numerical oracle.  N0 isolates the LoRA
    # delta so a large base result cannot hide a fused-boundary error.
    fixture.reset_hidden()
    fixture.invoke("N0")
    torch.cuda.synchronize()
    assert fixture.last_output is not None
    base_only = fixture.last_output.clone()
    candidate = _checked_output(
        fixture,
        "C2P",
        consumer_schedule=consumer_schedule,
        block_n=block_n,
        num_warps=num_warps,
    )

    reference_delta = base.float() - base_only.float()
    candidate_delta = candidate.float() - base_only.float()
    signal = float(reference_delta.abs().max().item())
    if signal == 0.0:
        raise AssertionError("C2P reference produced an all-zero LoRA delta")
    delta_atol = _strict_delta_atol(signal)
    delta_rtol = 2e-2
    torch.testing.assert_close(base, candidate, rtol=0.0, atol=3e-3)
    torch.testing.assert_close(
        reference_delta,
        candidate_delta,
        rtol=delta_rtol,
        atol=delta_atol,
    )
    delta_error = _max_abs_diff(reference_delta, candidate_delta)
    return {
        "c0_c2p_max_abs": _max_abs_diff(base, candidate),
        "c0_c2p_rtol": 0.0,
        "c0_c2p_atol": 3e-3,
        "reference_delta_max_abs": signal,
        "candidate_delta_max_abs": float(candidate_delta.abs().max().item()),
        "delta_max_abs_error": delta_error,
        "delta_error_over_signal": delta_error / signal if signal else None,
        "delta_rtol": delta_rtol,
        "delta_atol": delta_atol,
    }


def _benchmark(
    fixture: PipelineFixture,
    variant: str,
    *,
    consumer_schedule: str,
    block_n: int,
    num_warps: int,
    execution: str,
    warmup: int,
    samples: int,
) -> dict[str, object]:
    from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode

    fixture.reset_hidden()
    _invoke(
        fixture,
        variant,
        consumer_schedule=consumer_schedule,
        block_n=block_n,
        num_warps=num_warps,
    )
    torch.cuda.synchronize()
    eager_reference = fixture.last_output.clone()
    fixture.reset_hidden()

    with ExitStack() as stack:
        if execution == "cuda_graph":
            stack.enter_context(model_capture_mode())
        batch = make_batch(
            lambda: _invoke(
                fixture,
                variant,
                consumer_schedule=consumer_schedule,
                block_n=block_n,
                num_warps=num_warps,
            ),
            execution=execution,
            inner_iterations=1,
        )

    graph_correctness = None
    if execution == "cuda_graph":
        fixture.reset_hidden()
        batch.run()
        torch.cuda.synchronize()
        assert fixture.last_output is not None
        graph_diff = _max_abs_diff(eager_reference, fixture.last_output)
        torch.testing.assert_close(
            eager_reference, fixture.last_output, rtol=0.0, atol=3e-3
        )
        graph_correctness = {
            "eager_graph_max_abs": graph_diff,
            "rtol": 0.0,
            "atol": 3e-3,
        }

    timing = time_cuda_events(
        batch.run,
        launches_per_batch=1,
        warmup=warmup,
        samples=samples,
        before_sample=fixture.reset_hidden,
    )
    print(
        f"{fixture.case.case_id} {variant}/{consumer_schedule} "
        f"BN={block_n} W={num_warps} "
        f"{execution}: p50={timing.p50_us:.3f} us "
        f"p20/p80={timing.p20_us:.3f}/{timing.p80_us:.3f} us"
    )
    return {
        "variant": variant,
        "schedule": (
            {
                "consumer_schedule": consumer_schedule,
                "block_n": block_n,
                "num_warps": num_warps,
            }
            if variant == "C2P"
            else "not_applicable"
        ),
        "execution": execution,
        "timing": asdict(timing),
        "graph_correctness": graph_correctness,
    }


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(","))
    if not values or any(item <= 0 for item in values):
        raise ValueError("expected comma-separated positive integers")
    return values


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "h200", "gb300"), default="auto")
    parser.add_argument("--case-id")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), default="all")
    parser.add_argument(
        "--consumer-schedule", choices=CONSUMER_SCHEDULES, default="pair"
    )
    parser.add_argument("--block-n", default="16,32,64")
    parser.add_argument("--num-warps", default="4")
    parser.add_argument(
        "--execution", choices=("eager", "cuda_graph"), default="cuda_graph"
    )
    parser.add_argument("--mode", choices=("time", "nsys", "ncu"), default="time")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--counterbalance-repeats", type=int, default=2)
    parser.add_argument("--profile-iterations", type=int, default=1)
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    device = _detect_device(args.device)
    if args.list_cases:
        _list_cases(device)
        return 0

    case = _select_case(device, args.case_id)
    if case.model.num_slices != 2 or case.model.activation != "swiglu":
        raise NotImplementedError("C2P v1 requires the current gated SwiGLU contract")
    block_ns = _parse_csv_ints(args.block_n)
    warp_counts = _parse_csv_ints(args.num_warps)
    variants = VARIANTS if args.variant == "all" else (args.variant,)
    if args.mode != "time" and (
        len(variants) != 1 or len(block_ns) != 1 or len(warp_counts) != 1
    ):
        raise ValueError("profiling requires one variant, BLOCK_N, and warp count")
    if args.counterbalance_repeats <= 0:
        raise ValueError("--counterbalance-repeats must be positive")

    with _single_rank_runtime():
        fixture = _build_fixture(case, need_lora=True)
        checks = (
            None
            if args.skip_check
            else {
                f"bn{block_n}-w{warps}": _correctness(
                    fixture,
                    consumer_schedule=args.consumer_schedule,
                    block_n=block_n,
                    num_warps=warps,
                )
                for block_n in block_ns
                for warps in warp_counts
            }
        )

        runs = []
        comparisons = []
        if args.mode == "time":
            schedules = [
                (block_n, warps) for block_n in block_ns for warps in warp_counts
            ]
            if variants == VARIANTS:
                # Pair every candidate with a fresh control measurement and
                # alternate order.  This avoids treating thermal/cache drift
                # from an always-C0-first sweep as a C2P effect.
                pair_index = 0
                for block_n, warps in schedules:
                    for repeat in range(args.counterbalance_repeats):
                        order = ("C0", "C2P") if repeat % 2 == 0 else ("C2P", "C0")
                        pair_runs = {}
                        for variant in order:
                            run = _benchmark(
                                fixture,
                                variant,
                                consumer_schedule=args.consumer_schedule,
                                block_n=block_n,
                                num_warps=warps,
                                execution=args.execution,
                                warmup=args.warmup,
                                samples=args.samples,
                            )
                            run["pair_index"] = pair_index
                            run["pair_order"] = list(order)
                            runs.append(run)
                            pair_runs[variant] = run
                        c0_us = pair_runs["C0"]["timing"]["p50_us"]
                        c2p_us = pair_runs["C2P"]["timing"]["p50_us"]
                        comparisons.append(
                            {
                                "pair_index": pair_index,
                                "repeat": repeat,
                                "order": list(order),
                                "schedule": {
                                    "consumer_schedule": args.consumer_schedule,
                                    "block_n": block_n,
                                    "num_warps": warps,
                                },
                                "c0_p50_us": c0_us,
                                "c2p_p50_us": c2p_us,
                                "c2p_vs_c0_pct": (c2p_us / c0_us - 1.0) * 100.0,
                            }
                        )
                        pair_index += 1
            else:
                for block_n, warps in schedules:
                    runs.append(
                        _benchmark(
                            fixture,
                            variants[0],
                            consumer_schedule=args.consumer_schedule,
                            block_n=block_n,
                            num_warps=warps,
                            execution=args.execution,
                            warmup=args.warmup,
                            samples=args.samples,
                        )
                    )
        else:
            variant = variants[0]
            block_n, warps = block_ns[0], warp_counts[0]
            fixture.reset_hidden()
            _invoke(
                fixture,
                variant,
                consumer_schedule=args.consumer_schedule,
                block_n=block_n,
                num_warps=warps,
            )
            torch.cuda.synchronize()
            label = (
                f"sgl_lora_moe::C2P::{case.case_id}::{args.consumer_schedule}::"
                f"BN={block_n}::W={warps}"
                if variant == "C2P"
                else f"sgl_lora_moe::C0::{case.case_id}"
            )
            with cuda_profile_range(label):
                for _ in range(args.profile_iterations):
                    fixture.reset_hidden()
                    _invoke(
                        fixture,
                        variant,
                        consumer_schedule=args.consumer_schedule,
                        block_n=block_n,
                        num_warps=warps,
                    )
            runs.append(
                {
                    "variant": variant,
                    "schedule": {
                        "consumer_schedule": args.consumer_schedule,
                        "block_n": block_n,
                        "num_warps": warps,
                    },
                    "profile_label": label,
                    "profile_iterations": args.profile_iterations,
                }
            )

        comparison_summary = []
        for block_n in block_ns:
            for warps in warp_counts:
                matched = [
                    row
                    for row in comparisons
                    if row["schedule"]
                    == {
                        "consumer_schedule": args.consumer_schedule,
                        "block_n": block_n,
                        "num_warps": warps,
                    }
                ]
                if matched:
                    comparison_summary.append(
                        {
                            "schedule": {
                                "consumer_schedule": args.consumer_schedule,
                                "block_n": block_n,
                                "num_warps": warps,
                            },
                            "num_pairs": len(matched),
                            "median_paired_pct": median(
                                row["c2p_vs_c0_pct"] for row in matched
                            ),
                            "min_paired_pct": min(
                                row["c2p_vs_c0_pct"] for row in matched
                            ),
                            "max_paired_pct": max(
                                row["c2p_vs_c0_pct"] for row in matched
                            ),
                        }
                    )

        result = {
            "schema_version": 1,
            "scope": "M0_local_bf16_c2_consumer_partial_experimental",
            "contract": {
                "w13_layout": "masked_gate_first_contiguous",
                "activation": "ordinary_swiglu",
                "elided_bridges": ["gate_up_delta_TK2I", "activation_TKI"],
                "destinations": ["masked_bf16_w2_input", "canonical_TKR_down_b_input"],
                "remaining_unfused_boundary": "base_finalize_then_existing_down_b_expand_add",
                "planned_full_c2_completed": False,
                "routing_fairness": "shared_whole_forward_cache_matching_c0",
                "consumer_schedule": args.consumer_schedule,
                "scaling_semantics": (
                    "topk_weight_times_routed_scaling_applied_exactly_once_to_base_"
                    "and_lora;harness_factor_is_1"
                ),
                "production_dispatch_changed": False,
            },
            "case": _case_summary(case),
            "environment": _environment(args),
            "correctness": checks,
            "runs": runs,
            "counterbalanced_comparisons": comparisons,
            "counterbalanced_summary": comparison_summary,
        }
        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(
                json.dumps(result, indent=2, default=str) + "\n"
            )
        else:
            print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
