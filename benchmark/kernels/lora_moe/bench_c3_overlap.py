#!/usr/bin/env python3
"""Benchmark BF16 C3 gate-A/base overlap against C0/C1/C2P/C2F.

This driver measures the complete local-MoE boundary with identical weights,
routing, and DeepGEMM base work.  C3 changes only the execution topology: the
gate/up LoRA-A shrink runs on the LoRA side stream while the main stream runs
base prepare plus gate/up GEMM, then joins immediately before the fused C2
consumer.  Production dispatch is not changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path
from statistics import median
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from benchmark.kernels.lora_moe.bench_c2_down_finalize import (  # noqa: E402
    _invoke_full_c2,
    _invoke_partial_c2,
)
from benchmark.kernels.lora_moe.bench_moe_pipeline import (  # noqa: E402
    PipelineFixture,
    _build_fixture,
    _case_summary,
    _detect_device,
    _environment,
    _list_cases,
    _max_abs_diff,
    _select_case,
    _single_rank_runtime,
)
from benchmark.kernels.lora_moe.profiling import (  # noqa: E402
    cuda_profile_range,
    make_batch,
    time_cuda_events,
)

VARIANTS = ("C0", "C1", "C2P", "C2F", "C3")


def _with_rank(case, rank: int | None):
    if rank is None or rank == case.adapters.rank:
        return case
    adapters = replace(
        case.adapters,
        rank=rank,
        physical_rank=rank,
        max_rank=rank,
    )
    return replace(case, case_id=f"{case.case_id}-r{rank}", adapters=adapters)


def _invoke_c3(fixture: PipelineFixture, **schedule) -> None:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.lora.sgl_lora.experimental_c3 import (
        run_sgl_lora_moe_c3_experimental,
    )

    result = run_sgl_lora_moe_c3_experimental(
        StandardDispatchOutput(
            hidden_states=fixture.hidden_work,
            hidden_states_scale=None,
            topk_output=fixture.topk_output,
        ),
        fixture.sgl_quant_info,
        fixture.runner_config,
        fixture.lora_info,
        fixture.sgl_base,
        consumer_schedule=schedule["consumer_schedule"],
        consumer_block_size_n=schedule["consumer_block_n"],
        consumer_num_warps=schedule["consumer_warps"],
        finalize_block_size_h=schedule["finalize_block_h"],
        finalize_num_warps=schedule["finalize_warps"],
        has_base_rows=schedule.get("has_base_rows", True),
    )
    fixture.last_output = result.hidden_states


def _invoke(fixture: PipelineFixture, variant: str, **schedule) -> None:
    if variant in ("C0", "C1"):
        fixture.invoke(variant)
    elif variant == "C2P":
        _invoke_partial_c2(
            fixture,
            consumer_schedule=schedule["consumer_schedule"],
            consumer_block_n=schedule["consumer_block_n"],
            consumer_warps=schedule["consumer_warps"],
            has_base_rows=schedule.get("has_base_rows", True),
        )
    elif variant == "C2F":
        _invoke_full_c2(
            fixture,
            consumer_schedule=schedule["consumer_schedule"],
            consumer_block_n=schedule["consumer_block_n"],
            consumer_warps=schedule["consumer_warps"],
            finalize_block_h=schedule["finalize_block_h"],
            finalize_warps=schedule["finalize_warps"],
            has_base_rows=schedule.get("has_base_rows", True),
        )
    elif variant == "C3":
        _invoke_c3(fixture, **schedule)
    else:
        raise ValueError(f"unknown C3 benchmark variant {variant!r}")


def _checked_output(fixture: PipelineFixture, variant: str, **schedule) -> torch.Tensor:
    fixture.reset_hidden()
    _invoke(fixture, variant, **schedule)
    torch.cuda.synchronize()
    if fixture.last_output is None or not bool(
        torch.isfinite(fixture.last_output).all()
    ):
        raise AssertionError(f"{variant} produced no finite output")
    return fixture.last_output.clone()


def _check_independent_delta(
    checks: dict[str, object],
    prefix: str,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    base_only: torch.Tensor,
    *,
    oracle_nondeterminism: float,
    candidate_nondeterminism: float,
) -> None:
    """Apply a strict base-subtracted BF16 gate that rejects a lost delta.

    The full-model paths contain BF16 atomic rank reductions.  Their legal
    order changes are measured from two independent C0 executions, not hidden
    behind a generic loose tolerance.  The candidate allowance is twice that
    sum of the observed oracle and candidate envelopes, bounded to at most 1/4
    of the LoRA signal, so a dropped/all-zero delta still fails by at least 4x.
    """
    reference_delta = reference.float() - base_only.float()
    candidate_delta = candidate.float() - base_only.float()
    signal = float(reference_delta.abs().max().item())
    if signal <= 0.0:
        raise AssertionError("C3 reference produced an all-zero LoRA delta")
    error = _max_abs_diff(reference_delta, candidate_delta)
    atol = min(
        signal / 4.0,
        max(
            signal / 100.0,
            2.0**-11,
            2.0 * (oracle_nondeterminism + candidate_nondeterminism),
        ),
    )
    checks[f"{prefix}_reference_delta_max_abs"] = signal
    checks[f"{prefix}_candidate_delta_max_abs"] = float(
        candidate_delta.abs().max().item()
    )
    checks[f"{prefix}_delta_max_abs_error"] = error
    checks[f"{prefix}_delta_error_over_signal"] = error / signal
    checks[f"{prefix}_delta_rtol"] = 2e-2
    checks[f"{prefix}_delta_atol"] = atol
    checks[f"{prefix}_measured_c0_nondeterminism"] = oracle_nondeterminism
    checks[f"{prefix}_measured_candidate_nondeterminism"] = candidate_nondeterminism
    checks[f"{prefix}_zero_candidate_rejection_margin"] = signal / atol
    try:
        torch.testing.assert_close(
            reference_delta,
            candidate_delta,
            rtol=2e-2,
            atol=atol,
        )
    except AssertionError as exc:
        raise AssertionError(
            f"{prefix} independent delta check failed: error={error}, "
            f"signal={signal}, atol={atol}"
        ) from exc


def _correctness(fixture: PipelineFixture, **schedule) -> dict[str, object]:
    fixture.reset_hidden()
    fixture.invoke("N0")
    torch.cuda.synchronize()
    assert fixture.last_output is not None
    base_only = fixture.last_output.clone()
    reference = _checked_output(fixture, "C0", **schedule)
    reference_repeat = _checked_output(fixture, "C0", **schedule)
    oracle_nondeterminism = _max_abs_diff(
        reference.float() - base_only.float(),
        reference_repeat.float() - base_only.float(),
    )
    checks: dict[str, object] = {
        "oracle": "independent_production_C0_minus_matched_DeepGEMM_N0",
        "token_lora_mapping_rows": int(fixture.lora_info.token_lora_mapping.numel()),
        "active_rows": int((fixture.lora_info.token_lora_mapping >= 0).sum().item()),
        "base_only_rows": int((fixture.lora_info.token_lora_mapping < 0).sum().item()),
        "c0_repeat_delta_max_abs_error": oracle_nondeterminism,
        "correctness_tolerance_policy": (
            "max(1pct_signal,2^-11,2x_sum_of_measured_C0_and_candidate_"
            "repeat_noise),"
            "capped_at_25pct_signal"
        ),
    }
    candidate_repeats = {}
    for variant in ("C1", "C2P", "C2F", "C3"):
        candidate = _checked_output(fixture, variant, **schedule)
        candidate_repeat = _checked_output(fixture, variant, **schedule)
        candidate_nondeterminism = _max_abs_diff(candidate, candidate_repeat)
        candidate_repeats[variant] = (candidate, candidate_repeat)
        checks[f"c0_{variant.lower()}_max_abs"] = _max_abs_diff(reference, candidate)
        _check_independent_delta(
            checks,
            f"c0_{variant.lower()}",
            reference,
            candidate,
            base_only,
            oracle_nondeterminism=oracle_nondeterminism,
            candidate_nondeterminism=candidate_nondeterminism,
        )

    c3_repeat, c3_again = candidate_repeats["C3"]
    repeat_error = _max_abs_diff(c3_repeat, c3_again)
    reference_signal = float((reference.float() - base_only.float()).abs().max())
    repeat_limit = min(
        reference_signal / 4.0,
        max(5e-4, 2.0 * oracle_nondeterminism),
    )
    checks["c3_repeat_max_abs_error"] = repeat_error
    checks["c3_repeat_max_allowed_error"] = repeat_limit
    if repeat_error > repeat_limit:
        raise AssertionError(
            f"C3 repeat error {repeat_error} exceeds measured-noise limit "
            f"{repeat_limit}"
        )

    mapping = fixture.lora_info.token_lora_mapping
    base_mask = mapping < 0
    if bool(base_mask.any()):
        c3_base_error = _max_abs_diff(c3_repeat[base_mask], base_only[base_mask])
        checks["base_row_c3_vs_n0_max_abs"] = c3_base_error
        torch.testing.assert_close(
            c3_repeat[base_mask], base_only[base_mask], rtol=0.0, atol=5e-4
        )
    else:
        checks["base_row_c3_vs_n0_max_abs"] = "not_applicable"
    return checks


def _prepare_batch(
    fixture: PipelineFixture,
    variant: str,
    *,
    execution: str,
    check_graph: bool,
    **schedule,
):
    from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode

    fixture.reset_hidden()
    _invoke(fixture, variant, **schedule)
    torch.cuda.synchronize()
    assert fixture.last_output is not None
    eager = fixture.last_output.clone() if check_graph else None
    fixture.reset_hidden()

    with ExitStack() as stack:
        if execution == "cuda_graph":
            stack.enter_context(model_capture_mode())
        batch = make_batch(
            lambda: _invoke(fixture, variant, **schedule),
            execution=execution,
            inner_iterations=1,
        )
    capture_resource_count = len(batch.capture_resources)
    expected_c3_resources = int(execution == "cuda_graph" and variant == "C3")
    if variant == "C3" and capture_resource_count != expected_c3_resources:
        raise AssertionError(
            "C3 graph-scoped resource count does not match the captured topology: "
            f"actual={capture_resource_count}, expected={expected_c3_resources}"
        )

    graph_check = None
    if execution == "cuda_graph" and check_graph:
        replay_errors = []
        for _ in range(3):
            fixture.reset_hidden()
            batch.run()
            torch.cuda.synchronize()
            assert fixture.last_output is not None and eager is not None
            replay_errors.append(_max_abs_diff(eager, fixture.last_output))
            torch.testing.assert_close(eager, fixture.last_output, rtol=0.0, atol=3e-3)
        if len(batch.capture_resources) != capture_resource_count:
            raise AssertionError(
                "CUDA graph replay unexpectedly changed graph-scoped resources"
            )
        graph_check = {
            "replays_checked": len(replay_errors),
            "eager_graph_max_abs": max(replay_errors),
            "rtol": 0.0,
            "atol": 3e-3,
            "graph_scoped_resources_after_capture": capture_resource_count,
            "graph_scoped_resources_after_replays": len(batch.capture_resources),
        }
    return batch, graph_check


def _benchmark(
    fixture: PipelineFixture,
    variant: str,
    *,
    execution: str,
    warmup: int,
    samples: int,
    **schedule,
) -> dict[str, object]:
    batch, graph_check = _prepare_batch(
        fixture,
        variant,
        execution=execution,
        check_graph=True,
        **schedule,
    )
    timing = time_cuda_events(
        batch.run,
        launches_per_batch=1,
        warmup=warmup,
        samples=samples,
        before_sample=fixture.reset_hidden,
    )
    print(
        f"{fixture.case.case_id} {variant} {execution}: "
        f"p50={timing.p50_us:.3f} us "
        f"p20/p80={timing.p20_us:.3f}/{timing.p80_us:.3f} us"
    )
    return {
        "variant": variant,
        "execution": execution,
        "timing": asdict(timing),
        "graph_correctness": graph_check,
    }


def _counterbalanced_orders(variants: tuple[str, ...], repeats: int):
    """Rotate and reverse the candidate order to spread thermal/time drift."""
    for repeat in range(repeats):
        rotation = repeat % len(variants)
        order = variants[rotation:] + variants[:rotation]
        if (repeat // len(variants)) % 2:
            order = tuple(reversed(order))
        yield repeat, order


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "h200", "gb300"), default="auto")
    parser.add_argument("--case-id")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), default="all")
    parser.add_argument("--rank", type=int, choices=(32, 64, 128))
    parser.add_argument(
        "--c1-overlap-policy",
        choices=("production_auto", "force"),
        default="force",
    )
    parser.add_argument(
        "--consumer-schedule", choices=("pair", "aligned"), default="aligned"
    )
    parser.add_argument("--consumer-block-n", type=int, default=64)
    parser.add_argument("--consumer-warps", type=int, default=4)
    parser.add_argument("--finalize-block-h", type=int, default=32)
    parser.add_argument("--finalize-warps", type=int, default=4)
    parser.add_argument("--routed-scaling-factor", type=float, default=1.75)
    parser.add_argument(
        "--execution", choices=("eager", "cuda_graph", "both"), default="cuda_graph"
    )
    parser.add_argument("--mode", choices=("time", "nsys"), default="time")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--counterbalance-repeats", type=int, default=3)
    parser.add_argument("--profile-iterations", type=int, default=5)
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    device = _detect_device(args.device)
    if args.list_cases:
        _list_cases(device)
        return 0
    case = _with_rank(_select_case(device, args.case_id), args.rank)
    if not case.adapters.l_active:
        raise ValueError("C3 requires active LoRA rows")
    if case.model.num_slices != 2 or case.model.activation != "swiglu":
        raise NotImplementedError("C3 v1 requires gated SwiGLU")
    if args.counterbalance_repeats <= 0:
        raise ValueError("--counterbalance-repeats must be positive")
    variants = VARIANTS if args.variant == "all" else (args.variant,)
    executions = (
        ("eager", "cuda_graph") if args.execution == "both" else (args.execution,)
    )
    if args.mode == "nsys" and (len(variants) != 1 or len(executions) != 1):
        raise ValueError("Nsight mode requires one variant and one execution")

    schedule = {
        "consumer_schedule": args.consumer_schedule,
        "consumer_block_n": args.consumer_block_n,
        "consumer_warps": args.consumer_warps,
        "finalize_block_h": args.finalize_block_h,
        "finalize_warps": args.finalize_warps,
        # Fixed by the admitted case before capture.  Never scan the mapping
        # tensor inside an eager iteration or CUDA graph replay.
        "has_base_rows": bool(case.adapters.b_base),
    }
    with _single_rank_runtime():
        fixture = _build_fixture(
            case,
            need_lora=True,
            c1_overlap_policy=args.c1_overlap_policy,
        )
        fixture.runner_config.routed_scaling_factor = args.routed_scaling_factor
        checks = None if args.skip_check else _correctness(fixture, **schedule)
        runs = []
        comparisons = []
        if args.mode == "time":
            for execution in executions:
                for repeat, order in _counterbalanced_orders(
                    tuple(variants), args.counterbalance_repeats
                ):
                    group = {}
                    for variant in order:
                        run = _benchmark(
                            fixture,
                            variant,
                            execution=execution,
                            warmup=args.warmup,
                            samples=args.samples,
                            **schedule,
                        )
                        run.update(
                            {
                                "counterbalance_repeat": repeat,
                                "counterbalance_order": list(order),
                            }
                        )
                        runs.append(run)
                        group[variant] = run["timing"]["p50_us"]
                    if "C3" in group:
                        comparisons.append(
                            {
                                "execution": execution,
                                "repeat": repeat,
                                "order": list(order),
                                "p50_us": group,
                                "c3_vs_pct": {
                                    variant: (group["C3"] / value - 1.0) * 100.0
                                    for variant, value in group.items()
                                    if variant != "C3"
                                },
                            }
                        )
        else:
            variant = variants[0]
            execution = executions[0]
            batch, graph_check = _prepare_batch(
                fixture,
                variant,
                execution=execution,
                check_graph=True,
                **schedule,
            )
            label = (
                f"sgl_lora_moe::C3_overlap::{case.case_id}::{variant}::" f"{execution}"
            )
            with cuda_profile_range(label):
                for _ in range(args.profile_iterations):
                    fixture.reset_hidden()
                    batch.run()
            runs.append(
                {
                    "variant": variant,
                    "execution": execution,
                    "profile_label": label,
                    "profile_iterations": args.profile_iterations,
                    "graph_correctness": graph_check,
                }
            )

        summaries = []
        for execution in executions:
            matched = [row for row in comparisons if row["execution"] == execution]
            for baseline in VARIANTS:
                if baseline == "C3":
                    continue
                values = [
                    row["c3_vs_pct"][baseline]
                    for row in matched
                    if baseline in row["c3_vs_pct"]
                ]
                if values:
                    summaries.append(
                        {
                            "execution": execution,
                            "comparison": f"C3_vs_{baseline}",
                            "num_matched_groups": len(values),
                            "median_pct": median(values),
                            "min_pct": min(values),
                            "max_pct": max(values),
                            "winner_rule": (
                                "no_claim_when_interval_crosses_zero_or_effect_is_"
                                "smaller_than_observed_dispersion"
                            ),
                        }
                    )

        result = {
            "schema_version": 1,
            "scope": "M0_local_bf16_c3_gate_a_overlap_experimental",
            "contract": {
                "production_dispatch_changed": False,
                "overlapped_work": (
                    "gate_up_lora_A_shrink_vs_base_prepare_plus_gateup_gemm"
                ),
                "main_stream_before_fork": [
                    "all_gate_A_and_gate_B_route_metadata",
                    "gate_A_destination",
                    "down_A_accumulation_destination",
                    "join_event",
                ],
                "join_boundary": "immediately_before_fused_gate_B_activation_down_A",
                "post_join": [
                    "fused_gate_B_swiglu_down_A",
                    "base_down",
                    "fused_down_B_base_finalize",
                ],
                "side_stream_allocations": False,
                "capture_event_keepalive": True,
                "c1_overlap_policy": args.c1_overlap_policy,
                "routed_scaling_factor": args.routed_scaling_factor,
            },
            "case": _case_summary(case),
            "route_metadata": fixture.route_metadata,
            "environment": _environment(args),
            "schedule": schedule,
            "correctness": checks,
            "runs": runs,
            "counterbalanced_comparisons": comparisons,
            "comparison_summary": summaries,
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
