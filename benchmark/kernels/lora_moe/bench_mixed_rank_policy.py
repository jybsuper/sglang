#!/usr/bin/env python3
"""Compare full-R_max padding with load-time packed MoE-LoRA rank buckets.

Both candidates use the production BF16 SGL LoRA M0 runner.  The only changed
contract is resident factor rank: ``padded_rmax`` executes zero-tailed R_max
factors, while ``packed_bucket`` executes the active bucket's R_phys factors.
Factor transforms, rank planning, and graph selection happen before the timed
forward.  Timing is pairwise counterbalanced and raw per-sample values are
retained for audit and static-policy-regret analysis.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from contextlib import ExitStack
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from benchmark.kernels.lora_moe.bench_moe_pipeline import (
    PipelineFixture,
    _build_fixture,
    _detect_device,
    _single_rank_runtime,
)
from benchmark.kernels.lora_moe.matrix import mixed_rank_cases
from benchmark.kernels.lora_moe.mixed_rank_policy import (
    bind_factor_bundle,
    load_factor_rank_representation,
    logical_factor_bytes,
    poison_canonical_factor_source,
)
from benchmark.kernels.lora_moe.profiling import make_batch, summarize_timings_us
from sglang.srt.lora.sgl_lora.rank_policy import build_moe_lora_rank_plan

POLICIES = ("padded_rmax", "packed_bucket")
ORDERS = ("forward", "reverse")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(args: list[str], fallback: str = "unknown") -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return fallback


def _nvidia_query(field: str) -> str:
    # nvidia-smi indexes the physical inventory, while torch device 0 is the
    # first CUDA-visible device. Resolve the same physical lane used by torch.
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "0")
    physical_device = visible_devices.split(",", maxsplit=1)[0].strip() or "0"
    try:
        return subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={field}",
                "--format=csv,noheader,nounits",
                "-i",
                physical_device,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _source_provenance() -> dict[str, object]:
    paths = (
        Path(__file__),
        Path(__file__).with_name("mixed_rank_policy.py"),
        Path(__file__).with_name("matrix.py"),
        Path(__file__).with_name("cases.py"),
        Path(__file__).with_name("profiling.py"),
        Path(__file__).with_name("bench_moe_pipeline.py"),
        Path(__file__).with_name("run_mixed_rank_policy_matrix.sh"),
        Path(__file__).with_name("summarize_mixed_rank_policy.py"),
        REPO_ROOT / "python/sglang/srt/lora/sgl_lora/rank_policy.py",
        REPO_ROOT / "python/sglang/srt/lora/sgl_lora/moe_lora_runner.py",
        REPO_ROOT / "python/sglang/srt/lora/sgl_lora/base_gemm.py",
        REPO_ROOT / "python/sglang/srt/lora/sgl_lora/workspace.py",
        REPO_ROOT / "python/sglang/srt/lora/sgl_lora/quant_info.py",
        REPO_ROOT / "python/sglang/srt/lora/sgl_lora/triton_ops/virtual_experts.py",
        REPO_ROOT / "python/sglang/srt/lora/sgl_lora/triton_ops/expand.py",
        REPO_ROOT / "python/sglang/srt/lora/sgl_lora/triton_ops/act.py",
        REPO_ROOT / "python/sglang/srt/lora/lora_moe_runners.py",
        REPO_ROOT / "python/sglang/srt/model_executor/runner_utils/capture_mode.py",
        REPO_ROOT
        / "python/sglang/srt/model_executor/runner_utils/capture_resources.py",
    )
    return {
        "git_revision": os.getenv("SGLANG_LORA_BENCH_REVISION")
        or os.getenv("SGL_LORA_BENCH_REVISION")
        or _git(["rev-parse", "HEAD"]),
        "git_dirty": bool(_git(["status", "--porcelain"], "unknown")),
        "files": {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in paths},
    }


def _environment(args: argparse.Namespace) -> dict[str, object]:
    try:
        import triton

        triton_version = triton.__version__
    except (ImportError, AttributeError):
        triton_version = "unknown"
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": triton_version,
        "cuda_runtime": torch.version.cuda,
        "cuda_driver": _nvidia_query("driver_version"),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_uuid": _nvidia_query("uuid"),
        "gpu_clock_sm_mhz": _nvidia_query("clocks.current.sm"),
        "gpu_clock_memory_mhz": _nvidia_query("clocks.current.memory"),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "command": [sys.executable, *sys.argv],
        "cli": vars(args),
        "source": _source_provenance(),
    }


def _select_case(device: str, case_id: str):
    for case in mixed_rank_cases(device):
        if case.case_id == case_id:
            return case
    choices = ", ".join(case.case_id for case in mixed_rank_cases(device))
    raise ValueError(f"unknown case {case_id!r}; choose from {choices}")


def _slot_ranks(case) -> tuple[int, ...]:
    return (case.adapters.rank,) * case.adapters.l_active + (0,) * (
        case.adapters.l_capacity - case.adapters.l_active
    )


def _case_metadata(case) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "device": case.device,
        "model": case.model.key,
        "phase": case.phase,
        "T": case.t_local,
        "K": case.model.top_k,
        "H_moe": case.model.h_moe,
        "I": case.i_local,
        "E_local": case.e_local,
        "R": case.adapters.rank,
        "R_max": case.adapters.max_rank,
        "L_active": case.adapters.l_active,
        "B_base": case.adapters.b_base,
        "L_capacity": case.adapters.l_capacity,
        "slot_ranks": list(_slot_ranks(case)),
        "pipeline": "N0" if case.adapters.l_active == 0 else "C0",
        "semantic_boundary": "bf16_hidden_plus_standard_topk_to_bf16_T_by_H",
    }


def _assert_common_inputs(lhs: PipelineFixture, rhs: PipelineFixture) -> None:
    assert torch.equal(lhs.hidden_seed, rhs.hidden_seed)
    assert torch.equal(lhs.base_weights[0], rhs.base_weights[0])
    assert torch.equal(lhs.base_weights[1], rhs.base_weights[1])
    assert torch.equal(lhs.topk_output.topk_ids, rhs.topk_output.topk_ids)
    assert torch.equal(lhs.topk_output.topk_weights, rhs.topk_output.topk_weights)


def _build_candidates(case, *, order: str):
    has_lora = case.adapters.l_active > 0
    if not has_lora:
        fixtures = {
            policy: _build_fixture(
                case,
                need_lora=False,
                route_pattern="uniform_iid_without_replacement",
                route_seed=19,
            )
            for policy in POLICIES
        }
        _assert_common_inputs(fixtures["padded_rmax"], fixtures["packed_bucket"])
        plans = {
            policy: build_moe_lora_rank_plan(
                (0,) * case.adapters.l_capacity,
                allocated_rank=case.adapters.max_rank,
                policy=policy,
            )
            for policy in POLICIES
        }
        return (
            fixtures,
            plans,
            {
                policy: {
                    "resident_bytes": 0,
                    "logical_factor_bytes": 0,
                    "load_time_setup_ms_one_shot": 0.0,
                    "load_time_setup_measurement": "not_applicable_all_base",
                    "load_time_transform_order": list(
                        POLICIES if order == "forward" else tuple(reversed(POLICIES))
                    ),
                    "physical_rank": None,
                }
                for policy in POLICIES
            },
        )

    packed_case = replace(
        case,
        adapters=replace(
            case.adapters,
            max_rank=case.adapters.rank,
            physical_rank=case.adapters.rank,
        ),
    )
    padded_fixture = _build_fixture(
        case,
        need_lora=True,
        route_pattern="uniform_iid_without_replacement",
        route_seed=19,
    )
    packed_fixture = _build_fixture(
        packed_case,
        need_lora=True,
        route_pattern="uniform_iid_without_replacement",
        route_seed=19,
    )
    _assert_common_inputs(padded_fixture, packed_fixture)
    slot_ranks = _slot_ranks(case)
    canonical = poison_canonical_factor_source(
        padded_fixture.lora_weights,
        allocated_rank=case.adapters.max_rank,
        num_slices=case.model.num_slices,
        slot_ranks=slot_ranks,
    )
    logical_bytes = logical_factor_bytes(
        canonical,
        allocated_rank=case.adapters.max_rank,
        num_slices=case.model.num_slices,
        slot_ranks=slot_ranks,
    )
    plans = {
        policy: build_moe_lora_rank_plan(
            slot_ranks,
            allocated_rank=case.adapters.max_rank,
            policy=policy,
        )
        for policy in POLICIES
    }
    padded_rank = plans["padded_rmax"].buckets[0].physical_rank
    packed_rank = plans["packed_bucket"].buckets[0].physical_rank
    physical_ranks = {
        "padded_rmax": padded_rank,
        "packed_bucket": packed_rank,
    }
    representations = {
        "padded_rmax": "zero_tailed_full_rmax",
        "packed_bucket": "load_time_rank_bucket",
    }
    transform_order = POLICIES if order == "forward" else tuple(reversed(POLICIES))
    bundles = {}
    for policy in transform_order:
        bundles[policy] = load_factor_rank_representation(
            canonical,
            allocated_rank=case.adapters.max_rank,
            physical_rank=physical_ranks[policy],
            num_slices=case.model.num_slices,
            slot_ranks=slot_ranks,
            representation=representations[policy],
        )
    bind_factor_bundle(padded_fixture, bundles["padded_rmax"])
    bind_factor_bundle(packed_fixture, bundles["packed_bucket"])
    fixtures = {
        "padded_rmax": padded_fixture,
        "packed_bucket": packed_fixture,
    }
    factor_metadata = {
        policy: {
            "resident_bytes": bundle.resident_bytes,
            "logical_factor_bytes": logical_bytes,
            "resident_over_logical": bundle.resident_bytes / logical_bytes,
            "load_time_setup_ms_one_shot": bundle.setup_ms,
            "load_time_setup_measurement": (
                "descriptive_one_shot_counterbalanced_across_forward_reverse_"
                "artifacts;not_a_canonical_latency_claim"
            ),
            "load_time_transform_order": list(transform_order),
            "physical_rank": bundle.physical_rank,
            "representation": bundle.representation,
            "source_inactive_tails_poisoned": True,
            "active_physical_tails_zeroed": True,
            "inactive_slots_poisoned": case.adapters.l_active
            < case.adapters.l_capacity,
        }
        for policy, bundle in bundles.items()
    }
    del canonical, bundles
    return fixtures, plans, factor_metadata


def _invoke(fixture: PipelineFixture, pipeline: str) -> None:
    fixture.invoke(pipeline)


def _prepare_candidate(
    fixture: PipelineFixture,
    *,
    pipeline: str,
    execution: str,
    sustained_replays: int,
) -> tuple[object, torch.Tensor, torch.Tensor | None, dict[str, object]]:
    from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode

    base_reference = None
    if pipeline != "N0":
        fixture.reset_hidden()
        _invoke(fixture, "N0")
        torch.cuda.synchronize()
        assert fixture.last_output is not None
        base_reference = fixture.last_output.clone()

    fixture.reset_hidden()
    _invoke(fixture, pipeline)
    torch.cuda.synchronize()
    assert fixture.last_output is not None
    eager_reference = fixture.last_output.clone()
    assert torch.isfinite(eager_reference).all(), "poison reached the eager output"
    fixture.reset_hidden()

    if execution == "cuda_graph":
        with model_capture_mode():
            batch = make_batch(
                lambda: _invoke(fixture, pipeline),
                execution="cuda_graph",
                inner_iterations=1,
            )
        fixture.reset_hidden()
        batch.run()
        torch.cuda.synchronize()
        assert fixture.last_output is not None
        graph_diff = float((fixture.last_output - eager_reference).abs().max().item())
        torch.testing.assert_close(
            fixture.last_output, eager_reference, rtol=0.0, atol=3e-3
        )
        before_allocated = torch.cuda.memory_allocated()
        for _ in range(sustained_replays):
            fixture.reset_hidden()
            batch.run()
        torch.cuda.synchronize()
        after_allocated = torch.cuda.memory_allocated()
        assert fixture.last_output is not None
        sustained_diff = float(
            (fixture.last_output - eager_reference).abs().max().item()
        )
        torch.testing.assert_close(
            fixture.last_output, eager_reference, rtol=0.0, atol=3e-3
        )
        graph_check = {
            "capture_replay_max_abs": graph_diff,
            "sustained_replays": sustained_replays,
            "sustained_replay_max_abs": sustained_diff,
            "memory_allocated_before_replays": before_allocated,
            "memory_allocated_after_replays": after_allocated,
            "memory_allocated_delta_bytes": after_allocated - before_allocated,
            "rank_metadata_allocations_per_replay_by_construction": 0,
            "rank_metadata_allocation_evidence": (
                "static plan bound before capture; reviewed construction "
                "invariant, not allocator instrumentation"
            ),
        }
    else:
        batch = make_batch(
            lambda: _invoke(fixture, pipeline),
            execution="eager",
            inner_iterations=1,
        )
        graph_check = {
            "status": "not_applicable_eager",
            "rank_metadata_allocations_per_forward_by_construction": 0,
            "rank_metadata_allocation_evidence": (
                "static plan bound before launch; reviewed construction "
                "invariant, not allocator instrumentation"
            ),
        }
    return batch, eager_reference, base_reference, graph_check


def _time_counterbalanced(
    fixtures: dict[str, PipelineFixture],
    batches: dict[str, object],
    *,
    order: str,
    warmup: int,
    samples: int,
) -> dict[str, object]:
    resolved_order = POLICIES if order == "forward" else tuple(reversed(POLICIES))
    for _ in range(warmup):
        for policy in resolved_order:
            fixtures[policy].reset_hidden()
            batches[policy].run()
    torch.cuda.synchronize()

    starts = {
        policy: [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
        for policy in POLICIES
    }
    ends = {
        policy: [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
        for policy in POLICIES
    }
    for sample_index in range(samples):
        for policy in resolved_order:
            fixtures[policy].reset_hidden()
            starts[policy][sample_index].record()
            batches[policy].run()
            ends[policy][sample_index].record()
    torch.cuda.synchronize()

    result: dict[str, object] = {}
    for policy in POLICIES:
        raw = [
            start.elapsed_time(end) * 1000.0
            for start, end in zip(starts[policy], ends[policy], strict=True)
        ]
        result[policy] = {
            "timing": asdict(summarize_timings_us(raw, launches_per_batch=1)),
            "raw_samples_us": raw,
        }
    result["resolved_order"] = list(resolved_order)
    return result


def _correctness(
    eager_references: dict[str, torch.Tensor],
    base_references: dict[str, torch.Tensor | None],
    *,
    pipeline: str,
) -> dict[str, object]:
    padded = eager_references["padded_rmax"]
    packed = eager_references["packed_bucket"]
    diff = float((padded - packed).abs().max().item())
    tolerance = 0.0 if pipeline == "N0" else 5e-3
    torch.testing.assert_close(padded, packed, rtol=0.0, atol=tolerance)
    finite = bool(torch.isfinite(padded).all() and torch.isfinite(packed).all())
    if not finite:
        raise AssertionError("poisoned inactive factor data reached an output")

    active_delta = None
    if pipeline != "N0":
        padded_base = base_references["padded_rmax"]
        packed_base = base_references["packed_bucket"]
        assert padded_base is not None and packed_base is not None
        base_diff = float((padded_base - packed_base).abs().max().item())
        torch.testing.assert_close(padded_base, packed_base, rtol=0.0, atol=5e-3)
        padded_delta = padded.float() - padded_base.float()
        packed_delta = packed.float() - packed_base.float()
        delta_difference = padded_delta - packed_delta
        padded_l2 = float(torch.linalg.vector_norm(padded_delta).item())
        packed_l2 = float(torch.linalg.vector_norm(packed_delta).item())
        difference_l2 = float(torch.linalg.vector_norm(delta_difference).item())
        norm_product = padded_l2 * packed_l2
        cosine = (
            float(
                torch.dot(padded_delta.flatten(), packed_delta.flatten()).item()
                / norm_product
            )
            if norm_product > 0.0
            else 0.0
        )
        active_delta = {
            "max_abs": max(
                float(padded_delta.abs().max().item()),
                float(packed_delta.abs().max().item()),
            ),
            "l2": max(padded_l2, packed_l2),
            "nonzero": bool(
                torch.count_nonzero(padded_delta).item()
                and torch.count_nonzero(packed_delta).item()
            ),
            "padded": {
                "max_abs": float(padded_delta.abs().max().item()),
                "l2": padded_l2,
            },
            "packed": {
                "max_abs": float(packed_delta.abs().max().item()),
                "l2": packed_l2,
            },
            "difference": {
                "max_abs": float(delta_difference.abs().max().item()),
                "l2": difference_l2,
                "relative_l2": difference_l2 / max(padded_l2, packed_l2),
            },
            "cosine_similarity": cosine,
            "base_reference_max_abs_difference": base_diff,
        }
        if not active_delta["nonzero"]:
            raise AssertionError("active LoRA case produced a zero delta")
    return {
        "padded_vs_packed_max_abs": diff,
        "rtol": 0.0,
        "atol": tolerance,
        "outputs_finite": finite,
        "output_max_abs": max(
            float(padded.float().abs().max().item()),
            float(packed.float().abs().max().item()),
        ),
        "active_delta": active_delta,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--device", choices=("auto", "h200", "gb300"), default="auto")
    parser.add_argument("--case-id")
    parser.add_argument("--execution", choices=("eager", "cuda_graph"), default="eager")
    parser.add_argument("--order", choices=ORDERS, default="forward")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--sustained-replays", type=int, default=50)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args(argv)


def _main(args: argparse.Namespace) -> int:
    device = (
        args.device
        if args.device != "auto"
        else (_detect_device("auto") if torch.cuda.is_available() else "h200")
    )
    if args.list_cases:
        for case in mixed_rank_cases(device):
            print(case.case_id)
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("mixed-rank M0 benchmark requires CUDA")
    if args.case_id is None:
        raise ValueError("--case-id is required unless --list-cases is used")
    if args.warmup <= 0 or args.samples <= 0 or args.sustained_replays <= 0:
        raise ValueError("warmup, samples, and sustained-replays must be positive")
    device = _detect_device(args.device)
    case = _select_case(device, args.case_id)
    fixtures, plans, factor_metadata = _build_candidates(case, order=args.order)
    pipeline = "N0" if case.adapters.l_active == 0 else "C0"

    batches: dict[str, object] = {}
    eager_references: dict[str, torch.Tensor] = {}
    base_references: dict[str, torch.Tensor | None] = {}
    graph_checks: dict[str, object] = {}
    # Preparation order follows the timing order so forward/reverse artifacts
    # also counterbalance one-time JIT/capture effects (excluded from timing).
    preparation_order = (
        POLICIES if args.order == "forward" else tuple(reversed(POLICIES))
    )
    for policy in preparation_order:
        batch, reference, base_reference, graph_check = _prepare_candidate(
            fixtures[policy],
            pipeline=pipeline,
            execution=args.execution,
            sustained_replays=args.sustained_replays,
        )
        batches[policy] = batch
        eager_references[policy] = reference
        base_references[policy] = base_reference
        graph_checks[policy] = graph_check
    correctness = _correctness(
        eager_references,
        base_references,
        pipeline=pipeline,
    )
    timings = _time_counterbalanced(
        fixtures,
        batches,
        order=args.order,
        warmup=args.warmup,
        samples=args.samples,
    )

    for policy in POLICIES:
        p50 = timings[policy]["timing"]["p50_us"]
        factor_metadata[policy]["rank_plan"] = {
            **asdict(plans[policy]),
            "graph_key": repr(plans[policy].graph_key),
            "plan_lifetime": "adapter_publish_or_graph_family_setup",
            "selection_lifetime": "before_eager_launch_or_cuda_graph_capture",
            "per_forward_rank_metadata_work": "none",
        }
        factor_metadata[policy]["graph_correctness"] = graph_checks[policy]
        factor_metadata[policy]["timing"] = timings[policy]
        factor_metadata[policy]["p50_us"] = p50

    padded_p50 = factor_metadata["padded_rmax"]["p50_us"]
    packed_p50 = factor_metadata["packed_bucket"]["p50_us"]
    winner = "padded_rmax" if padded_p50 <= packed_p50 else "packed_bucket"
    oracle = min(padded_p50, packed_p50)
    comparison = {
        "winner": winner,
        "oracle_p50_us": oracle,
        "packed_vs_padded_percent": 100.0 * (packed_p50 / padded_p50 - 1.0),
        "padded_regret_percent": 100.0 * (padded_p50 / oracle - 1.0),
        "packed_regret_percent": 100.0 * (packed_p50 / oracle - 1.0),
    }
    result = {
        "schema": "sgl_lora_mixed_rank_policy_v2",
        "environment": _environment(args),
        "case": _case_metadata(case),
        "scope": "M0 local MoE",
        "comparison_contract": {
            "common_runner": "production BF16 SGL LoRA M0",
            "common_base_weights_inputs_routes": True,
            "padded_candidate": "R_max resident factors with zero active tails",
            "packed_candidate": "load-time resident R_phys bucket",
            "load_time_transform_included_in_forward_timing": False,
            "rank_plan_or_metadata_included_in_forward_timing": False,
            "logical_rank_metadata_consumed_by_current_padded_kernels": False,
            "actual_rank_semantics": (
                "logical R selects valid factor prefixes; padded kernels execute "
                "zero-tailed R_max, packed kernels execute R_phys"
            ),
            "production_dispatch_modified": False,
        },
        "order": args.order,
        "resolved_order": timings["resolved_order"],
        "correctness": correctness,
        "policies": factor_metadata,
        "comparison": comparison,
        "limitations": [
            "uniform_active_rank_per_forward",
            "bf16_qwen_shape",
            "ws1_local_m0_no_communication",
            "serial_c0_for_active_lora",
            "rank_policy_not_promoted_to_production_dispatch",
            "graph_key_serialized_but_graph_cache_selection_not_exercised",
            "active_graph_key_changes_with_exact_slot_rank_assignment",
        ],
    }
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(
        f"{case.case_id} {args.execution}/{args.order}: "
        f"padded={padded_p50:.3f} us packed={packed_p50:.3f} us "
        f"packed-vs-padded={comparison['packed_vs_padded_percent']:+.2f}%"
    )
    del batches, eager_references, base_references, fixtures
    gc.collect()
    torch.cuda.empty_cache()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    with ExitStack() as stack:
        if not args.list_cases:
            stack.enter_context(_single_rank_runtime())
        return _main(args)


if __name__ == "__main__":
    raise SystemExit(main())
