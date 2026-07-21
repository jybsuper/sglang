#!/usr/bin/env python3
"""Tune BF16 virtual-expert LoRA-A (shrink) launch schedules.

This benchmark calls the production private Triton kernel with explicit launch
metadata.  It does not copy or modify the kernel.  Routing is prepared once,
outside the measured region, for the candidate's exact ``BLOCK_SIZE_M``.

Examples::

    python benchmark/kernels/lora_moe/bench_shrink_schedules.py --list-configs
    python benchmark/kernels/lora_moe/bench_shrink_schedules.py \
      --case-id p0-qwen3.5-35b-a3b-sparse-h200 --site gate \
      --config production --execution cuda_graph
    python benchmark/kernels/lora_moe/bench_shrink_schedules.py \
      --config bn64-sk4 --block-k 128 --num-warps 4

``--all-configs`` is a small, curated sweep.  The script deliberately does not
form a Cartesian product; wider searches belong in a separate orchestrator.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from benchmark.kernels.lora_moe.bench_local import (
    SiteFixture,
    _build_fixture,
    _case_summary,
    _detect_device,
    _environment,
    _select_case,
)
from benchmark.kernels.lora_moe.profiling import (
    RunConfig,
    cuda_profile_range,
    make_batch,
    time_cuda_events,
)


@dataclass(frozen=True, slots=True)
class ShrinkSchedule:
    key: str
    block_m: int
    block_n: int
    block_k: int
    split_k: int
    num_warps: int
    num_stages: int
    note: str = ""

    def validate(self) -> None:
        for name in ("block_m", "block_n", "block_k"):
            value = getattr(self, name)
            if value < 16 or value & (value - 1):
                raise ValueError(f"{name} must be a power of two >= 16, got {value}")
        if self.split_k < 1:
            raise ValueError("split_k must be positive")
        if self.num_warps not in (2, 4, 8):
            raise ValueError("num_warps must be one of 2, 4, or 8")
        if self.num_stages < 1:
            raise ValueError("num_stages must be positive")


# Explicit, intentionally sparse candidates.  BN16/32/64 exercise multiple N
# tiles for gate-A rank 64 (N=128); BN128 mirrors today's one-N-tile launcher.
CURATED_SCHEDULES: tuple[ShrinkSchedule, ...] = (
    ShrinkSchedule("bn16-sk1", 16, 16, 256, 1, 4, 3, "narrow N tiles, no split-K"),
    ShrinkSchedule("bn32-sk1", 16, 32, 256, 1, 4, 3, "two/four N tiles, no split-K"),
    ShrinkSchedule("bn64-sk1", 16, 64, 256, 1, 4, 3, "two N tiles at gate rank 64"),
    ShrinkSchedule("bn128-sk1", 16, 128, 256, 1, 4, 3, "current gate rank-64 tile"),
    ShrinkSchedule("bn32-sk2", 16, 32, 256, 2, 4, 3),
    ShrinkSchedule("bn64-sk2", 16, 64, 256, 2, 4, 3),
    ShrinkSchedule("bn64-sk4", 16, 64, 256, 4, 4, 3),
    ShrinkSchedule("bn64-sk8", 16, 64, 256, 8, 4, 3),
    ShrinkSchedule("bk128-bn64-sk4", 16, 64, 128, 4, 4, 3),
    ShrinkSchedule("bk64-bn64-sk8", 16, 64, 64, 8, 4, 3),
    ShrinkSchedule("bm32-bn64-sk2", 32, 64, 256, 2, 4, 3),
    ShrinkSchedule("bm32-bn128-sk1-w2", 32, 128, 256, 1, 2, 2),
)
_SCHEDULES_BY_KEY = {schedule.key: schedule for schedule in CURATED_SCHEDULES}


@dataclass(slots=True)
class RoutingPlan:
    sorted_token_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_tokens_post_padded: torch.Tensor
    virtual_topk_ids: torch.Tensor
    block_m: int
    virtual_num_experts: int

    def metadata(self) -> dict[str, int]:
        actual_pairs = int(self.num_tokens_post_padded.item())
        return {
            "block_m": self.block_m,
            "virtual_num_experts": self.virtual_num_experts,
            "allocated_pair_slots": self.sorted_token_ids.numel(),
            "allocated_expert_blocks": self.expert_ids.numel(),
            "post_padding_pair_slots": actual_pairs,
        }


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _align_virtual_experts(
    virtual_topk_ids: torch.Tensor,
    *,
    block_m: int,
    virtual_num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mirror production alignment while keeping route construction untimed."""
    from sglang.srt.lora.sgl_lora.triton_ops.virtual_experts import (
        _align_block_size_large,
    )

    if virtual_num_experts >= 1024:
        return _align_block_size_large(virtual_topk_ids, block_m, virtual_num_experts)

    from sglang.kernels.ops.moe import moe_align_block_size

    num_pairs = virtual_topk_ids.numel()
    bucket_count = virtual_num_experts + 1
    if num_pairs < bucket_count:
        max_pairs_padded = num_pairs * block_m
    else:
        max_pairs_padded = num_pairs + bucket_count * (block_m - 1)
    sorted_token_ids = torch.empty(
        max_pairs_padded, dtype=torch.int32, device=virtual_topk_ids.device
    )
    expert_ids = torch.empty(
        _ceil_div(max_pairs_padded, block_m),
        dtype=torch.int32,
        device=virtual_topk_ids.device,
    )
    num_tokens_post_padded = torch.empty(
        1, dtype=torch.int32, device=virtual_topk_ids.device
    )
    cumsum_buffer = torch.empty(
        virtual_num_experts + 2,
        dtype=torch.int32,
        device=virtual_topk_ids.device,
    )
    moe_align_block_size(
        virtual_topk_ids,
        bucket_count,
        block_m,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        cumsum_buffer,
        True,
    )
    return sorted_token_ids, expert_ids, num_tokens_post_padded


def _build_a_routing_plan(fixture: SiteFixture, block_m: int) -> RoutingPlan:
    """Prebuild the production-equivalent A route for one candidate BM."""
    from sglang.srt.lora.sgl_lora.triton_ops.virtual_experts import (
        _fused_virtual_topk_ids,
        fused_sanitize_expert_ids,
    )

    num_experts = fixture.lora_a.shape[1]
    max_loras = fixture.lora_a.shape[0]
    shared_outer = fixture.case.adapters.shared_outer
    virtual_topk_ids, _, virtual_num_experts = _fused_virtual_topk_ids(
        fixture.topk_ids,
        fixture.token_lora_mapping,
        num_experts,
        shared_outer,
        max_loras,
        local_expert_offset=0,
        local_num_experts=fixture.case.e_local,
    )
    sorted_ids, expert_ids, post_padded = _align_virtual_experts(
        virtual_topk_ids,
        block_m=block_m,
        virtual_num_experts=virtual_num_experts,
    )

    # Match production's local, non-EP tight upper bound.  The scalar post-pad
    # count remains on device; trimming uses only static allocation dimensions.
    num_pairs = fixture.topk_ids.numel()
    max_nonempty = min(num_pairs, virtual_num_experts)
    tight_padded = (
        _ceil_div(num_pairs + max_nonempty * (block_m - 1), block_m) * block_m
    )
    sorted_ids = sorted_ids[:tight_padded]
    expert_ids = expert_ids[: tight_padded // block_m]
    if max_loras != 1:
        expert_ids = fused_sanitize_expert_ids(expert_ids, virtual_num_experts)
    return RoutingPlan(
        sorted_token_ids=sorted_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=post_padded,
        virtual_topk_ids=virtual_topk_ids,
        block_m=block_m,
        virtual_num_experts=virtual_num_experts,
    )


def _production_schedule(fixture: SiteFixture, plan: RoutingPlan) -> ShrinkSchedule:
    from sglang.srt.lora.sgl_lora.triton_ops.virtual_experts import (
        _get_moe_lora_shrink_split_k,
    )

    weight = _merged_a_weight(fixture)
    n = weight.shape[1]
    block_m = 16 if fixture.case.t_local < 512 else 32
    if plan.block_m != block_m:
        raise ValueError("production route plan does not match production BM")
    config = {"BLOCK_SIZE_M": block_m}
    split_k = _get_moe_lora_shrink_split_k(weight, plan.sorted_token_ids, config)
    return ShrinkSchedule(
        key="production",
        block_m=block_m,
        block_n=_next_power_of_two(n),
        block_k=256,
        split_k=split_k,
        num_warps=(
            (4 if fixture.case.t_local <= 4 or n >= 32 else 2)
            if fixture.case.t_local < 512
            else 2
        ),
        num_stages=3 if fixture.case.t_local < 512 else 2,
        note="resolved production launcher",
    )


def _merged_a_weight(fixture: SiteFixture) -> torch.Tensor:
    weight = fixture.lora_a
    return weight.reshape(weight.shape[0] * weight.shape[1], *weight.shape[2:])


def _apply_overrides(
    schedule: ShrinkSchedule, args: argparse.Namespace
) -> ShrinkSchedule:
    replacements = {
        "block_m": args.block_m,
        "block_n": args.block_n,
        "block_k": args.block_k,
        "split_k": args.split_k,
        "num_warps": args.num_warps,
        "num_stages": args.num_stages,
    }
    replacements = {
        key: value for key, value in replacements.items() if value is not None
    }
    if replacements:
        schedule = replace(schedule, key=f"{schedule.key}+override", **replacements)
    schedule.validate()
    return schedule


@dataclass(slots=True)
class PreparedShrink:
    schedule: ShrinkSchedule
    plan: RoutingPlan
    output: torch.Tensor
    launch: Callable[[], None]
    grid: tuple[int]
    num_m_blocks: int
    num_n_blocks: int


def _prepare_shrink(
    fixture: SiteFixture,
    schedule: ShrinkSchedule,
    plan: RoutingPlan,
) -> PreparedShrink:
    from sglang.srt.lora.sgl_lora.triton_ops.virtual_experts import (
        _get_pdl_launch_metadata,
        _moe_lora_shrink_splitk_kernel,
    )

    weight = _merged_a_weight(fixture)
    n, k = weight.shape[1:]
    output = torch.empty(
        (fixture.topk_ids.numel(), n),
        dtype=fixture.hidden_states.dtype,
        device=fixture.hidden_states.device,
    )
    num_m_blocks = _ceil_div(plan.sorted_token_ids.shape[0], schedule.block_m)
    num_n_blocks = _ceil_div(n, schedule.block_n)
    grid = (schedule.split_k * num_m_blocks * num_n_blocks,)
    input_top_k = (
        1
        if fixture.hidden_states.shape[0] == fixture.topk_ids.numel()
        else fixture.topk_ids.shape[1]
    )
    enable_pdl, pdl_kwargs = _get_pdl_launch_metadata()

    def launch() -> None:
        # This zero is part of the operator semantics and measured/captured.
        # Every split writes the same destination through atomic_add.
        if schedule.split_k > 1:
            output.zero_()
        _moe_lora_shrink_splitk_kernel[grid](
            fixture.hidden_states,
            weight,
            output,
            plan.sorted_token_ids,
            plan.expert_ids,
            plan.num_tokens_post_padded,
            n,
            k,
            fixture.topk_ids.numel(),
            fixture.hidden_states.stride(0),
            fixture.hidden_states.stride(1),
            weight.stride(0),
            weight.stride(1),
            weight.stride(2),
            output.stride(0),
            output.stride(1),
            top_k=input_top_k,
            BLOCK_SIZE_M=schedule.block_m,
            BLOCK_SIZE_N=schedule.block_n,
            BLOCK_SIZE_K=schedule.block_k,
            GROUP_SIZE_M=1,
            SPLIT_K=schedule.split_k,
            ENABLE_PDL=enable_pdl,
            num_warps=schedule.num_warps,
            num_stages=schedule.num_stages,
            **pdl_kwargs,
        )

    return PreparedShrink(
        schedule=schedule,
        plan=plan,
        output=output,
        launch=launch,
        grid=grid,
        num_m_blocks=num_m_blocks,
        num_n_blocks=num_n_blocks,
    )


def _production_reference(fixture: SiteFixture) -> torch.Tensor:
    fixture.invoke("routing", direct=True)
    fixture.invoke("shrink", direct=True)
    torch.cuda.synchronize()
    return fixture.intermediate.view(-1, fixture.intermediate.shape[-1]).clone()


def _check(op: PreparedShrink, reference: torch.Tensor) -> dict[str, float]:
    op.launch()
    torch.cuda.synchronize()
    torch.testing.assert_close(op.output, reference, rtol=3e-2, atol=3e-2)
    error = (op.output.float() - reference.float()).abs()
    return {
        "max_abs_error": float(error.max().item()),
        "mean_abs_error": float(error.mean().item()),
    }


def _run_schedule(
    fixture: SiteFixture,
    reference: torch.Tensor,
    schedule: ShrinkSchedule,
    plan: RoutingPlan,
    args: argparse.Namespace,
) -> dict[str, object]:
    op = _prepare_shrink(fixture, schedule, plan)
    op.launch()
    torch.cuda.synchronize()
    correctness = None if args.skip_check else _check(op, reference)
    run_config = RunConfig(
        mode=args.mode,
        execution=args.execution,
        warmup=args.warmup,
        samples=args.samples,
        inner_iterations=args.inner_iterations,
        profile_iterations=args.profile_iterations,
    )
    batch = make_batch(
        op.launch,
        execution=run_config.execution,
        inner_iterations=run_config.inner_iterations,
    )
    n, k = _merged_a_weight(fixture).shape[1:]
    result: dict[str, object] = {
        "config": asdict(schedule),
        "dimensions": {"M_valid": fixture.topk_ids.numel(), "N": n, "K": k},
        "grid": {
            "num_m_blocks_allocated": op.num_m_blocks,
            "num_n_blocks": op.num_n_blocks,
            "split_k": schedule.split_k,
            "programs": op.grid[0],
        },
        "routing": {
            "valid_pair_slots": fixture.topk_ids.numel(),
            **plan.metadata(),
        },
        "zero_in_semantic_launch": schedule.split_k > 1,
        "correctness": correctness,
    }
    if run_config.mode == "time":
        timing = time_cuda_events(
            batch.run,
            launches_per_batch=batch.launches_per_batch,
            warmup=run_config.warmup,
            samples=run_config.samples,
        )
        result["timing"] = asdict(timing)
        print(
            f"{schedule.key:<28} BM/BN/BK={schedule.block_m}/{schedule.block_n}/"
            f"{schedule.block_k} SK={schedule.split_k} "
            f"W/S={schedule.num_warps}/{schedule.num_stages}: "
            f"p50={timing.p50_us:.3f} us "
            f"p20/p80={timing.p20_us:.3f}/{timing.p80_us:.3f} us"
        )
    else:
        label = (
            f"sgl_lora_moe::K0::shrink::{fixture.site}::{fixture.case.case_id}::"
            f"{schedule.key}::{args.execution}::pdl=auto"
        )
        with cuda_profile_range(label):
            for _ in range(run_config.profile_iterations):
                batch.run()
        result["profile"] = {
            "mode": run_config.mode,
            "label": label,
            "iterations": run_config.profile_iterations,
        }
        print(f"captured {label}")
    return result


def _list_configs() -> None:
    print("production                    resolved from the current production policy")
    for schedule in CURATED_SCHEDULES:
        print(
            f"{schedule.key:<28} BM/BN/BK={schedule.block_m}/{schedule.block_n}/"
            f"{schedule.block_k} SK={schedule.split_k} "
            f"W/S={schedule.num_warps}/{schedule.num_stages}  {schedule.note}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-configs", action="store_true")
    parser.add_argument("--all-configs", action="store_true")
    parser.add_argument("--device", choices=("auto", "h200", "gb300"), default="auto")
    parser.add_argument("--case-id")
    parser.add_argument("--site", choices=("gate", "down"), default="gate")
    parser.add_argument(
        "--config", choices=("production", *_SCHEDULES_BY_KEY), default="production"
    )
    parser.add_argument("--block-m", type=int)
    parser.add_argument("--block-n", type=int)
    parser.add_argument("--block-k", type=int)
    parser.add_argument("--split-k", type=int)
    parser.add_argument("--num-warps", type=int)
    parser.add_argument("--num-stages", type=int)
    parser.add_argument("--mode", choices=("time", "nsys", "ncu"), default="time")
    parser.add_argument("--execution", choices=("eager", "cuda_graph"), default="eager")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--inner-iterations", type=int, default=10)
    parser.add_argument("--profile-iterations", type=int, default=1)
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args(argv)


def _has_overrides(args: argparse.Namespace) -> bool:
    return any(
        getattr(args, name) is not None
        for name in (
            "block_m",
            "block_n",
            "block_k",
            "split_k",
            "num_warps",
            "num_stages",
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_configs:
        _list_configs()
        return 0
    if args.all_configs and _has_overrides(args):
        raise ValueError(
            "explicit launch overrides cannot be combined with --all-configs"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    device = _detect_device(args.device)
    case = _select_case(device, args.case_id)
    fixture = _build_fixture(case, args.site)
    reference = _production_reference(fixture)

    templates: list[ShrinkSchedule | None]
    if args.all_configs:
        templates = [None, *CURATED_SCHEDULES]
    else:
        templates = [
            None if args.config == "production" else _SCHEDULES_BY_KEY[args.config]
        ]

    plan_cache: dict[int, RoutingPlan] = {}

    def plan_for(block_m: int) -> RoutingPlan:
        if block_m not in plan_cache:
            plan_cache[block_m] = _build_a_routing_plan(fixture, block_m)
            torch.cuda.synchronize()
        return plan_cache[block_m]

    results = []
    for template in templates:
        if template is None:
            production_bm = 16 if case.t_local < 512 else 32
            plan = plan_for(production_bm)
            schedule = _production_schedule(fixture, plan)
        else:
            schedule = template
            plan = plan_for(schedule.block_m)
        schedule = _apply_overrides(schedule, args)
        # An override may have changed BM after the initial plan was selected.
        plan = plan_for(schedule.block_m)
        results.append(_run_schedule(fixture, reference, schedule, plan, args))

    report = {
        "environment": _environment(args),
        "case": _case_summary(case),
        "site": args.site,
        "reference": "production shrink stage",
        "pdl_policy": "architecture_auto",
        "results": results,
    }
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
