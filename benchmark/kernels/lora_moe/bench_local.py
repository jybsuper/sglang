#!/usr/bin/env python3
"""Benchmark the current BF16 virtual-expert LoRA A/B operators.

This is the first local (single-rank) checkpoint for the SGL LoRA redesign.  It
calls the production ``merged_experts_fused_moe_lora_add`` entrypoint; no kernel
is copied into the benchmark package.

Examples::

    python benchmark/kernels/lora_moe/bench_local.py --list-cases
    python benchmark/kernels/lora_moe/bench_local.py --target gate_ab
    python benchmark/kernels/lora_moe/bench_local.py \
      --case-id p0-qwen3.5-35b-a3b-sparse-gb300 --target gate_b \
      --variant direct --execution cuda_graph --json-output result.json

Use ``--mode nsys`` under ``nsys profile --capture-range=cudaProfilerApi`` and
``--mode ncu`` under ``ncu --profile-from-start off``.  Timing mode is always
unprofiled and reports CUDA-event quantiles.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from benchmark.kernels.lora_moe.cases import (
    AdapterBatch,
    ModelShape,
    MoeLoraBenchCase,
)
from benchmark.kernels.lora_moe.matrix import model_shape_cases, p0_cases
from benchmark.kernels.lora_moe.profiling import (
    RunConfig,
    cuda_profile_range,
    make_batch,
    time_cuda_events,
)

TARGETS = (
    "routing",
    "gate_a",
    "gate_b",
    "gate_ab",
    "down_a",
    "down_b",
    "down_ab",
)
VARIANTS = ("production", "direct", "generic")


def _smoke_case(device: str) -> MoeLoraBenchCase:
    model = ModelShape(
        key="synthetic-smoke",
        h_model=64,
        h_moe=64,
        intermediate_size=192,
        num_experts=8,
        top_k=2,
        num_slices=2,
        activation="swiglu",
        moe_layers=1,
    )
    return MoeLoraBenchCase(
        case_id=f"smoke-bf16-{device}",
        model=model,
        adapters=AdapterBatch(
            l_active=1,
            b_base=1,
            l_capacity=2,
            rank=16,
            max_rank=16,
            physical_rank=16,
        ),
        t_local=4,
        phase="decode",
        device=device,
        provider="deepgemm_bf16",
        scope="K0",
        stage="A1/B1/A2/B2",
        pipeline="C0",
        graph_mode="eager",
        routing="balanced",
        cache_state="hot",
    )


def _cases(device: str) -> tuple[MoeLoraBenchCase, ...]:
    return (_smoke_case(device),) + p0_cases(device) + model_shape_cases(device)


def _select_case(device: str, case_id: str | None) -> MoeLoraBenchCase:
    resolved_id = case_id or f"smoke-bf16-{device}"
    for case in _cases(device):
        if case.case_id == resolved_id:
            return case
    choices = ", ".join(case.case_id for case in _cases(device))
    raise ValueError(f"unknown case {resolved_id!r}; choose from {choices}")


def _detect_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if not torch.cuda.is_available():
        raise RuntimeError("--device is required when CUDA is unavailable")
    name = torch.cuda.get_device_name().lower()
    if "gb300" in name:
        return "gb300"
    if "h200" in name:
        return "h200"
    raise RuntimeError(f"unsupported benchmark GPU {torch.cuda.get_device_name()!r}")


def _git_value(args: list[str], fallback: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return fallback


def _environment(args: argparse.Namespace) -> dict[str, object]:
    revision = os.getenv("SGL_LORA_BENCH_REVISION") or _git_value(
        ["rev-parse", "HEAD"], "unknown"
    )
    dirty_env = os.getenv("SGL_LORA_BENCH_DIRTY")
    dirty = (
        dirty_env not in (None, "0", "false", "False")
        if dirty_env is not None
        else bool(_git_value(["status", "--porcelain"], "unknown"))
    )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "gpu_capability": list(torch.cuda.get_device_capability()),
        "git_revision": revision,
        "git_dirty": dirty,
        "pdl_policy": "architecture_auto",
        "cli": vars(args),
    }


def _make_routing(case: MoeLoraBenchCase, device: torch.device):
    tokens = torch.arange(case.t_local, dtype=torch.int32, device=device)
    slots = torch.arange(case.model.top_k, dtype=torch.int32, device=device)
    topk_ids = (tokens[:, None] * 13 + slots[None, :] * 7) % case.e_local
    generator = torch.Generator(device=device)
    generator.manual_seed(17)
    topk_weights = torch.rand(
        (case.t_local, case.model.top_k),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)

    identities = list(range(case.adapters.l_active))
    if case.adapters.b_base:
        identities.append(case.adapters.l_active)
    identity_tensor = torch.tensor(identities, dtype=torch.int32, device=device)
    token_lora_mapping = identity_tensor[tokens.long() % len(identities)]
    return topk_ids.contiguous(), topk_weights, token_lora_mapping


def _random_factor(
    shape: tuple[int, ...], generator: torch.Generator, device: torch.device
) -> torch.Tensor:
    result = torch.empty(shape, dtype=torch.bfloat16, device=device)
    result.uniform_(-0.02, 0.02, generator=generator)
    return result


@dataclass(slots=True)
class SiteFixture:
    case: MoeLoraBenchCase
    site: str
    hidden_states: torch.Tensor
    lora_a: torch.Tensor
    lora_b: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    token_lora_mapping: torch.Tensor
    output: torch.Tensor
    base_output: torch.Tensor | None
    intermediate: torch.Tensor
    routing_cache: dict

    @property
    def is_down(self) -> bool:
        return self.site == "down"

    @property
    def num_slices(self) -> int:
        return 1 if self.is_down else self.case.model.num_slices

    def reset_output(self) -> None:
        if self.base_output is not None:
            self.output.copy_(self.base_output)

    def invoke(self, stage: str, *, direct: bool) -> torch.Tensor | None:
        from sglang.srt.lora.sgl_lora.triton_ops.virtual_experts import (
            merged_experts_fused_moe_lora_add,
        )

        return merged_experts_fused_moe_lora_add(
            output=self.output,
            hidden_states=self.hidden_states,
            lora_a=self.lora_a,
            lora_b=self.lora_b,
            topk_ids=self.topk_ids,
            topk_weights=self.topk_weights,
            token_lora_mapping=self.token_lora_mapping,
            mul_routed_weight=self.is_down,
            experts_shared_outer_loras_a=False,
            experts_shared_outer_loras_b=False,
            routing_cache=self.routing_cache,
            fuse_add_to_output=False,
            fuse_sum_all_reduce=self.is_down,
            use_direct_expand_add=direct,
            num_output_slices=self.num_slices,
            local_expert_offset=0,
            local_num_experts=self.case.e_local,
            stage=stage,
            intermediate_buffer=(None if stage == "routing" else self.intermediate),
        )


def _build_fixture(case: MoeLoraBenchCase, site: str) -> SiteFixture:
    device = torch.device("cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(20260721)
    shapes = case.factor_shapes
    topk_ids, topk_weights, token_map = _make_routing(case, device)
    if site == "gate":
        a_shape, b_shape = shapes.gate_up_a, shapes.gate_up_b
        hidden = torch.empty(
            (case.t_local, case.model.h_moe),
            dtype=torch.bfloat16,
            device=device,
        ).uniform_(-0.1, 0.1, generator=generator)
        output = torch.empty(
            (
                case.t_local,
                case.model.top_k,
                case.model.num_slices * case.i_physical,
            ),
            dtype=torch.bfloat16,
            device=device,
        )
        base_output = None
        intermediate_shape = (
            case.t_local,
            case.model.top_k,
            case.model.num_slices * case.adapters.max_rank,
        )
    else:
        a_shape, b_shape = shapes.down_a, shapes.down_b
        hidden = torch.empty(
            (case.pair_capacity, case.i_physical),
            dtype=torch.bfloat16,
            device=device,
        ).uniform_(-0.1, 0.1, generator=generator)
        base_output = torch.empty(
            (case.t_local, case.model.h_moe),
            dtype=torch.bfloat16,
            device=device,
        ).uniform_(-0.1, 0.1, generator=generator)
        output = base_output.clone()
        intermediate_shape = (
            case.t_local,
            case.model.top_k,
            case.adapters.max_rank,
        )

    lora_a = _random_factor(a_shape, generator, device)
    lora_b = _random_factor(b_shape, generator, device)
    if case.adapters.b_base:
        base_slot = case.adapters.l_active
        lora_a[base_slot].zero_()
        lora_b[base_slot].zero_()
    return SiteFixture(
        case=case,
        site=site,
        hidden_states=hidden,
        lora_a=lora_a,
        lora_b=lora_b,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        token_lora_mapping=token_map,
        output=output,
        base_output=base_output,
        intermediate=torch.empty(
            intermediate_shape, dtype=torch.bfloat16, device=device
        ),
        routing_cache={},
    )


def _resolve_direct(variant: str, rank: int) -> bool:
    if variant == "direct":
        return True
    if variant == "generic":
        return False
    return rank <= 64


@dataclass(slots=True)
class PreparedOp:
    fixture: SiteFixture
    target: str
    scope: str
    direct: bool
    launch: Callable[[], None]
    before_sample: Callable[[], None] | None


def _build_op(
    fixture: SiteFixture, *, target: str, variant: str, scope: str
) -> PreparedOp:
    direct = _resolve_direct(variant, fixture.case.adapters.rank)
    is_b = target.endswith("_b")
    is_ab = target.endswith("_ab")

    # Seed both A and B route plans, then compile the A producer needed by a
    # B-only target. None of this is part of K0 timing.
    fixture.invoke("routing", direct=direct)
    if is_b:
        fixture.invoke("shrink", direct=direct)
    torch.cuda.synchronize()

    if target == "routing":
        stage = "routing"
    elif is_ab:
        stage = "all"
    elif is_b:
        stage = "expand"
    else:
        stage = "shrink"

    if scope == "O0":
        if target not in ("routing", "gate_ab", "down_ab"):
            raise ValueError("O0 supports routing or a complete A+B operator")

        def launch() -> None:
            fixture.routing_cache.clear()
            fixture.invoke(stage, direct=direct)

    else:

        def launch() -> None:
            fixture.invoke(stage, direct=direct)

    reset = (
        fixture.reset_output if fixture.is_down and stage in ("expand", "all") else None
    )
    return PreparedOp(
        fixture=fixture,
        target=target,
        scope=scope,
        direct=direct,
        launch=launch,
        before_sample=reset,
    )


def _check_smoke(op: PreparedOp) -> None:
    """Use direct-vs-generic and staged-vs-combined checks on the tiny case."""
    fixture = op.fixture
    if fixture.case.model.key != "synthetic-smoke" or op.target == "routing":
        return

    fixture.reset_output()
    fixture.invoke("shrink", direct=op.direct)
    if op.target.endswith("_a"):
        actual = fixture.intermediate.clone()
        fixture.invoke("shrink", direct=op.direct)
        torch.testing.assert_close(fixture.intermediate, actual, rtol=3e-2, atol=3e-2)
        return

    fixture.invoke("expand", direct=op.direct)
    staged = fixture.output.clone()
    fixture.reset_output()
    fixture.invoke("shrink", direct=op.direct)
    fixture.invoke("expand", direct=not op.direct)
    alternate = fixture.output.clone()
    torch.testing.assert_close(staged, alternate, rtol=6e-2, atol=6e-2)

    if op.target.endswith("_ab"):
        fixture.reset_output()
        fixture.invoke("all", direct=op.direct)
        combined = fixture.output.clone()
        torch.testing.assert_close(staged, combined, rtol=6e-2, atol=6e-2)


def _case_summary(case: MoeLoraBenchCase) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "model": case.model.key,
        "T": case.t_local,
        "H_moe": case.model.h_moe,
        "I": case.i_local,
        "E_local": case.e_local,
        "K": case.model.top_k,
        "R": case.adapters.rank,
        "L_active": case.adapters.l_active,
        "B_base": case.adapters.b_base,
        "L_capacity": case.adapters.l_capacity,
    }


def _list_cases(device: str) -> None:
    for case in _cases(device):
        row = _case_summary(case)
        print(
            f"{row['case_id']:<52} T={row['T']:<5} H={row['H_moe']:<5} "
            f"I={row['I']:<5} E={row['E_local']:<4} K={row['K']:<2} "
            f"R={row['R']:<3} L={row['L_active']}+{row['B_base']}/{row['L_capacity']}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--device", choices=("auto", "h200", "gb300"), default="auto")
    parser.add_argument("--case-id")
    parser.add_argument("--target", choices=TARGETS, default="gate_ab")
    parser.add_argument("--variant", choices=VARIANTS, default="production")
    parser.add_argument("--scope", choices=("K0", "O0"), default="K0")
    parser.add_argument("--mode", choices=("time", "nsys", "ncu"), default="time")
    parser.add_argument("--execution", choices=("eager", "cuda_graph"), default="eager")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--inner-iterations", type=int, default=10)
    parser.add_argument("--profile-iterations", type=int, default=1)
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_cases:
        device = args.device if args.device != "auto" else "h200"
        _list_cases(device)
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    device = _detect_device(args.device)
    case = _select_case(device, args.case_id)
    if args.scope == "O0" and args.execution == "cuda_graph":
        raise ValueError("route-inclusive O0 is eager-only in this first checkpoint")
    if args.scope == "O0" and args.inner_iterations != 1:
        raise ValueError("route-inclusive O0 requires --inner-iterations 1")

    site = "down" if args.target.startswith("down") else "gate"
    fixture = _build_fixture(case, site)
    op = _build_op(fixture, target=args.target, variant=args.variant, scope=args.scope)
    op.launch()
    torch.cuda.synchronize()
    if not args.skip_check:
        _check_smoke(op)
        torch.cuda.synchronize()

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
    result: dict[str, object] = {
        "environment": _environment(args),
        "case": _case_summary(case),
        "target": args.target,
        "scope": args.scope,
        "requested_variant": args.variant,
        "effective_expand": "direct" if op.direct else "generic",
        "factor_shapes": {
            "a": list(fixture.lora_a.shape),
            "b": list(fixture.lora_b.shape),
        },
    }

    if run_config.mode == "time":
        timing = time_cuda_events(
            batch.run,
            launches_per_batch=batch.launches_per_batch,
            warmup=run_config.warmup,
            samples=run_config.samples,
            before_sample=op.before_sample,
        )
        result["timing"] = asdict(timing)
        print(
            f"{case.case_id} {args.scope}/{args.target} "
            f"{result['effective_expand']} {args.execution}: "
            f"p50={timing.p50_us:.3f} us "
            f"p20/p80={timing.p20_us:.3f}/{timing.p80_us:.3f} us"
        )
    else:
        if op.before_sample is not None:
            op.before_sample()
        label = (
            f"sgl_lora_moe::{args.scope}::{args.target}::{case.case_id}::"
            f"{result['effective_expand']}::{args.execution}::pdl=auto"
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

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
