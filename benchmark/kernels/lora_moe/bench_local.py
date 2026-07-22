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
unprofiled: K0 reports CUDA-event quantiles, while O0 reports isolated
host-to-device-completion wall-clock quantiles.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

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
    time_isolated_cuda_wall,
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
B_CONFIG_SELECTORS = ("logical-t", "flat-tk", "explicit")
B_INPUT_SOURCES = ("production-a", "synthetic")
SYNTHETIC_B_INPUT_SEED = 20260722
_MISSING_SERVER_ARGS_ERROR = "Global server args is not set yet!"


def _ensure_benchmark_server_args() -> None:
    """Publish the minimal scheduler context required by production MoE code."""
    runtime_context = import_module("sglang.srt.runtime_context")

    try:
        runtime_context.get_server_args()
    except ValueError as exc:
        if str(exc) != _MISSING_SERVER_ARGS_ERROR:
            raise
        server_args = import_module("sglang.srt.server_args")

        runtime_context.get_context().set_server_args(
            server_args.ServerArgs(model_path="dummy")
        )


@dataclass(frozen=True, slots=True)
class ExplicitBConfig:
    block_m: int = 64
    block_n: int = 64
    block_k: int = 64
    group_size_m: int = 1
    num_warps: int = 4
    num_stages: int = 4

    def kernel_config(self) -> dict[str, int]:
        return {
            "BLOCK_SIZE_M": self.block_m,
            "BLOCK_SIZE_N": self.block_n,
            "BLOCK_SIZE_K": self.block_k,
            "GROUP_SIZE_M": self.group_size_m,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
        }


@dataclass(frozen=True, slots=True)
class BConfigSelection:
    selector: str
    lookup_m: int | None
    resolved_config: dict[str, Any]
    resolution_status: str
    resolution_error: str | None
    held_override: dict[str, Any] | None


def _local_b_fallback(merged_shape: tuple[int, ...]) -> dict[str, int]:
    _, n_dim, k_dim = merged_shape
    default_block_k = 256 if k_dim >= 1024 else 64 if k_dim >= 64 else max(16, k_dim)
    return {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": min(64, max(16, n_dim)),
        "BLOCK_SIZE_K": min(default_block_k, max(16, k_dim)),
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 4,
    }


def _resolve_production_b_config(
    merged_shape: tuple[int, ...], dtype: torch.dtype, lookup_m: int
) -> dict[str, Any]:
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
        get_config_dtype_str,
        try_get_optimal_moe_config,
    )

    return dict(
        try_get_optimal_moe_config(
            merged_shape,
            merged_shape,
            1,
            get_config_dtype_str(dtype=dtype),
            lookup_m,
        )
    )


def _select_b_config(
    fixture: SiteFixture, selector: str, explicit: ExplicitBConfig
) -> BConfigSelection:
    if selector not in B_CONFIG_SELECTORS:
        raise ValueError(f"unknown B config selector {selector!r}")
    weight = fixture.lora_b
    merged_shape = (weight.shape[0] * weight.shape[1], *weight.shape[2:])
    logical_m = fixture.case.t_local
    flat_pair_m = fixture.topk_ids.numel()
    if selector == "explicit":
        resolved = explicit.kernel_config()
        return BConfigSelection(
            selector=selector,
            lookup_m=None,
            resolved_config=resolved,
            resolution_status="benchmark_explicit",
            resolution_error=None,
            held_override=resolved,
        )

    lookup_m = logical_m if selector == "logical-t" else flat_pair_m
    try:
        resolved = _resolve_production_b_config(
            merged_shape, fixture.hidden_states.dtype, lookup_m
        )
        status = "production_resolver"
        error = None
    except ValueError as exc:
        if str(exc) == _MISSING_SERVER_ARGS_ERROR:
            raise
        resolved = _local_b_fallback(merged_shape)
        status = "local_fallback_after_value_error"
        error = f"{type(exc).__name__}: {exc}"
    return BConfigSelection(
        selector=selector,
        lookup_m=lookup_m,
        resolved_config=resolved,
        resolution_status=status,
        resolution_error=error,
        held_override=resolved if selector == "flat-tk" else None,
    )


@contextmanager
def _exit_context_normally(manager) -> Iterator[None]:
    """Resume a generator context normally even when the body raises."""
    manager.__enter__()
    try:
        yield
    finally:
        manager.__exit__(None, None, None)


@contextmanager
def _b_config_override(config: dict[str, Any] | None) -> Iterator[None]:
    if config is None:
        yield
        return
    from sglang.srt.layers.moe.moe_runner.triton_utils import override_config

    # Production's generator does not restore on exceptional `throw()`. Force
    # a normal resume in our outer finally without changing that shared utility.
    with _exit_context_normally(override_config(config)):
        yield


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _effective_b_config(
    fixture: SiteFixture, *, direct: bool, resolved: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Describe the launch constants actually consumed by the selected B path."""
    if resolved is None:
        return None
    if not direct:
        effective = dict(resolved)
        effective.setdefault("num_warps", 4)
        effective.setdefault("num_stages", 3)
        return {"kernel_family": "generic_fused_moe", **effective}

    n = fixture.lora_b.shape[2]
    rank = fixture.lora_b.shape[3]
    block_n_from_config = resolved["BLOCK_SIZE_N"]
    block_n_uses_config = True
    if fixture.num_slices == 2:
        slice_n = n // 2
        if n % 2 == 0 and slice_n % 16 == 0:
            schedule = "aligned_flat"
            if n % 128 == 0:
                block_n = 128
                block_n_uses_config = False
            else:
                block_n = block_n_from_config
            while block_n > 16 and slice_n % block_n != 0:
                block_n //= 2
        else:
            schedule = "two_slice"
            block_n = min(64, max(16, _next_power_of_two(slice_n)))
            block_n_uses_config = False
    else:
        schedule = "flat"
        if n % 128 == 0:
            block_n = 128
            block_n_uses_config = False
        else:
            block_n = block_n_from_config

    ignored_fields = ["BLOCK_SIZE_K", "num_stages"]
    if not block_n_uses_config:
        ignored_fields.append("BLOCK_SIZE_N")

    return {
        "kernel_family": "direct_lora_b",
        "schedule": schedule,
        "BLOCK_SIZE_M": resolved["BLOCK_SIZE_M"],
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_R": _next_power_of_two(rank),
        "GROUP_SIZE_M": resolved.get("GROUP_SIZE_M", 1),
        "num_warps": resolved.get("num_warps", 4),
        "num_stages": 1,
        "resolved_fields_not_consumed": ignored_fields,
    }


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


def _fill_synthetic_b_intermediate(intermediate: torch.Tensor) -> None:
    generator = torch.Generator(device=intermediate.device)
    generator.manual_seed(SYNTHETIC_B_INPUT_SEED)
    intermediate.uniform_(-0.1, 0.1, generator=generator)


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
    strict_reference_delta: torch.Tensor | None

    @property
    def is_down(self) -> bool:
        return self.site == "down"

    @property
    def num_slices(self) -> int:
        return 1 if self.is_down else self.case.model.num_slices

    def reset_output(self) -> None:
        if self.base_output is not None:
            self.output.copy_(self.base_output)

    def routing_metrics(self) -> list[dict[str, int | bool]]:
        metrics = []
        for key, value in self.routing_cache.items():
            num_experts, shared_outer, block_m = key
            sorted_token_ids, expert_ids, num_tokens_post_padded, _ = value
            post_padding_pairs = int(num_tokens_post_padded.item())
            metrics.append(
                {
                    "num_experts": num_experts,
                    "shared_outer": shared_outer,
                    "block_m": block_m,
                    "allocated_pair_slots": sorted_token_ids.numel(),
                    "allocated_expert_blocks": expert_ids.numel(),
                    "actual_pair_slots": post_padding_pairs,
                    "post_padding_pair_slots": post_padding_pairs,
                    "padding_pair_slots": post_padding_pairs - self.topk_ids.numel(),
                }
            )
        return metrics

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


def _build_fixture(
    case: MoeLoraBenchCase,
    site: str,
    *,
    b_input_source: str = "production-a",
) -> SiteFixture:
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
    fixture = SiteFixture(
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
        strict_reference_delta=None,
    )
    if b_input_source == "synthetic":
        # Isolate B from unsupported or suboptimal production-A schedules while
        # retaining the exact routed [token, top-k, rank] consumer layout.
        _fill_synthetic_b_intermediate(fixture.intermediate)
    return fixture


def _resolve_direct(variant: str, rank: int) -> bool:
    if variant == "direct":
        return True
    if variant == "generic":
        return False
    return rank <= 64


def _fixture_b_rank(fixture: SiteFixture) -> int:
    weight = getattr(fixture, "lora_b", None)
    if weight is not None:
        return weight.shape[-1]
    adapters = fixture.case.adapters
    return getattr(adapters, "physical_rank", adapters.rank)


def _reference_direct(
    fixture: SiteFixture,
    *,
    target: str,
    variant: str,
    b_input_source: str,
) -> bool:
    direct = _resolve_direct(variant, _fixture_b_rank(fixture))
    if b_input_source == "synthetic" and target.endswith("_b"):
        return not direct
    if (
        variant == "generic"
        and target in ("gate_b", "gate_ab")
        and fixture.num_slices > 1
        and _fixture_b_rank(fixture) <= 64
    ):
        return True
    return direct


def _target_stage(target: str) -> str:
    if target == "routing":
        return "routing"
    if target.endswith("_ab"):
        return "all"
    return "expand" if target.endswith("_b") else "shrink"


@dataclass(slots=True)
class PreparedOp:
    fixture: SiteFixture
    target: str
    scope: str
    direct: bool
    launch: Callable[[], None]
    before_sample: Callable[[], None] | None
    b_input_source: str = "production-a"


def _build_op(
    fixture: SiteFixture,
    *,
    target: str,
    variant: str,
    scope: str,
    b_input_source: str = "production-a",
) -> PreparedOp:
    direct = _resolve_direct(variant, _fixture_b_rank(fixture))
    is_b = target.endswith("_b")

    # Seed both A and B route plans, then (for the production source) compile
    # and run the A producer needed by a B-only target. This is outside K0.
    fixture.invoke("routing", direct=direct)
    if is_b and b_input_source == "production-a":
        fixture.invoke("shrink", direct=direct)
    torch.cuda.synchronize()

    stage = _target_stage(target)

    if scope == "O0":
        if target not in ("routing", "gate_b", "gate_ab", "down_b", "down_ab"):
            raise ValueError(
                "O0 supports routing, route-inclusive B with precomputed A, "
                "or a complete A+B operator"
            )

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
        b_input_source=b_input_source,
    )


def _clone_operator_value(fixture: SiteFixture, target: str) -> torch.Tensor | None:
    if target == "routing":
        return None
    value = fixture.intermediate if target.endswith("_a") else fixture.output
    return value.clone()


def _production_config_reference(
    fixture: SiteFixture,
    *,
    target: str,
    variant: str,
    b_input_source: str = "production-a",
) -> torch.Tensor | None:
    """Compute a safe reference before installing an experimental B config."""
    direct = _reference_direct(
        fixture,
        target=target,
        variant=variant,
        b_input_source=b_input_source,
    )
    # Synthetic B-only runs always use the opposite B family as their oracle.
    # Production-A generic gate/up uses direct B when rank <= 64, where A can
    # still produce the intermediate without the known rank-128 resource error.
    fixture.routing_cache.clear()
    fixture.invoke("routing", direct=direct)
    if target.endswith("_b") and b_input_source == "production-a":
        fixture.invoke("shrink", direct=direct)

    if target != "routing":
        fixture.reset_output()
        fixture.invoke(_target_stage(target), direct=direct)
    torch.cuda.synchronize()
    reference = _clone_operator_value(fixture, target)
    if getattr(fixture, "is_down", False) and target in ("down_b", "down_ab"):
        # Capture the LoRA delta into a zero destination as well as the real
        # nonzero-base result above. Subtracting two BF16 base-added tensors
        # cannot recover contributions below the base destination's ULP and
        # makes atomic reduction-order noise look like a semantic failure.
        fixture.output.zero_()
        fixture.invoke(_target_stage(target), direct=direct)
        torch.cuda.synchronize()
        fixture.strict_reference_delta = fixture.output.clone()
        fixture.reset_output()
    fixture.routing_cache.clear()
    return reference


def _operator_delta(
    fixture: SiteFixture, target: str, value: torch.Tensor
) -> torch.Tensor:
    """Return the value whose loss the correctness gate must detect.

    Down-B writes into a nonzero base destination, so comparing the full BF16
    output can make a missing LoRA contribution invisible.  Gate/up B already
    produces a standalone delta and needs no subtraction.
    """
    if getattr(fixture, "is_down", False) and target in ("down_b", "down_ab"):
        assert fixture.base_output is not None
        return value.float() - fixture.base_output.float()
    return value.float()


def _check_operator(
    op: PreparedOp, reference: torch.Tensor | None
) -> dict[str, float | str] | None:
    """Compare the selected operator with its safe production-config oracle."""
    fixture = op.fixture
    if op.target == "routing":
        op.launch()
        torch.cuda.synchronize()
        return None

    def run_once() -> torch.Tensor:
        fixture.reset_output()
        op.launch()
        torch.cuda.synchronize()
        value = _clone_operator_value(fixture, op.target)
        assert value is not None
        return value

    first = run_once()
    second = run_once()
    assert reference is not None
    if op.target.endswith("_b") or op.target.endswith("_ab"):
        is_down_b = getattr(fixture, "is_down", False)
        if is_down_b:
            reference_delta = getattr(fixture, "strict_reference_delta", None)
            assert reference_delta is not None

            def run_zero_destination_once() -> torch.Tensor:
                fixture.output.zero_()
                op.launch()
                torch.cuda.synchronize()
                return fixture.output.clone().float()

            first_delta = run_zero_destination_once()
            second_delta = run_zero_destination_once()
            fixture.reset_output()
            reference_delta = reference_delta.float()
        else:
            reference_delta = _operator_delta(fixture, op.target, reference)
            first_delta = _operator_delta(fixture, op.target, first)
            second_delta = _operator_delta(fixture, op.target, second)
        signal = float(reference_delta.abs().max().item())
        # A fully dropped delta must fail. Keep the established BF16-friendly
        # relative bound, but cap absolute slack at one tenth of the observed
        # signal. Down-B is checked in a zero destination so the base tensor's
        # much larger BF16 ULP cannot hide or manufacture a LoRA contribution.
        rtol = 3e-2
        atol = min(2e-4, signal / 10.0) if signal else 0.0
        torch.testing.assert_close(
            first_delta, reference_delta, rtol=rtol, atol=atol
        )
        torch.testing.assert_close(second_delta, first_delta, rtol=rtol, atol=atol)
        return {
            "reference": "base_subtracted_lora_delta",
            "reference_delta_max_abs": signal,
            "candidate_delta_max_abs": float(first_delta.abs().max().item()),
            "candidate_delta_max_abs_error": float(
                (first_delta - reference_delta).abs().max().item()
            ),
            "repeat_delta_max_abs_error": float(
                (second_delta - first_delta).abs().max().item()
            ),
            "full_output_max_abs_error": float(
                (first.float() - reference.float()).abs().max().item()
            ),
            "rtol": rtol,
            "atol": atol,
        }

    rtol = atol = 3e-2
    torch.testing.assert_close(first, reference, rtol=rtol, atol=atol)
    torch.testing.assert_close(second, first, rtol=rtol, atol=atol)
    return {
        "reference": "operator_value",
        "reference_max_abs": float(reference.float().abs().max().item()),
        "candidate_max_abs_error": float(
            (first.float() - reference.float()).abs().max().item()
        ),
        "repeat_max_abs_error": float(
            (second.float() - first.float()).abs().max().item()
        ),
        "rtol": rtol,
        "atol": atol,
    }


def _case_summary(case: MoeLoraBenchCase) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "model": case.model.key,
        "T": case.t_local,
        "H_moe": case.model.h_moe,
        "I": case.i_local,
        "E_local": case.e_local,
        "K": case.model.top_k,
        "P_capacity": case.pair_capacity,
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
    parser.add_argument(
        "--b-input-source",
        choices=B_INPUT_SOURCES,
        default="production-a",
        help=(
            "B-only input source: run the production A producer, or use a "
            "deterministic precomputed synthetic intermediate"
        ),
    )
    parser.add_argument(
        "--b-config-selector",
        choices=B_CONFIG_SELECTORS,
        default="logical-t",
        help=(
            "LoRA-B config policy: current logical T lookup, alternate flattened "
            "T*K lookup, or benchmark-owned explicit fields"
        ),
    )
    for name, choices, default in (
        ("block-m", (16, 32, 64, 128), 64),
        ("block-n", (16, 32, 64, 128, 256, 512), 64),
        ("block-k", (16, 32, 64, 128, 256), 64),
        ("group-size-m", (1, 2, 4, 8, 16, 32), 1),
        ("num-warps", (2, 4, 8), 4),
        ("num-stages", (1, 2, 3, 4, 5), 4),
    ):
        parser.add_argument(
            f"--b-{name}",
            type=int,
            choices=choices,
            default=default,
            help="benchmark-owned explicit B config field",
        )
    parser.add_argument("--scope", choices=("K0", "O0"), default="K0")
    parser.add_argument("--mode", choices=("time", "nsys", "ncu"), default="time")
    parser.add_argument("--execution", choices=("eager", "cuda_graph"), default="eager")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--inner-iterations", type=int)
    parser.add_argument("--profile-iterations", type=int, default=1)
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args(argv)


def _validate_b_input_source(target: str, source: str) -> None:
    if source == "synthetic" and target not in ("gate_b", "down_b"):
        raise ValueError("synthetic B input is only valid for gate_b or down_b")


def _execute_benchmark(
    args: argparse.Namespace,
    case: MoeLoraBenchCase,
    fixture: SiteFixture,
    selection: BConfigSelection,
    reference: torch.Tensor | None,
) -> dict[str, object]:
    op = _build_op(
        fixture,
        target=args.target,
        variant=args.variant,
        scope=args.scope,
        b_input_source=args.b_input_source,
    )
    if args.skip_check:
        op.launch()
        correctness = {"skipped": True}
    else:
        correctness = _check_operator(op, reference)
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
    is_b_only = args.target.endswith("_b")
    oracle_expand = (
        "direct"
        if _reference_direct(
            fixture,
            target=args.target,
            variant=args.variant,
            b_input_source=args.b_input_source,
        )
        else "generic"
    )
    result: dict[str, object] = {
        "environment": _environment(args),
        "case": _case_summary(case),
        "target": args.target,
        "scope": args.scope,
        "requested_variant": args.variant,
        "effective_expand": "direct" if op.direct else "generic",
        "b_intermediate": {
            "source": args.b_input_source if is_b_only else "not_applicable",
            "seed": (
                SYNTHETIC_B_INPUT_SEED
                if is_b_only and args.b_input_source == "synthetic"
                else None
            ),
            "shape": list(fixture.intermediate.shape) if is_b_only else None,
            "a_producer_launched": (
                args.b_input_source == "production-a" if is_b_only else None
            ),
        },
        "correctness_oracle_expand": (
            oracle_expand
            if not args.skip_check and (is_b_only or args.target.endswith("_ab"))
            else "not_applicable"
        ),
        "correctness": correctness,
        "b_config": {
            "requested_selector": selection.selector,
            "site": fixture.site,
            "lookup_m": selection.lookup_m,
            "resolution_status": selection.resolution_status,
            "resolution_error": selection.resolution_error,
            "held_override": selection.held_override is not None,
            "resolved_config": selection.resolved_config,
            "effective_config": _effective_b_config(
                fixture, direct=op.direct, resolved=selection.resolved_config
            ),
        },
        "route_inclusion": (
            "prebuilt"
            if args.scope == "K0"
            else (
                (
                    "route_inclusive_b_with_precomputed_a"
                    if args.b_input_source == "production-a"
                    else "route_inclusive_b_with_synthetic_intermediate"
                )
                if args.target.endswith("_b")
                else "route_inclusive"
            )
        ),
        "cache_measurement": (
            "single_plan_diagnostic"
            if args.scope == "K0"
            else (
                "b_route_rebuild_with_fixed_a_intermediate"
                if args.target.endswith("_b")
                else "producer_realistic_route_rebuild"
            )
        ),
        "factor_shapes": {
            "a": list(fixture.lora_a.shape),
            "b": list(fixture.lora_b.shape),
        },
        "factor_bytes_allocated": (
            fixture.lora_a.numel() * fixture.lora_a.element_size()
            + fixture.lora_b.numel() * fixture.lora_b.element_size()
        ),
        "routing": {
            "valid_pairs": fixture.topk_ids.numel(),
            "experts_hit": int(torch.unique(fixture.topk_ids).numel()),
            "virtual_expert_capacity": (case.e_local * case.adapters.l_capacity),
            "prewarm": "a_and_b",
            "a_execution": (
                "skipped" if args.b_input_source == "synthetic" else "production"
            ),
        },
    }

    if run_config.mode == "time":
        timing_helper = (
            time_isolated_cuda_wall if args.scope == "O0" else time_cuda_events
        )
        timing = timing_helper(
            batch.run,
            launches_per_batch=batch.launches_per_batch,
            warmup=run_config.warmup,
            samples=run_config.samples,
            before_sample=op.before_sample,
        )
        result["timing"] = asdict(timing)
        result["timing_domain"] = (
            "isolated_wall_host_to_device_completion"
            if args.scope == "O0"
            else "cuda_event_device"
        )
        print(
            f"{case.case_id} {args.scope}/{args.target} "
            f"{result['effective_expand']} B={selection.selector} "
            f"BInput={args.b_input_source} {args.execution}: "
            f"p50={timing.p50_us:.3f} us "
            f"p20/p80={timing.p20_us:.3f}/{timing.p80_us:.3f} us"
        )
    else:
        if op.before_sample is not None:
            op.before_sample()
        label = (
            f"sgl_lora_moe::{args.scope}::{args.target}::{case.case_id}::"
            f"{result['effective_expand']}::B={selection.selector}::"
            f"BInput={args.b_input_source}::{args.execution}::pdl=auto"
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

    result["routing"]["plans"] = fixture.routing_metrics()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_cases:
        device = args.device if args.device != "auto" else "h200"
        _list_cases(device)
        return 0
    _validate_b_input_source(args.target, args.b_input_source)
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    _ensure_benchmark_server_args()
    device = _detect_device(args.device)
    case = _select_case(device, args.case_id)
    if args.inner_iterations is None:
        args.inner_iterations = 1 if args.scope == "O0" else 10
    if args.scope == "O0" and args.execution == "cuda_graph":
        raise ValueError("route-inclusive O0 is eager-only")
    if args.scope == "O0" and args.inner_iterations != 1:
        raise ValueError("route-inclusive O0 requires --inner-iterations 1")

    site = "down" if args.target.startswith("down") else "gate"
    fixture = _build_fixture(case, site, b_input_source=args.b_input_source)
    explicit = ExplicitBConfig(
        block_m=args.b_block_m,
        block_n=args.b_block_n,
        block_k=args.b_block_k,
        group_size_m=args.b_group_size_m,
        num_warps=args.b_num_warps,
        num_stages=args.b_num_stages,
    )
    reference = (
        None
        if args.skip_check
        else _production_config_reference(
            fixture,
            target=args.target,
            variant=args.variant,
            b_input_source=args.b_input_source,
        )
    )
    selection = _select_b_config(fixture, args.b_config_selector, explicit)
    # The production reference left only compiled kernels behind. Per-config
    # routing starts from an empty cache under the selected held override.
    fixture.routing_cache.clear()
    with _b_config_override(selection.held_override):
        result = _execute_benchmark(args, case, fixture, selection, reference)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
