#!/usr/bin/env python3
"""Benchmark the Phase-1a BF16 SGL LoRA full-MoE pipeline (M0).

This local, single-rank driver compares three execution plans with identical
base weights, inputs, and routing:

* ``N0``: stock BF16 DeepGEMM through ``MoeRunner(DEEP_GEMM)``;
* ``C0``: the serial ``run_sgl_lora_moe`` pipeline;
* ``C1``: the same SGL LoRA pipeline under either the production-auto or a
  benchmark-forced two-stream decision.

``C0`` and ``C1`` contain active LoRA work and are therefore compared with one
another for correctness. ``N0`` is the matched base-only latency reference, not
a numerical reference for an active-adapter result. The initial M0 checkpoint
is intentionally limited to gated SwiGLU, BF16, and TP=EP=MoE-DP=1.

Examples::

    python benchmark/kernels/lora_moe/bench_moe_pipeline.py --list-cases
    python benchmark/kernels/lora_moe/bench_moe_pipeline.py \
      --case-id p0-qwen3.5-35b-a3b-cap1-h200 --pipeline all \
      --execution cuda_graph --json-output result.json
    python benchmark/kernels/lora_moe/bench_moe_pipeline.py \
      --case-id p0-qwen3.5-35b-a3b-cap1-h200 --pipeline C1 \
      --a-provider indexed --execution cuda_graph
    python benchmark/kernels/lora_moe/bench_moe_pipeline.py \
      --case-id p0-qwen3.5-35b-a3b-prefill-h200 --pipeline all \
      --c1-overlap-policy force --execution cuda_graph

For a trace, choose exactly one pipeline and wrap this script with Nsight using
the CUDA-profiler capture range, as in ``bench_local.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Sequence

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

PIPELINES = ("N0", "C0", "C1")
C1_OVERLAP_POLICIES = ("production_auto", "force")
B_VARIANTS = ("production", "direct", "generic")
_B_CONFIG_FIELDS = (
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "GROUP_SIZE_M",
    "num_warps",
    "num_stages",
)

_INDEXED_AUTO_CONFIG_KEYS = {
    "h200": {
        "gate": "bn32-bk128-w4",
        "down": "bn16-bk128-w8",
    },
    "gb300": {
        "gate": "bn32-bk128-w8",
        "down": "bn8-bk128-w8",
    },
}

if TYPE_CHECKING:
    from benchmark.kernels.lora_moe.bench_indexed_shrink import IndexedShrinkConfig


@dataclass(frozen=True, slots=True)
class IndexedAConfigs:
    gate: IndexedShrinkConfig
    down: IndexedShrinkConfig
    gate_source: str
    down_source: str

    def metadata(self) -> dict[str, object]:
        return {
            "gate": {"selection": self.gate_source, **asdict(self.gate)},
            "down": {"selection": self.down_source, **asdict(self.down)},
        }


@dataclass(frozen=True, slots=True)
class BScheduleOverrides:
    gate_variant: str = "production"
    down_variant: str = "production"
    gate_config: dict[str, int] | None = None
    down_config: dict[str, int] | None = None

    @property
    def applied(self) -> bool:
        return (
            self.gate_variant != "production"
            or self.down_variant != "production"
            or self.gate_config is not None
            or self.down_config is not None
        )

    @property
    def routing_config_overridden(self) -> bool:
        return self.gate_config is not None or self.down_config is not None

    def for_call(self, *, mul_routed_weight: bool) -> tuple[str, dict[str, int] | None]:
        if mul_routed_weight:
            return self.down_variant, self.down_config
        return self.gate_variant, self.gate_config

    def metadata(
        self,
        *,
        production_variant: str | None = None,
        shared_outer_b: bool = False,
    ) -> dict[str, object]:
        def site_metadata(
            variant: str, config: dict[str, int] | None
        ) -> dict[str, object]:
            selected = production_variant if variant == "production" else variant
            effective = (
                "generic" if selected == "direct" and shared_outer_b else selected
            )
            return {
                "requested_variant": variant,
                "effective_variant": effective,
                "config": config,
                "routing_config_overridden": config is not None,
            }

        return {
            "gate": site_metadata(self.gate_variant, self.gate_config),
            "down": site_metadata(self.down_variant, self.down_config),
            "applied": self.applied,
            "benchmark_only": self.applied,
            "routing_config_overridden": self.routing_config_overridden,
            "substitution_scope": (
                "gate_and_down_lora_b_family_and_config_only"
                if self.applied
                else "none"
            ),
            "production_policy_changed": False,
        }


def _parse_b_config(value: str | None) -> dict[str, int] | None:
    if value is None:
        return None
    try:
        values = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise ValueError("B config must contain six comma-separated integers") from exc
    if len(values) != len(_B_CONFIG_FIELDS) or any(item <= 0 for item in values):
        raise ValueError(
            "B config must be positive BM,BN,BK,GROUP_SIZE_M,num_warps,num_stages"
        )
    return dict(zip(_B_CONFIG_FIELDS, values, strict=True))


def _smoke_case(device: str) -> MoeLoraBenchCase:
    return MoeLoraBenchCase(
        case_id=f"m0-smoke-bf16-{device}",
        model=ModelShape(
            key="m0-synthetic-smoke",
            h_model=64,
            h_moe=64,
            intermediate_size=192,
            num_experts=8,
            top_k=2,
            num_slices=2,
            activation="swiglu",
            moe_layers=1,
        ),
        adapters=AdapterBatch(
            l_active=1,
            b_base=0,
            l_capacity=1,
            rank=16,
            max_rank=16,
            physical_rank=16,
        ),
        t_local=4,
        phase="decode",
        device=device,
        provider="deepgemm_bf16",
        scope="M0",
        stage="M0",
        pipeline="C0",
        graph_mode="eager",
        routing="seeded_iid",
        cache_state="hot",
    )


def _cases(device: str) -> tuple[MoeLoraBenchCase, ...]:
    return (_smoke_case(device),) + p0_cases(device) + model_shape_cases(device)


def _select_case(device: str, case_id: str | None) -> MoeLoraBenchCase:
    resolved_id = case_id or f"m0-smoke-bf16-{device}"
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


def _case_summary(case: MoeLoraBenchCase) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "model": case.model.key,
        "phase": case.phase,
        "T": case.t_local,
        "H_moe": case.model.h_moe,
        "I": case.i_local,
        "E_local": case.e_local,
        "K": case.model.top_k,
        "R": case.adapters.rank,
        "L_active": case.adapters.l_active,
        "B_base": case.adapters.b_base,
        "L_capacity": case.adapters.l_capacity,
        "activation": case.model.activation,
        "num_slices": case.model.num_slices,
        "tp_size": case.tp_size,
        "ep_size": case.ep_size,
        "moe_dp_size": case.moe_dp_size,
    }


def _list_cases(device: str) -> None:
    for case in _cases(device):
        row = _case_summary(case)
        support = "gated" if case.model.num_slices == 2 else "M0-gap:non-gated"
        print(
            f"{row['case_id']:<52} T={row['T']:<5} H={row['H_moe']:<5} "
            f"I={row['I']:<5} E={row['E_local']:<4} K={row['K']:<2} "
            f"R={row['R']:<3} L={row['L_active']}+{row['B_base']}/"
            f"{row['L_capacity']} {support}"
        )


def _resolve_indexed_a_configs(
    device: str, gate_key: str, down_key: str
) -> IndexedAConfigs:
    from benchmark.kernels.lora_moe.bench_indexed_shrink import INDEXED_CONFIGS

    configs_by_key = {config.key: config for config in INDEXED_CONFIGS}

    def resolve(site: str, requested: str):
        if requested == "auto":
            key = _INDEXED_AUTO_CONFIG_KEYS[device][site]
            source = "auto_cold_cache_shortlist"
        else:
            key = requested
            source = "explicit"
        try:
            return configs_by_key[key], source
        except KeyError as exc:
            choices = ", ".join(("auto", *configs_by_key))
            raise ValueError(
                f"unknown indexed {site} config {key!r}; choose from {choices}"
            ) from exc

    gate, gate_source = resolve("gate", gate_key)
    down, down_source = resolve("down", down_key)
    return IndexedAConfigs(
        gate=gate,
        down=down,
        gate_source=gate_source,
        down_source=down_source,
    )


def _resolve_c1_overlap(policy: str, num_tokens: int) -> bool:
    """Resolve C1 once so warmup, capture, replay, and metadata agree."""
    if policy == "force":
        return True
    if policy == "production_auto":
        from sglang.srt.lora.sgl_lora.moe_lora_runner import (
            resolve_lora_two_stream_auto,
        )

        return resolve_lora_two_stream_auto(requested=True, num_tokens=num_tokens)
    raise ValueError(f"unknown C1 overlap policy {policy!r}")


def _make_routing(
    case: MoeLoraBenchCase, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    # Match the producer contract: active adapters use their weight slot and
    # base-only rows use -1. Configured-but-inactive capacity is never routed.
    identities = list(range(case.adapters.l_active))
    if case.adapters.b_base:
        identities.append(-1)
    identity_tensor = torch.tensor(identities, dtype=torch.int32, device=device)
    token_lora_mapping = identity_tensor[tokens.long() % len(identities)]
    return topk_ids.contiguous(), topk_weights, token_lora_mapping


def _random_bf16(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    device: torch.device,
    scale: float,
) -> torch.Tensor:
    tensor = torch.empty(shape, dtype=torch.bfloat16, device=device)
    tensor.uniform_(-scale, scale, generator=generator)
    return tensor


@dataclass(slots=True)
class PipelineFixture:
    case: MoeLoraBenchCase
    c1_overlap_policy: str
    c1_two_stream_enabled: bool
    hidden_seed: torch.Tensor
    hidden_work: torch.Tensor
    topk_output: object
    runner_config: object
    base_quant_info: object
    base_runner: object
    sgl_quant_info: object | None
    sgl_base: object | None
    lora_info: object | None
    base_weights: tuple[torch.Tensor, torch.Tensor]
    lora_weights: tuple[torch.Tensor, ...]
    indexed_down_intermediate: torch.Tensor | None = None
    last_output: torch.Tensor | None = None

    def reset_hidden(self) -> None:
        """Restore the provider input outside the measured interval.

        The stock DeepGEMM runner disposes its input storage in eager mode. A
        graph-captured input keeps its storage, so copy into that stable address.
        """
        if self.hidden_work.shape != self.hidden_seed.shape:
            self.hidden_work = self.hidden_seed.clone()
        else:
            self.hidden_work.copy_(self.hidden_seed)

    def invoke(self, pipeline: str) -> None:
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardDispatchOutput,
        )

        dispatch_output = StandardDispatchOutput(
            hidden_states=self.hidden_work,
            hidden_states_scale=None,
            topk_output=self.topk_output,
        )
        if pipeline == "N0":
            result = self.base_runner.run(dispatch_output, self.base_quant_info)
        else:
            if self.lora_info is None or self.sgl_quant_info is None:
                raise RuntimeError("active LoRA fixture was not constructed")
            from sglang.srt.lora.sgl_lora.moe_lora_runner import run_sgl_lora_moe

            result = run_sgl_lora_moe(
                dispatch_output,
                self.sgl_quant_info,
                self.runner_config,
                self.lora_info,
                self.sgl_base,
                two_stream_enabled=(pipeline == "C1" and self.c1_two_stream_enabled),
            )
        self.last_output = result.hidden_states


def _pipeline_two_stream_metadata(
    fixture: PipelineFixture, pipeline: str
) -> dict[str, object]:
    from sglang.srt.lora.sgl_lora.moe_lora_runner import (
        LORA_TWO_STREAM_AUTO_MAX_TOKENS,
        resolve_lora_two_stream_auto,
    )

    requested = pipeline == "C1"
    production_auto_enabled = resolve_lora_two_stream_auto(
        requested=requested,
        num_tokens=fixture.case.t_local,
    )
    effective = requested and fixture.c1_two_stream_enabled
    forced = requested and fixture.c1_overlap_policy == "force"
    fallback_reason = None
    if requested and not effective:
        fallback_reason = "production_auto_token_threshold"
    return {
        "requested": requested,
        "policy": fixture.c1_overlap_policy if requested else "serial",
        "production_auto_max_tokens": LORA_TWO_STREAM_AUTO_MAX_TOKENS,
        "production_auto_enabled": production_auto_enabled,
        "effective": effective,
        "benchmark_force_requested": forced,
        "benchmark_force_changed_decision": forced and not production_auto_enabled,
        "fallback_reason": fallback_reason,
        "decision_scope": "fixed_shape_before_eager_or_cuda_graph_capture",
        "overlap_scope": (
            "gate_up_lora_a_b_vs_base_prepare_gateup;down_lora_serial"
            if effective
            else None
        ),
        "production_default_unchanged": True,
    }


def _build_fixture(
    case: MoeLoraBenchCase,
    *,
    need_lora: bool,
    need_indexed_a: bool = False,
    c1_overlap_policy: str = "production_auto",
) -> PipelineFixture:
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.deep_gemm import DeepGemmMoeQuantInfo
    from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.moe.utils import MoeRunnerBackend

    device = torch.device("cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(20260721)

    hidden_seed = _random_bf16(
        (case.t_local, case.model.h_moe),
        generator=generator,
        device=device,
        scale=1.0,
    )
    topk_ids, topk_weights, token_lora_mapping = _make_routing(case, device)
    topk_output = StandardTopKOutput(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        router_logits=torch.empty(0, dtype=torch.float32, device=device),
    )

    w13 = _random_bf16(
        (case.e_local, 2 * case.i_physical, case.model.h_moe),
        generator=generator,
        device=device,
        scale=0.02,
    )
    w2 = _random_bf16(
        (case.e_local, case.model.h_moe, case.i_physical),
        generator=generator,
        device=device,
        scale=0.02,
    )
    config = MoeRunnerConfig(
        num_experts=case.model.num_experts,
        num_local_experts=case.e_local,
        hidden_size=case.model.h_moe,
        intermediate_size_per_partition=case.i_physical,
        top_k=case.model.top_k,
        activation="silu",
        is_gated=True,
        inplace=False,
        no_combine=False,
        routed_scaling_factor=1.0,
        gate_up_interleaved=False,
    )
    base_quant_info = DeepGemmMoeQuantInfo(
        w13_weight=w13,
        w2_weight=w2,
        use_fp8=False,
    )
    # Force the non-fused DeepGEMM core even if a fused registration is added
    # later; N0 is specifically the matched staged-provider reference.
    base_runner = MoeRunner(MoeRunnerBackend.DEEP_GEMM, config, lora_enabled=True)

    sgl_quant_info = None
    sgl_base = None
    lora_info = None
    lora_weights: tuple[torch.Tensor, ...] = ()
    indexed_down_intermediate = None
    if need_lora:
        from sglang.srt.lora.lora_moe_runners import LoRAInfo
        from sglang.srt.lora.sgl_lora.base_gemm import resolve_base_gemm
        from sglang.srt.lora.sgl_lora.quant_info import SglLoraBf16QuantInfo

        shapes = case.factor_shapes
        gate_a = _random_bf16(
            shapes.gate_up_a, generator=generator, device=device, scale=0.02
        )
        gate_b = _random_bf16(
            shapes.gate_up_b, generator=generator, device=device, scale=0.02
        )
        down_a = _random_bf16(
            shapes.down_a, generator=generator, device=device, scale=0.02
        )
        down_b = _random_bf16(
            shapes.down_b, generator=generator, device=device, scale=0.02
        )
        lora_weights = (gate_a, gate_b, down_a, down_b)

        lora_ranks = torch.zeros(
            case.adapters.l_capacity, dtype=torch.int32, device=device
        )
        adapter_enabled = torch.zeros_like(lora_ranks)
        if case.adapters.l_active:
            lora_ranks[: case.adapters.l_active] = case.adapters.rank
            adapter_enabled[: case.adapters.l_active] = 1
        lora_info = LoRAInfo(
            gate_up_lora_a_weights=gate_a,
            gate_up_lora_b_weights=gate_b,
            down_lora_a_weights=down_a,
            down_lora_b_weights=down_b,
            seg_indptr=torch.tensor(
                [0, case.t_local], dtype=torch.int32, device=device
            ),
            req_to_lora=torch.tensor([0], dtype=torch.int32, device=device),
            lora_ranks=lora_ranks,
            adapter_enabled=adapter_enabled,
            token_lora_mapping=token_lora_mapping,
            max_lora_rank=case.adapters.max_rank,
            num_experts=case.e_local,
            has_active_lora=True,
            experts_shared_outer_loras=case.adapters.shared_outer,
            tp_size=1,
            tp_rank=0,
            hidden_size=case.model.h_moe,
            lora_use_virtual_experts=True,
        )
        sgl_quant_info = SglLoraBf16QuantInfo(
            w13_weight=w13,
            w2_weight=w2,
            num_local_experts=case.e_local,
            intermediate_size=case.i_physical,
            hidden_size=case.model.h_moe,
        )
        sgl_base = resolve_base_gemm(sgl_quant_info, config)
        if need_indexed_a:
            indexed_down_intermediate = torch.empty(
                (
                    case.t_local,
                    case.model.top_k,
                    down_a.shape[2],
                ),
                dtype=hidden_seed.dtype,
                device=device,
            )

    return PipelineFixture(
        case=case,
        c1_overlap_policy=c1_overlap_policy,
        c1_two_stream_enabled=_resolve_c1_overlap(c1_overlap_policy, case.t_local),
        hidden_seed=hidden_seed,
        hidden_work=hidden_seed.clone(),
        topk_output=topk_output,
        runner_config=config,
        base_quant_info=base_quant_info,
        base_runner=base_runner,
        sgl_quant_info=sgl_quant_info,
        sgl_base=sgl_base,
        lora_info=lora_info,
        base_weights=(w13, w2),
        lora_weights=lora_weights,
        indexed_down_intermediate=indexed_down_intermediate,
    )


@contextmanager
def _exit_context_normally(manager) -> Iterator[None]:
    """Resume a generator context normally even when the wrapped call raises."""
    manager.__enter__()
    try:
        yield
    finally:
        manager.__exit__(None, None, None)


@contextmanager
def _held_b_config(config: dict[str, int] | None) -> Iterator[None]:
    if config is None:
        yield
        return
    from sglang.srt.layers.moe.moe_runner.triton_utils import override_config

    with _exit_context_normally(override_config(config)):
        yield


@contextmanager
def _b_schedule_override(schedules: BScheduleOverrides) -> Iterator[None]:
    """Override only the benchmarked B family/config at each LoRA MoE site."""
    if not schedules.applied:
        yield
        return

    from sglang.srt.lora.sgl_lora.triton_ops import virtual_experts

    wrapped_ab = virtual_experts.merged_experts_fused_moe_lora_add

    def scheduled_b(*args, **kwargs):
        if args:
            raise TypeError("B schedule benchmark wrapper requires keyword arguments")
        variant, config = schedules.for_call(
            mul_routed_weight=kwargs["mul_routed_weight"]
        )
        call_kwargs = dict(kwargs)
        if variant != "production":
            call_kwargs["use_direct_expand_add"] = variant == "direct"
        with _held_b_config(config):
            return wrapped_ab(**call_kwargs)

    virtual_experts.merged_experts_fused_moe_lora_add = scheduled_b
    try:
        yield
    finally:
        virtual_experts.merged_experts_fused_moe_lora_add = wrapped_ab


@contextmanager
def _indexed_a_override(
    fixture: PipelineFixture, configs: IndexedAConfigs
) -> Iterator[None]:
    """Replace only LoRA-A inside the delegated A+B entrypoint.

    C1's ``stage="routing"`` call deliberately stays production-equivalent. It
    therefore retains the now-unused production A route prewarm as conservative
    overhead, while the subsequent ``stage="all"`` call uses indexed A followed
    by the delegated B ``stage="expand"`` path. A benchmark-only B schedule may
    be installed beneath this wrapper; its configuration is then held only while
    delegated routing/B work runs, never while indexed A runs.
    """
    from benchmark.kernels.lora_moe.bench_indexed_shrink import (
        invoke_indexed_lora_a,
    )
    from sglang.srt.lora.sgl_lora.triton_ops import virtual_experts

    down_intermediate = fixture.indexed_down_intermediate
    if down_intermediate is None:
        raise RuntimeError("indexed A requires a retained down intermediate")

    delegated_ab = virtual_experts.merged_experts_fused_moe_lora_add

    def indexed_a_delegated_b(*args, **kwargs):
        if args:
            raise TypeError("indexed A benchmark wrapper requires keyword arguments")
        stage = kwargs.get("stage", "all")
        if stage == "routing":
            return delegated_ab(**kwargs)
        if stage != "all":
            raise ValueError(
                "indexed A benchmark wrapper supports only stage='routing' or "
                f"stage='all', got {stage!r}"
            )
        if kwargs["experts_shared_outer_loras_a"]:
            raise NotImplementedError("indexed A supports per-expert factors only")

        num_output_slices = kwargs["num_output_slices"]
        if num_output_slices == 2:
            site = "gate"
            intermediate = kwargs.get("intermediate_buffer")
            if intermediate is None:
                raise RuntimeError("indexed gate A requires the runner intermediate")
        elif num_output_slices == 1:
            site = "down"
            intermediate = down_intermediate
        else:
            raise ValueError(f"unsupported indexed A output slices {num_output_slices}")

        invoke_indexed_lora_a(
            kwargs["hidden_states"],
            kwargs["lora_a"],
            kwargs["topk_ids"],
            kwargs["token_lora_mapping"],
            intermediate,
            config=configs.gate if site == "gate" else configs.down,
            local_expert_offset=kwargs.get("local_expert_offset", 0),
        )

        expand_kwargs = dict(kwargs)
        expand_kwargs["stage"] = "expand"
        expand_kwargs["intermediate_buffer"] = intermediate
        return delegated_ab(**expand_kwargs)

    virtual_experts.merged_experts_fused_moe_lora_add = indexed_a_delegated_b
    try:
        yield
    finally:
        virtual_experts.merged_experts_fused_moe_lora_add = delegated_ab


def _run_checked(fixture: PipelineFixture, pipeline: str) -> torch.Tensor:
    fixture.reset_hidden()
    fixture.invoke(pipeline)
    torch.cuda.synchronize()
    assert fixture.last_output is not None
    if not bool(torch.isfinite(fixture.last_output).all()):
        raise AssertionError(f"{pipeline} produced a non-finite output")
    return fixture.last_output.clone()


def _capture_production_c0_reference(
    fixture: PipelineFixture,
) -> tuple[torch.Tensor | None, dict[str, object]]:
    """Run the production C0 oracle, preserving expected resource failures."""
    from triton.runtime.errors import OutOfResources

    try:
        reference = _run_checked(fixture, "C0")
    except OutOfResources as exc:
        return None, {
            "status": "unsupported",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    return reference, {"status": "available"}


def _max_abs_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return float((lhs.float() - rhs.float()).abs().max().item())


def _check_lora_delta(
    checks: dict[str, object],
    prefix: str,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    base_only: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> None:
    """Compare active LoRA deltas so the much larger base cannot mask errors."""
    reference_delta = reference.float() - base_only.float()
    candidate_delta = candidate.float() - base_only.float()
    signal = float(reference_delta.abs().max().item())
    error = _max_abs_diff(reference_delta, candidate_delta)
    checks[f"{prefix}_reference_delta_max_abs"] = signal
    checks[f"{prefix}_candidate_delta_max_abs"] = float(
        candidate_delta.abs().max().item()
    )
    checks[f"{prefix}_delta_max_abs_error"] = error
    checks[f"{prefix}_delta_error_over_signal"] = error / signal if signal else None
    checks[f"{prefix}_delta_rtol"] = rtol
    checks[f"{prefix}_delta_atol"] = atol
    torch.testing.assert_close(
        reference_delta,
        candidate_delta,
        rtol=rtol,
        atol=atol,
    )


def _check_pipelines(
    fixture: PipelineFixture,
    pipelines: tuple[str, ...],
    *,
    production_c0_reference: torch.Tensor | None = None,
    pre_b_override_reference: torch.Tensor | None = None,
) -> dict[str, object]:
    checks: dict[str, object] = {
        "n0_role": "matched_base_only_latency_reference_not_active_lora_reference",
        "all_outputs_finite": True,
    }
    if "N0" in pipelines:
        first = _run_checked(fixture, "N0")
        second = _run_checked(fixture, "N0")
        checks["n0_repeat_max_abs"] = _max_abs_diff(first, second)
        torch.testing.assert_close(first, second, rtol=1e-2, atol=1e-2)

    if "C0" in pipelines or "C1" in pipelines:
        # With every row mapped to -1, the SGL pipeline must reduce to the
        # matched stock base runner. This checks the decomposed base pipeline
        # independently of the active-adapter C0-vs-C1 comparison below.
        assert fixture.lora_info is not None
        active_mapping = fixture.lora_info.token_lora_mapping.clone()
        base_reference = _run_checked(fixture, "N0")
        try:
            fixture.lora_info.token_lora_mapping.fill_(-1)
            sgl_base_only = _run_checked(fixture, "C0")
        finally:
            fixture.lora_info.token_lora_mapping.copy_(active_mapping)
        checks["n0_c0_zero_lora_max_abs"] = _max_abs_diff(base_reference, sgl_base_only)
        checks["n0_c0_zero_lora_rtol"] = 6e-2
        checks["n0_c0_zero_lora_atol"] = 6e-2
        torch.testing.assert_close(base_reference, sgl_base_only, rtol=6e-2, atol=6e-2)

        serial = _run_checked(fixture, "C0")
        if production_c0_reference is not None:
            checks["production_c0_candidate_c0_max_abs"] = _max_abs_diff(
                production_c0_reference, serial
            )
            checks["production_c0_candidate_c0_rtol"] = 6e-2
            checks["production_c0_candidate_c0_atol"] = 6e-2
            torch.testing.assert_close(
                production_c0_reference, serial, rtol=6e-2, atol=6e-2
            )
        if pre_b_override_reference is not None:
            checks["pre_b_override_candidate_c0_max_abs"] = _max_abs_diff(
                pre_b_override_reference, serial
            )
            _check_lora_delta(
                checks,
                "pre_b_override_candidate_c0",
                pre_b_override_reference,
                serial,
                sgl_base_only,
                rtol=6e-2,
                atol=1.5e-3,
            )
        overlap = _run_checked(fixture, "C1")
        checks["c0_c1_max_abs"] = _max_abs_diff(serial, overlap)
        checks["c0_c1_rtol"] = 6e-2
        checks["c0_c1_atol"] = 6e-2
        torch.testing.assert_close(serial, overlap, rtol=6e-2, atol=6e-2)
    return checks


@contextmanager
def _single_rank_runtime() -> Iterator[None]:
    from sglang.srt.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.lora.sgl_lora.runtime import init_lora_two_stream_resources
    from sglang.srt.runtime_context import get_context
    from sglang.srt.utils.network import get_open_port

    if torch.distributed.is_initialized():
        raise RuntimeError("M0 driver requires a fresh standalone process")

    torch.cuda.set_device(0)
    port = get_open_port()
    dist_initialized = False
    model_parallel_initialized = False
    try:
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            backend="nccl",
            distributed_init_method=f"tcp://127.0.0.1:{port}",
        )
        dist_initialized = True
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            moe_data_model_parallel_size=1,
        )
        model_parallel_initialized = True
        init_lora_two_stream_resources(torch.device("cuda", 0))
        with get_context().override_server_args(enable_fused_moe_sum_all_reduce=False):
            yield
    finally:
        if model_parallel_initialized:
            destroy_model_parallel()
        if dist_initialized:
            destroy_distributed_environment()


def _resolve_pipelines(requested: str, case: MoeLoraBenchCase) -> tuple[str, ...]:
    if requested == "all":
        return PIPELINES if case.adapters.l_active else ("N0",)
    if requested != "N0" and not case.adapters.l_active:
        raise ValueError(f"{requested} requires at least one active adapter")
    return (requested,)


def _validate_case(case: MoeLoraBenchCase) -> None:
    if case.model.num_slices != 2 or case.model.activation != "swiglu":
        raise NotImplementedError(
            "initial M0 supports gated SwiGLU only; non-gated activation is a "
            "recorded follow-up gap"
        )
    if (case.tp_size, case.ep_size, case.moe_dp_size) != (1, 1, 1):
        raise NotImplementedError("initial M0 is WS1 (TP=EP=MoE-DP=1) only")
    if case.provider != "deepgemm_bf16":
        raise NotImplementedError("initial M0 supports the BF16 provider only")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--device", choices=("auto", "h200", "gb300"), default="auto")
    parser.add_argument("--case-id")
    parser.add_argument("--pipeline", choices=(*PIPELINES, "all"), default="all")
    parser.add_argument(
        "--c1-overlap-policy",
        choices=C1_OVERLAP_POLICIES,
        default="production_auto",
        help=(
            "C1 execution decision; force is a benchmark-only override of the "
            "production token threshold"
        ),
    )
    parser.add_argument(
        "--a-provider",
        choices=("production", "indexed"),
        default="production",
        help="LoRA-A implementation; production remains the default",
    )
    parser.add_argument(
        "--indexed-gate-config",
        default="auto",
        metavar="AUTO_OR_KEY",
        help="indexed gate-A schedule (default: device-specific cold-cache shortlist)",
    )
    parser.add_argument(
        "--indexed-down-config",
        default="auto",
        metavar="AUTO_OR_KEY",
        help="indexed down-A schedule (default: device-specific cold-cache shortlist)",
    )
    parser.add_argument(
        "--gate-b-variant",
        choices=B_VARIANTS,
        default="production",
        help="gate/up LoRA-B family; benchmark-only when not production",
    )
    parser.add_argument(
        "--down-b-variant",
        choices=B_VARIANTS,
        default="production",
        help="down LoRA-B family; benchmark-only when not production",
    )
    parser.add_argument(
        "--gate-b-config",
        metavar="BM,BN,BK,G,W,S",
        help="optional explicit gate/up B launch config",
    )
    parser.add_argument(
        "--down-b-config",
        metavar="BM,BN,BK,G,W,S",
        help="optional explicit down B launch config",
    )
    parser.add_argument("--mode", choices=("time", "nsys", "ncu"), default="time")
    parser.add_argument("--execution", choices=("eager", "cuda_graph"), default="eager")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--profile-iterations", type=int, default=1)
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args(argv)


def _benchmark_pipeline(
    fixture: PipelineFixture,
    pipeline: str,
    run_config: RunConfig,
    case: MoeLoraBenchCase,
    *,
    check: bool,
    a_provider: str = "production",
    b_schedules: BScheduleOverrides = BScheduleOverrides(),
) -> dict[str, object]:
    from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode

    effective_a_provider = "not_applicable" if pipeline == "N0" else a_provider
    two_stream = _pipeline_two_stream_metadata(fixture, pipeline)
    two_stream_effective = bool(two_stream["effective"])

    # Compile/JIT and initialize all lazy resources before timing or capture.
    fixture.reset_hidden()
    fixture.invoke(pipeline)
    torch.cuda.synchronize()
    assert fixture.last_output is not None
    eager_reference = fixture.last_output.clone() if check else None
    fixture.reset_hidden()

    if run_config.execution == "cuda_graph":
        with model_capture_mode():
            batch = make_batch(
                lambda: fixture.invoke(pipeline),
                execution="cuda_graph",
                inner_iterations=1,
            )
    else:
        batch = make_batch(
            lambda: fixture.invoke(pipeline),
            execution="eager",
            inner_iterations=1,
        )

    result: dict[str, object] = {
        "pipeline": pipeline,
        "description": {
            "N0": "matched_base_only_deepgemm",
            "C0": f"sgl_lora_serial_{a_provider}_a",
            "C1": (
                f"sgl_lora_c1_{fixture.c1_overlap_policy}_"
                f"{'two_stream' if two_stream_effective else 'serial'}_"
                f"{a_provider}_a"
            ),
        }[pipeline],
        "a_provider": effective_a_provider,
        "b_schedule": (
            b_schedules.metadata(
                production_variant=(
                    "direct" if case.adapters.max_rank <= 64 else "generic"
                ),
                shared_outer_b=case.adapters.shared_outer,
            )
            if pipeline != "N0"
            else "not_applicable"
        ),
        "retained_components": (
            None
            if pipeline == "N0"
            else [
                (
                    "production_lora_b"
                    if not b_schedules.applied
                    else "production_lora_b_with_benchmark_schedule_override"
                ),
                "production_swiglu_activation",
                "production_deepgemm_base",
            ]
        ),
        "two_stream_requested": two_stream["requested"],
        "two_stream_overlap_effective": two_stream_effective,
        "two_stream_policy": two_stream,
        "c1_route_prewarm": (
            "production_a_and_b_routes_including_conservative_unused_a_overhead"
            if two_stream_effective and a_provider == "indexed"
            else (
                "production"
                if two_stream_effective
                else (
                    "not_run_resolved_serial_production_auto_threshold"
                    if pipeline == "C1"
                    else None
                )
            )
        ),
        "logical_invocations_per_batch": 1,
    }

    if run_config.execution == "cuda_graph" and check:
        fixture.reset_hidden()
        batch.run()
        torch.cuda.synchronize()
        assert fixture.last_output is not None and eager_reference is not None
        graph_diff = _max_abs_diff(eager_reference, fixture.last_output)
        result["graph_correctness"] = {
            "eager_graph_max_abs": graph_diff,
            "rtol": 0.0,
            "atol": 3e-3,
        }
        torch.testing.assert_close(
            eager_reference, fixture.last_output, rtol=0.0, atol=3e-3
        )

    if run_config.mode == "time":
        timing = time_cuda_events(
            batch.run,
            launches_per_batch=1,
            warmup=run_config.warmup,
            samples=run_config.samples,
            before_sample=fixture.reset_hidden,
        )
        result["timing"] = asdict(timing)
        print(
            f"{case.case_id} M0/{pipeline} {run_config.execution}: "
            f"p50={timing.p50_us:.3f} us "
            f"p20/p80={timing.p20_us:.3f}/{timing.p80_us:.3f} us"
        )
    else:
        fixture.reset_hidden()
        label = (
            f"sgl_lora_moe::M0::{pipeline}::{case.case_id}::A={effective_a_provider}::"
            f"GateB={b_schedules.gate_variant}::DownB={b_schedules.down_variant}::"
            f"C1_POLICY={two_stream['policy']}::overlap={two_stream_effective}::"
            f"{run_config.execution}::pdl=auto"
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
    _validate_case(case)
    indexed_configs = None
    if args.a_provider == "indexed":
        if case.adapters.shared_outer:
            raise NotImplementedError(
                "indexed A is a per-expert benchmark candidate and does not support "
                "shared-outer factors"
            )
        indexed_configs = _resolve_indexed_a_configs(
            device, args.indexed_gate_config, args.indexed_down_config
        )
    b_schedules = BScheduleOverrides(
        gate_variant=args.gate_b_variant,
        down_variant=args.down_b_variant,
        gate_config=_parse_b_config(args.gate_b_config),
        down_config=_parse_b_config(args.down_b_config),
    )
    pipelines = _resolve_pipelines(args.pipeline, case)
    if b_schedules.applied and not any(pipeline != "N0" for pipeline in pipelines):
        raise ValueError("B schedule overrides require a benchmarked LoRA pipeline")
    if args.mode != "time" and len(pipelines) != 1:
        raise ValueError("Nsight capture requires one explicit --pipeline")
    if (
        args.mode != "time"
        and args.execution == "eager"
        and args.profile_iterations != 1
    ):
        raise ValueError(
            "eager M0 capture requires --profile-iterations 1 because the "
            "DeepGEMM provider may dispose its input"
        )

    run_config = RunConfig(
        mode=args.mode,
        execution=args.execution,
        warmup=args.warmup,
        samples=args.samples,
        inner_iterations=1,
        profile_iterations=args.profile_iterations,
    )

    with ExitStack() as stack:
        stack.enter_context(_single_rank_runtime())
        need_lora = any(pipeline != "N0" for pipeline in pipelines)
        indexed_applied = args.a_provider == "indexed" and need_lora
        fixture = _build_fixture(
            case,
            need_lora=need_lora,
            need_indexed_a=indexed_applied,
            c1_overlap_policy=args.c1_overlap_policy,
        )
        c1_two_stream = _pipeline_two_stream_metadata(fixture, "C1")
        effective_indexed_c1 = (
            indexed_applied and "C1" in pipelines and bool(c1_two_stream["effective"])
        )

        # Establish the active-adapter production result before installing the
        # module-symbol wrapper. This catches any chain-level indexed-A drift in
        # addition to the existing base-only and C0/C1 checks.
        production_c0_reference = None
        production_c0_reference_status: dict[str, object] = {"status": "not_applicable"}
        if indexed_applied and not args.skip_check:
            (
                production_c0_reference,
                production_c0_reference_status,
            ) = _capture_production_c0_reference(fixture)
        elif indexed_applied:
            production_c0_reference_status = {"status": "skipped"}

        # Capture the same A provider with production B before installing the
        # candidate B schedule. This becomes the chain-level semantic oracle.
        # Indexed A is entered temporarily, then re-entered outside the B
        # configuration below so its launch is never affected by that global
        # benchmark override.
        pre_b_override_reference = None
        pre_b_override_reference_status: dict[str, object] = {
            "status": "not_applicable"
        }
        if b_schedules.applied and need_lora and not args.skip_check:
            if indexed_applied:
                assert indexed_configs is not None
                with _indexed_a_override(fixture, indexed_configs):
                    (
                        pre_b_override_reference,
                        pre_b_override_reference_status,
                    ) = _capture_production_c0_reference(fixture)
            else:
                (
                    pre_b_override_reference,
                    pre_b_override_reference_status,
                ) = _capture_production_c0_reference(fixture)
        elif b_schedules.applied and need_lora:
            pre_b_override_reference_status = {"status": "skipped"}

        # Install B first and indexed A second: indexed A is the outer call
        # wrapper and delegates only routing/B work into the held B config.
        if b_schedules.applied:
            stack.enter_context(_b_schedule_override(b_schedules))
        if indexed_applied:
            assert indexed_configs is not None
            stack.enter_context(_indexed_a_override(fixture, indexed_configs))

        correctness = (
            {"skipped": True}
            if args.skip_check
            else _check_pipelines(
                fixture,
                pipelines,
                production_c0_reference=production_c0_reference,
                pre_b_override_reference=pre_b_override_reference,
            )
        )
        if indexed_applied:
            correctness["production_c0_reference"] = production_c0_reference_status
        if b_schedules.applied:
            correctness["pre_b_override_reference"] = pre_b_override_reference_status
        torch.cuda.synchronize()

        substitutions = []
        if indexed_applied:
            substitutions.append("indexed A")
        if b_schedules.applied:
            substitutions.append("benchmark-selected B schedules")
        substitution_text = (
            f" with {' and '.join(substitutions)}" if substitutions else ""
        )
        b_routing = (
            "benchmark_selected_b_routing"
            if b_schedules.routing_config_overridden
            else "production_b_routing"
        )

        result: dict[str, object] = {
            "environment": _environment(args),
            "case": _case_summary(case),
            "scope": "M0",
            "comparison": (
                "matched DeepGEMM N0 versus SGL LoRA C0/C1"
                f"{substitution_text}; C1 policy={args.c1_overlap_policy}"
            ),
            "execution_policy": {
                **c1_two_stream,
                "pipeline": "C1",
                "benchmarked": "C1" in pipelines,
            },
            "phase_semantics": (
                "synthetic_fixed_local_token_shape; phase is metadata and is not "
                "passed to the M0 runner"
            ),
            "a_provider": {
                "name": args.a_provider,
                "applied": indexed_applied,
                "configs": (
                    indexed_configs.metadata() if indexed_configs is not None else None
                ),
                "substitution_scope": (
                    "gate_and_down_lora_a_only" if indexed_applied else "none"
                ),
                "retained_components": (
                    [
                        (
                            "production_lora_b"
                            if not b_schedules.applied
                            else "production_lora_b_with_benchmark_schedule_override"
                        ),
                        "production_swiglu_activation",
                        "production_deepgemm_base",
                    ]
                    if need_lora
                    else []
                ),
                "production_policy_changed": False,
                "c1_route_overhead": (
                    "not_applicable"
                    if "C1" not in pipelines
                    else (
                        "not_run_resolved_serial_production_auto_threshold"
                        if not fixture.c1_two_stream_enabled
                        else (
                            "conservative unused production A route prewarm retained"
                            if effective_indexed_c1
                            else "production"
                        )
                    )
                ),
            },
            "b_schedule": b_schedules.metadata(
                production_variant=(
                    "direct" if case.adapters.max_rank <= 64 else "generic"
                ),
                shared_outer_b=case.adapters.shared_outer,
            ),
            "route_inclusion": (
                f"raw_route_indexed_a_plus_{b_routing}; effective C1 "
                "retains conservative unused production A route prewarm"
                if effective_indexed_c1
                else (
                    f"raw_route_indexed_a_plus_{b_routing}"
                    if indexed_applied
                    else (
                        "full_pipeline_including_benchmark_selected_b_route_planning"
                        if b_schedules.routing_config_overridden
                        else "full_pipeline_including_lora_route_planning"
                    )
                )
            ),
            "correctness": correctness,
            "base_weight_bytes": sum(
                tensor.numel() * tensor.element_size()
                for tensor in fixture.base_weights
            ),
            "lora_factor_bytes": sum(
                tensor.numel() * tensor.element_size()
                for tensor in fixture.lora_weights
            ),
            "factor_shapes": (
                {
                    "gate_up_a": list(case.factor_shapes.gate_up_a),
                    "gate_up_b": list(case.factor_shapes.gate_up_b),
                    "down_a": list(case.factor_shapes.down_a),
                    "down_b": list(case.factor_shapes.down_b),
                }
                if fixture.lora_weights
                else None
            ),
            "limitations": [
                "gated_swiglu_only",
                "bf16_only",
                "tp1_ep1_moe_dp1_only",
                "non_gated_models_pending",
                "fp8_nvfp4_w4a16_pending",
                *(["benchmark_b_schedule_override"] if b_schedules.applied else []),
                *(
                    [
                        "indexed_a_per_expert_only",
                        "indexed_effective_c1_retains_unused_production_a_route_prewarm",
                    ]
                    if effective_indexed_c1
                    else ["indexed_a_per_expert_only"] if indexed_applied else []
                ),
            ],
            "e0_server_driver": "sglang.benchmark.one_batch_server",
            "pipelines": {},
        }
        pipeline_results = result["pipelines"]
        assert isinstance(pipeline_results, dict)
        for pipeline in pipelines:
            pipeline_results[pipeline] = _benchmark_pipeline(
                fixture,
                pipeline,
                run_config,
                case,
                check=not args.skip_check,
                a_provider=args.a_provider,
                b_schedules=b_schedules,
            )

        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(
                json.dumps(result, indent=2, default=str) + "\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
