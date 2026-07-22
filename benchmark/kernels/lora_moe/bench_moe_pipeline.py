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

The opt-in ``--neutral-baselines`` bracket adds whole-M0 controls without
changing the existing SGL result schema:

* ``legacy_triton``: stock Triton ``N0`` and the classic legacy-hook ``C0``;
* ``experimental_trtllm``: provider-matched TRTLLM BF16 ``N0/C0/C1``.

All providers start from the same canonical BF16 weights and standard top-k.
Provider-private load-time weight conversion is excluded and reported; every
per-forward route/alignment/top-k pack needed by a provider remains inside M0.

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
    python benchmark/kernels/lora_moe/bench_moe_pipeline.py \
      --case-id p0-qwen3.5-35b-a3b-cap1-h200 --pipeline all \
      --neutral-baselines all --execution cuda_graph

For a trace, choose exactly one pipeline and wrap this script with Nsight using
the CUDA-profiler capture range, as in ``bench_local.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, replace
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
NEUTRAL_BASELINES = ("legacy_triton", "experimental_trtllm")
NEUTRAL_BASELINE_CHOICES = ("none", *NEUTRAL_BASELINES, "all")
ROUTE_PATTERNS = (
    "lattice_control",
    "uniform_iid_without_replacement",
    "skewed_iid_without_replacement",
)
PIPELINE_ORDERS = ("forward", "reverse")
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


@dataclass(frozen=True, slots=True)
class NeutralBaselineSpec:
    """One provider-matched whole-M0 control.

    ``pipelines`` names only execution topology. The provider-private weight
    representation and all per-forward conversions are described separately so
    an ``N0`` number is never reused across providers.
    """

    key: str
    base_provider: str
    active_provider: str
    pipelines: tuple[str, ...]
    supported_devices: tuple[str, ...]
    runtime_requirements: tuple[str, ...]
    weight_layout: str
    per_forward_preparation: tuple[str, ...]


_NEUTRAL_BASELINE_SPECS = {
    "legacy_triton": NeutralBaselineSpec(
        key="legacy_triton",
        base_provider="stock_triton_bf16",
        active_provider="legacy_triton_classic_lora_hooks",
        pipelines=("N0", "C0"),
        supported_devices=("h200", "gb300"),
        runtime_requirements=(
            "triton_moe_available",
            "exact_shape_tuned_configs_optional_but_required_for_peak_performance",
        ),
        weight_layout="canonical_standard_bf16",
        per_forward_preparation=(
            "stock_or_staged_triton_expert_alignment",
            "classic_lora_adapter_expert_alignment_for_active_pipeline",
        ),
    ),
    "experimental_trtllm": NeutralBaselineSpec(
        key="experimental_trtllm",
        base_provider="flashinfer_trtllm_bf16_routed",
        active_provider="experimental_sgl_trtllm_bf16_lora",
        pipelines=("N0", "C0", "C1"),
        # The branch's FlashInfer BF16 routed entrypoint unconditionally loads
        # its SM100 module.  The installed H200 toolchain therefore cannot emit
        # a compatible kernel; do not turn that compile failure into a timing.
        supported_devices=("gb300",),
        runtime_requirements=(
            "sm100_flashinfer_bf16_routed_moe",
            "h_moe_and_intermediate_divisible_by_128",
            "vendored_experimental_runner_must_match_flashinfer_header_abi",
        ),
        weight_layout="flashinfer_trtllm_block_major_k_bf16",
        per_forward_preparation=(
            "standard_topk_to_trtllm_packed_topk",
            "virtual_expert_lora_route_plan_for_active_pipeline",
        ),
    ),
}


def _resolve_neutral_baselines(requested: str) -> tuple[str, ...]:
    if requested == "none":
        return ()
    if requested == "all":
        return NEUTRAL_BASELINES
    if requested in _NEUTRAL_BASELINE_SPECS:
        return (requested,)
    raise ValueError(f"unknown neutral baseline {requested!r}")


def _resolve_baseline_pipelines(
    baseline: str,
    requested_pipeline: str,
    case: MoeLoraBenchCase,
) -> tuple[str, ...]:
    """Intersect requested topologies with what one neutral provider supports."""
    spec = _NEUTRAL_BASELINE_SPECS[baseline]
    requested = _resolve_pipelines(requested_pipeline, case)
    return tuple(pipeline for pipeline in requested if pipeline in spec.pipelines)


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
        "experimental_lora_master_effective": os.getenv(
            "SGLANG_EXPERIMENTAL_LORA_OPTI", "0"
        ),
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


@contextmanager
def _experimental_trtllm_environment(enabled: bool) -> Iterator[None]:
    """Enable the existing experimental defaults for an isolated benchmark run.

    The direct benchmark dispatch does not alter production selectors or install
    the package-wide monkey patches. It does need the same master gate that makes
    the experimental backend's default routing/packing optimizations visible.
    """
    if not enabled:
        yield
        return
    name = "SGLANG_EXPERIMENTAL_LORA_OPTI"
    previous = os.environ.get(name)
    os.environ[name] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


@contextmanager
def _host_contention(workers: int) -> Iterator[None]:
    """Optionally keep host cores busy during an eager launch bracket.

    This is a reproducible launch-sensitivity diagnostic, not a server-load
    model.  E0 remains the authority for scheduler and request contention.
    """
    if workers < 0:
        raise ValueError("host load workers must be non-negative")
    processes: list[subprocess.Popen] = []
    try:
        for _ in range(workers):
            processes.append(
                subprocess.Popen(
                    [sys.executable, "-c", "while True: pass"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )
        if processes:
            time.sleep(0.25)
        yield
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def _make_routing(
    case: MoeLoraBenchCase,
    device: torch.device,
    *,
    pattern: str = "lattice_control",
    route_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    """Resolve one fixed top-k route independently of all other fixture data.

    ``lattice_control`` retains the original deterministic arithmetic lattice;
    it is a structured control, not IID. ``uniform_iid_without_replacement``
    draws every token's K distinct experts with equal probability.
    ``skewed_iid_without_replacement`` uses ``torch.multinomial`` with Zipf-1.2
    expert weights, which is a sequential weighted-without-replacement draw.
    Hidden states, top-k weights, adapter mapping, base weights, and LoRA factors
    use separate fixed seeds, so changing only ``route_seed`` changes only IDs.
    """
    tokens = torch.arange(case.t_local, dtype=torch.int32, device=device)
    slots = torch.arange(case.model.top_k, dtype=torch.int32, device=device)
    distribution = "deterministic_arithmetic_lattice"
    seed_effective = False
    if pattern == "lattice_control":
        topk_ids = (tokens[:, None] * 13 + slots[None, :] * 7) % case.e_local
    elif pattern in (
        "uniform_iid_without_replacement",
        "skewed_iid_without_replacement",
    ):
        route_generator = torch.Generator(device=device)
        route_generator.manual_seed(route_seed)
        if pattern == "uniform_iid_without_replacement":
            expert_weights = torch.ones(
                case.e_local, dtype=torch.float32, device=device
            )
            distribution = "uniform"
        else:
            expert_rank = torch.arange(
                1, case.e_local + 1, dtype=torch.float32, device=device
            )
            expert_weights = expert_rank.pow(-1.2)
            distribution = "zipf_alpha_1.2"
        topk_ids = torch.multinomial(
            expert_weights.expand(case.t_local, -1),
            num_samples=case.model.top_k,
            replacement=False,
            generator=route_generator,
        ).to(torch.int32)
        seed_effective = True
    else:
        raise ValueError(f"unknown route pattern {pattern!r}")

    topk_weight_generator = torch.Generator(device=device)
    topk_weight_generator.manual_seed(17)
    topk_weights = torch.rand(
        (case.t_local, case.model.top_k),
        dtype=torch.float32,
        device=device,
        generator=topk_weight_generator,
    )
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)

    # Match the producer contract: active adapters use their weight slot and
    # base-only rows use -1. Configured-but-inactive capacity is never routed.
    identities = list(range(case.adapters.l_active))
    if case.adapters.b_base:
        identities.append(-1)
    identity_tensor = torch.tensor(identities, dtype=torch.int32, device=device)
    token_lora_mapping = identity_tensor[tokens.long() % len(identities)]
    topk_ids = topk_ids.contiguous()
    route_bytes = topk_ids.detach().cpu().numpy().tobytes()
    metadata: dict[str, object] = {
        "pattern": pattern,
        "route_seed": route_seed,
        "seed_effective": seed_effective,
        "sampling": distribution,
        "without_replacement_within_token": True,
        "E_hit": int(torch.unique(topk_ids).numel()),
        "resolved_route_hash": hashlib.sha256(route_bytes).hexdigest(),
        "route_hash_algorithm": "sha256_int32_row_major",
        "topk_weight_seed": 17,
        "fixture_tensor_seed": 20260721,
        "non_route_inputs_fixed_across_route_seeds": True,
    }
    return topk_ids, topk_weights, token_lora_mapping, metadata


def _run_length_encode_token_mapping(
    token_lora_mapping: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a token-domain adapter map to minimal contiguous segments."""
    num_tokens = token_lora_mapping.shape[0]
    if num_tokens == 0:
        return (
            torch.zeros(1, dtype=torch.int32, device=token_lora_mapping.device),
            token_lora_mapping.clone(),
        )
    changes = torch.nonzero(
        token_lora_mapping[1:] != token_lora_mapping[:-1], as_tuple=False
    ).flatten()
    segment_indptr = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=token_lora_mapping.device),
            (changes + 1).to(torch.int32),
            torch.full(
                (1,),
                num_tokens,
                dtype=torch.int32,
                device=token_lora_mapping.device,
            ),
        )
    )
    segment_to_lora = token_lora_mapping[segment_indptr[:-1].long()].clone()
    return segment_indptr, segment_to_lora


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


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _prepare_trtllm_bf16_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
    *,
    is_gated: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Create TRTLLM's load-time BF16 BlockMajorK representation.

    This mirrors unquantized MoE post-load processing. The conversion is a
    provider-residency cost, not a forward operation; its elapsed setup time and
    bytes are reported so it cannot be mistaken for free per-forward work.
    """
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        convert_to_block_layout,
        get_w2_permute_indices_with_cache,
    )

    torch.cuda.synchronize()
    started = time.perf_counter()
    index_cache: dict = {}

    def convert(weight: torch.Tensor, *, gate_up: bool) -> torch.Tensor:
        converted = []
        for expert_weight in weight:
            bytes_view = expert_weight.view(torch.uint8)
            if gate_up:
                indices = _maybe_get_cached_w3_w1_permute_indices(
                    index_cache,
                    bytes_view,
                    128,
                    is_gated_act_gemm=is_gated,
                )
            else:
                indices = get_w2_permute_indices_with_cache(
                    index_cache,
                    bytes_view,
                    128,
                )
            permuted = bytes_view[indices.to(weight.device)].contiguous()
            blocked = convert_to_block_layout(permuted, 128)
            converted.append(blocked.view(torch.bfloat16))
        return torch.stack(converted).contiguous()

    trtllm_w13 = convert(w13, gate_up=True)
    trtllm_w2 = convert(w2, gate_up=False)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1e3
    canonical_bytes = _tensor_bytes(w13) + _tensor_bytes(w2)
    resident_bytes = _tensor_bytes(trtllm_w13) + _tensor_bytes(trtllm_w2)
    metadata: dict[str, object] = {
        "source_layout": "canonical_standard_bf16",
        "resident_layout": "flashinfer_trtllm_block_major_k_bf16",
        "lifetime": "load_time_provider_residency",
        "included_in_m0_timing": False,
        "conversion": "row_permute_then_block_major_k_128",
        "canonical_source_bytes": canonical_bytes,
        "provider_resident_bytes": resident_bytes,
        "elapsed_setup_ms": elapsed_ms,
        "per_forward_weight_conversion": False,
    }
    return trtllm_w13, trtllm_w2, metadata


@dataclass(slots=True)
class PipelineFixture:
    case: MoeLoraBenchCase
    route_metadata: dict[str, object]
    c1_overlap_policy: str
    c1_two_stream_enabled: bool
    hidden_seed: torch.Tensor
    hidden_work: torch.Tensor
    topk_output: object
    runner_config: object
    base_quant_info: object
    base_runner: object
    legacy_base_runner: object | None
    legacy_lora_runner: object | None
    legacy_quant_info: object | None
    legacy_lora_info: object | None
    trtllm_quant_info: object | None
    provider_representations: dict[str, dict[str, object]]
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

    def _dispatch_output(self):
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardDispatchOutput,
        )

        return StandardDispatchOutput(
            hidden_states=self.hidden_work,
            hidden_states_scale=None,
            topk_output=self.topk_output,
        )

    def invoke(self, pipeline: str) -> None:
        dispatch_output = self._dispatch_output()
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

    def invoke_neutral(self, baseline: str, pipeline: str) -> None:
        """Run one provider-matched neutral control at the M0 boundary."""
        dispatch_output = self._dispatch_output()
        if baseline == "legacy_triton":
            if self.legacy_quant_info is None or self.legacy_base_runner is None:
                raise RuntimeError("legacy Triton baseline was not constructed")
            if pipeline == "N0":
                result = self.legacy_base_runner.run(
                    dispatch_output, self.legacy_quant_info
                )
            elif pipeline == "C0":
                if self.legacy_lora_runner is None or self.legacy_lora_info is None:
                    raise RuntimeError(
                        "legacy Triton LoRA baseline was not constructed"
                    )
                result = self.legacy_lora_runner.run(
                    dispatch_output,
                    self.legacy_quant_info,
                    lora_info=self.legacy_lora_info,
                )
            else:
                raise ValueError("legacy Triton has no C1 two-stream topology")
        elif baseline == "experimental_trtllm":
            if self.trtllm_quant_info is None:
                raise RuntimeError("experimental TRTLLM baseline was not constructed")
            if pipeline == "N0":
                from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
                    fused_experts_none_to_flashinfer_trtllm_bf16,
                )

                result = fused_experts_none_to_flashinfer_trtllm_bf16(
                    dispatch_output,
                    self.trtllm_quant_info,
                    self.runner_config,
                    use_routed_topk=True,
                )
            elif pipeline in ("C0", "C1"):
                if self.lora_info is None:
                    raise RuntimeError(
                        "experimental TRTLLM LoRA baseline was not constructed"
                    )
                import sglang.srt.lora.trtllm_lora_temp.lora_dispatch as trt_dispatch
                from sglang.srt.lora.trtllm_lora_temp import (
                    get_original_bf16_moe_lora_func,
                )

                serial_dispatch = get_original_bf16_moe_lora_func()
                if serial_dispatch is None:
                    serial_dispatch = (
                        trt_dispatch.fused_experts_none_to_experimental_sgl_trtllm_bf16_lora
                    )

                if pipeline == "C1" and self.c1_two_stream_enabled:
                    import sglang.srt.lora.trtllm_lora_temp.moe_overlap as overlap

                    # C1 is an explicitly resolved fixed-shape topology in this
                    # benchmark. Bypass the legacy module's internal scalar gate
                    # so ``force`` and production-auto mean the same thing for all
                    # providers; restore immediately after the call.
                    original_gate = overlap.is_two_stream_active
                    overlap.is_two_stream_active = lambda _hidden: True
                    try:
                        two_stream_dispatch = getattr(
                            overlap,
                            "fused_experts_none_to_experimental_sgl_"
                            "trtllm_bf16_lora_two_stream",
                        )
                        result = two_stream_dispatch(
                            dispatch_output,
                            self.trtllm_quant_info,
                            self.runner_config,
                            self.lora_info,
                        )
                    finally:
                        overlap.is_two_stream_active = original_gate
                else:
                    result = serial_dispatch(
                        dispatch_output,
                        self.trtllm_quant_info,
                        self.runner_config,
                        self.lora_info,
                    )
            else:
                raise ValueError(f"unknown TRTLLM pipeline {pipeline!r}")
        else:
            raise ValueError(f"unknown neutral baseline {baseline!r}")
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
    neutral_baselines: tuple[str, ...] = (),
    route_pattern: str = "lattice_control",
    route_seed: int = 0,
    zero_lora_factors: bool = False,
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
    topk_ids, topk_weights, token_lora_mapping, route_metadata = _make_routing(
        case,
        device,
        pattern=route_pattern,
        route_seed=route_seed,
    )
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
        # FlashInfer's BF16 TRTLLM entrypoint requires an explicit zero rather
        # than the MoeRunnerConfig default (None).  M0 intentionally excludes
        # model-level fused shared experts, so make the common fixture contract
        # unambiguous for every provider.
        num_fused_shared_experts=0,
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

    canonical_base_bytes = _tensor_bytes(w13) + _tensor_bytes(w2)
    provider_representations: dict[str, dict[str, object]] = {
        "sgl": {
            "base_provider": "deepgemm_bf16_staged",
            "active_provider": "sgl_lora_deepgemm_bf16",
            "source_layout": "canonical_standard_bf16",
            "resident_layout": "canonical_standard_bf16",
            "canonical_source_bytes": canonical_base_bytes,
            "provider_resident_bytes": canonical_base_bytes,
            "load_time_conversion": "none",
            "load_time_conversion_included_in_m0": False,
            "per_forward_preparation_included_in_m0": [
                "deepgemm_prepare_and_finalize",
                "virtual_expert_lora_route_plan_for_active_pipeline",
            ],
            "output_contract": "bf16_token_domain_T_by_H",
            "output_conversion": "none",
        }
    }

    legacy_base_runner = None
    legacy_lora_runner = None
    legacy_quant_info = None
    legacy_lora_info = None
    if "legacy_triton" in neutral_baselines:
        from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo

        legacy_quant_info = TritonMoeQuantInfo(w13_weight=w13, w2_weight=w2)
        legacy_base_runner = MoeRunner(
            MoeRunnerBackend.TRITON, config, lora_enabled=False
        )
        if need_lora:
            legacy_lora_runner = MoeRunner(
                MoeRunnerBackend.TRITON, config, lora_enabled=True
            )
        spec = _NEUTRAL_BASELINE_SPECS["legacy_triton"]
        provider_representations["legacy_triton"] = {
            "base_provider": spec.base_provider,
            "active_provider": spec.active_provider,
            "source_layout": "canonical_standard_bf16",
            "resident_layout": spec.weight_layout,
            "canonical_source_bytes": canonical_base_bytes,
            "provider_resident_bytes": canonical_base_bytes,
            "load_time_conversion": "none",
            "load_time_conversion_included_in_m0": False,
            "per_forward_preparation_included_in_m0": list(
                spec.per_forward_preparation
            ),
            "output_contract": "bf16_token_domain_T_by_H",
            "output_conversion": "none",
            "lora_route_representation": "classic_segment_to_adapter_expert_alignment",
        }

    trtllm_quant_info = None
    if "experimental_trtllm" in neutral_baselines:
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            FlashInferTrtllmBf16MoeQuantInfo,
        )
        from sglang.srt.lora.trtllm_lora_temp.environ import lora_envs

        trtllm_w13, trtllm_w2, conversion = _prepare_trtllm_bf16_weights(
            w13, w2, is_gated=config.is_gated
        )
        trtllm_quant_info = FlashInferTrtllmBf16MoeQuantInfo(
            gemm1_weights=trtllm_w13,
            gemm2_weights=trtllm_w2,
            global_num_experts=case.model.num_experts,
            local_expert_offset=case.global_expert_offset,
        )
        spec = _NEUTRAL_BASELINE_SPECS["experimental_trtllm"]
        provider_representations["experimental_trtllm"] = {
            "base_provider": spec.base_provider,
            "active_provider": spec.active_provider,
            **conversion,
            "load_time_conversion": conversion["conversion"],
            "load_time_conversion_included_in_m0": False,
            "per_forward_preparation_included_in_m0": list(
                spec.per_forward_preparation
            ),
            "output_contract": "bf16_token_domain_T_by_H",
            "output_conversion": "none",
            "lora_route_representation": "virtual_expert_aligned_plan",
            "experimental_policy": {
                "shrink_split_k": lora_envs.SGLANG_ENABLE_LORA_SHRINK_SPLIT_K.get(),
                "fused_merged_align": lora_envs.SGLANG_OPT_LORA_FUSED_MERGED_ALIGN.get(),
                "fused_topk_pack": lora_envs.SGLANG_OPT_LORA_FUSED_TOPK_PACK.get(),
                "prefill_routing_reuse": lora_envs.SGLANG_OPT_LORA_PREFILL_ROUTING_REUSE.get(),
                "gated_split_fix": lora_envs.SGLANG_ENABLE_LORA_MOE_GATEUP_GATED_SPLIT.get(),
                "legacy_two_stream_max_tokens": lora_envs.SGLANG_TWO_STREAM_MAX_TOKENS.get(),
                "benchmark_c1_policy": c1_overlap_policy,
            },
        }

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

        def make_factor(shape: tuple[int, ...]) -> torch.Tensor:
            if zero_lora_factors:
                return torch.zeros(shape, dtype=torch.bfloat16, device=device)
            return _random_bf16(shape, generator=generator, device=device, scale=0.02)

        gate_a = make_factor(shapes.gate_up_a)
        gate_b = make_factor(shapes.gate_up_b)
        down_a = make_factor(shapes.down_a)
        down_b = make_factor(shapes.down_b)
        lora_weights = (gate_a, gate_b, down_a, down_b)

        lora_ranks = torch.zeros(
            case.adapters.l_capacity, dtype=torch.int32, device=device
        )
        adapter_enabled = torch.zeros_like(lora_ranks)
        if case.adapters.l_active:
            lora_ranks[: case.adapters.l_active] = case.adapters.rank
            adapter_enabled[: case.adapters.l_active] = 1
        # The classic runner consumes request segments.  Build the minimal
        # run-length encoding of the canonical token mapping: one-token
        # segments are semantically valid but can dramatically overstate the
        # legacy orchestration cost for a single-adapter batch.
        segment_indptr, segment_to_lora = _run_length_encode_token_mapping(
            token_lora_mapping
        )
        lora_info = LoRAInfo(
            gate_up_lora_a_weights=gate_a,
            gate_up_lora_b_weights=gate_b,
            down_lora_a_weights=down_a,
            down_lora_b_weights=down_b,
            seg_indptr=segment_indptr,
            req_to_lora=segment_to_lora,
            lora_ranks=lora_ranks,
            adapter_enabled=adapter_enabled,
            token_lora_mapping=token_lora_mapping,
            max_lora_rank=case.adapters.max_rank,
            num_experts=case.e_local,
            has_active_lora=not zero_lora_factors,
            experts_shared_outer_loras=case.adapters.shared_outer,
            tp_size=1,
            tp_rank=0,
            hidden_size=case.model.h_moe,
            lora_use_virtual_experts=True,
        )
        if "legacy_triton" in neutral_baselines:
            legacy_lora_info = replace(
                lora_info,
                lora_use_virtual_experts=False,
                cg_buffers=None,
            )
            provider_representations["legacy_triton"].update(
                {
                    "adapter_segment_encoding": "minimal_contiguous_run_length",
                    "adapter_segment_count": int(segment_to_lora.numel()),
                    "token_count": case.t_local,
                }
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
        route_metadata=route_metadata,
        c1_overlap_policy=c1_overlap_policy,
        c1_two_stream_enabled=_resolve_c1_overlap(c1_overlap_policy, case.t_local),
        hidden_seed=hidden_seed,
        hidden_work=hidden_seed.clone(),
        topk_output=topk_output,
        runner_config=config,
        base_quant_info=base_quant_info,
        base_runner=base_runner,
        legacy_base_runner=legacy_base_runner,
        legacy_lora_runner=legacy_lora_runner,
        legacy_quant_info=legacy_quant_info,
        legacy_lora_info=legacy_lora_info,
        trtllm_quant_info=trtllm_quant_info,
        provider_representations=provider_representations,
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


class _CuTeStaticGateAOverride:
    """Benchmark-only grouped Tensor Core gate-A upper-bound bracket.

    The installed Blackwell grouped GEMM consumes one static compact route.
    That is sufficient to answer whether its gather/GEMM/unpack boundary can
    survive the complete M0 pipeline, but it is *not* a serving-compatible
    provider: a CUDA-graph replay with different expert IDs or adapter mapping
    would need a new route and new runtime problem descriptors.  The result is
    therefore recorded as an upper bound and can never directly select a
    production dispatch policy.

    Only gate/up A is replaced.  Gate/up B, activation, both down LoRA stages,
    and the base MoE path remain production code.  A retained output allocation
    gives the CuTe plan graph-stable pointers; the production call's transient
    gate-A intermediate is intentionally unused.
    """

    def __init__(self, fixture: PipelineFixture) -> None:
        from benchmark.kernels.lora_moe.cutedsl_grouped_tensorcore import (
            GroupedTactic,
        )
        from benchmark.kernels.lora_moe.cutedsl_moe_boundary import (
            build_compact_grouped_route,
        )

        if fixture.lora_info is None or not fixture.lora_weights:
            raise RuntimeError("CuTe gate-A override requires active LoRA factors")
        case = fixture.case
        gate_a = fixture.lora_weights[0]
        self.fixture = fixture
        self.route = build_compact_grouped_route(
            fixture.topk_output.topk_ids,
            fixture.lora_info.token_lora_mapping,
            num_experts=case.e_local,
            num_adapters=case.adapters.l_capacity,
            local_expert_offset=case.global_expert_offset,
        )
        self.weight = gate_a.reshape(
            case.adapters.l_capacity * case.e_local,
            gate_a.shape[-2],
            gate_a.shape[-1],
        )
        self.tactic = GroupedTactic(mma_m=128, mma_n=128)
        self.output = torch.empty(
            (case.t_local, case.model.top_k, gate_a.shape[-2]),
            dtype=gate_a.dtype,
            device=gate_a.device,
        )
        self.boundary = None
        self._input_ptr: int | None = None
        self._passthrough_depth = 0
        self._delegated = None

    def _ensure_boundary(self, hidden_states: torch.Tensor) -> None:
        from benchmark.kernels.lora_moe.cutedsl_moe_boundary import (
            GroupedGemmBoundary,
        )

        pointer = hidden_states.data_ptr()
        if self.boundary is not None and pointer == self._input_ptr:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "CuTe gate-A input address changed during CUDA graph capture"
            )
        self.boundary = GroupedGemmBoundary(
            hidden_states,
            self.weight,
            self.output.view(-1, self.output.shape[-1]),
            self.route,
            input_pair_major=False,
            top_k=self.fixture.case.model.top_k,
            tactic=self.tactic,
        )
        self._input_ptr = pointer

    @contextmanager
    def production_passthrough(self) -> Iterator[None]:
        """Temporarily restore production for the mutable all-base oracle."""

        self._passthrough_depth += 1
        try:
            yield
        finally:
            self._passthrough_depth -= 1

    def __enter__(self) -> "_CuTeStaticGateAOverride":
        from sglang.srt.lora.sgl_lora.triton_ops import virtual_experts

        self._delegated = virtual_experts.merged_experts_fused_moe_lora_add

        def cutedsl_gate_a_delegated_b(*args, **kwargs):
            if args:
                raise TypeError("CuTe gate-A benchmark wrapper requires keywords")
            assert self._delegated is not None
            if self._passthrough_depth:
                return self._delegated(**kwargs)
            stage = kwargs.get("stage", "all")
            if stage == "routing":
                # C1 retains production prewarm.  This lane benchmarks serial C0;
                # keeping it here makes accidental C1 use conservative, not faster.
                return self._delegated(**kwargs)
            if kwargs["num_output_slices"] != 2:
                return self._delegated(**kwargs)
            if stage != "all":
                raise ValueError(
                    "static CuTe gate-A supports serial stage='all' only, got "
                    f"{stage!r}"
                )
            if kwargs["experts_shared_outer_loras_a"]:
                raise NotImplementedError(
                    "static CuTe gate-A supports per-expert factors only"
                )
            self._ensure_boundary(kwargs["hidden_states"])
            assert self.boundary is not None
            self.boundary.invoke_boundary()
            expand_kwargs = dict(kwargs)
            expand_kwargs["stage"] = "expand"
            expand_kwargs["intermediate_buffer"] = self.output
            return self._delegated(**expand_kwargs)

        virtual_experts.merged_experts_fused_moe_lora_add = cutedsl_gate_a_delegated_b
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        from sglang.srt.lora.sgl_lora.triton_ops import virtual_experts

        assert self._delegated is not None
        virtual_experts.merged_experts_fused_moe_lora_add = self._delegated

    def metadata(self) -> dict[str, object]:
        return {
            "selection": "fixed_qwen_prefill_r128_k0_winner",
            "provider": "blackwell_cutedsl_grouped_tensorcore",
            "scope": "gate_up_lora_a_only",
            "route_contract": "static_compact_route_built_outside_m0",
            "dispatch_eligible": False,
            "reason_not_dispatch_eligible": (
                "runtime expert IDs and adapter mapping can change between graph replays"
            ),
            "tactic": asdict(self.tactic),
            "boundary": self.boundary.metadata() if self.boundary is not None else None,
        }


def _run_checked(fixture: PipelineFixture, pipeline: str) -> torch.Tensor:
    fixture.reset_hidden()
    fixture.invoke(pipeline)
    torch.cuda.synchronize()
    assert fixture.last_output is not None
    if not bool(torch.isfinite(fixture.last_output).all()):
        raise AssertionError(f"{pipeline} produced a non-finite output")
    return fixture.last_output.clone()


def _run_neutral_checked(
    fixture: PipelineFixture, baseline: str, pipeline: str
) -> torch.Tensor:
    fixture.reset_hidden()
    fixture.invoke_neutral(baseline, pipeline)
    torch.cuda.synchronize()
    assert fixture.last_output is not None
    if not bool(torch.isfinite(fixture.last_output).all()):
        raise AssertionError(f"{baseline}/{pipeline} produced a non-finite output")
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


_BF16_DELTA_ABS_FLOOR = 2.0**-11
_STRICT_DELTA_RTOL = 2e-2


def _strict_delta_atol(signal: float) -> float:
    """Return an absolute tolerance that cannot hide a dropped LoRA delta.

    ``2**-11`` covers two BF16 rounding steps around the fixture's typical
    output scale: provider deltas subtract two independently rounded outputs.
    The signal/10 cap still guarantees an all-zero candidate fails.  A 1%
    signal component keeps the rule scale-aware above that quantization floor.
    """
    if not signal > 0.0:
        raise AssertionError("active LoRA reference produced an all-zero delta")
    return min(signal / 10.0, max(signal / 100.0, _BF16_DELTA_ABS_FLOOR))


def _check_lora_delta(
    checks: dict[str, object],
    prefix: str,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    base_only: torch.Tensor,
    *,
    rtol: float = _STRICT_DELTA_RTOL,
    atol: float | None = None,
) -> None:
    """Compare active LoRA deltas so the much larger base cannot mask errors."""
    reference_delta = reference.float() - base_only.float()
    candidate_delta = candidate.float() - base_only.float()
    signal = float(reference_delta.abs().max().item())
    strict_atol = _strict_delta_atol(signal)
    effective_rtol = min(rtol, _STRICT_DELTA_RTOL)
    effective_atol = strict_atol if atol is None else min(atol, strict_atol)
    error = _max_abs_diff(reference_delta, candidate_delta)
    checks[f"{prefix}_reference_delta_max_abs"] = signal
    checks[f"{prefix}_candidate_delta_max_abs"] = float(
        candidate_delta.abs().max().item()
    )
    checks[f"{prefix}_delta_max_abs_error"] = error
    checks[f"{prefix}_delta_error_over_signal"] = error / signal if signal else None
    checks[f"{prefix}_delta_rtol"] = effective_rtol
    checks[f"{prefix}_delta_atol"] = effective_atol
    checks[f"{prefix}_delta_atol_signal_cap"] = signal / 10.0
    checks[f"{prefix}_bf16_abs_floor"] = _BF16_DELTA_ABS_FLOOR
    torch.testing.assert_close(
        reference_delta,
        candidate_delta,
        rtol=effective_rtol,
        atol=effective_atol,
    )


def _check_provider_lora_delta(
    checks: dict[str, object],
    prefix: str,
    reference_active: torch.Tensor,
    reference_base: torch.Tensor,
    candidate_active: torch.Tensor,
    candidate_base: torch.Tensor,
    *,
    rtol: float = _STRICT_DELTA_RTOL,
    atol: float | None = None,
) -> None:
    """Compare LoRA contributions after subtracting each provider's own N0."""
    reference_delta = reference_active.float() - reference_base.float()
    candidate_delta = candidate_active.float() - candidate_base.float()
    signal = float(reference_delta.abs().max().item())
    strict_atol = _strict_delta_atol(signal)
    effective_rtol = min(rtol, _STRICT_DELTA_RTOL)
    effective_atol = strict_atol if atol is None else min(atol, strict_atol)
    error = _max_abs_diff(reference_delta, candidate_delta)
    checks[f"{prefix}_reference_delta_max_abs"] = signal
    checks[f"{prefix}_candidate_delta_max_abs"] = float(
        candidate_delta.abs().max().item()
    )
    checks[f"{prefix}_delta_max_abs_error"] = error
    checks[f"{prefix}_delta_error_over_signal"] = error / signal if signal else None
    checks[f"{prefix}_delta_rtol"] = effective_rtol
    checks[f"{prefix}_delta_atol"] = effective_atol
    checks[f"{prefix}_delta_atol_signal_cap"] = signal / 10.0
    checks[f"{prefix}_bf16_abs_floor"] = _BF16_DELTA_ABS_FLOOR
    torch.testing.assert_close(
        reference_delta,
        candidate_delta,
        rtol=effective_rtol,
        atol=effective_atol,
    )


def _check_pipelines(
    fixture: PipelineFixture,
    pipelines: tuple[str, ...],
    *,
    production_c0_reference: torch.Tensor | None = None,
    pre_b_override_reference: torch.Tensor | None = None,
    static_a_override: _CuTeStaticGateAOverride | None = None,
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
            if static_a_override is None:
                sgl_base_only = _run_checked(fixture, "C0")
            else:
                # The CuTe upper-bound route is intentionally immutable.  The
                # base-only oracle still validates the retained production path.
                with static_a_override.production_passthrough():
                    sgl_base_only = _run_checked(fixture, "C0")
        finally:
            fixture.lora_info.token_lora_mapping.copy_(active_mapping)
        checks["n0_c0_zero_lora_max_abs"] = _max_abs_diff(base_reference, sgl_base_only)
        checks["n0_c0_zero_lora_rtol"] = 6e-2
        checks["n0_c0_zero_lora_atol"] = 6e-2
        torch.testing.assert_close(base_reference, sgl_base_only, rtol=6e-2, atol=6e-2)

        serial = _run_checked(fixture, "C0")
        # Full-output tolerances can conceal a missing LoRA contribution behind
        # the much larger base result.  Record and require a non-zero C0 delta
        # before comparing any candidate execution plan.
        _check_lora_delta(
            checks,
            "c0_active",
            serial,
            serial,
            sgl_base_only,
        )
        if production_c0_reference is not None:
            checks["production_c0_candidate_c0_max_abs"] = _max_abs_diff(
                production_c0_reference, serial
            )
            checks["production_c0_candidate_c0_rtol"] = 6e-2
            checks["production_c0_candidate_c0_atol"] = 6e-2
            torch.testing.assert_close(
                production_c0_reference, serial, rtol=6e-2, atol=6e-2
            )
            _check_lora_delta(
                checks,
                "production_c0_candidate_c0",
                production_c0_reference,
                serial,
                sgl_base_only,
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
            )
        overlap = _run_checked(fixture, "C1")
        checks["c0_c1_max_abs"] = _max_abs_diff(serial, overlap)
        checks["c0_c1_rtol"] = 6e-2
        checks["c0_c1_atol"] = 6e-2
        torch.testing.assert_close(serial, overlap, rtol=6e-2, atol=6e-2)
        _check_lora_delta(
            checks,
            "c0_c1",
            serial,
            overlap,
            sgl_base_only,
        )
    return checks


def _check_neutral_baselines(
    fixture: PipelineFixture,
    neutral_baselines: tuple[str, ...],
    requested_pipeline: str,
) -> dict[str, object]:
    """Check every provider at the common final-BF16 M0 boundary.

    Full base outputs are compared directly. Active LoRA is checked both as a
    full output and as ``C0(provider)-N0(provider)`` so a base-provider rounding
    difference cannot hide or manufacture LoRA parity.
    """
    if not neutral_baselines:
        return {}
    reference_n0 = _run_checked(fixture, "N0")
    need_active_reference = any(
        any(
            pipeline != "N0"
            for pipeline in _resolve_baseline_pipelines(
                baseline, requested_pipeline, fixture.case
            )
        )
        for baseline in neutral_baselines
    )
    reference_c0 = _run_checked(fixture, "C0") if need_active_reference else None
    results: dict[str, object] = {}
    for baseline in neutral_baselines:
        pipelines = _resolve_baseline_pipelines(
            baseline, requested_pipeline, fixture.case
        )
        checks: dict[str, object] = {
            "common_input_contract": "bf16_hidden_plus_standard_topk",
            "common_output_contract": "bf16_token_domain_T_by_H",
            "provider_matched_n0": True,
            "pipelines_checked": list(pipelines),
        }
        provider_n0 = _run_neutral_checked(fixture, baseline, "N0")
        checks["sgl_n0_provider_n0_max_abs"] = _max_abs_diff(reference_n0, provider_n0)
        checks["base_rtol"] = 8e-2
        checks["base_atol"] = 8e-2
        torch.testing.assert_close(reference_n0, provider_n0, rtol=8e-2, atol=8e-2)

        if "C0" in pipelines or "C1" in pipelines:
            assert reference_c0 is not None
            provider_c0 = _run_neutral_checked(fixture, baseline, "C0")
            checks["sgl_c0_provider_c0_max_abs"] = _max_abs_diff(
                reference_c0, provider_c0
            )
            checks["active_rtol"] = 8e-2
            checks["active_atol"] = 8e-2
            torch.testing.assert_close(reference_c0, provider_c0, rtol=8e-2, atol=8e-2)
            _check_provider_lora_delta(
                checks,
                "sgl_c0_provider_c0",
                reference_c0,
                reference_n0,
                provider_c0,
                provider_n0,
            )
            if "C1" in pipelines:
                provider_c1 = _run_neutral_checked(fixture, baseline, "C1")
                checks["provider_c0_c1_max_abs"] = _max_abs_diff(
                    provider_c0, provider_c1
                )
                torch.testing.assert_close(
                    provider_c0, provider_c1, rtol=8e-2, atol=8e-2
                )
        results[baseline] = checks
    return results


def _check_all_base_sgl_c0_sentinel(
    fixture: PipelineFixture,
) -> dict[str, object]:
    """Verify the opt-in captured SGL path for an all-base adapter batch."""
    if fixture.lora_info is None or not fixture.lora_weights:
        raise RuntimeError("all-base SGL C0 sentinel requires a LoRA fixture")
    if not bool((fixture.lora_info.token_lora_mapping == -1).all()):
        raise AssertionError("all-base sentinel requires token_lora_mapping=-1")
    if any(bool(torch.count_nonzero(weight)) for weight in fixture.lora_weights):
        raise AssertionError("all-base sentinel requires zero LoRA factors")

    n0 = _run_checked(fixture, "N0")
    c0 = _run_checked(fixture, "C0")
    max_abs = _max_abs_diff(n0, c0)
    rtol = 2e-2
    atol = 3e-3
    torch.testing.assert_close(n0, c0, rtol=rtol, atol=atol)
    return {
        "status": "available",
        "n0_c0_max_abs": max_abs,
        "rtol": rtol,
        "atol": atol,
        "token_lora_mapping": "all_-1",
        "lora_factors": "all_zero",
        "semantic_role": "captured_all_base_graph_tax_sentinel",
    }


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


def _validate_neutral_case(
    case: MoeLoraBenchCase, neutral_baselines: tuple[str, ...]
) -> None:
    if "experimental_trtllm" not in neutral_baselines:
        return
    spec = _NEUTRAL_BASELINE_SPECS["experimental_trtllm"]
    if case.device not in spec.supported_devices:
        raise NotImplementedError(
            "experimental TRTLLM BF16 on this branch loads FlashInfer's SM100 "
            f"module and cannot run on {case.device}; supported benchmark devices: "
            f"{', '.join(spec.supported_devices)}"
        )
    if case.model.h_moe % 128 or case.i_physical % 128:
        raise ValueError(
            "experimental TRTLLM BF16 needs H_moe and I_physical divisible by 128; "
            f"case {case.case_id!r} resolves to H={case.model.h_moe}, "
            f"I={case.i_physical}. Select a model-scale P0 case instead of the "
            "synthetic smoke case."
        )


def _validate_all_base_sentinel(
    case: MoeLoraBenchCase,
    *,
    enabled: bool,
    execution: str,
    a_provider: str,
    b_schedules: BScheduleOverrides,
) -> None:
    if not enabled:
        return
    if case.adapters.l_active != 0 or case.adapters.b_base != 1:
        raise ValueError(
            "--all-base-sgl-c0-sentinel requires an all-base P0 case "
            "(L_active=0, B_base=1)"
        )
    if execution != "cuda_graph":
        raise ValueError("--all-base-sgl-c0-sentinel requires --execution cuda_graph")
    if a_provider != "production" or b_schedules.applied:
        raise ValueError(
            "--all-base-sgl-c0-sentinel requires unchanged production SGL A/B"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--device", choices=("auto", "h200", "gb300"), default="auto")
    parser.add_argument("--case-id")
    parser.add_argument("--pipeline", choices=(*PIPELINES, "all"), default="all")
    parser.add_argument(
        "--neutral-baselines",
        choices=NEUTRAL_BASELINE_CHOICES,
        default="none",
        help=(
            "opt-in provider-matched whole-M0 bracket; load-time provider weight "
            "conversion is reported and per-forward preparation stays timed"
        ),
    )
    parser.add_argument(
        "--route-pattern",
        choices=ROUTE_PATTERNS,
        default="lattice_control",
        help=(
            "fixed route family: the historical lattice control, true uniform "
            "IID top-k without replacement, or Zipf-1.2 weighted IID top-k "
            "without replacement"
        ),
    )
    parser.add_argument(
        "--route-seed",
        type=int,
        default=0,
        help="seed for IID expert-ID draws; recorded but ignored by lattice_control",
    )
    parser.add_argument(
        "--all-base-sgl-c0-sentinel",
        action="store_true",
        help=(
            "opt in only on an all-base case: build zero LoRA factors with every "
            "token mapped to -1, capture SGL C0, and report its tax over matched N0"
        ),
    )
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
        choices=("production", "indexed", "cutedsl_static_gate"),
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
    parser.add_argument(
        "--pdl",
        choices=("auto", "on", "off"),
        default="auto",
        help="programmatic PDL policy for the SGL virtual-expert chain",
    )
    parser.add_argument(
        "--cache-state",
        choices=("hot", "cold"),
        default="hot",
        help=(
            "working-set state before every measured M0 invocation; cold evicts "
            "the complete block working set outside the timing events"
        ),
    )
    parser.add_argument(
        "--pipeline-order",
        choices=PIPELINE_ORDERS,
        default="forward",
        help="counterbalance provider-local pipeline timing order across processes",
    )
    parser.add_argument(
        "--host-load-workers",
        type=int,
        default=0,
        help=(
            "launch-sensitivity diagnostic: number of independent busy host "
            "processes; use only with eager timing"
        ),
    )
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
    neutral_baseline: str | None = None,
    expect_active_lora: bool = True,
    cache_control=None,
) -> dict[str, object]:
    from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode

    provider_key = neutral_baseline or "sgl"
    is_sgl = neutral_baseline is None
    effective_a_provider = (
        "not_applicable" if pipeline == "N0" or not is_sgl else a_provider
    )
    two_stream = _pipeline_two_stream_metadata(fixture, pipeline)
    two_stream_effective = bool(two_stream["effective"])

    def invoke() -> None:
        if neutral_baseline is None:
            fixture.invoke(pipeline)
        else:
            fixture.invoke_neutral(neutral_baseline, pipeline)

    def invoke_base() -> None:
        if neutral_baseline is None:
            fixture.invoke("N0")
        else:
            fixture.invoke_neutral(neutral_baseline, "N0")

    # Compile/JIT and initialize all lazy resources before timing or capture.
    eager_base_reference = None
    if check and pipeline != "N0":
        fixture.reset_hidden()
        invoke_base()
        torch.cuda.synchronize()
        assert fixture.last_output is not None
        eager_base_reference = fixture.last_output.clone()
    fixture.reset_hidden()
    invoke()
    torch.cuda.synchronize()
    assert fixture.last_output is not None
    eager_reference = fixture.last_output.clone() if check else None
    fixture.reset_hidden()

    if run_config.execution == "cuda_graph":
        with model_capture_mode():
            batch = make_batch(
                invoke,
                execution="cuda_graph",
                inner_iterations=1,
            )
    else:
        batch = make_batch(
            invoke,
            execution="eager",
            inner_iterations=1,
        )

    if neutral_baseline is None:
        description = (
            "sgl_c0_all_base_zero_factor_graph_sentinel"
            if pipeline == "C0" and not expect_active_lora
            else {
                "N0": "matched_base_only_deepgemm",
                "C0": f"sgl_lora_serial_{a_provider}_a",
                "C1": (
                    f"sgl_lora_c1_{fixture.c1_overlap_policy}_"
                    f"{'two_stream' if two_stream_effective else 'serial'}_"
                    f"{a_provider}_a"
                ),
            }[pipeline]
        )
    else:
        spec = _NEUTRAL_BASELINE_SPECS[neutral_baseline]
        provider = spec.base_provider if pipeline == "N0" else spec.active_provider
        topology = (
            "two_stream" if pipeline == "C1" and two_stream_effective else "serial"
        )
        description = f"{provider}_{pipeline.lower()}_{topology}"

    if not two_stream_effective:
        route_prewarm = (
            "not_run_resolved_serial_production_auto_threshold"
            if pipeline == "C1"
            else None
        )
    elif is_sgl and a_provider == "indexed":
        route_prewarm = (
            "production_a_and_b_routes_including_conservative_unused_a_overhead"
        )
    elif is_sgl:
        route_prewarm = "production"
    else:
        route_prewarm = "provider_native"

    result: dict[str, object] = {
        "pipeline": pipeline,
        "provider_key": provider_key,
        "description": description,
        "semantic_boundary": "bf16_hidden_plus_standard_topk_to_bf16_T_by_H",
        "provider_representation": fixture.provider_representations[provider_key],
        "a_provider": effective_a_provider,
        "b_schedule": (
            b_schedules.metadata(
                production_variant=(
                    "direct" if case.adapters.max_rank <= 64 else "generic"
                ),
                shared_outer_b=case.adapters.shared_outer,
            )
            if pipeline != "N0" and is_sgl
            else "not_applicable"
        ),
        "retained_components": (
            None
            if pipeline == "N0"
            else (
                [
                    (
                        "production_lora_b"
                        if not b_schedules.applied
                        else "production_lora_b_with_benchmark_schedule_override"
                    ),
                    "production_swiglu_activation",
                    "production_deepgemm_base",
                ]
                if is_sgl
                else [
                    _NEUTRAL_BASELINE_SPECS[neutral_baseline].active_provider,
                    _NEUTRAL_BASELINE_SPECS[neutral_baseline].base_provider,
                    "provider_native_swiglu",
                ]
            )
        ),
        "two_stream_requested": two_stream["requested"],
        "two_stream_overlap_effective": two_stream_effective,
        "two_stream_policy": two_stream,
        "c1_route_prewarm": route_prewarm,
        "logical_invocations_per_batch": 1,
        "cache_control": (
            cache_control.metadata() if cache_control is not None else {"state": "hot"}
        ),
    }

    if run_config.execution == "cuda_graph" and check:
        fixture.reset_hidden()
        batch.run()
        torch.cuda.synchronize()
        assert fixture.last_output is not None and eager_reference is not None
        graph_diff = _max_abs_diff(eager_reference, fixture.last_output)
        graph_atol = 3e-3
        result["graph_correctness"] = {
            "eager_graph_max_abs": graph_diff,
            "rtol": 0.0,
            "atol": graph_atol,
        }
        torch.testing.assert_close(
            eager_reference, fixture.last_output, rtol=0.0, atol=graph_atol
        )
        if pipeline != "N0" and expect_active_lora:
            assert eager_base_reference is not None
            active_delta_checks: dict[str, object] = {}
            _check_lora_delta(
                active_delta_checks,
                "eager_graph_active",
                eager_reference,
                fixture.last_output,
                eager_base_reference,
            )
            result["graph_correctness"]["active_delta"] = active_delta_checks

    if run_config.mode == "time":

        def prepare_sample() -> None:
            fixture.reset_hidden()
            if cache_control is not None:
                cache_control.evict()

        timing = time_cuda_events(
            batch.run,
            launches_per_batch=1,
            warmup=run_config.warmup,
            samples=run_config.samples,
            before_sample=prepare_sample,
        )
        result["timing"] = asdict(timing)
        print(
            f"{case.case_id} M0/{provider_key}/{pipeline} {run_config.execution}: "
            f"p50={timing.p50_us:.3f} us "
            f"p20/p80={timing.p20_us:.3f}/{timing.p80_us:.3f} us"
        )
    else:
        fixture.reset_hidden()
        gate_b_label = b_schedules.gate_variant if is_sgl else "provider_native"
        down_b_label = b_schedules.down_variant if is_sgl else "provider_native"
        label = (
            f"sgl_lora_moe::M0::{provider_key}::{pipeline}::{case.case_id}::"
            f"A={effective_a_provider}::"
            f"GateB={gate_b_label}::DownB={down_b_label}::"
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


def _matched_latency_summary(
    pipeline_results: dict[str, object],
) -> dict[str, object]:
    """Derive provider-local LoRA tax without ever borrowing another N0."""
    n0 = pipeline_results.get("N0")
    if not isinstance(n0, dict) or not isinstance(n0.get("timing"), dict):
        return {
            "status": "unavailable_without_timed_provider_n0",
            "provider_matched": True,
        }
    n0_p50 = float(n0["timing"]["p50_us"])
    summary: dict[str, object] = {
        "status": "available",
        "provider_matched": True,
        "n0_p50_us": n0_p50,
        "active": {},
    }
    active = summary["active"]
    assert isinstance(active, dict)
    for pipeline in ("C0", "C1"):
        candidate = pipeline_results.get(pipeline)
        if not isinstance(candidate, dict) or not isinstance(
            candidate.get("timing"), dict
        ):
            continue
        p50 = float(candidate["timing"]["p50_us"])
        active[pipeline] = {
            "p50_us": p50,
            "active_over_n0_p50_us": p50 - n0_p50,
            "n0_retention_percent": 100.0 * n0_p50 / p50,
        }
    return summary


def _main(args: argparse.Namespace) -> int:
    if args.list_cases:
        device = args.device if args.device != "auto" else "h200"
        _list_cases(device)
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    if args.host_load_workers < 0:
        raise ValueError("--host-load-workers must be non-negative")
    if args.host_load_workers and (args.mode != "time" or args.execution != "eager"):
        raise ValueError("--host-load-workers is an eager timing diagnostic only")
    if args.cache_state == "cold" and args.mode != "time":
        raise ValueError("cold M0 profiling is not supported; profile the timed winner")

    device = _detect_device(args.device)
    case = _select_case(device, args.case_id)
    _validate_case(case)
    neutral_baselines = _resolve_neutral_baselines(args.neutral_baselines)
    _validate_neutral_case(case, neutral_baselines)
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
    if args.pipeline_order == "reverse":
        pipelines = tuple(reversed(pipelines))
    _validate_all_base_sentinel(
        case,
        enabled=args.all_base_sgl_c0_sentinel,
        execution=args.execution,
        a_provider=args.a_provider,
        b_schedules=b_schedules,
    )
    if neutral_baselines and (args.a_provider != "production" or b_schedules.applied):
        raise ValueError(
            "neutral baselines require the unchanged production SGL A/B path; "
            "run benchmark-only A/B substitutions in a separate bracket"
        )
    if args.a_provider == "cutedsl_static_gate" and b_schedules.applied:
        raise ValueError(
            "the CuTe static-route M0 upper bound must retain production B schedules"
        )
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
        stack.enter_context(
            _experimental_trtllm_environment("experimental_trtllm" in neutral_baselines)
        )
        stack.enter_context(_single_rank_runtime())
        stack.enter_context(_host_contention(args.host_load_workers))
        from benchmark.kernels.lora_moe.bench_shrink_schedules import (
            _make_cache_control,
        )

        cache_control = _make_cache_control(args.cache_state, torch.device("cuda"))
        need_lora = any(pipeline != "N0" for pipeline in pipelines) or (
            args.all_base_sgl_c0_sentinel
        )
        indexed_applied = args.a_provider == "indexed" and need_lora
        cutedsl_applied = args.a_provider == "cutedsl_static_gate" and need_lora
        a_substitution_applied = indexed_applied or cutedsl_applied
        fixture = _build_fixture(
            case,
            need_lora=need_lora,
            need_indexed_a=indexed_applied,
            c1_overlap_policy=args.c1_overlap_policy,
            neutral_baselines=neutral_baselines,
            route_pattern=args.route_pattern,
            route_seed=args.route_seed,
            zero_lora_factors=args.all_base_sgl_c0_sentinel,
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
        if a_substitution_applied and not args.skip_check:
            (
                production_c0_reference,
                production_c0_reference_status,
            ) = _capture_production_c0_reference(fixture)
        elif a_substitution_applied:
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
        cutedsl_override = None
        if cutedsl_applied:
            cutedsl_override = stack.enter_context(_CuTeStaticGateAOverride(fixture))
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
                static_a_override=cutedsl_override,
            )
        )
        if a_substitution_applied:
            correctness["production_c0_reference"] = production_c0_reference_status
        if b_schedules.applied:
            correctness["pre_b_override_reference"] = pre_b_override_reference_status
        neutral_correctness = (
            {"skipped": True}
            if args.skip_check and neutral_baselines
            else _check_neutral_baselines(
                fixture,
                neutral_baselines,
                args.pipeline,
            )
        )
        all_base_sentinel_correctness = (
            _check_all_base_sgl_c0_sentinel(fixture)
            if args.all_base_sgl_c0_sentinel and not args.skip_check
            else (
                {"skipped": True}
                if args.all_base_sgl_c0_sentinel
                else {"status": "not_requested"}
            )
        )
        torch.cuda.synchronize()

        substitutions = []
        if indexed_applied:
            substitutions.append("indexed A")
        if cutedsl_applied:
            substitutions.append("static-route CuTe gate A")
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
            "routing": fixture.route_metadata,
            "pdl_policy": args.pdl,
            "all_base_sgl_c0_sentinel": {
                "requested": args.all_base_sgl_c0_sentinel,
                "normal_pipeline_resolution_unchanged": True,
                "correctness": all_base_sentinel_correctness,
                "pipeline": None,
                "matched_latency": None,
            },
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
            "measurement_conditions": {
                "cache": cache_control.metadata(),
                "pipeline_order": args.pipeline_order,
                "resolved_pipeline_order": list(pipelines),
                "host_load_workers": args.host_load_workers,
                "host_load_scope": (
                    "synthetic_cpu_launch_contention_diagnostic"
                    if args.host_load_workers
                    else "quiet_host"
                ),
                "e0_still_required": True,
            },
            "phase_semantics": (
                "synthetic_fixed_local_token_shape; phase is metadata and is not "
                "passed to the M0 runner"
            ),
            "a_provider": {
                "name": args.a_provider,
                "applied": a_substitution_applied,
                "configs": (
                    indexed_configs.metadata()
                    if indexed_configs is not None
                    else (
                        cutedsl_override.metadata()
                        if cutedsl_override is not None
                        else None
                    )
                ),
                "substitution_scope": (
                    "gate_and_down_lora_a_only"
                    if indexed_applied
                    else "gate_up_lora_a_only_static_route_upper_bound"
                    if cutedsl_applied
                    else "none"
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
                "static_compact_route_built_outside_m0_upper_bound_plus_"
                "production_b_route_planning"
                if cutedsl_applied
                else (
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
                )
            ),
            "correctness": correctness,
            "neutral_baselines": {
                "requested": list(neutral_baselines),
                "common_input_contract": "bf16_hidden_plus_standard_topk",
                "common_output_contract": "bf16_token_domain_T_by_H",
                "canonical_weight_identity": (
                    "all providers derive from the same seeded canonical BF16 w13/w2"
                ),
                "accounting_rule": (
                    "load-time provider weight conversion is reported but excluded; "
                    "all per-forward route/alignment/topk packing remains inside M0"
                ),
                "correctness": neutral_correctness,
                "providers": {},
            },
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
                *(
                    ["opt_in_all_base_sgl_c0_graph_sentinel"]
                    if args.all_base_sgl_c0_sentinel
                    else []
                ),
                *(["benchmark_b_schedule_override"] if b_schedules.applied else []),
                *(
                    [
                        "cutedsl_gate_a_static_route_upper_bound_only",
                        "cutedsl_route_not_dynamic_across_graph_replays",
                        "cutedsl_blackwell_only",
                    ]
                    if cutedsl_applied
                    else []
                ),
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
                cache_control=cache_control,
            )
        result["matched_latency"] = _matched_latency_summary(pipeline_results)

        if args.all_base_sgl_c0_sentinel:
            sentinel_pipeline = _benchmark_pipeline(
                fixture,
                "C0",
                run_config,
                case,
                check=not args.skip_check,
                expect_active_lora=False,
                cache_control=cache_control,
            )
            sentinel_section = result["all_base_sgl_c0_sentinel"]
            assert isinstance(sentinel_section, dict)
            sentinel_section["pipeline"] = sentinel_pipeline
            sentinel_section["matched_latency"] = _matched_latency_summary(
                {"N0": pipeline_results["N0"], "C0": sentinel_pipeline}
            )

        neutral_section = result["neutral_baselines"]
        assert isinstance(neutral_section, dict)
        provider_results = neutral_section["providers"]
        assert isinstance(provider_results, dict)
        for baseline in neutral_baselines:
            baseline_pipelines = _resolve_baseline_pipelines(
                baseline, args.pipeline, case
            )
            if args.pipeline_order == "reverse":
                baseline_pipelines = tuple(reversed(baseline_pipelines))
            timed: dict[str, object] = {}
            for pipeline in baseline_pipelines:
                timed[pipeline] = _benchmark_pipeline(
                    fixture,
                    pipeline,
                    run_config,
                    case,
                    check=not args.skip_check,
                    neutral_baseline=baseline,
                    cache_control=cache_control,
                )
            requested_for_case = _resolve_pipelines(args.pipeline, case)
            unsupported = [
                pipeline
                for pipeline in requested_for_case
                if pipeline not in _NEUTRAL_BASELINE_SPECS[baseline].pipelines
            ]
            provider_results[baseline] = {
                "spec": asdict(_NEUTRAL_BASELINE_SPECS[baseline]),
                "representation": fixture.provider_representations[baseline],
                "unsupported_requested_pipelines": unsupported,
                "pipelines": timed,
                "matched_latency": _matched_latency_summary(timed),
            }

        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(
                json.dumps(result, indent=2, default=str) + "\n"
            )
        return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from sglang.srt.lora.sgl_lora.triton_ops.virtual_experts import lora_pdl_policy

    policy = {"auto": None, "on": True, "off": False}[args.pdl]
    with lora_pdl_policy(policy):
        return _main(args)


if __name__ == "__main__":
    raise SystemExit(main())
