#!/usr/bin/env python3
"""Audit shared-outer MoE-LoRA repeated work without changing production.

The current virtual-expert implementation is the ``production`` control.
FP32/PyTorch oracles and benchmark-only Triton candidates expose two algebraic
opportunities for a future specialized production path:

* gate/up A: repeated pair work versus one token/adapter result, optionally
  materialized back to today's ``[T,K,2R]`` contract;
* down B: one B GEMM per pair versus weighted rank reduction followed by one B
  GEMM per token/adapter.

The PyTorch variants are executable algebra controls, not candidate kernels.
The Triton variants are diagnostic candidates outside ``python/sglang``.
Neither may select production dispatch: they answer whether the identity is
valuable enough for an integrated implementation and make the exact
consumer-contract cost visible.

Examples::

    python benchmark/kernels/lora_moe/bench_shared_outer.py --list-cases
    python benchmark/kernels/lora_moe/bench_shared_outer.py \
      --case-id qwen35-t32-r64-l4 --site all --variant all \
      --execution cuda_graph --json-output /tmp/shared_outer_h200.json
    python benchmark/kernels/lora_moe/bench_shared_outer.py \
      --case-id qwen35-t256-r64-l3 --site down-b \
      --variant weighted-rank-reduce --execution cuda_graph

For traces, choose one site and one variant, then wrap the script with either::

    nsys profile --capture-range=cudaProfilerApi --trace=cuda,nvtx,osrt \
      python benchmark/kernels/lora_moe/bench_shared_outer.py \
        --mode nsys --site down-b --variant production
    ncu --profile-from-start off --set full \
      python benchmark/kernels/lora_moe/bench_shared_outer.py \
        --mode ncu --site gate-a --variant production
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from benchmark.kernels.lora_moe.bench_local import (
    _detect_device,
    _ensure_benchmark_server_args,
    _environment,
)
from benchmark.kernels.lora_moe.bench_shrink_schedules import (
    _CacheControl,
    _make_cache_control,
)
from benchmark.kernels.lora_moe.profiling import (
    RunConfig,
    cuda_profile_range,
    make_batch,
    time_cuda_events,
    time_isolated_cuda_wall,
)
from benchmark.kernels.lora_moe.shared_outer import (
    AdapterSpan,
    arithmetic_work,
    contiguous_adapter_spans,
    down_b_repeated_pair_reference,
    gate_up_a_deduplicated_reference,
    gate_up_a_repeated_pair_reference,
    token_lora_mapping_from_spans,
)
from benchmark.kernels.lora_moe.shared_outer_triton import (
    AUTO_DOWN_B_CONFIG,
    AUTO_GATE_A_CONFIG,
    DOWN_B_CONFIGS,
    DOWN_B_CONFIGS_BY_KEY,
    GATE_A_CONFIGS,
    GATE_A_CONFIGS_BY_KEY,
    SharedDownBConfig,
    SharedGateAConfig,
    SharedOuterTilePlan,
    build_shared_outer_tile_plan,
    invoke_shared_down_b_weighted_rank,
    invoke_shared_gate_a,
    materialize_shared_gate_pairs,
)

SITES = ("gate-a", "down-b")
GATE_VARIANTS = (
    "production",
    "repeated-pair",
    "dedup-materialized",
    "dedup-token",
    "triton-token-materialized",
    "triton-token",
)
DOWN_VARIANTS = (
    "production",
    "repeated-pair",
    "weighted-rank-reduce",
    "triton-weighted-rank-reduce",
)
ALL_VARIANTS = tuple(dict.fromkeys((*GATE_VARIANTS, *DOWN_VARIANTS)))


@dataclass(frozen=True, slots=True)
class SharedOuterBenchCase:
    case_id: str
    model: str
    tokens: int
    hidden_size: int
    num_experts: int
    top_k: int
    rank: int
    active_adapters: int
    include_base: bool
    lora_capacity: int
    phase: str

    def __post_init__(self) -> None:
        for name in (
            "tokens",
            "hidden_size",
            "num_experts",
            "top_k",
            "rank",
            "lora_capacity",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.active_adapters < 0:
            raise ValueError("active_adapters must be nonnegative")
        if self.active_adapters > self.lora_capacity:
            raise ValueError("active_adapters cannot exceed lora_capacity")
        if self.active_adapters + int(self.include_base) > self.tokens:
            raise ValueError("each adapter/base group needs at least one token")
        if self.top_k > self.num_experts:
            raise ValueError("top_k cannot exceed num_experts")

    @property
    def spans(self) -> tuple[AdapterSpan, ...]:
        return contiguous_adapter_spans(
            num_tokens=self.tokens,
            active_adapters=self.active_adapters,
            include_base=self.include_base,
        )

    @property
    def active_tokens(self) -> int:
        return sum(
            span.stop - span.start for span in self.spans if span.adapter_id is not None
        )


SHARED_OUTER_CASES: tuple[SharedOuterBenchCase, ...] = (
    SharedOuterBenchCase(
        "smoke-t4-r16-l1",
        "synthetic",
        4,
        64,
        8,
        2,
        16,
        1,
        True,
        2,
        "decode",
    ),
    SharedOuterBenchCase(
        "qwen35-t1-r32-l1",
        "qwen3.5-35b-a3b",
        1,
        2048,
        256,
        8,
        32,
        1,
        False,
        1,
        "decode",
    ),
    SharedOuterBenchCase(
        "qwen35-t32-r64-l1",
        "qwen3.5-35b-a3b",
        32,
        2048,
        256,
        8,
        64,
        1,
        False,
        1,
        "decode",
    ),
    SharedOuterBenchCase(
        "qwen35-t32-r64-l4",
        "qwen3.5-35b-a3b",
        32,
        2048,
        256,
        8,
        64,
        4,
        False,
        8,
        "decode",
    ),
    SharedOuterBenchCase(
        "qwen35-t32-r64-l4-base",
        "qwen3.5-35b-a3b",
        32,
        2048,
        256,
        8,
        64,
        4,
        True,
        8,
        "decode",
    ),
    SharedOuterBenchCase(
        "qwen35-t32-r128-l8",
        "qwen3.5-35b-a3b",
        32,
        2048,
        256,
        8,
        128,
        8,
        False,
        8,
        "decode",
    ),
    SharedOuterBenchCase(
        "qwen35-t256-r64-l3",
        "qwen3.5-35b-a3b",
        256,
        2048,
        256,
        8,
        64,
        3,
        False,
        8,
        "decode",
    ),
    SharedOuterBenchCase(
        "qwen35-t2048-r64-l3",
        "qwen3.5-35b-a3b",
        2048,
        2048,
        256,
        8,
        64,
        3,
        False,
        8,
        "prefill",
    ),
    SharedOuterBenchCase(
        "qwen397-t32-r64-l4",
        "qwen3.5-397b-a17b",
        32,
        4096,
        512,
        10,
        64,
        4,
        False,
        8,
        "decode",
    ),
    SharedOuterBenchCase(
        "kimi-t32-r64-l4",
        "kimi-k2.5",
        32,
        7168,
        384,
        8,
        64,
        4,
        False,
        8,
        "decode",
    ),
)
_CASES_BY_ID = {case.case_id: case for case in SHARED_OUTER_CASES}


def _select_cases(
    case_id: str | None, all_cases: bool
) -> tuple[SharedOuterBenchCase, ...]:
    if all_cases:
        return SHARED_OUTER_CASES
    resolved = case_id or SHARED_OUTER_CASES[0].case_id
    try:
        return (_CASES_BY_ID[resolved],)
    except KeyError as exc:
        raise ValueError(f"unknown shared-outer case {resolved!r}") from exc


def _random_bf16(
    shape: tuple[int, ...], *, generator: torch.Generator, scale: float
) -> torch.Tensor:
    result = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    result.uniform_(-scale, scale, generator=generator)
    return result


@dataclass(slots=True)
class SharedOuterFixture:
    case: SharedOuterBenchCase
    spans: tuple[AdapterSpan, ...]
    token_lora_mapping: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    hidden_states: torch.Tensor
    shared_gate_a: torch.Tensor
    gate_pair_output: torch.Tensor
    gate_token_output: torch.Tensor
    pair_rank: torch.Tensor
    shared_down_b: torch.Tensor
    down_base: torch.Tensor
    down_output: torch.Tensor
    dummy_gate_b: torch.Tensor
    dummy_down_a: torch.Tensor
    dummy_down_hidden: torch.Tensor
    gate_routing_cache: dict
    down_routing_cache: dict
    gate_oracle: torch.Tensor
    down_oracle: torch.Tensor

    def reset_gate_pair(self) -> None:
        self.gate_pair_output.zero_()

    def reset_gate_token(self) -> None:
        self.gate_token_output.zero_()

    def reset_gate_materialized(self) -> None:
        # A base-only span has no gate-A producer.  Clear both representations
        # before materialization so those rows stay the canonical zero delta.
        self.gate_token_output.zero_()
        self.gate_pair_output.zero_()

    def reset_down(self) -> None:
        self.down_output.copy_(self.down_base)


def _build_fixture(case: SharedOuterBenchCase) -> SharedOuterFixture:
    generator = torch.Generator(device="cuda").manual_seed(20260722)
    spans = case.spans
    mapping = token_lora_mapping_from_spans(spans, device="cuda")
    token_ids = torch.arange(case.tokens, dtype=torch.int32, device="cuda")
    slots = torch.arange(case.top_k, dtype=torch.int32, device="cuda")
    topk_ids = (token_ids[:, None] * 13 + slots[None, :] * 7) % case.num_experts
    topk_weights = torch.rand(
        (case.tokens, case.top_k),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)

    hidden = _random_bf16(
        (case.tokens, case.hidden_size), generator=generator, scale=0.1
    )
    shared_a = _random_bf16(
        (case.lora_capacity, 1, 2 * case.rank, case.hidden_size),
        generator=generator,
        scale=0.02,
    )
    pair_rank = _random_bf16(
        (case.tokens, case.top_k, case.rank), generator=generator, scale=0.1
    )
    shared_b = _random_bf16(
        (case.lora_capacity, 1, case.hidden_size, case.rank),
        generator=generator,
        scale=0.02,
    )
    gate_pair = torch.empty(
        case.tokens,
        case.top_k,
        2 * case.rank,
        dtype=torch.bfloat16,
        device="cuda",
    )
    gate_token = torch.empty(
        case.tokens, 2 * case.rank, dtype=torch.bfloat16, device="cuda"
    )
    down_base = _random_bf16(
        (case.tokens, case.hidden_size), generator=generator, scale=0.1
    )
    down_output = down_base.clone()

    fixture = SharedOuterFixture(
        case=case,
        spans=spans,
        token_lora_mapping=mapping,
        topk_ids=topk_ids.contiguous(),
        topk_weights=topk_weights,
        hidden_states=hidden,
        shared_gate_a=shared_a,
        gate_pair_output=gate_pair,
        gate_token_output=gate_token,
        pair_rank=pair_rank,
        shared_down_b=shared_b,
        down_base=down_base,
        down_output=down_output,
        # These factors are shape carriers for the unmeasured stage of the
        # production A+B entrypoint.  Their compact dimensions avoid allocating
        # model-sized factors that the selected stage never reads.
        dummy_gate_b=torch.empty(
            case.lora_capacity,
            case.num_experts,
            1,
            case.rank,
            dtype=torch.bfloat16,
            device="cuda",
        ),
        dummy_down_a=torch.empty(
            case.lora_capacity,
            case.num_experts,
            case.rank,
            1,
            dtype=torch.bfloat16,
            device="cuda",
        ),
        dummy_down_hidden=torch.empty(
            case.tokens * case.top_k, 1, dtype=torch.bfloat16, device="cuda"
        ),
        gate_routing_cache={},
        down_routing_cache={},
        gate_oracle=torch.empty(0, device="cuda"),
        down_oracle=torch.empty(0, device="cuda"),
    )
    fixture.gate_oracle = gate_up_a_repeated_pair_reference(
        hidden, shared_a, spans, top_k=case.top_k
    )
    fixture.down_oracle = down_base.float() + down_b_repeated_pair_reference(
        pair_rank, shared_b, topk_weights, spans
    )
    return fixture


def _invoke_production_gate(
    fixture: SharedOuterFixture, *, route_inclusive: bool, stage: str = "shrink"
) -> None:
    from sglang.srt.lora.sgl_lora.triton_ops.virtual_experts import (
        merged_experts_fused_moe_lora_add,
    )

    merged_experts_fused_moe_lora_add(
        output=fixture.gate_pair_output,
        hidden_states=fixture.hidden_states,
        lora_a=fixture.shared_gate_a,
        lora_b=fixture.dummy_gate_b,
        topk_ids=fixture.topk_ids,
        topk_weights=fixture.topk_weights,
        token_lora_mapping=fixture.token_lora_mapping,
        mul_routed_weight=False,
        experts_shared_outer_loras_a=True,
        experts_shared_outer_loras_b=False,
        routing_cache=None if route_inclusive else fixture.gate_routing_cache,
        fuse_add_to_output=False,
        fuse_sum_all_reduce=False,
        use_direct_expand_add=False,
        num_output_slices=2,
        local_expert_offset=0,
        local_num_experts=fixture.case.num_experts,
        stage=stage,
        intermediate_buffer=(None if stage == "routing" else fixture.gate_pair_output),
    )


def _invoke_production_down(
    fixture: SharedOuterFixture, *, route_inclusive: bool, stage: str = "expand"
) -> None:
    from sglang.srt.lora.sgl_lora.triton_ops.virtual_experts import (
        merged_experts_fused_moe_lora_add,
    )

    merged_experts_fused_moe_lora_add(
        output=fixture.down_output,
        hidden_states=fixture.dummy_down_hidden,
        lora_a=fixture.dummy_down_a,
        lora_b=fixture.shared_down_b,
        topk_ids=fixture.topk_ids,
        topk_weights=fixture.topk_weights,
        token_lora_mapping=fixture.token_lora_mapping,
        mul_routed_weight=True,
        experts_shared_outer_loras_a=False,
        experts_shared_outer_loras_b=True,
        routing_cache=None if route_inclusive else fixture.down_routing_cache,
        fuse_add_to_output=False,
        fuse_sum_all_reduce=True,
        use_direct_expand_add=False,
        num_output_slices=1,
        local_expert_offset=0,
        local_num_experts=fixture.case.num_experts,
        stage=stage,
        intermediate_buffer=(None if stage == "routing" else fixture.pair_rank),
    )


def _torch_gate_repeated_pair(fixture: SharedOuterFixture) -> None:
    for span in fixture.spans:
        if span.adapter_id is None:
            continue
        hidden = fixture.hidden_states[span.start : span.stop]
        factor_t = fixture.shared_gate_a[span.adapter_id, 0].T
        for topk_slot in range(fixture.case.top_k):
            fixture.gate_pair_output[span.start : span.stop, topk_slot].copy_(
                hidden @ factor_t
            )


def _torch_gate_dedup_token(fixture: SharedOuterFixture) -> None:
    for span in fixture.spans:
        if span.adapter_id is None:
            continue
        fixture.gate_token_output[span.start : span.stop].copy_(
            fixture.hidden_states[span.start : span.stop]
            @ fixture.shared_gate_a[span.adapter_id, 0].T
        )


def _torch_gate_dedup_materialized(fixture: SharedOuterFixture) -> None:
    _torch_gate_dedup_token(fixture)
    fixture.gate_pair_output.copy_(
        fixture.gate_token_output[:, None, :].expand(-1, fixture.case.top_k, -1)
    )


def _torch_down_repeated_pair(fixture: SharedOuterFixture) -> None:
    for span in fixture.spans:
        if span.adapter_id is None:
            continue
        factor_t = fixture.shared_down_b[span.adapter_id, 0].T
        for topk_slot in range(fixture.case.top_k):
            contribution = (
                fixture.pair_rank[span.start : span.stop, topk_slot] @ factor_t
            )
            fixture.down_output[span.start : span.stop].add_(
                contribution
                * fixture.topk_weights[span.start : span.stop, topk_slot, None].to(
                    contribution.dtype
                )
            )


def _torch_down_weighted_rank_reduce(fixture: SharedOuterFixture) -> None:
    reduced_rank = torch.sum(
        fixture.pair_rank
        * fixture.topk_weights.to(fixture.pair_rank.dtype).unsqueeze(-1),
        dim=1,
    )
    for span in fixture.spans:
        if span.adapter_id is None:
            continue
        fixture.down_output[span.start : span.stop].add_(
            reduced_rank[span.start : span.stop]
            @ fixture.shared_down_b[span.adapter_id, 0].T
        )


@dataclass(frozen=True, slots=True)
class PreparedVariant:
    site: str
    variant: str
    launch: Callable[[], None]
    reset: Callable[[], None]
    result: Callable[[], torch.Tensor]
    contract: str
    implementation_role: str
    config: SharedGateAConfig | SharedDownBConfig | None
    tile_plan: SharedOuterTilePlan | None


def _prepare_variant(
    fixture: SharedOuterFixture,
    *,
    site: str,
    variant: str,
    scope: str,
    gate_config: SharedGateAConfig,
    down_config: SharedDownBConfig,
) -> PreparedVariant:
    route_inclusive = scope == "O0"
    if site == "gate-a":
        gate_plan = None
        if variant in ("triton-token", "triton-token-materialized"):
            gate_plan = build_shared_outer_tile_plan(
                fixture.spans,
                output_width=fixture.shared_gate_a.shape[2],
                block_m=gate_config.block_m,
                block_n=gate_config.block_n,
                device=fixture.hidden_states.device,
            )
        if variant == "production":
            launch = lambda: _invoke_production_gate(
                fixture, route_inclusive=route_inclusive
            )
            role = "current_virtual_expert_control"
            contract = "pair_major_[T,K,2R]"
        elif variant == "repeated-pair":
            launch = lambda: _torch_gate_repeated_pair(fixture)
            role = "executable_algebra_control_not_candidate_kernel"
            contract = "pair_major_[T,K,2R]"
        elif variant == "dedup-materialized":
            launch = lambda: _torch_gate_dedup_materialized(fixture)
            role = "executable_algebra_control_not_candidate_kernel"
            contract = "token_compute_then_pair_materialize_[T,K,2R]"
        elif variant == "dedup-token":
            launch = lambda: _torch_gate_dedup_token(fixture)
            role = "executable_algebra_lower_bound_not_candidate_kernel"
            contract = "token_owned_[T,2R]_requires_gate_b_consumer_change"
        elif variant == "triton-token":
            launch = lambda: invoke_shared_gate_a(
                fixture.hidden_states,
                fixture.shared_gate_a,
                fixture.gate_token_output,
                gate_plan,
                config=gate_config,
            )
            role = "benchmark_only_triton_candidate"
            contract = "token_owned_[T,2R]_requires_gate_b_consumer_change"
        elif variant == "triton-token-materialized":

            def launch() -> None:
                invoke_shared_gate_a(
                    fixture.hidden_states,
                    fixture.shared_gate_a,
                    fixture.gate_token_output,
                    gate_plan,
                    config=gate_config,
                )
                materialize_shared_gate_pairs(
                    fixture.gate_token_output, fixture.gate_pair_output
                )

            role = "benchmark_only_triton_candidate"
            contract = "token_compute_then_separately_charged_pair_materialize_[T,K,2R]"
        else:
            raise ValueError(f"variant {variant!r} is invalid for gate-a")
        token_owned = variant in ("dedup-token", "triton-token")
        materialized_from_token = variant in (
            "dedup-materialized",
            "triton-token-materialized",
        )
        return PreparedVariant(
            site,
            variant,
            launch,
            (
                fixture.reset_gate_token
                if token_owned
                else (
                    fixture.reset_gate_materialized
                    if materialized_from_token
                    else fixture.reset_gate_pair
                )
            ),
            lambda: (
                fixture.gate_token_output if token_owned else fixture.gate_pair_output
            ),
            contract,
            role,
            gate_config if variant.startswith("triton-") else None,
            gate_plan,
        )

    down_plan = None
    if variant == "triton-weighted-rank-reduce":
        down_plan = build_shared_outer_tile_plan(
            fixture.spans,
            output_width=fixture.shared_down_b.shape[2],
            block_m=down_config.block_m,
            block_n=down_config.block_n,
            device=fixture.pair_rank.device,
        )
    if variant == "production":
        launch = lambda: _invoke_production_down(
            fixture, route_inclusive=route_inclusive
        )
        role = "current_virtual_expert_generic_b_control"
    elif variant == "repeated-pair":
        launch = lambda: _torch_down_repeated_pair(fixture)
        role = "executable_algebra_control_not_candidate_kernel"
    elif variant == "weighted-rank-reduce":
        launch = lambda: _torch_down_weighted_rank_reduce(fixture)
        role = "executable_algebra_control_not_candidate_kernel"
    elif variant == "triton-weighted-rank-reduce":
        launch = lambda: invoke_shared_down_b_weighted_rank(
            fixture.pair_rank,
            fixture.shared_down_b,
            fixture.topk_weights,
            fixture.down_output,
            down_plan,
            config=down_config,
        )
        role = "benchmark_only_triton_candidate"
    else:
        raise ValueError(f"variant {variant!r} is invalid for down-b")
    return PreparedVariant(
        site,
        variant,
        launch,
        fixture.reset_down,
        lambda: fixture.down_output,
        "token_output_[T,H]",
        role,
        down_config if variant.startswith("triton-") else None,
        down_plan,
    )


def _max_abs(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.float() - expected.float()).abs().max().item())


def _check_variant(
    fixture: SharedOuterFixture, prepared: PreparedVariant
) -> dict[str, float | str]:
    prepared.reset()
    prepared.launch()
    torch.cuda.synchronize()
    actual = prepared.result().float()
    if prepared.site == "gate-a":
        expected = (
            gate_up_a_deduplicated_reference(
                fixture.hidden_states,
                fixture.shared_gate_a,
                fixture.spans,
                top_k=fixture.case.top_k,
                materialize_pairs=False,
            )
            if actual.ndim == 2
            else fixture.gate_oracle
        )
    else:
        expected = fixture.down_oracle

    max_abs = _max_abs(actual, expected)
    # Production and executable controls operate in BF16.  The reference is
    # FP32 and the down identity changes reduction order, so keep a BF16-level
    # tolerance while recording the exact maximum error.
    rtol = 4e-2
    atol = 3e-2
    if not bool(torch.allclose(actual, expected, rtol=rtol, atol=atol)):
        raise AssertionError(
            f"{prepared.site}/{prepared.variant} disagrees with FP32 oracle: "
            f"max_abs={max_abs} rtol={rtol} atol={atol}"
        )
    return {
        "reference": "FP32 repeated-pair algebra oracle",
        "max_abs": max_abs,
        "rtol": rtol,
        "atol": atol,
    }


def _combined_before_sample(
    prepared: PreparedVariant, cache_control: _CacheControl
) -> Callable[[], None]:
    def before_sample() -> None:
        prepared.reset()
        cache_control.evict()

    return before_sample


def _routing_metrics(cache: dict) -> list[dict[str, int | bool]]:
    metrics = []
    for (num_experts, shared_outer, block_m), values in cache.items():
        sorted_ids, expert_ids, num_post_padded, _ = values
        metrics.append(
            {
                "num_experts_for_factor": num_experts,
                "shared_outer": shared_outer,
                "block_m": block_m,
                "allocated_pair_slots": sorted_ids.numel(),
                "allocated_expert_blocks": expert_ids.numel(),
                "post_padding_pair_slots": int(num_post_padded.item()),
            }
        )
    return metrics


def _run_variant(
    fixture: SharedOuterFixture,
    *,
    site: str,
    variant: str,
    scope: str,
    run_config: RunConfig,
    cache_control: _CacheControl,
    check: bool,
    gate_config: SharedGateAConfig,
    down_config: SharedDownBConfig,
) -> dict[str, object]:
    prepared = _prepare_variant(
        fixture,
        site=site,
        variant=variant,
        scope=scope,
        gate_config=gate_config,
        down_config=down_config,
    )

    # Seed/compile outside the measured region.  K0 retains the route cache;
    # O0 passes no cache to production and therefore rebuilds its selected
    # stage's route on every measured invocation.
    if variant == "production" and scope == "K0":
        if site == "gate-a":
            _invoke_production_gate(fixture, route_inclusive=False, stage="routing")
        else:
            _invoke_production_down(fixture, route_inclusive=False, stage="routing")
    prepared.reset()
    prepared.launch()
    torch.cuda.synchronize()

    correctness = _check_variant(fixture, prepared) if check else None
    prepared.reset()
    batch = make_batch(
        prepared.launch,
        execution=run_config.execution,
        inner_iterations=run_config.inner_iterations,
    )
    before_sample = _combined_before_sample(prepared, cache_control)
    label = (
        f"sgl_lora_shared_outer::{site}::{variant}::{fixture.case.case_id}::"
        f"config={prepared.config.key if prepared.config else 'none'}::"
        f"{scope}::{run_config.execution}"
    )

    result: dict[str, object] = {
        "site": site,
        "variant": variant,
        "scope": scope,
        "contract": prepared.contract,
        "implementation_role": prepared.implementation_role,
        "production_policy_changed": False,
        "candidate_config": asdict(prepared.config) if prepared.config else None,
        "tile_plan": (
            {
                "num_tiles": prepared.tile_plan.num_tiles,
                "block_m": prepared.tile_plan.block_m,
                "block_n": prepared.tile_plan.block_n,
                "planner_timing": "excluded_from_K0",
                "status": "benchmark_metadata_not_runtime_ABI",
            }
            if prepared.tile_plan is not None
            else None
        ),
        "correctness": correctness,
        "cache": cache_control.metadata(),
        "route_inclusion": (
            "current_production_stage_route_rebuilt"
            if variant == "production" and scope == "O0"
            else (
                "current_production_route_prebuilt"
                if variant == "production"
                else "static_adapter_spans_no_synthetic_route_sort"
            )
        ),
    }
    if variant == "production":
        result["routing"] = _routing_metrics(
            fixture.gate_routing_cache
            if site == "gate-a"
            else fixture.down_routing_cache
        )

    if run_config.mode == "time":
        timer = time_isolated_cuda_wall if scope == "O0" else time_cuda_events
        timing = timer(
            batch.run,
            launches_per_batch=batch.launches_per_batch,
            warmup=run_config.warmup,
            samples=run_config.samples,
            before_sample=before_sample,
        )
        result["timing"] = asdict(timing)
        print(
            f"{fixture.case.case_id} {site}/{variant} {scope} "
            f"config={prepared.config.key if prepared.config else 'none'} "
            f"p50={timing.p50_us:.3f} us "
            f"p20/p80={timing.p20_us:.3f}/{timing.p80_us:.3f} us"
        )
    else:
        before_sample()
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


def _case_metadata(case: SharedOuterBenchCase) -> dict[str, object]:
    metadata = asdict(case)
    metadata["active_tokens"] = case.active_tokens
    metadata["spans"] = [asdict(span) for span in case.spans]
    metadata["arithmetic_work"] = arithmetic_work(
        active_tokens=case.active_tokens,
        top_k=case.top_k,
        hidden_size=case.hidden_size,
        rank=case.rank,
    )
    metadata["factor_shapes"] = {
        "gate_up_a": [case.lora_capacity, 1, 2 * case.rank, case.hidden_size],
        "down_b": [case.lora_capacity, 1, case.hidden_size, case.rank],
    }
    return metadata


def _json_environment(args: argparse.Namespace) -> dict[str, object]:
    """Make the shared benchmark environment JSON-safe.

    ``bench_local._environment`` intentionally preserves parsed CLI objects;
    this driver has a ``Path`` output argument, so normalize just that metadata
    copy rather than weakening the shared helper's type fidelity.
    """

    environment = _environment(args)
    cli = environment.get("cli")
    if isinstance(cli, dict):
        environment["cli"] = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in cli.items()
        }
    return environment


def _site_variants(site: str, requested: str) -> tuple[str, ...]:
    allowed = GATE_VARIANTS if site == "gate-a" else DOWN_VARIANTS
    if requested == "all":
        return allowed
    if requested == "candidate-set":
        return (
            ("production", "triton-token-materialized", "triton-token")
            if site == "gate-a"
            else ("production", "triton-weighted-rank-reduce")
        )
    if requested not in allowed:
        raise ValueError(f"variant {requested!r} is invalid for {site}")
    return (requested,)


def _variant_configs(
    *,
    site: str,
    variant: str,
    gate_selector: str,
    down_selector: str,
    device: str,
) -> tuple[tuple[SharedGateAConfig, SharedDownBConfig], ...]:
    default_gate = GATE_A_CONFIGS_BY_KEY[AUTO_GATE_A_CONFIG[device]]
    default_down = DOWN_B_CONFIGS_BY_KEY[AUTO_DOWN_B_CONFIG[device]]
    if variant in ("triton-token", "triton-token-materialized"):
        gate_configs = (
            GATE_A_CONFIGS
            if gate_selector == "all"
            else (
                GATE_A_CONFIGS_BY_KEY[
                    (
                        AUTO_GATE_A_CONFIG[device]
                        if gate_selector == "auto"
                        else gate_selector
                    )
                ],
            )
        )
        return tuple((config, default_down) for config in gate_configs)
    if variant == "triton-weighted-rank-reduce":
        down_configs = (
            DOWN_B_CONFIGS
            if down_selector == "all"
            else (
                DOWN_B_CONFIGS_BY_KEY[
                    (
                        AUTO_DOWN_B_CONFIG[device]
                        if down_selector == "auto"
                        else down_selector
                    )
                ],
            )
        )
        return tuple((default_gate, config) for config in down_configs)
    return ((default_gate, default_down),)


def _resource_limit_result(
    *,
    site: str,
    variant: str,
    scope: str,
    gate_config: SharedGateAConfig,
    down_config: SharedDownBConfig,
    error: Exception,
) -> dict[str, object]:
    """Record a compile-time resource limit without dropping the matrix tail."""

    config = (
        gate_config
        if variant in ("triton-token", "triton-token-materialized")
        else (down_config if variant == "triton-weighted-rank-reduce" else None)
    )
    return {
        "site": site,
        "variant": variant,
        "scope": scope,
        "status": "unsupported_resource_limit",
        "error_type": type(error).__name__,
        "error": str(error),
        "candidate_config": asdict(config) if config is not None else None,
        "production_policy_changed": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--all-cases", action="store_true")
    parser.add_argument("--case-id", choices=tuple(_CASES_BY_ID))
    parser.add_argument("--site", choices=(*SITES, "all"), default="all")
    parser.add_argument(
        "--variant",
        choices=(*ALL_VARIANTS, "all", "candidate-set"),
        default="all",
    )
    parser.add_argument("--scope", choices=("K0", "O0"), default="K0")
    parser.add_argument("--device", choices=("auto", "h200", "gb300"), default="auto")
    parser.add_argument("--cache-state", choices=("hot", "cold"), default="hot")
    parser.add_argument("--execution", choices=("eager", "cuda_graph"), default="eager")
    parser.add_argument("--mode", choices=("time", "nsys", "ncu"), default="time")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--profile-iterations", type=int, default=1)
    parser.add_argument(
        "--gate-config",
        choices=("auto", "all", *GATE_A_CONFIGS_BY_KEY),
        default="auto",
    )
    parser.add_argument(
        "--down-config",
        choices=("auto", "all", *DOWN_B_CONFIGS_BY_KEY),
        default="auto",
    )
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args(argv)


def _list_cases() -> None:
    for case in SHARED_OUTER_CASES:
        print(
            f"{case.case_id:<29} T={case.tokens:<5} H={case.hidden_size:<5} "
            f"K={case.top_k:<2} R={case.rank:<3} "
            f"L={case.active_adapters}+{int(case.include_base)}/{case.lora_capacity} "
            f"{case.phase}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_cases:
        _list_cases()
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("shared-outer timing requires CUDA")
    device = _detect_device(args.device)
    _ensure_benchmark_server_args()
    if args.mode != "time" and (args.site == "all" or args.variant == "all"):
        raise ValueError("Nsight capture requires one explicit site and variant")
    if args.mode != "time" and (args.gate_config == "all" or args.down_config == "all"):
        raise ValueError("Nsight capture requires one explicit candidate config")
    if args.scope == "O0" and args.execution == "cuda_graph":
        raise ValueError("O0 route/allocation timing must use eager execution")

    run_config = RunConfig(
        mode=args.mode,
        execution=args.execution,
        warmup=args.warmup,
        samples=args.samples,
        inner_iterations=1,
        profile_iterations=args.profile_iterations,
    )
    cases = _select_cases(args.case_id, args.all_cases)
    sites = SITES if args.site == "all" else (args.site,)
    results = []
    for case in cases:
        fixture = _build_fixture(case)
        cache_control = _make_cache_control(args.cache_state, torch.device("cuda"))
        case_results = []
        for site in sites:
            for variant in _site_variants(site, args.variant):
                for gate_config, down_config in _variant_configs(
                    site=site,
                    variant=variant,
                    gate_selector=args.gate_config,
                    down_selector=args.down_config,
                    device=device,
                ):
                    try:
                        case_results.append(
                            _run_variant(
                                fixture,
                                site=site,
                                variant=variant,
                                scope=args.scope,
                                run_config=run_config,
                                cache_control=cache_control,
                                check=not args.skip_check,
                                gate_config=gate_config,
                                down_config=down_config,
                            )
                        )
                    except Exception as error:
                        if type(error).__name__ != "OutOfResources":
                            raise
                        unsupported = _resource_limit_result(
                            site=site,
                            variant=variant,
                            scope=args.scope,
                            gate_config=gate_config,
                            down_config=down_config,
                            error=error,
                        )
                        case_results.append(unsupported)
                        print(
                            f"{case.case_id} {site}/{variant} {args.scope} "
                            f"unsupported_resource_limit: {error}"
                        )
        results.append(
            {
                "case": _case_metadata(case),
                "device_label": device,
                "results": case_results,
            }
        )

    payload = {
        "schema": "sgl_lora_shared_outer_lab_v1",
        "environment": _json_environment(args),
        "methodology": {
            "production": "current shared-outer virtual-expert stage",
            "repeated_pair": "PyTorch executable algebra control",
            "factorized": "PyTorch executable algebra control, not a candidate kernel",
            "selection_status": "diagnostic_only; no production dispatch change",
            "timing": (
                "isolated host-to-completion wall clock"
                if args.scope == "O0"
                else "CUDA event, prebuilt routing for production"
            ),
        },
        "runs": results,
    }
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
