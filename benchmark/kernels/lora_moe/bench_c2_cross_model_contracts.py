#!/usr/bin/env python3
"""Cross-model semantic/provider guardrails for experimental BF16 C2.

This is a benchmark-only K0 consumer study.  It does not alter production
dispatch.  Gated SwiGLU presets exercise the existing two-slice C2 kernel;
non-gated Nemotron presets exercise the specialized one-slice ReLU2 kernel.
Both use identical provider-destination and pair/aligned route metadata.

The matrix intentionally includes R16 as a compilation guardrail, R32/R64,
T1/T32/T256, invalid and global/local expert IDs, base-only rows, non-dense
provider destinations, and Nemotron Nano/odd-I provider padding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from benchmark.kernels.lora_moe.c2_semantic_contracts import C2SemanticContract
from benchmark.kernels.lora_moe.matrix import MODEL_PRESETS
from benchmark.kernels.lora_moe.profiling import make_batch, time_cuda_events

MODEL_KEYS = (
    "qwen3.5-397b-a17b",
    "kimi-k2.5",
    "glm-5.2",
    "nemotron-3-super",
    "nemotron-3-nano",
    "odd-provider-padding",
)


@dataclass(frozen=True, slots=True)
class GuardrailCase:
    model: str
    activation: str
    logical_i: int
    physical_i: int
    experts: int
    top_k: int
    tokens: int
    rank: int

    @property
    def case_id(self) -> str:
        return (
            f"{self.model}-T{self.tokens}-R{self.rank}-"
            f"I{self.logical_i}p{self.physical_i}"
        )


@dataclass(slots=True)
class Fixture:
    case: GuardrailCase
    contract: C2SemanticContract
    value: torch.Tensor
    value_a: torch.Tensor
    value_b: torch.Tensor
    down_a: torch.Tensor
    act_out: torch.Tensor
    down_rank: torch.Tensor
    src2dst: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    mapping: torch.Tensor
    sorted_pair_ids: torch.Tensor
    virtual_expert_ids: torch.Tensor
    num_pairs_post_padded: torch.Tensor
    route_block_m: int
    local_expert_offset: int


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    items = tuple(int(item) for item in value.split(","))
    if not items or any(item <= 0 for item in items):
        raise ValueError("expected comma-separated positive integers")
    return items


def _parse_csv(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("expected a non-empty comma-separated list")
    return items


def _resolve_models(value: str) -> tuple[str, ...]:
    models = MODEL_KEYS if value == "all" else _parse_csv(value)
    unknown = set(models).difference(MODEL_KEYS)
    if unknown:
        raise ValueError(f"unknown model keys: {sorted(unknown)}")
    return models


def _model_dims(model: str) -> tuple[str, int, int, int]:
    if model == "odd-provider-padding":
        return "relu2", 1877, 1920, 17
    preset = MODEL_PRESETS[model]
    physical_i = preset.intermediate_size
    # Nano's logical I is not 128-aligned; explicitly retain a provider-padded
    # slice so physical width cannot be inferred from model metadata.
    if model == "nemotron-3-nano":
        physical_i = ((preset.intermediate_size + 127) // 128) * 128
    return (
        preset.activation,
        preset.intermediate_size,
        physical_i,
        preset.num_experts,
    )


def _cases(
    models: tuple[str, ...], tokens: tuple[int, ...], ranks: tuple[int, ...]
) -> list[GuardrailCase]:
    result = []
    for model in models:
        activation, logical_i, physical_i, experts = _model_dims(model)
        top_k = 5 if model == "odd-provider-padding" else MODEL_PRESETS[model].top_k
        # The synthetic odd-I case is one focused provider ABI guardrail. The
        # real presets carry the complete T/R cross product.
        model_tokens = (32,) if model == "odd-provider-padding" else tokens
        model_ranks = (32,) if model == "odd-provider-padding" else ranks
        for token_count in model_tokens:
            for rank in model_ranks:
                result.append(
                    GuardrailCase(
                        model=model,
                        activation=activation,
                        logical_i=logical_i,
                        physical_i=physical_i,
                        experts=experts,
                        top_k=top_k,
                        tokens=token_count,
                        rank=rank,
                    )
                )
    return result


def _random_bf16(
    shape: tuple[int, ...], generator: torch.Generator, scale: float = 0.02
) -> torch.Tensor:
    result = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    result.uniform_(-scale, scale, generator=generator)
    return result


def _build_aligned_route(
    topk_ids: torch.Tensor,
    mapping: torch.Tensor,
    *,
    experts: int,
    local_offset: int,
    block_m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_pairs = topk_ids.numel()
    top_k = topk_ids.shape[1]
    ids_cpu = topk_ids.cpu().reshape(-1).tolist()
    mapping_cpu = mapping.cpu().tolist()
    buckets: dict[int, list[int]] = {}
    for pair_idx, global_expert in enumerate(ids_cpu):
        local_expert = global_expert - local_offset
        adapter = mapping_cpu[pair_idx // top_k]
        virtual = (
            adapter * experts + local_expert
            if adapter >= 0 and 0 <= local_expert < experts
            else -1
        )
        # Model the production LoRA route: base-only/invalid pairs are absent.
        # The aligned consumer owns their activation fill independently.
        if virtual >= 0:
            buckets.setdefault(virtual, []).append(pair_idx)
    sorted_pairs: list[int] = []
    route_experts: list[int] = []
    for virtual in sorted(buckets):
        pairs = buckets[virtual]
        for start in range(0, len(pairs), block_m):
            block = pairs[start : start + block_m]
            sorted_pairs.extend(block + [num_pairs] * (block_m - len(block)))
            route_experts.append(virtual)
    return (
        torch.tensor(sorted_pairs, dtype=torch.int32, device="cuda"),
        torch.tensor(route_experts, dtype=torch.int32, device="cuda"),
        torch.tensor([len(sorted_pairs)], dtype=torch.int32, device="cuda"),
    )


def _build_fixture(case: GuardrailCase) -> Fixture:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260722)
    slices = 2 if case.activation == "swiglu" else 1
    logical_slices = ("gate", "up") if slices == 2 else ("value",)
    contract = C2SemanticContract(
        activation=case.activation,
        logical_slices=logical_slices,
        lora_target_slices=logical_slices,
        logical_intermediate_size=case.logical_i,
        physical_intermediate_size=case.physical_i,
    )
    num_pairs = case.tokens * case.top_k
    destination_rows = num_pairs + 7
    loras = 2
    local_offset = 11
    value = _random_bf16(
        (destination_rows, slices * case.physical_i), generator, scale=0.1
    )
    value_a = _random_bf16(
        (case.tokens, case.top_k, slices * case.rank), generator
    )
    value_b = _random_bf16(
        (loras, case.experts, slices * case.physical_i, case.rank), generator
    )
    down_a = _random_bf16(
        (loras, case.experts, case.rank, case.physical_i), generator
    )
    # Padding contains large values to prove it cannot leak into the explicit
    # logical-width ReLU2 kernel. Gated v1 has no padded provider ABI.
    if case.logical_i < case.physical_i:
        value[:, case.logical_i : case.physical_i] = 4.0
        value_b[:, :, case.logical_i : case.physical_i] = 4.0
        down_a[:, :, :, case.logical_i : case.physical_i] = 4.0

    token = torch.arange(case.tokens, dtype=torch.int64, device="cuda")
    slot = torch.arange(case.top_k, dtype=torch.int64, device="cuda")
    local_ids = (token[:, None] * 13 + slot[None, :] * 7) % case.experts
    topk_ids = (local_ids + local_offset).to(torch.int32)
    # Performance routes use the provider-neutral negative invalid sentinel.
    # Dedicated kernel tests additionally exercise positive global IDs outside
    # the local EP range; both pair and aligned consumers reject those IDs
    # before consuming the provider-private ``src2dst`` value.
    topk_ids[::7, -1] = -1
    topk_weights = torch.rand(
        (case.tokens, case.top_k),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)
    mapping = (token % loras).to(torch.int32)
    if case.tokens > 1:
        mapping[1::5] = -1

    permutation = torch.randperm(destination_rows, generator=generator, device="cuda")
    src2dst = permutation[:num_pairs].to(torch.int32)
    invalid = (topk_ids < local_offset) | (topk_ids >= local_offset + case.experts)
    src2dst[invalid.reshape(-1)] = destination_rows + 999
    route_block_m = 16
    route = _build_aligned_route(
        topk_ids,
        mapping,
        experts=case.experts,
        local_offset=local_offset,
        block_m=route_block_m,
    )
    return Fixture(
        case=case,
        contract=contract,
        value=value,
        value_a=value_a,
        value_b=value_b,
        down_a=down_a,
        act_out=torch.full(
            (destination_rows, case.physical_i),
            -123.0,
            dtype=torch.bfloat16,
            device="cuda",
        ),
        down_rank=torch.zeros(
            case.tokens,
            case.top_k,
            case.rank,
            dtype=torch.bfloat16,
            device="cuda",
        ),
        src2dst=src2dst,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        mapping=mapping,
        sorted_pair_ids=route[0],
        virtual_expert_ids=route[1],
        num_pairs_post_padded=route[2],
        route_block_m=route_block_m,
        local_expert_offset=local_offset,
    )


def _invoke(fixture: Fixture, schedule: str, block_n: int) -> None:
    fixture.down_rank.zero_()
    if fixture.case.activation == "swiglu":
        from sglang.srt.lora.sgl_lora.triton_ops.fused_c2 import (
            fused_gate_up_b_swiglu_down_a,
            fused_gate_up_b_swiglu_down_a_aligned,
        )

        if schedule == "pair":
            fused_gate_up_b_swiglu_down_a(
                fixture.value,
                fixture.value_a,
                fixture.value_b,
                fixture.down_a,
                fixture.act_out,
                fixture.down_rank,
                fixture.src2dst,
                fixture.topk_ids,
                fixture.mapping,
                local_expert_offset=fixture.local_expert_offset,
                block_size_n=block_n,
            )
        else:
            fused_gate_up_b_swiglu_down_a_aligned(
                fixture.value,
                fixture.value_a,
                fixture.value_b,
                fixture.down_a,
                fixture.act_out,
                fixture.down_rank,
                fixture.src2dst,
                fixture.topk_ids,
                fixture.sorted_pair_ids,
                fixture.virtual_expert_ids,
                fixture.num_pairs_post_padded,
                route_block_size_m=fixture.route_block_m,
                token_lora_mapping=fixture.mapping,
                local_expert_offset=fixture.local_expert_offset,
                block_size_n=block_n,
            )
    else:
        from sglang.srt.lora.sgl_lora.triton_ops.fused_c2_relu2 import (
            fused_value_b_relu2_down_a,
            fused_value_b_relu2_down_a_aligned,
        )

        if schedule == "pair":
            fused_value_b_relu2_down_a(
                fixture.value,
                fixture.value_a,
                fixture.value_b,
                fixture.down_a,
                fixture.act_out,
                fixture.down_rank,
                fixture.src2dst,
                fixture.topk_ids,
                fixture.mapping,
                logical_intermediate_size=fixture.case.logical_i,
                local_expert_offset=fixture.local_expert_offset,
                block_size_n=block_n,
            )
        else:
            fused_value_b_relu2_down_a_aligned(
                fixture.value,
                fixture.value_a,
                fixture.value_b,
                fixture.down_a,
                fixture.act_out,
                fixture.down_rank,
                fixture.src2dst,
                fixture.topk_ids,
                fixture.sorted_pair_ids,
                fixture.virtual_expert_ids,
                fixture.num_pairs_post_padded,
                logical_intermediate_size=fixture.case.logical_i,
                route_block_size_m=fixture.route_block_m,
                token_lora_mapping=fixture.mapping,
                local_expert_offset=fixture.local_expert_offset,
                block_size_n=block_n,
            )


def _sample_pair_indices(fixture: Fixture) -> list[int]:
    num_pairs = fixture.topk_ids.numel()
    candidates = {0, num_pairs - 1, num_pairs // 2}
    ids = fixture.topk_ids.reshape(-1)
    adapters = fixture.mapping[:, None].expand_as(fixture.topk_ids).reshape(-1)
    valid = (ids >= fixture.local_expert_offset) & (
        ids < fixture.local_expert_offset + fixture.case.experts
    )
    for mask in (valid & (adapters >= 0), valid & (adapters < 0), ~valid):
        positions = torch.nonzero(mask, as_tuple=False).reshape(-1)
        if positions.numel():
            candidates.add(int(positions[0]))
            candidates.add(int(positions[-1]))
    return sorted(candidates)


def _strict_check(fixture: Fixture, schedule: str, block_n: int) -> dict[str, object]:
    fixture.act_out.fill_(-123.0)
    _invoke(fixture, schedule, block_n)
    torch.cuda.synchronize()
    num_pairs = fixture.topk_ids.numel()
    top_k = fixture.case.top_k
    value_flat = fixture.value.view(-1, fixture.contract.provider_output_width)
    value_a_flat = fixture.value_a.view(num_pairs, -1)
    down_flat = fixture.down_rank.view(num_pairs, fixture.case.rank)
    ids_flat = fixture.topk_ids.reshape(-1)
    src_flat = fixture.src2dst.reshape(-1)
    act_errors: list[float] = []
    down_errors: list[float] = []
    act_signals: list[float] = []
    down_signals: list[float] = []

    for pair_idx in _sample_pair_indices(fixture):
        local_expert = int(ids_flat[pair_idx]) - fixture.local_expert_offset
        adapter = int(fixture.mapping[pair_idx // top_k])
        if not 0 <= local_expert < fixture.case.experts:
            assert bool((down_flat[pair_idx] == 0).all())
            continue
        destination = int(src_flat[pair_idx])
        slices: dict[str, torch.Tensor] = {}
        for name in fixture.contract.logical_slices:
            start = fixture.contract.provider_slice_offset(name)
            slices[name] = value_flat[
                destination, start : start + fixture.case.logical_i
            ].float()
        if adapter >= 0:
            for name in fixture.contract.lora_target_slices:
                a_start = fixture.contract.lora_slice_offset(name, fixture.case.rank)
                b_start = fixture.contract.lora_slice_offset(
                    name, fixture.case.physical_i
                )
                delta = (
                    fixture.value_b[
                        adapter,
                        local_expert,
                        b_start : b_start + fixture.case.logical_i,
                    ].float()
                    @ value_a_flat[
                        pair_idx, a_start : a_start + fixture.case.rank
                    ].float()
                ).to(torch.bfloat16)
                slices[name] += delta.float()
        if fixture.case.activation == "swiglu":
            expected_act = (F.silu(slices["gate"]) * slices["up"]).to(
                torch.bfloat16
            )
        else:
            expected_act = torch.relu(slices["value"]).square().to(torch.bfloat16)
        actual_act = fixture.act_out[destination, : fixture.case.logical_i]
        act_errors.append(float((actual_act.float() - expected_act.float()).abs().max()))
        act_signals.append(float(expected_act.float().abs().max()))
        if fixture.case.logical_i < fixture.case.physical_i:
            assert bool(
                (
                    fixture.act_out[destination, fixture.case.logical_i :] == 0
                ).all()
            )
        if adapter < 0:
            assert bool((down_flat[pair_idx] == 0).all())
            continue
        expected_down = torch.zeros(
            fixture.case.rank, dtype=torch.bfloat16, device="cuda"
        )
        for start in range(0, fixture.case.logical_i, block_n):
            stop = min(start + block_n, fixture.case.logical_i)
            partial = (
                fixture.down_a[
                    adapter, local_expert, :, start:stop
                ].float()
                @ expected_act[start:stop].float()
            ).to(torch.bfloat16)
            expected_down = (
                expected_down.float() + partial.float()
            ).to(torch.bfloat16)
        actual_down = down_flat[pair_idx]
        down_errors.append(
            float((actual_down.float() - expected_down.float()).abs().max())
        )
        down_signals.append(float(expected_down.float().abs().max()))

    act_error = max(act_errors, default=0.0)
    down_error = max(down_errors, default=0.0)
    act_signal = max(act_signals, default=0.0)
    down_signal = max(down_signals, default=0.0)
    if act_error > max(2e-4, 0.03 * act_signal):
        raise AssertionError(f"activation error {act_error} over signal {act_signal}")
    if down_error > max(5e-4, 0.05 * down_signal):
        raise AssertionError(f"down-A error {down_error} over signal {down_signal}")

    # Validate the explicit downstream scaling/dtype contract independently of
    # the consumer: one fixed sample must scale linearly and preserve FP32.
    selected = next(
        (
            pair
            for pair in _sample_pair_indices(fixture)
            if int(fixture.mapping[pair // top_k]) >= 0
            and 0
            <= int(ids_flat[pair]) - fixture.local_expert_offset
            < fixture.case.experts
        ),
        None,
    )
    scaling_check = None
    if selected is not None:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(1701)
        down_b = _random_bf16((37, fixture.case.rank), generator)
        unscaled = down_b.float() @ down_flat[selected].float()
        routed = (
            unscaled
            * fixture.topk_weights.reshape(-1)[selected]
            * 1.75
        ).to(torch.float32)
        expected = (
            unscaled * fixture.topk_weights.reshape(-1)[selected]
        ).to(torch.float32) * 1.75
        scale_error = float((routed - expected).abs().max())
        if scale_error > 2e-6:
            raise AssertionError(f"routed scaling error {scale_error}")
        scaling_check = {
            "factor": 1.75,
            "destination_dtype": str(routed.dtype),
            "max_abs_error": scale_error,
            "applied_stage": "down_b_finalize_not_fused_consumer",
        }
    return {
        "sampled_pair_indices": _sample_pair_indices(fixture),
        "activation_max_abs_error": act_error,
        "activation_signal_max_abs": act_signal,
        "activation_error_over_signal": act_error / act_signal if act_signal else None,
        "down_rank_max_abs_error": down_error,
        "down_rank_signal_max_abs": down_signal,
        "down_rank_error_over_signal": down_error / down_signal if down_signal else None,
        "routed_scaling": scaling_check,
        "invalid_pairs": int(
            (
                (fixture.topk_ids < fixture.local_expert_offset)
                | (
                    fixture.topk_ids
                    >= fixture.local_expert_offset + fixture.case.experts
                )
            ).sum()
        ),
        "base_only_tokens": int((fixture.mapping < 0).sum()),
    }


def _benchmark(
    fixture: Fixture,
    schedule: str,
    block_n: int,
    *,
    execution: str,
    warmup: int,
    samples: int,
) -> dict[str, object]:
    invoke: Callable[[], None] = lambda: _invoke(fixture, schedule, block_n)
    invoke()
    torch.cuda.synchronize()
    eager_act = fixture.act_out.clone()
    eager_down = fixture.down_rank.clone()
    batch = make_batch(invoke, execution=execution, inner_iterations=1)
    graph_error = None
    if execution == "cuda_graph":
        batch.run()
        torch.cuda.synchronize()
        graph_error = {
            "activation_max_abs": float(
                (fixture.act_out.float() - eager_act.float()).abs().max()
            ),
            "down_rank_max_abs": float(
                (fixture.down_rank.float() - eager_down.float()).abs().max()
            ),
        }
        if graph_error["activation_max_abs"] > 0.0:
            raise AssertionError("activation differs under CUDA graph replay")
        if graph_error["down_rank_max_abs"] > 5e-4:
            raise AssertionError("down rank differs under CUDA graph replay")
    timing = time_cuda_events(
        batch.run,
        launches_per_batch=1,
        warmup=warmup,
        samples=samples,
    )
    print(
        f"{fixture.case.case_id} {schedule}/BN{block_n}: "
        f"p50={timing.p50_us:.3f} us "
        f"p20/p80={timing.p20_us:.3f}/{timing.p80_us:.3f} us",
        flush=True,
    )
    return {
        "schedule": schedule,
        "block_n": block_n,
        "launch_topology": (
            "down_rank_zero_then_pair_consumer"
            if schedule == "pair"
            else "down_rank_zero_then_base_only_activation_then_aligned_consumer"
        ),
        "timing": asdict(timing),
        "graph_correctness": graph_error,
    }


def _environment(args: argparse.Namespace) -> dict[str, object]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "gpu_capability": list(torch.cuda.get_device_capability()),
        "git_revision": os.getenv("SGL_LORA_BENCH_REVISION", "working-tree"),
        "cli": vars(args),
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="all")
    parser.add_argument("--tokens", default="1,32,256")
    parser.add_argument("--ranks", default="16,32,64")
    parser.add_argument("--schedules", default="pair,aligned")
    parser.add_argument("--block-n", default="16,32,64")
    parser.add_argument(
        "--execution", choices=("eager", "cuda_graph"), default="cuda_graph"
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--json-output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index must be in [0, num_shards)")
    models = _resolve_models(args.models)
    tokens = _parse_csv_ints(args.tokens)
    ranks = _parse_csv_ints(args.ranks)
    schedules = _parse_csv(args.schedules)
    block_ns = _parse_csv_ints(args.block_n)
    if set(schedules).difference(("pair", "aligned")):
        raise ValueError("schedules must be pair and/or aligned")
    selected_cases = [
        case
        for index, case in enumerate(_cases(models, tokens, ranks))
        if index % args.num_shards == args.shard_index
    ]
    records = []
    for case in selected_cases:
        fixture = _build_fixture(case)
        runs = []
        checks = {}
        for block_n in block_ns:
            for schedule in schedules:
                key = f"{schedule}-bn{block_n}"
                if not args.skip_check:
                    checks[key] = _strict_check(fixture, schedule, block_n)
                runs.append(
                    _benchmark(
                        fixture,
                        schedule,
                        block_n,
                        execution=args.execution,
                        warmup=args.warmup,
                        samples=args.samples,
                    )
                )
        fastest = min(runs, key=lambda run: run["timing"]["p50_us"])
        route_bytes = fixture.topk_ids.cpu().numpy().tobytes()
        records.append(
            {
                "case": asdict(case),
                "semantic_contract": {
                    "activation": fixture.contract.activation,
                    "logical_slices": fixture.contract.logical_slices,
                    "lora_target_slices": fixture.contract.lora_target_slices,
                    "logical_intermediate_size": case.logical_i,
                    "physical_intermediate_size": case.physical_i,
                    "provider_slice_order": fixture.contract.resolved_provider_slice_order,
                    "provider_row_domain": fixture.contract.provider_row_domain,
                    "provider_padding_policy": fixture.contract.provider_padding_policy,
                    "kernel_family": fixture.contract.benchmark_kernel_family(),
                },
                "route": {
                    "pattern": "deterministic_lattice_with_invalid_guardrail",
                    "local_expert_offset": fixture.local_expert_offset,
                    "hash_sha256_int32_row_major": hashlib.sha256(route_bytes).hexdigest(),
                    "route_block_m": fixture.route_block_m,
                    "pairs": fixture.topk_ids.numel(),
                    "pairs_post_padded": int(fixture.num_pairs_post_padded.item()),
                },
                "checks": checks,
                "runs": runs,
                "selected_schedule": {
                    "schedule": fastest["schedule"],
                    "block_n": fastest["block_n"],
                    "p50_us": fastest["timing"]["p50_us"],
                    "selection_scope": "device_and_resolved_shape_guardrail_only",
                },
            }
        )
        del fixture
        torch.cuda.empty_cache()

    payload = {
        "schema_version": 1,
        "environment": _environment(args),
        "scope": {
            "kind": "benchmark_only_C2_K0_cross_model_guardrail",
            "production_dispatch_changed": False,
            "required_dimensions": {
                "models": list(models),
                "tokens": list(tokens),
                "ranks": list(ranks),
                "schedules": list(schedules),
                "block_n": list(block_ns),
            },
            "abi_conclusion": (
                "provider_semantics_keyed_not_model_name_keyed: standard contiguous "
                "two-slice SwiGLU shares gated_swiglu_v1; one-slice ReLU2 and "
                "explicit logical/physical padding share nongated_relu2_v1; "
                "partial or reordered slices require another provider specialization"
            ),
            "base_only_contract": (
                "aligned LoRA routes may omit base-only rows; measured aligned "
                "timings include a separate masked base-only activation fill"
            ),
        },
        "records": records,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"wrote {args.json_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
