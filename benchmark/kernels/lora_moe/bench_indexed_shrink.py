#!/usr/bin/env python3
"""Benchmark a raw-route, indexed BF16 MoE LoRA-A candidate.

The candidate consumes ``topk_ids`` and ``token_lora_mapping`` directly.  One
program computes one routed ``(token, top-k slot, N tile)`` result, so it needs
no virtual-expert ids, sorting, alignment, descriptors, or padded pair slots.
It is intentionally benchmark-only; the production grouped shrink remains the
correctness reference.

The per-expert local API is::

    hidden_states:       [T, H] (gate) or [T * K, H] (down)
    factors:             [L_capacity, E_local, N, H]
    topk_ids:            [T, K] global expert ids
    token_lora_mapping:  [T]
    output:              [T, K, N]

Negative/out-of-range adapter ids and negative/non-local expert ids preserve
the corresponding output row.  ``--scope K0`` and ``--scope O0`` execute the
same single kernel: raw route address resolution is already inside the kernel,
so O0 has no separate route-plan work to add.

Examples::

    python benchmark/kernels/lora_moe/bench_indexed_shrink.py --list-configs
    python benchmark/kernels/lora_moe/bench_indexed_shrink.py \
      --case-id p0-qwen3.5-35b-a3b-cap1-h200 --site gate \
      --config bn16-bk64-w4 --execution cuda_graph
    python benchmark/kernels/lora_moe/bench_indexed_shrink.py \
      --site down --all-configs --json-output indexed.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from itertools import product
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
from benchmark.kernels.lora_moe.bench_shrink_schedules import (
    _CacheControl,
    _make_cache_control,
    _torch_reference,
)
from benchmark.kernels.lora_moe.profiling import (
    RunConfig,
    cuda_profile_range,
    make_batch,
    time_cuda_events,
)

try:
    import triton
    import triton.language as tl
except ImportError:  # Keep --list-configs runnable in a CPU-only checkout.
    triton = None
    tl = None


@dataclass(frozen=True, slots=True)
class IndexedShrinkConfig:
    key: str
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int = 3


INDEXED_CONFIGS: tuple[IndexedShrinkConfig, ...] = tuple(
    IndexedShrinkConfig(
        key=f"bn{block_n}-bk{block_k}-w{num_warps}",
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
    )
    for block_n, block_k, num_warps in product(
        (8, 16, 32),
        (32, 64, 128),
        (2, 4, 8),
    )
)
_CONFIGS_BY_KEY = {config.key: config for config in INDEXED_CONFIGS}
DEFAULT_CONFIG = "bn16-bk64-w4"


if triton is not None:

    @triton.jit
    def _indexed_lora_a_kernel(
        hidden_states_ptr,
        factors_ptr,
        topk_ids_ptr,
        token_lora_mapping_ptr,
        output_ptr,
        lora_capacity,
        local_num_experts,
        local_expert_offset,
        stride_xm,
        stride_xh,
        stride_fl,
        stride_fe,
        stride_fn,
        stride_fh,
        stride_ot,
        stride_ok,
        stride_on,
        N: tl.constexpr,
        H: tl.constexpr,
        TOP_K: tl.constexpr,
        INPUT_IS_PAIR_MAJOR: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        ENABLE_PDL: tl.constexpr,
    ):
        # Adjacent route pairs can select unrelated (adapter, expert) weights,
        # so they cannot share a conventional 16-row tl.dot tile without first
        # regrouping or doing redundant work.  This candidate deliberately uses
        # a direct BN-by-BK vector reduction to measure that no-plan tradeoff.
        pair_id = tl.program_id(0)
        n_block_id = tl.program_id(1)
        token_id = pair_id // TOP_K
        topk_slot = pair_id % TOP_K

        if ENABLE_PDL:
            tl.extra.cuda.gdc_wait()

        adapter_id = tl.load(token_lora_mapping_ptr + token_id)
        global_expert_id = tl.load(topk_ids_ptr + pair_id)
        local_expert_id = global_expert_id - local_expert_offset
        valid_route = (
            (adapter_id >= 0)
            & (adapter_id < lora_capacity)
            & (local_expert_id >= 0)
            & (local_expert_id < local_num_experts)
        )

        # Clamp pointer-only indices as well as masking loads.  This keeps the
        # address itself in allocation bounds for every invalid route.
        safe_adapter_id = tl.minimum(tl.maximum(adapter_id, 0), lora_capacity - 1)
        safe_local_expert_id = tl.minimum(
            tl.maximum(local_expert_id, 0), local_num_experts - 1
        )

        n_offsets = n_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
        k_offsets = tl.arange(0, BLOCK_K)
        input_row = pair_id if INPUT_IS_PAIR_MAJOR else token_id
        accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for k_block_id in range(0, tl.cdiv(H, BLOCK_K)):
            current_k = k_block_id * BLOCK_K + k_offsets
            x = tl.load(
                hidden_states_ptr + input_row * stride_xm + current_k * stride_xh,
                mask=valid_route & (current_k < H),
                other=0.0,
            )
            factor = tl.load(
                factors_ptr
                + safe_adapter_id * stride_fl
                + safe_local_expert_id * stride_fe
                + n_offsets[:, None] * stride_fn
                + current_k[None, :] * stride_fh,
                mask=valid_route & (n_offsets[:, None] < N) & (current_k[None, :] < H),
                other=0.0,
            )
            accumulator += tl.sum(
                factor.to(tl.float32) * x[None, :].to(tl.float32), axis=1
            )

        if ENABLE_PDL:
            tl.extra.cuda.gdc_launch_dependents()

        tl.store(
            output_ptr
            + token_id * stride_ot
            + topk_slot * stride_ok
            + n_offsets * stride_on,
            accumulator.to(output_ptr.dtype.element_ty),
            mask=valid_route & (n_offsets < N),
        )

else:
    _indexed_lora_a_kernel = None


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def invoke_indexed_lora_a(
    hidden_states: torch.Tensor,
    factors: torch.Tensor,
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    output: torch.Tensor,
    *,
    config: IndexedShrinkConfig,
    local_expert_offset: int = 0,
) -> None:
    """Launch the benchmark-only per-expert indexed LoRA-A candidate."""
    if _indexed_lora_a_kernel is None or triton is None:
        raise RuntimeError("the indexed shrink candidate requires Triton")
    if factors.ndim != 4:
        raise ValueError("factors must have shape [L_capacity, E_local, N, H]")
    if topk_ids.ndim != 2 or token_lora_mapping.ndim != 1:
        raise ValueError("topk_ids and token_lora_mapping must have rank 2 and 1")
    tokens, top_k = topk_ids.shape
    lora_capacity, local_num_experts, n, h = factors.shape
    if token_lora_mapping.shape[0] != tokens:
        raise ValueError("token_lora_mapping length must equal topk_ids.shape[0]")
    if output.shape != (tokens, top_k, n):
        raise ValueError(f"output must have shape {(tokens, top_k, n)}")
    if hidden_states.ndim != 2 or hidden_states.shape[1] != h:
        raise ValueError(f"hidden_states must have H={h}")
    if hidden_states.shape[0] not in (tokens, tokens * top_k):
        raise ValueError("hidden_states rows must equal T (gate) or T * K (down)")
    if lora_capacity == 0 or local_num_experts == 0:
        raise ValueError("factors must have nonempty adapter and expert dimensions")

    from sglang.srt.lora.sgl_lora.triton_ops.virtual_experts import (
        _get_pdl_launch_metadata,
    )

    input_is_pair_major = hidden_states.shape[0] == tokens * top_k
    enable_pdl, pdl_kwargs = _get_pdl_launch_metadata()
    grid = (tokens * top_k, _ceil_div(n, config.block_n))
    _indexed_lora_a_kernel[grid](
        hidden_states,
        factors,
        topk_ids,
        token_lora_mapping,
        output,
        lora_capacity,
        local_num_experts,
        local_expert_offset,
        hidden_states.stride(0),
        hidden_states.stride(1),
        factors.stride(0),
        factors.stride(1),
        factors.stride(2),
        factors.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        N=n,
        H=h,
        TOP_K=top_k,
        INPUT_IS_PAIR_MAJOR=input_is_pair_major,
        BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        ENABLE_PDL=enable_pdl,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
        **pdl_kwargs,
    )


@dataclass(slots=True)
class PreparedIndexedShrink:
    fixture: SiteFixture
    config: IndexedShrinkConfig
    output: torch.Tensor
    launch: Callable[[], None]


def _prepare_candidate(
    fixture: SiteFixture,
    config: IndexedShrinkConfig,
    *,
    topk_ids: torch.Tensor | None = None,
    token_lora_mapping: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    local_expert_offset: int = 0,
) -> PreparedIndexedShrink:
    route_ids = fixture.topk_ids if topk_ids is None else topk_ids
    token_map = (
        fixture.token_lora_mapping if token_lora_mapping is None else token_lora_mapping
    )
    if output is None:
        output = torch.empty(
            (
                route_ids.shape[0],
                route_ids.shape[1],
                fixture.lora_a.shape[2],
            ),
            dtype=fixture.hidden_states.dtype,
            device=fixture.hidden_states.device,
        )

    def launch() -> None:
        invoke_indexed_lora_a(
            fixture.hidden_states,
            fixture.lora_a,
            route_ids,
            token_map,
            output,
            config=config,
            local_expert_offset=local_expert_offset,
        )

    return PreparedIndexedShrink(fixture, config, output, launch)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


def _check_preserve_semantics(
    fixture: SiteFixture,
    config: IndexedShrinkConfig,
    reference: torch.Tensor,
) -> dict[str, object]:
    """Exercise local-offset and invalid-route semantics outside timing."""
    sentinel = 5.25
    offset = 17
    shifted_ids = fixture.topk_ids + offset

    shifted_output = torch.full_like(reference, sentinel)
    shifted = _prepare_candidate(
        fixture,
        config,
        topk_ids=shifted_ids,
        output=shifted_output,
        local_expert_offset=offset,
    )
    shifted.launch()
    torch.cuda.synchronize()
    _assert_close(shifted.output, reference)

    base_map = fixture.token_lora_mapping.clone()
    base_map[0] = -1
    base_output = torch.full_like(reference, sentinel)
    base = _prepare_candidate(
        fixture,
        config,
        topk_ids=shifted_ids,
        token_lora_mapping=base_map,
        output=base_output,
        local_expert_offset=offset,
    )
    base.launch()
    torch.cuda.synchronize()
    expected_base = reference.clone()
    expected_base[0].fill_(sentinel)
    _assert_close(base.output, expected_base)

    invalid_ids = shifted_ids.clone()
    invalid_ids.view(-1)[0] = -1
    invalid_ids.view(-1)[1] = offset + fixture.lora_a.shape[1]
    invalid_expert_output = torch.full_like(reference, sentinel)
    invalid_expert_op = _prepare_candidate(
        fixture,
        config,
        topk_ids=invalid_ids,
        output=invalid_expert_output,
        local_expert_offset=offset,
    )
    invalid_expert_op.launch()
    torch.cuda.synchronize()
    expected_invalid_expert = reference.clone()
    expected_invalid_expert.view(-1, expected_invalid_expert.shape[-1])[:2].fill_(
        sentinel
    )
    _assert_close(invalid_expert_op.output, expected_invalid_expert)

    invalid_map = fixture.token_lora_mapping.clone()
    invalid_map[0] = fixture.lora_a.shape[0]
    invalid_output = torch.full_like(reference, sentinel)
    invalid = _prepare_candidate(
        fixture,
        config,
        topk_ids=shifted_ids,
        token_lora_mapping=invalid_map,
        output=invalid_output,
        local_expert_offset=offset,
    )
    invalid.launch()
    torch.cuda.synchronize()
    _assert_close(invalid.output, expected_base)

    return {
        "semantic_mode": "preserve_invalid",
        "local_expert_offset_checked": offset,
        "negative_adapter_preserved": True,
        "out_of_range_adapter_preserved": True,
        "negative_expert_preserved": True,
        "nonlocal_expert_preserved": True,
    }


def _check_candidate(
    op: PreparedIndexedShrink,
    reference: torch.Tensor,
) -> dict[str, object]:
    op.output.fill_(5.25)
    op.launch()
    torch.cuda.synchronize()
    _assert_close(op.output, reference)
    error = (op.output.float() - reference.float()).abs()
    result: dict[str, object] = {
        "max_abs_error": float(error.max().item()),
        "mean_abs_error": float(error.mean().item()),
    }
    result.update(_check_preserve_semantics(op.fixture, op.config, reference))
    return result


def _run_config(
    fixture: SiteFixture,
    reference: torch.Tensor,
    config: IndexedShrinkConfig,
    cache_control: _CacheControl,
    args: argparse.Namespace,
) -> dict[str, object]:
    op = _prepare_candidate(fixture, config)
    op.launch()
    torch.cuda.synchronize()
    correctness = None if args.skip_check else _check_candidate(op, reference)
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
    tokens, top_k = fixture.topk_ids.shape
    n, h = fixture.lora_a.shape[2:]
    result: dict[str, object] = {
        "config": asdict(config),
        "dimensions": {
            "T": tokens,
            "top_k": top_k,
            "route_pairs": tokens * top_k,
            "N": n,
            "H": h,
        },
        "grid": {
            "route_pair_programs": tokens * top_k,
            "n_tiles_per_pair": _ceil_div(n, config.block_n),
            "programs": tokens * top_k * _ceil_div(n, config.block_n),
        },
        "route_inputs": "raw_topk_ids_and_token_lora_mapping",
        "compute_primitive": "indexed_vector_reduction_no_tensor_core",
        "routing_preprocessing_kernels": 0,
        "descriptors": False,
        "pair_padding": 0,
        "semantic_mode": "preserve_invalid",
        "correctness": correctness,
    }
    if run_config.mode == "time":
        timing = time_cuda_events(
            batch.run,
            launches_per_batch=batch.launches_per_batch,
            warmup=run_config.warmup,
            samples=run_config.samples,
            before_sample=cache_control.before_sample(),
        )
        result["timing"] = asdict(timing)
        print(
            f"{config.key:<18} {args.scope}/{fixture.site} {args.execution}: "
            f"p50={timing.p50_us:.3f} us "
            f"p20/p80={timing.p20_us:.3f}/{timing.p80_us:.3f} us"
        )
    else:
        label = (
            f"sgl_lora_moe::{args.scope}::indexed_shrink::{fixture.site}::"
            f"{fixture.case.case_id}::{config.key}::{args.execution}::"
            f"route=inline_raw::cache={args.cache_state}::pdl=auto"
        )
        if cache_control.state == "cold":
            cache_control.evict()
            torch.cuda.synchronize()
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
    for config in INDEXED_CONFIGS:
        default = " (default)" if config.key == DEFAULT_CONFIG else ""
        print(
            f"{config.key:<18} BN/BK={config.block_n}/{config.block_k} "
            f"warps/stages={config.num_warps}/{config.num_stages}{default}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-configs", action="store_true")
    parser.add_argument("--all-configs", action="store_true")
    parser.add_argument("--device", choices=("auto", "h200", "gb300"), default="auto")
    parser.add_argument("--case-id")
    parser.add_argument("--site", choices=("gate", "down"), default="gate")
    parser.add_argument(
        "--config", choices=tuple(_CONFIGS_BY_KEY), default=DEFAULT_CONFIG
    )
    parser.add_argument("--scope", choices=("K0", "O0"), default="K0")
    parser.add_argument("--mode", choices=("time", "nsys", "ncu"), default="time")
    parser.add_argument("--execution", choices=("eager", "cuda_graph"), default="eager")
    parser.add_argument("--cache-state", choices=("hot", "cold"), default="hot")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--inner-iterations", type=int)
    parser.add_argument("--profile-iterations", type=int, default=1)
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_configs:
        _list_configs()
        return 0
    if args.inner_iterations is None:
        args.inner_iterations = 1 if args.cache_state == "cold" else 10
    elif args.cache_state == "cold" and args.inner_iterations != 1:
        raise ValueError("cold-cache runs require --inner-iterations 1")
    if (
        args.cache_state == "cold"
        and args.mode != "time"
        and args.profile_iterations != 1
    ):
        raise ValueError("cold-cache profiling requires --profile-iterations 1")
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    device = _detect_device(args.device)
    case = _select_case(device, args.case_id)
    if args.scope == "O0" and args.execution == "cuda_graph":
        raise ValueError("route-inclusive O0 is eager-only for baseline comparability")
    if args.scope == "O0" and args.inner_iterations != 1:
        raise ValueError("route-inclusive O0 requires --inner-iterations 1")
    fixture = _build_fixture(case, args.site)
    reference = _torch_reference(fixture).view(
        fixture.topk_ids.shape[0], fixture.topk_ids.shape[1], -1
    )
    cache_control = _make_cache_control(args.cache_state, fixture.hidden_states.device)

    configs = INDEXED_CONFIGS if args.all_configs else (_CONFIGS_BY_KEY[args.config],)
    results = []
    for config in configs:
        try:
            results.append(_run_config(fixture, reference, config, cache_control, args))
        except Exception as exc:
            from triton.runtime.errors import OutOfResources

            if not args.all_configs or not isinstance(exc, OutOfResources):
                raise
            error = {
                "config": asdict(config),
                "status": "unsupported",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            results.append(error)
            print(f"{config.key:<18} unsupported: {exc}")
    report = {
        "environment": _environment(args),
        "case": _case_summary(case),
        "site": args.site,
        "scope": args.scope,
        "reference": "chunked PyTorch FP32 route-pair oracle",
        "route_inclusion": (
            "raw_route_inputs_prebuilt"
            if args.scope == "K0"
            else "raw_route_addressing_inline_no_separate_plan"
        ),
        "scope_note": (
            "K0 and O0 launch the same candidate kernel because adapter/expert "
            "address resolution is inline and no route plan exists"
        ),
        "pdl_policy": "architecture_auto",
        "cache_control": cache_control.metadata(),
        "baseline_driver": "benchmark/kernels/lora_moe/bench_shrink_schedules.py",
        "results": results,
    }
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
