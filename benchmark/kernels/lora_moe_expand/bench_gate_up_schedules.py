"""A/B benchmark for the direct gate/up LoRA expand schedules.

This benchmark isolates the expand kernel: both implementations consume the
same prebuilt virtual-expert routing metadata, so alignment/sorting time is not
included.  It compares:

* ``ISO_TILE``: flat midpoint and two-slice grids both use ``BLOCK_SIZE_N=64``.
  This holds the number of output tiles constant for H=192 and isolates the
  grid/slice mapping change.
* ``POLICY``: each implementation uses its default policy.  For H=192, the
  flat midpoint kernel falls back to 64 so no tile crosses the midpoint.  The
  two-slice policy is also capped at 64 after the GB300 sweep showed that its
  128-column tile regresses ranks 32 and 64.
* ``WIDE_TILE`` (optional): flat uses its midpoint-safe default while two-slice
  is forced to 128.  This reproduces the fewer-CTA/more-masked-work tradeoff.

The production-relevant shape is gate/up output N=2H with H=192, BF16, top-k
8, ranks 16/32/64, and decode batches 1/16/32 (8/128/256 routed rows).
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import torch
import triton

from sglang.srt.lora.sgl_lora.triton_ops.expand import (
    invoke_moe_lora_expand_add_flat_for_benchmark,
    invoke_moe_lora_expand_add_sliced_for_benchmark,
)


@dataclass
class ExpandCase:
    intermediate: torch.Tensor
    weight: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    sorted_token_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_tokens_post_padded: torch.Tensor
    pair_expert_ids: torch.Tensor
    config: dict[str, int]
    num_tokens: int
    top_k: int
    rank: int
    hidden_size: int

    @property
    def num_pairs(self) -> int:
        return self.num_tokens * self.top_k

    @property
    def output_size(self) -> int:
        return 2 * self.hidden_size


@dataclass
class Quantiles:
    p20_us: float
    p50_us: float
    p80_us: float


def _git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_is_dirty() -> bool | None:
    try:
        return bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def _environment_metadata(args: argparse.Namespace) -> dict[str, object]:
    device = torch.cuda.current_device()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "git_revision": _git_revision(),
        "git_dirty": _git_is_dirty(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "gpu_capability": list(torch.cuda.get_device_capability(device)),
        "cli": vars(args),
    }


def _build_routing(
    num_tokens: int,
    top_k: int,
    num_virtual_experts: int,
    block_size_m: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build deterministic, padded routing metadata outside the timed region."""
    num_pairs = num_tokens * top_k
    token = torch.arange(num_tokens, dtype=torch.int64)
    slot = torch.arange(top_k, dtype=torch.int64)
    # Spread a batch across virtual experts while keeping each token's top-k
    # choices distinct for the default E>=top_k configuration.
    pair_expert_ids_cpu = (
        token[:, None] * 13 + slot[None, :] * 7
    ) % num_virtual_experts
    flat_experts = pair_expert_ids_cpu.flatten()

    sorted_chunks: list[torch.Tensor] = []
    block_experts: list[int] = []
    sentinel = num_pairs
    for expert_id in range(num_virtual_experts):
        pair_ids = torch.nonzero(flat_experts == expert_id, as_tuple=False).flatten()
        for begin in range(0, pair_ids.numel(), block_size_m):
            chunk = pair_ids[begin : begin + block_size_m]
            padded = torch.full((block_size_m,), sentinel, dtype=torch.int32)
            padded[: chunk.numel()] = chunk.to(torch.int32)
            sorted_chunks.append(padded)
            block_experts.append(expert_id)

    sorted_token_ids = torch.cat(sorted_chunks).to(device)
    expert_ids = torch.tensor(block_experts, dtype=torch.int32, device=device)
    num_tokens_post_padded = torch.tensor(
        [sorted_token_ids.numel()], dtype=torch.int32, device=device
    )
    return (
        pair_expert_ids_cpu.to(device=device, dtype=torch.int32),
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    )


def make_case(
    *,
    num_tokens: int,
    top_k: int,
    rank: int,
    hidden_size: int,
    num_virtual_experts: int,
    block_size_m: int,
    block_size_n: int,
    num_warps: int,
    device: torch.device,
    seed: int,
) -> ExpandCase:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    num_pairs = num_tokens * top_k
    pair_expert_ids, sorted_ids, block_experts, post_padded = _build_routing(
        num_tokens,
        top_k,
        num_virtual_experts,
        block_size_m,
        device,
    )

    intermediate = torch.randn(
        (num_pairs, 2 * rank),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    weight = torch.randn(
        (num_virtual_experts, 2 * hidden_size, rank),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    topk_weights = torch.ones((num_tokens, top_k), dtype=torch.float32, device=device)
    return ExpandCase(
        intermediate=intermediate,
        weight=weight,
        topk_weights=topk_weights,
        topk_ids=pair_expert_ids.view(num_tokens, top_k),
        sorted_token_ids=sorted_ids,
        expert_ids=block_experts,
        num_tokens_post_padded=post_padded,
        pair_expert_ids=pair_expert_ids.flatten(),
        config={
            "BLOCK_SIZE_M": block_size_m,
            "BLOCK_SIZE_N": block_size_n,
            "GROUP_SIZE_M": 1,
            "num_warps": num_warps,
        },
        num_tokens=num_tokens,
        top_k=top_k,
        rank=rank,
        hidden_size=hidden_size,
    )


def _launch(
    launcher: Callable[..., None],
    case: ExpandCase,
    output: torch.Tensor,
    force_block_size_n: int | None,
) -> None:
    launcher(
        case.intermediate,
        case.weight,
        output,
        case.topk_weights,
        case.topk_ids,
        case.sorted_token_ids,
        case.expert_ids,
        case.num_tokens_post_padded,
        case.config,
        mul_routed_weight=False,
        fuse_sum_all_reduce=False,
        force_block_size_n=force_block_size_n,
    )


def _reference(case: ExpandCase) -> torch.Tensor:
    """Independent gate-A/up-A reference for the routed-row output."""
    expected = torch.empty(
        (case.num_pairs, case.output_size),
        dtype=case.intermediate.dtype,
        device=case.intermediate.device,
    )
    for expert_id in torch.unique(case.pair_expert_ids).tolist():
        rows = torch.nonzero(
            case.pair_expert_ids == expert_id, as_tuple=False
        ).flatten()
        expected[rows, : case.hidden_size] = (
            case.intermediate[rows, : case.rank]
            @ case.weight[expert_id, : case.hidden_size].T
        )
        expected[rows, case.hidden_size :] = (
            case.intermediate[rows, case.rank :]
            @ case.weight[expert_id, case.hidden_size :].T
        )
    return expected


def check_correctness(
    case: ExpandCase,
    flat_force_bn: int | None,
    sliced_force_bn: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat_output = torch.empty(
        (case.num_pairs, case.output_size),
        dtype=case.intermediate.dtype,
        device=case.intermediate.device,
    )
    sliced_output = torch.empty_like(flat_output)
    _launch(
        invoke_moe_lora_expand_add_flat_for_benchmark,
        case,
        flat_output,
        flat_force_bn,
    )
    _launch(
        invoke_moe_lora_expand_add_sliced_for_benchmark,
        case,
        sliced_output,
        sliced_force_bn,
    )
    torch.cuda.synchronize()

    expected = _reference(case)
    torch.testing.assert_close(flat_output, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(sliced_output, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(sliced_output, flat_output, rtol=2e-2, atol=2e-2)
    return flat_output, sliced_output


def _timed_sample_us(
    batch_fn: Callable[[], None],
    start: torch.cuda.Event,
    end: torch.cuda.Event,
    launches_per_batch: int,
) -> float:
    start.record()
    batch_fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / launches_per_batch


def _capture_repeated_launches(
    fn: Callable[[], None], inner_iterations: int
) -> torch.cuda.CUDAGraph:
    """Capture only already-JIT-compiled launches into a reusable graph."""
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(inner_iterations):
            fn()
    return graph


def _make_timed_batch(
    fn: Callable[[], None],
    *,
    execution: str,
    inner_iterations: int,
) -> Callable[[], None]:
    if execution == "cuda_graph":
        graph = _capture_repeated_launches(fn, inner_iterations)
        return graph.replay

    def eager_batch() -> None:
        for _ in range(inner_iterations):
            fn()

    return eager_batch


def benchmark_pair(
    flat_fn: Callable[[], None],
    sliced_fn: Callable[[], None],
    *,
    execution: str,
    warmup: int,
    samples: int,
    inner_iterations: int,
) -> tuple[Quantiles, Quantiles]:
    flat_batch = _make_timed_batch(
        flat_fn, execution=execution, inner_iterations=inner_iterations
    )
    sliced_batch = _make_timed_batch(
        sliced_fn, execution=execution, inner_iterations=inner_iterations
    )
    for _ in range(warmup):
        flat_batch()
        sliced_batch()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    flat_samples: list[float] = []
    sliced_samples: list[float] = []
    # Alternate order to reduce clock/temperature drift bias between A and B.
    for sample in range(samples):
        if sample % 2 == 0:
            flat_samples.append(
                _timed_sample_us(flat_batch, start, end, inner_iterations)
            )
            sliced_samples.append(
                _timed_sample_us(sliced_batch, start, end, inner_iterations)
            )
        else:
            sliced_samples.append(
                _timed_sample_us(sliced_batch, start, end, inner_iterations)
            )
            flat_samples.append(
                _timed_sample_us(flat_batch, start, end, inner_iterations)
            )

    def summarize(values: list[float]) -> Quantiles:
        q = torch.tensor(values).quantile(torch.tensor([0.2, 0.5, 0.8])).tolist()
        return Quantiles(p20_us=q[0], p50_us=q[1], p80_us=q[2])

    return summarize(flat_samples), summarize(sliced_samples)


def _old_midpoint_policy_bn(output_size: int, half_size: int, config_bn: int) -> int:
    block_size_n = 128 if output_size % 128 == 0 else config_bn
    while block_size_n > 16 and half_size % block_size_n != 0:
        block_size_n //= 2
    return block_size_n


def _sliced_policy_bn(half_size: int) -> int:
    return min(64, max(16, 1 << (half_size - 1).bit_length()))


def _print_row(result: dict[str, object]) -> None:
    print(
        f"{result['comparison']:9s} "
        f"T={result['num_tokens']:>2d} pairs={result['num_pairs']:>3d} "
        f"R={result['rank']:>2d} "
        f"flat={result['flat']['p50_us']:>8.3f} us "
        f"slice={result['sliced']['p50_us']:>8.3f} us "
        f"sliced_speedup={result['sliced_speedup_p50']:>6.3f}x "
        f"sliced_delta={result['sliced_delta_pct']:>+6.2f}%"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 16, 32])
    parser.add_argument("--ranks", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--num-virtual-experts", type=int, default=64)
    parser.add_argument("--block-size-m", type=int, default=16)
    parser.add_argument("--block-size-n", type=int, default=128)
    parser.add_argument("--iso-block-size-n", type=int, default=64)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument(
        "--comparisons",
        nargs="+",
        choices=("iso_tile", "policy", "wide_tile"),
        default=["iso_tile", "policy"],
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--inner-iterations", type=int, default=20)
    parser.add_argument(
        "--execution",
        choices=("cuda_graph", "eager"),
        default="cuda_graph",
        help=(
            "CUDA graphs remove Python enqueue gaps from the default kernel-only "
            "measurement; eager keeps the diagnostic launch path"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")
    if args.num_virtual_experts < args.top_k:
        raise ValueError("num_virtual_experts must be at least top_k")
    if args.hidden_size <= 0:
        raise ValueError("hidden_size must be positive")

    device = torch.device("cuda")
    metadata = _environment_metadata(args)
    print(json.dumps(metadata, indent=2, default=str))
    results: list[dict[str, object]] = []

    for rank in args.ranks:
        for num_tokens in args.tokens:
            case = make_case(
                num_tokens=num_tokens,
                top_k=args.top_k,
                rank=rank,
                hidden_size=args.hidden_size,
                num_virtual_experts=args.num_virtual_experts,
                block_size_m=args.block_size_m,
                block_size_n=args.block_size_n,
                num_warps=args.num_warps,
                device=device,
                seed=args.seed + 1000 * rank + num_tokens,
            )
            for comparison in args.comparisons:
                if comparison == "iso_tile":
                    flat_force_bn = args.iso_block_size_n
                    sliced_force_bn = args.iso_block_size_n
                    flat_effective_bn = sliced_effective_bn = args.iso_block_size_n
                elif comparison == "policy":
                    flat_force_bn = None
                    sliced_force_bn = None
                    flat_effective_bn = _old_midpoint_policy_bn(
                        case.output_size, case.hidden_size, args.block_size_n
                    )
                    sliced_effective_bn = _sliced_policy_bn(case.hidden_size)
                else:
                    flat_force_bn = None
                    sliced_force_bn = 128
                    flat_effective_bn = _old_midpoint_policy_bn(
                        case.output_size, case.hidden_size, args.block_size_n
                    )
                    sliced_effective_bn = 128

                flat_output, sliced_output = check_correctness(
                    case, flat_force_bn, sliced_force_bn
                )
                flat_fn = lambda: _launch(
                    invoke_moe_lora_expand_add_flat_for_benchmark,
                    case,
                    flat_output,
                    flat_force_bn,
                )
                sliced_fn = lambda: _launch(
                    invoke_moe_lora_expand_add_sliced_for_benchmark,
                    case,
                    sliced_output,
                    sliced_force_bn,
                )
                flat_q, sliced_q = benchmark_pair(
                    flat_fn,
                    sliced_fn,
                    execution=args.execution,
                    warmup=args.warmup,
                    samples=args.samples,
                    inner_iterations=args.inner_iterations,
                )
                result = {
                    "comparison": comparison.upper(),
                    "num_tokens": num_tokens,
                    "top_k": args.top_k,
                    "num_pairs": case.num_pairs,
                    "rank": rank,
                    "hidden_size": args.hidden_size,
                    "output_size": case.output_size,
                    "execution": args.execution,
                    "expected_flat_block_size_n": flat_effective_bn,
                    "expected_sliced_block_size_n": sliced_effective_bn,
                    "flat": asdict(flat_q),
                    "sliced": asdict(sliced_q),
                    "sliced_speedup_p50": flat_q.p50_us / sliced_q.p50_us,
                    "sliced_delta_pct": 100.0 * (flat_q.p50_us / sliced_q.p50_us - 1.0),
                }
                results.append(result)
                _print_row(result)

    payload = {"metadata": metadata, "results": results}
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2, default=str) + "\n")
        print(f"Wrote {args.json_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
