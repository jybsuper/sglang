"""Prototype a general sliced routed LoRA-B expand kernel.

This file is deliberately benchmark-only.  It tests whether one Triton kernel
can cover the common stacked-projection shape:

    C[:, n_begin:n_end] = A[:, a_begin:a_begin + R] @ B[:, n_begin:n_end, :].T

One ``@triton.jit`` body has three compile-time schedule modes:

* ``ALIGNED_FLAT`` traverses a contiguous equal-slice output when every slice
  boundary is aligned to ``BLOCK_SIZE_N``. Slice and A offsets are arithmetic.
* ``UNIFORM_RAGGED`` restarts the N grid for any number of equal-width slices,
  masks each slice tail independently, and also uses arithmetic offsets.
* ``GENERAL_RAGGED`` consumes runtime output/A/tile prefixes for unequal
  Q/K/V-style slices. Its ragged grid launches no empty widest-slice CTAs.

The first two modes test the intended one-source/multiple-compiled-schedules
design: their unused metadata branches and pointer loads are compile-time dead.
The third mode measures the flexibility tax instead of imposing it on hot
equal-slice shapes.

At identical ``BLOCK_SIZE_N``, the benchmark compares compiled ``ALIGNED_FLAT``
with the production flat kernel and compiled ``UNIFORM_RAGGED`` with the
production two-slice kernel. For the 48-column-per-slice case it sweeps
16/32/64/128; the flat schedule is valid only for 16, while ragged schedules
can mask each slice tail.

This file is a testbed, not production dispatch policy.
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Callable, Sequence

import torch
import triton
import triton.language as tl

from sglang.srt.lora.sgl_lora.triton_ops.expand import (
    _invoke_flat,
    _invoke_two_slice,
)

_invoke_flat_for_benchmark = partial(
    _invoke_flat,
    mul_routed_weight=False,
    fuse_sum_all_reduce=False,
    gated_midpoint=True,
)
_invoke_sliced_for_benchmark = partial(
    _invoke_two_slice,
    mul_routed_weight=False,
    fuse_sum_all_reduce=False,
)

ALIGNED_FLAT = 0
UNIFORM_RAGGED = 1
GENERAL_RAGGED = 2


@triton.jit
def _moe_lora_expand_add_scheduled_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    output_prefix_ptr,
    a_offset_ptr,
    tile_prefix_ptr,
    R: tl.constexpr,
    num_valid_tokens,
    total_slice_tiles,
    stride_am,
    stride_ar,
    stride_be,
    stride_bn,
    stride_br,
    stride_cm,
    stride_cn,
    router_topk: tl.constexpr,
    SCHEDULE: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    UNIFORM_SLICE_N: tl.constexpr,
    TOTAL_N: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    FUSE_SUM_ALL_REDUCE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
):
    """One kernel body compiled into flat, uniform, or general schedules."""
    pid = tl.program_id(0)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)

    if SCHEDULE == 0:  # ALIGNED_FLAT
        num_pid_n = tl.cdiv(TOTAL_N, BLOCK_SIZE_N)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
        output_begin = 0
        output_end = TOTAL_N
        local_tile_id = pid_n
        slice_id = (pid_n * BLOCK_SIZE_N) // UNIFORM_SLICE_N
        a_begin = slice_id * R
    elif SCHEDULE == 1:  # UNIFORM_RAGGED
        num_pid_n = tl.cdiv(UNIFORM_SLICE_N, BLOCK_SIZE_N)
        pid_m = pid // num_pid_n
        local_tile_id = pid % num_pid_n
        slice_id = tl.program_id(1)
        output_begin = slice_id * UNIFORM_SLICE_N
        output_end = output_begin + UNIFORM_SLICE_N
        a_begin = slice_id * R
    else:
        pid_m = pid // total_slice_tiles
        global_tile_id = pid % total_slice_tiles

        # NUM_SLICES is constexpr, so this small runtime-prefix search is
        # unrolled. Only GENERAL_RAGGED retains these metadata loads.
        slice_id = global_tile_id * 0
        for slice_idx in tl.static_range(1, NUM_SLICES):
            slice_tile_begin = tl.load(tile_prefix_ptr + slice_idx)
            slice_id += (global_tile_id >= slice_tile_begin).to(tl.int32)

        tile_begin = tl.load(tile_prefix_ptr + slice_id).to(tl.int64)
        output_begin = tl.load(output_prefix_ptr + slice_id).to(tl.int64)
        output_end = tl.load(output_prefix_ptr + slice_id + 1).to(tl.int64)
        a_begin = tl.load(a_offset_ptr + slice_id).to(tl.int64)
        local_tile_id = global_tile_id - tile_begin

    if pid_m >= num_pid_m:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    offs_n = (
        output_begin
        + local_tile_id * BLOCK_SIZE_N
        + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    )
    output_mask = offs_n < output_end

    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_expert == -1:
        if not FUSE_SUM_ALL_REDUCE:
            c_ptrs = (
                c_ptr + offs_token[:, None] * stride_cm + offs_n[None, :] * stride_cn
            )
            c_mask = token_mask[:, None] & output_mask[None, :]
            zeros = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=c_ptr.dtype.element_ty)
            tl.store(c_ptrs, zeros, mask=c_mask)
        return

    offs_r = tl.arange(0, BLOCK_SIZE_R).to(tl.int64)
    rank_mask = offs_r < R
    a = tl.load(
        a_ptr
        + offs_token[:, None] * stride_am
        + (a_begin + offs_r)[None, :] * stride_ar,
        mask=token_mask[:, None] & rank_mask[None, :],
        other=0.0,
    )
    b = tl.load(
        b_ptr
        + off_expert * stride_be
        + offs_n[None, :] * stride_bn
        + offs_r[:, None] * stride_br,
        mask=output_mask[None, :] & rank_mask[:, None],
        other=0.0,
    )

    accumulator = tl.dot(a, b, out_dtype=tl.float32)
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator *= moe_weight[:, None]

    if FUSE_SUM_ALL_REDUCE:
        offs_token_out = offs_token // router_topk
    else:
        offs_token_out = offs_token
    c_ptrs = c_ptr + offs_token_out[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = token_mask[:, None] & output_mask[None, :]
    if FUSE_SUM_ALL_REDUCE:
        tl.atomic_add(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=c_mask)
    else:
        tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=c_mask)


@dataclass
class Quantiles:
    p20_us: float
    p50_us: float
    p80_us: float


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
    output_prefix: torch.Tensor
    a_offsets: torch.Tensor
    slice_widths: tuple[int, ...]
    num_tokens: int
    top_k: int
    rank: int
    block_size_m: int
    num_warps: int
    tile_prefixes: dict[int, torch.Tensor] = field(default_factory=dict)

    @property
    def num_pairs(self) -> int:
        return self.num_tokens * self.top_k

    @property
    def output_size(self) -> int:
        return sum(self.slice_widths)

    @property
    def num_slices(self) -> int:
        return len(self.slice_widths)

    @property
    def config(self) -> dict[str, int]:
        return {
            "BLOCK_SIZE_M": self.block_size_m,
            "BLOCK_SIZE_N": 128,
            "GROUP_SIZE_M": 1,
            "num_warps": self.num_warps,
        }

    def tile_prefix(self, block_size_n: int) -> torch.Tensor:
        cached = self.tile_prefixes.get(block_size_n)
        if cached is not None:
            return cached
        prefix = [0]
        for width in self.slice_widths:
            prefix.append(prefix[-1] + triton.cdiv(width, block_size_n))
        cached = torch.tensor(
            prefix,
            dtype=torch.int32,
            device=self.intermediate.device,
        )
        self.tile_prefixes[block_size_n] = cached
        return cached


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


def _build_routing(
    num_tokens: int,
    top_k: int,
    num_virtual_experts: int,
    block_size_m: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_pairs = num_tokens * top_k
    token = torch.arange(num_tokens, dtype=torch.int64)
    slot = torch.arange(top_k, dtype=torch.int64)
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
    slice_widths: tuple[int, ...],
    num_tokens: int,
    top_k: int,
    rank: int,
    num_virtual_experts: int,
    block_size_m: int,
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
    num_slices = len(slice_widths)
    intermediate = torch.randn(
        (num_pairs, num_slices * rank),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    weight = torch.randn(
        (num_virtual_experts, sum(slice_widths), rank),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    output_prefix_cpu = [0]
    for width in slice_widths:
        output_prefix_cpu.append(output_prefix_cpu[-1] + width)
    return ExpandCase(
        intermediate=intermediate,
        weight=weight,
        topk_weights=torch.ones(
            (num_tokens, top_k), dtype=torch.float32, device=device
        ),
        topk_ids=pair_expert_ids.view(num_tokens, top_k),
        sorted_token_ids=sorted_ids,
        expert_ids=block_experts,
        num_tokens_post_padded=post_padded,
        pair_expert_ids=pair_expert_ids.flatten(),
        output_prefix=torch.tensor(output_prefix_cpu, dtype=torch.int32, device=device),
        a_offsets=torch.arange(num_slices, dtype=torch.int32, device=device) * rank,
        slice_widths=slice_widths,
        num_tokens=num_tokens,
        top_k=top_k,
        rank=rank,
        block_size_m=block_size_m,
        num_warps=num_warps,
    )


def _launch_scheduled(
    case: ExpandCase,
    output: torch.Tensor,
    block_size_n: int,
    schedule: int,
) -> None:
    tile_prefix = case.tile_prefix(block_size_n)
    total_slice_tiles = sum(
        triton.cdiv(width, block_size_n) for width in case.slice_widths
    )
    num_pid_m = triton.cdiv(case.sorted_token_ids.shape[0], case.block_size_m)
    equal_slices = len(set(case.slice_widths)) == 1
    uniform_slice_n = case.slice_widths[0] if equal_slices else 1
    if schedule == ALIGNED_FLAT:
        if not equal_slices or uniform_slice_n % block_size_n != 0:
            raise ValueError("ALIGNED_FLAT requires equal, BLOCK_SIZE_N-aligned slices")
        grid = (num_pid_m * triton.cdiv(case.output_size, block_size_n),)
    elif schedule == UNIFORM_RAGGED:
        if not equal_slices:
            raise ValueError("UNIFORM_RAGGED requires equal-width slices")
        grid = (
            num_pid_m * triton.cdiv(uniform_slice_n, block_size_n),
            case.num_slices,
        )
    else:
        grid = (num_pid_m * total_slice_tiles,)

    _moe_lora_expand_add_scheduled_kernel[grid](
        case.intermediate,
        case.weight,
        output,
        case.topk_weights,
        case.sorted_token_ids,
        case.expert_ids,
        case.num_tokens_post_padded,
        case.output_prefix,
        case.a_offsets,
        tile_prefix,
        case.rank,
        case.topk_ids.numel(),
        total_slice_tiles,
        case.intermediate.stride(0),
        case.intermediate.stride(1),
        case.weight.stride(0),
        case.weight.stride(1),
        case.weight.stride(2),
        output.stride(0),
        output.stride(1),
        router_topk=case.top_k,
        SCHEDULE=schedule,
        NUM_SLICES=case.num_slices,
        UNIFORM_SLICE_N=uniform_slice_n,
        TOTAL_N=case.output_size,
        MUL_ROUTED_WEIGHT=False,
        FUSE_SUM_ALL_REDUCE=False,
        BLOCK_SIZE_M=case.block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_R=triton.next_power_of_2(case.rank),
        num_warps=case.num_warps,
        num_stages=1,
    )


def _launch_general(case: ExpandCase, output: torch.Tensor, block_size_n: int) -> None:
    _launch_scheduled(case, output, block_size_n, GENERAL_RAGGED)


def _launch_uniform(case: ExpandCase, output: torch.Tensor, block_size_n: int) -> None:
    _launch_scheduled(case, output, block_size_n, UNIFORM_RAGGED)


def _launch_aligned(case: ExpandCase, output: torch.Tensor, block_size_n: int) -> None:
    _launch_scheduled(case, output, block_size_n, ALIGNED_FLAT)


def _launch_specialized(
    case: ExpandCase, output: torch.Tensor, block_size_n: int
) -> None:
    _invoke_sliced_for_benchmark(
        case.intermediate,
        case.weight,
        output,
        case.topk_weights,
        case.topk_ids,
        case.sorted_token_ids,
        case.expert_ids,
        case.num_tokens_post_padded,
        case.config,
        force_block_size_n=block_size_n,
    )


def _launch_flat(case: ExpandCase, output: torch.Tensor, block_size_n: int) -> None:
    _invoke_flat_for_benchmark(
        case.intermediate,
        case.weight,
        output,
        case.topk_weights,
        case.topk_ids,
        case.sorted_token_ids,
        case.expert_ids,
        case.num_tokens_post_padded,
        case.config,
        force_block_size_n=block_size_n,
    )


def _launch_single(case: ExpandCase, output: torch.Tensor, block_size_n: int) -> None:
    """Launch the production one-slice schedule as the generic S=1 baseline."""
    _invoke_flat(
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
        gated_midpoint=False,
        force_block_size_n=block_size_n,
    )


def _reference(case: ExpandCase) -> torch.Tensor:
    expected = torch.empty(
        (case.num_pairs, case.output_size),
        dtype=case.intermediate.dtype,
        device=case.intermediate.device,
    )
    output_begin = 0
    for slice_id, width in enumerate(case.slice_widths):
        output_end = output_begin + width
        a_begin = slice_id * case.rank
        for expert_id in torch.unique(case.pair_expert_ids).tolist():
            rows = torch.nonzero(
                case.pair_expert_ids == expert_id, as_tuple=False
            ).flatten()
            expected[rows, output_begin:output_end] = (
                case.intermediate[rows, a_begin : a_begin + case.rank]
                @ case.weight[expert_id, output_begin:output_end].T
            )
        output_begin = output_end
    return expected


def _capture_repeated_launches(
    fn: Callable[[], None], inner_iterations: int
) -> torch.cuda.CUDAGraph:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(inner_iterations):
            fn()
    return graph


def _make_timed_batch(
    fn: Callable[[], None], execution: str, inner_iterations: int
) -> Callable[[], None]:
    if execution == "cuda_graph":
        return _capture_repeated_launches(fn, inner_iterations).replay

    def eager_batch() -> None:
        for _ in range(inner_iterations):
            fn()

    return eager_batch


def benchmark_variants(
    variants: dict[str, Callable[[], None]],
    *,
    execution: str,
    warmup: int,
    samples: int,
    inner_iterations: int,
) -> dict[str, Quantiles]:
    batches = {
        name: _make_timed_batch(fn, execution, inner_iterations)
        for name, fn in variants.items()
    }
    for _ in range(warmup):
        for batch in batches.values():
            batch()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    timings = {name: [] for name in batches}
    names = list(batches)
    for sample in range(samples):
        # Rotate order to limit clock/temperature bias across variants.
        ordered_names = names[sample % len(names) :] + names[: sample % len(names)]
        for name in ordered_names:
            start.record()
            batches[name]()
            end.record()
            end.synchronize()
            timings[name].append(start.elapsed_time(end) * 1000.0 / inner_iterations)

    result: dict[str, Quantiles] = {}
    for name, values in timings.items():
        quantiles = torch.tensor(values).quantile(torch.tensor([0.2, 0.5, 0.8]))
        result[name] = Quantiles(*quantiles.tolist())
    return result


def _case_widths(args: argparse.Namespace) -> dict[str, tuple[int, ...]]:
    available = {
        "single": (args.equal_width,),
        "equal2": (args.equal_width, args.equal_width),
        "equal3": (args.equal_width, args.equal_width, args.equal_width),
        "unequal3": tuple(args.unequal_widths),
        "h48": (48, 48),
    }
    return {name: available[name] for name in args.cases}


def _environment_metadata(args: argparse.Namespace) -> dict[str, object]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "git_revision": _git_revision(),
        "git_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=False,
                capture_output=True,
                text=True,
            ).stdout
        ),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "gpu_capability": list(torch.cuda.get_device_capability()),
        "cli": vars(args),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("single", "equal2", "equal3", "unequal3", "h48"),
        default=["single", "equal2", "equal3", "unequal3", "h48"],
    )
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 16, 32])
    parser.add_argument("--ranks", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--block-sizes", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--equal-width", type=int, default=192)
    parser.add_argument("--unequal-widths", nargs="+", type=int, default=[192, 64, 64])
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--num-virtual-experts", type=int, default=64)
    parser.add_argument("--block-size-m", type=int, default=16)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--inner-iterations", type=int, default=20)
    parser.add_argument(
        "--execution", choices=("cuda_graph", "eager"), default="cuda_graph"
    )
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")
    if args.num_virtual_experts < args.top_k:
        raise ValueError("num_virtual_experts must be at least top_k")
    if any(width <= 0 for width in [args.equal_width, *args.unequal_widths]):
        raise ValueError("slice widths must be positive")
    if any(block_n <= 0 or block_n & (block_n - 1) for block_n in args.block_sizes):
        raise ValueError("block sizes must be positive powers of two")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_args(args)
    device = torch.device("cuda")
    print(json.dumps(_environment_metadata(args), indent=2, default=str))
    results: list[dict[str, object]] = []

    for case_id, slice_widths in _case_widths(args).items():
        for rank in args.ranks:
            for num_tokens in args.tokens:
                case = make_case(
                    slice_widths=slice_widths,
                    num_tokens=num_tokens,
                    top_k=args.top_k,
                    rank=rank,
                    num_virtual_experts=args.num_virtual_experts,
                    block_size_m=args.block_size_m,
                    num_warps=args.num_warps,
                    device=device,
                    seed=args.seed + 1000 * rank + num_tokens + len(slice_widths),
                )
                expected = _reference(case)
                for block_size_n in args.block_sizes:
                    outputs: dict[str, torch.Tensor] = {
                        "general_ragged": torch.empty_like(expected)
                    }
                    variants: dict[str, Callable[[], None]] = {
                        "general_ragged": lambda output=outputs[
                            "general_ragged"
                        ]: _launch_general(case, output, block_size_n)
                    }
                    equal_slices = len(set(case.slice_widths)) == 1
                    if equal_slices:
                        outputs["uniform_ragged"] = torch.empty_like(expected)
                        variants["uniform_ragged"] = lambda output=outputs[
                            "uniform_ragged"
                        ]: _launch_uniform(case, output, block_size_n)
                        if case.slice_widths[0] % block_size_n == 0:
                            outputs["aligned_flat"] = torch.empty_like(expected)
                            variants["aligned_flat"] = lambda output=outputs[
                                "aligned_flat"
                            ]: _launch_aligned(case, output, block_size_n)

                    if case.num_slices == 1:
                        outputs["production_single_flat"] = torch.empty_like(expected)
                        variants["production_single_flat"] = lambda output=outputs[
                            "production_single_flat"
                        ]: _launch_single(case, output, block_size_n)
                    elif case.num_slices == 2 and equal_slices:
                        outputs["production_two_slice"] = torch.empty_like(expected)
                        variants["production_two_slice"] = lambda output=outputs[
                            "production_two_slice"
                        ]: _launch_specialized(case, output, block_size_n)
                        if case.slice_widths[0] % block_size_n == 0:
                            outputs["production_flat"] = torch.empty_like(expected)
                            variants["production_flat"] = lambda output=outputs[
                                "production_flat"
                            ]: _launch_flat(case, output, block_size_n)

                    for name, launch in variants.items():
                        launch()
                        torch.cuda.synchronize()
                        torch.testing.assert_close(
                            outputs[name], expected, rtol=2e-2, atol=2e-2
                        )

                    timings = benchmark_variants(
                        variants,
                        execution=args.execution,
                        warmup=args.warmup,
                        samples=args.samples,
                        inner_iterations=args.inner_iterations,
                    )
                    speedups: dict[str, float] = {}
                    for prototype, production in (
                        ("uniform_ragged", "production_two_slice"),
                        ("aligned_flat", "production_flat"),
                        ("uniform_ragged", "production_single_flat"),
                        ("aligned_flat", "production_single_flat"),
                    ):
                        if prototype in timings and production in timings:
                            speedups[f"{prototype}_vs_{production}"] = (
                                timings[production].p50_us / timings[prototype].p50_us
                            )
                    result = {
                        "case": case_id,
                        "slice_widths": list(case.slice_widths),
                        "num_slices": case.num_slices,
                        "num_tokens": num_tokens,
                        "num_pairs": case.num_pairs,
                        "rank": rank,
                        "block_size_n": block_size_n,
                        "execution": args.execution,
                        "timings": {
                            name: asdict(quantiles)
                            for name, quantiles in timings.items()
                        },
                        "prototype_speedup_vs_production_p50": speedups,
                    }
                    results.append(result)
                    timing_text = " ".join(
                        f"{name}={quantiles.p50_us:.3f}us"
                        for name, quantiles in timings.items()
                    )
                    print(
                        f"{case_id:8s} widths={slice_widths!s:>15s} "
                        f"T={num_tokens:>2d} R={rank:>2d} BN={block_size_n:>3d} "
                        f"{timing_text}"
                    )

    payload = {"metadata": _environment_metadata(args), "results": results}
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2, default=str) + "\n")
        print(f"Wrote {args.json_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
