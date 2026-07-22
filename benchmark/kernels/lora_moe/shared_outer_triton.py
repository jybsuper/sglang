"""Benchmark-only Triton candidates for shared-outer MoE-LoRA factors.

These kernels deliberately live outside ``python/sglang`` and are never
selected by production dispatch.  They test two factorization opportunities:

* shared gate/up A is a token/adapter GEMM producing ``[T, 2R]``;
* shared down B can reduce weighted top-k rank vectors inside the output GEMM,
  applying B once per token instead of once per routed pair.

Adapter spans are supplied by the benchmark's already-segmented request
metadata.  A prebuilt descriptor grid covers every active adapter in one
kernel launch while preserving tensor-core GEMM tiles; no synthetic adapter
sort is hidden in the candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch

from benchmark.kernels.lora_moe.shared_outer import AdapterSpan

try:
    import triton
    import triton.language as tl
except ImportError:  # Keep CPU-only test discovery working.
    triton = None
    tl = None


@dataclass(frozen=True, slots=True)
class SharedGateAConfig:
    key: str
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int = 3


@dataclass(frozen=True, slots=True)
class SharedDownBConfig:
    key: str
    block_m: int
    block_n: int
    block_r: int
    num_warps: int
    num_stages: int = 3


@dataclass(frozen=True, slots=True)
class SharedOuterTilePlan:
    """Prebuilt tensor-core tiles for already-segmented adapter spans.

    The plan is benchmark metadata, not a proposed public runtime ABI.  A
    production implementation could derive the same schedule from existing
    segment pointers or absorb it into a persistent descriptor cache.
    """

    adapter_ids: torch.Tensor
    token_starts: torch.Tensor
    token_counts: torch.Tensor
    n_block_ids: torch.Tensor
    block_m: int
    block_n: int
    output_width: int

    @property
    def num_tiles(self) -> int:
        return self.adapter_ids.numel()


def build_shared_outer_tile_plan(
    spans: tuple[AdapterSpan, ...],
    *,
    output_width: int,
    block_m: int,
    block_n: int,
    device: torch.device,
) -> SharedOuterTilePlan:
    """Build a one-launch tile map outside the measured K0 kernel region."""

    if output_width <= 0 or block_m <= 0 or block_n <= 0:
        raise ValueError("output_width and tile dimensions must be positive")
    adapter_ids: list[int] = []
    token_starts: list[int] = []
    token_counts: list[int] = []
    n_block_ids: list[int] = []
    num_n_blocks = (output_width + block_n - 1) // block_n
    for span in spans:
        if span.adapter_id is None:
            continue
        for token_start in range(span.start, span.stop, block_m):
            token_count = min(block_m, span.stop - token_start)
            for n_block_id in range(num_n_blocks):
                adapter_ids.append(span.adapter_id)
                token_starts.append(token_start)
                token_counts.append(token_count)
                n_block_ids.append(n_block_id)
    if not adapter_ids:
        raise ValueError("tile plan requires at least one active adapter span")

    def tensor(values: list[int]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.int32, device=device)

    return SharedOuterTilePlan(
        adapter_ids=tensor(adapter_ids),
        token_starts=tensor(token_starts),
        token_counts=tensor(token_counts),
        n_block_ids=tensor(n_block_ids),
        block_m=block_m,
        block_n=block_n,
        output_width=output_width,
    )


GATE_A_CONFIGS: tuple[SharedGateAConfig, ...] = tuple(
    SharedGateAConfig(
        key=f"bm{block_m}-bn{block_n}-bk{block_k}-w{num_warps}",
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
    )
    for block_m, block_n, block_k, num_warps in (
        (16, 32, 64, 4),
        (16, 64, 64, 4),
        (16, 128, 64, 4),
        (32, 32, 64, 4),
        (32, 64, 64, 4),
        (32, 128, 64, 8),
        (64, 64, 64, 4),
        (64, 128, 64, 8),
        (16, 64, 128, 4),
        (32, 64, 128, 4),
    )
)
DOWN_B_CONFIGS: tuple[SharedDownBConfig, ...] = tuple(
    SharedDownBConfig(
        key=f"bm{block_m}-bn{block_n}-br{block_r}-w{num_warps}",
        block_m=block_m,
        block_n=block_n,
        block_r=block_r,
        num_warps=num_warps,
    )
    for block_m, block_n, block_r, num_warps in (
        (16, 32, 32, 4),
        (16, 64, 32, 4),
        (16, 128, 32, 4),
        (16, 64, 64, 4),
        (16, 128, 64, 4),
        (16, 128, 64, 8),
        (32, 64, 32, 4),
        (32, 128, 32, 4),
        (32, 64, 64, 4),
        (32, 128, 64, 8),
        (64, 64, 64, 4),
        (64, 128, 64, 8),
    )
)
GATE_A_CONFIGS_BY_KEY = {config.key: config for config in GATE_A_CONFIGS}
DOWN_B_CONFIGS_BY_KEY = {config.key: config for config in DOWN_B_CONFIGS}

DEFAULT_GATE_A_CONFIG = "bm16-bn32-bk64-w4"
DEFAULT_DOWN_B_CONFIG = "bm16-bn128-br64-w4"
AUTO_GATE_A_CONFIG = {
    "h200": "bm16-bn32-bk64-w4",
    "gb300": "bm16-bn32-bk64-w4",
}
AUTO_DOWN_B_CONFIG = {
    "h200": "bm16-bn128-br64-w8",
    "gb300": "bm64-bn128-br64-w8",
}


if triton is not None:

    @triton.jit
    def _shared_gate_a_kernel(
        hidden_ptr,
        factor_ptr,
        output_ptr,
        adapter_ids_ptr,
        token_starts_ptr,
        token_counts_ptr,
        n_block_ids_ptr,
        stride_xt,
        stride_xh,
        stride_fl,
        stride_fn,
        stride_fh,
        stride_ot,
        stride_on,
        N: tl.constexpr,
        H: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        ENABLE_PDL: tl.constexpr,
    ):
        tile_id = tl.program_id(0)
        adapter_id = tl.load(adapter_ids_ptr + tile_id)
        token_start = tl.load(token_starts_ptr + tile_id)
        token_count = tl.load(token_counts_ptr + tile_id)
        n_block_id = tl.load(n_block_ids_ptr + tile_id)
        local_m = tl.arange(0, BLOCK_M)
        offsets_m = token_start + local_m
        offsets_n = n_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_k = tl.arange(0, BLOCK_K)

        if ENABLE_PDL:
            tl.extra.cuda.gdc_wait()

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_block in range(0, tl.cdiv(H, BLOCK_K)):
            current_k = k_block * BLOCK_K + offsets_k
            hidden = tl.load(
                hidden_ptr
                + offsets_m[:, None] * stride_xt
                + current_k[None, :] * stride_xh,
                mask=(local_m[:, None] < token_count) & (current_k[None, :] < H),
                other=0.0,
            )
            factor = tl.load(
                factor_ptr
                + adapter_id * stride_fl
                + offsets_n[:, None] * stride_fn
                + current_k[None, :] * stride_fh,
                mask=(offsets_n[:, None] < N) & (current_k[None, :] < H),
                other=0.0,
            )
            accumulator += tl.dot(hidden, tl.trans(factor))

        tl.store(
            output_ptr
            + offsets_m[:, None] * stride_ot
            + offsets_n[None, :] * stride_on,
            accumulator.to(output_ptr.dtype.element_ty),
            mask=(local_m[:, None] < token_count) & (offsets_n[None, :] < N),
        )
        if ENABLE_PDL:
            tl.extra.cuda.gdc_launch_dependents()

    @triton.jit
    def _materialize_gate_pairs_kernel(
        token_ptr,
        pair_ptr,
        num_elements,
        N: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK: tl.constexpr,
        ENABLE_PDL: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < num_elements
        feature = offsets % N
        token = offsets // (TOP_K * N)
        if ENABLE_PDL:
            tl.extra.cuda.gdc_wait()
        value = tl.load(token_ptr + token * N + feature, mask=valid)
        tl.store(pair_ptr + offsets, value, mask=valid)
        if ENABLE_PDL:
            tl.extra.cuda.gdc_launch_dependents()

    @triton.jit
    def _shared_down_b_weighted_rank_kernel(
        pair_rank_ptr,
        factor_ptr,
        topk_weights_ptr,
        output_ptr,
        adapter_ids_ptr,
        token_starts_ptr,
        token_counts_ptr,
        n_block_ids_ptr,
        stride_zt,
        stride_zk,
        stride_zr,
        stride_fl,
        stride_fn,
        stride_fr,
        stride_wt,
        stride_wk,
        stride_ot,
        stride_on,
        N: tl.constexpr,
        R: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_R: tl.constexpr,
        ENABLE_PDL: tl.constexpr,
    ):
        tile_id = tl.program_id(0)
        adapter_id = tl.load(adapter_ids_ptr + tile_id)
        token_start = tl.load(token_starts_ptr + tile_id)
        token_count = tl.load(token_counts_ptr + tile_id)
        n_block_id = tl.load(n_block_ids_ptr + tile_id)
        local_m = tl.arange(0, BLOCK_M)
        offsets_m = token_start + local_m
        offsets_n = n_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_r = tl.arange(0, BLOCK_R)

        if ENABLE_PDL:
            tl.extra.cuda.gdc_wait()

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for r_block in range(0, tl.cdiv(R, BLOCK_R)):
            current_r = r_block * BLOCK_R + offsets_r
            reduced_rank = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
            for topk_slot in range(0, TOP_K):
                routed_weight = tl.load(
                    topk_weights_ptr + offsets_m * stride_wt + topk_slot * stride_wk,
                    mask=local_m < token_count,
                    other=0.0,
                )
                pair_rank = tl.load(
                    pair_rank_ptr
                    + offsets_m[:, None] * stride_zt
                    + topk_slot * stride_zk
                    + current_r[None, :] * stride_zr,
                    mask=(local_m[:, None] < token_count) & (current_r[None, :] < R),
                    other=0.0,
                )
                reduced_rank += pair_rank.to(tl.float32) * routed_weight[:, None]

            factor = tl.load(
                factor_ptr
                + adapter_id * stride_fl
                + offsets_n[:, None] * stride_fn
                + current_r[None, :] * stride_fr,
                mask=(offsets_n[:, None] < N) & (current_r[None, :] < R),
                other=0.0,
            )
            accumulator += tl.dot(reduced_rank.to(tl.bfloat16), tl.trans(factor))

        base = tl.load(
            output_ptr
            + offsets_m[:, None] * stride_ot
            + offsets_n[None, :] * stride_on,
            mask=(local_m[:, None] < token_count) & (offsets_n[None, :] < N),
            other=0.0,
        )
        tl.store(
            output_ptr
            + offsets_m[:, None] * stride_ot
            + offsets_n[None, :] * stride_on,
            (base.to(tl.float32) + accumulator).to(output_ptr.dtype.element_ty),
            mask=(local_m[:, None] < token_count) & (offsets_n[None, :] < N),
        )
        if ENABLE_PDL:
            tl.extra.cuda.gdc_launch_dependents()

else:
    _shared_gate_a_kernel = None
    _materialize_gate_pairs_kernel = None
    _shared_down_b_weighted_rank_kernel = None


def _require_triton() -> None:
    if triton is None or _shared_gate_a_kernel is None:
        raise RuntimeError("shared-outer candidates require Triton")


def _pdl_metadata() -> tuple[bool, dict[str, bool]]:
    from sglang.srt.lora.sgl_lora.triton_ops.virtual_experts import (
        _get_pdl_launch_metadata,
    )

    return _get_pdl_launch_metadata()


def invoke_shared_gate_a(
    hidden_states: torch.Tensor,
    shared_a: torch.Tensor,
    output: torch.Tensor,
    plan: SharedOuterTilePlan,
    *,
    config: SharedGateAConfig,
) -> None:
    """Launch all token/adapter tensor-core tiles in one descriptor grid."""

    _require_triton()
    if hidden_states.ndim != 2 or shared_a.ndim != 4 or shared_a.shape[1] != 1:
        raise ValueError("expected hidden [T,H] and shared_a [L,1,N,H]")
    tokens, hidden_size = hidden_states.shape
    output_width = shared_a.shape[2]
    if shared_a.shape[3] != hidden_size or output.shape != (tokens, output_width):
        raise ValueError("shared gate-A shapes do not match")
    if (
        plan.block_m != config.block_m
        or plan.block_n != config.block_n
        or plan.output_width != output_width
    ):
        raise ValueError("gate-A tile plan does not match config/output")
    enable_pdl, pdl_kwargs = _pdl_metadata()
    _shared_gate_a_kernel[(plan.num_tiles,)](
        hidden_states,
        shared_a,
        output,
        plan.adapter_ids,
        plan.token_starts,
        plan.token_counts,
        plan.n_block_ids,
        hidden_states.stride(0),
        hidden_states.stride(1),
        shared_a.stride(0),
        shared_a.stride(2),
        shared_a.stride(3),
        output.stride(0),
        output.stride(1),
        N=output_width,
        H=hidden_size,
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        ENABLE_PDL=enable_pdl,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
        **pdl_kwargs,
    )


def materialize_shared_gate_pairs(
    token_output: torch.Tensor, pair_output: torch.Tensor
) -> None:
    """Charge the explicit ``[T,N] -> [T,K,N]`` compatibility copy."""

    _require_triton()
    if token_output.ndim != 2 or pair_output.ndim != 3:
        raise ValueError("expected token [T,N] and pair [T,K,N] outputs")
    if (
        pair_output.shape[0] != token_output.shape[0]
        or pair_output.shape[2] != token_output.shape[1]
    ):
        raise ValueError("token and pair outputs do not match")
    if not token_output.is_contiguous() or not pair_output.is_contiguous():
        raise ValueError("pair materialization requires contiguous outputs")
    num_elements = pair_output.numel()
    block = 512
    enable_pdl, pdl_kwargs = _pdl_metadata()
    _materialize_gate_pairs_kernel[(triton.cdiv(num_elements, block),)](
        token_output,
        pair_output,
        num_elements,
        N=token_output.shape[1],
        TOP_K=pair_output.shape[1],
        BLOCK=block,
        ENABLE_PDL=enable_pdl,
        num_warps=4,
        num_stages=1,
        **pdl_kwargs,
    )


def invoke_shared_down_b_weighted_rank(
    pair_rank: torch.Tensor,
    shared_b: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
    plan: SharedOuterTilePlan,
    *,
    config: SharedDownBConfig,
) -> None:
    """Fuse weighted top-k rank reduction with one shared-B GEMM per token."""

    _require_triton()
    if pair_rank.ndim != 3 or shared_b.ndim != 4 or shared_b.shape[1] != 1:
        raise ValueError("expected pair_rank [T,K,R] and shared_b [L,1,H,R]")
    tokens, top_k, rank = pair_rank.shape
    hidden_size = shared_b.shape[2]
    if shared_b.shape[3] != rank:
        raise ValueError("shared-B rank does not match pair rank")
    if topk_weights.shape != (tokens, top_k) or output.shape != (tokens, hidden_size):
        raise ValueError("shared down-B shapes do not match")
    if (
        plan.block_m != config.block_m
        or plan.block_n != config.block_n
        or plan.output_width != hidden_size
    ):
        raise ValueError("down-B tile plan does not match config/output")
    enable_pdl, pdl_kwargs = _pdl_metadata()
    _shared_down_b_weighted_rank_kernel[(plan.num_tiles,)](
        pair_rank,
        shared_b,
        topk_weights,
        output,
        plan.adapter_ids,
        plan.token_starts,
        plan.token_counts,
        plan.n_block_ids,
        pair_rank.stride(0),
        pair_rank.stride(1),
        pair_rank.stride(2),
        shared_b.stride(0),
        shared_b.stride(2),
        shared_b.stride(3),
        topk_weights.stride(0),
        topk_weights.stride(1),
        output.stride(0),
        output.stride(1),
        N=hidden_size,
        R=rank,
        TOP_K=top_k,
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_R=config.block_r,
        ENABLE_PDL=enable_pdl,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
        **pdl_kwargs,
    )


__all__ = [
    "AUTO_DOWN_B_CONFIG",
    "AUTO_GATE_A_CONFIG",
    "DEFAULT_DOWN_B_CONFIG",
    "DEFAULT_GATE_A_CONFIG",
    "DOWN_B_CONFIGS",
    "DOWN_B_CONFIGS_BY_KEY",
    "GATE_A_CONFIGS",
    "GATE_A_CONFIGS_BY_KEY",
    "SharedDownBConfig",
    "SharedGateAConfig",
    "SharedOuterTilePlan",
    "build_shared_outer_tile_plan",
    "invoke_shared_down_b_weighted_rank",
    "invoke_shared_gate_a",
    "materialize_shared_gate_pairs",
]
