"""Benchmark-only raw indexed MoE-LoRA GEMV written in CuTe DSL.

This is a deliberately simple capability/performance probe, not a serving
kernel.  One CUDA thread owns one output element and resolves its adapter and
expert directly from the canonical route.  It establishes whether the local
CuTe DSL toolchain can express, compile, graph-capture, and execute the raw
indexed contract before investing in a Tensor-Core/warp-specialized version.
"""

from __future__ import annotations

from typing import Any

import torch

try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
except ImportError as exc:  # H200 image capability record.
    cuda = None
    cutlass = None
    cute = None
    from_dlpack = None
    _IMPORT_ERROR: Exception | None = exc
else:
    _IMPORT_ERROR = None


_COMPILED: dict[tuple[int, ...], Any] = {}
_THREADS = 128


if cute is not None:

    @cute.kernel
    def _indexed_kernel(
        mX: cute.Tensor,
        mW: cute.Tensor,
        mTopk: cute.Tensor,
        mMapping: cute.Tensor,
        mOut: cute.Tensor,
        pairs: cutlass.Constexpr,
        n_size: cutlass.Constexpr,
        k_size: cutlass.Constexpr,
        experts: cutlass.Constexpr,
        adapters: cutlass.Constexpr,
        top_k: cutlass.Constexpr,
        input_pair_major: cutlass.Constexpr,
    ):
        n_block, pair, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        n = n_block * _THREADS + tid
        token = pair // top_k
        expert = cutlass.Int32(mTopk[pair])
        adapter = cutlass.Int32(mMapping[token])
        valid = (
            (n < n_size)
            and (pair < pairs)
            and (expert >= 0)
            and (expert < experts)
            and (adapter >= 0)
            and (adapter < adapters)
        )
        if valid:
            group = adapter * experts + expert
            x_row = pair if input_pair_major else token
            acc = cutlass.Float32(0.0)
            for k in cutlass.range(k_size, unroll=1):
                acc += cutlass.Float32(mX[(x_row, k)]) * cutlass.Float32(
                    mW[(group, n, k)]
                )
            mOut[(pair, n)] = cutlass.BFloat16(acc)
        elif (n < n_size) and (pair < pairs):
            mOut[(pair, n)] = cutlass.BFloat16(0.0)

    @cute.jit
    def _indexed_host(
        mX: cute.Tensor,
        mW: cute.Tensor,
        mTopk: cute.Tensor,
        mMapping: cute.Tensor,
        mOut: cute.Tensor,
        stream: cuda.CUstream,
        pairs: cutlass.Constexpr,
        n_size: cutlass.Constexpr,
        k_size: cutlass.Constexpr,
        experts: cutlass.Constexpr,
        adapters: cutlass.Constexpr,
        top_k: cutlass.Constexpr,
        input_pair_major: cutlass.Constexpr,
    ):
        _indexed_kernel(
            mX,
            mW,
            mTopk,
            mMapping,
            mOut,
            pairs,
            n_size,
            k_size,
            experts,
            adapters,
            top_k,
            input_pair_major,
        ).launch(
            grid=[cute.ceil_div(n_size, _THREADS), pairs, 1],
            block=[_THREADS, 1, 1],
            stream=stream,
        )


def capability() -> dict[str, object]:
    return {
        "available": _IMPORT_ERROR is None,
        "error_type": type(_IMPORT_ERROR).__name__ if _IMPORT_ERROR else None,
        "error": str(_IMPORT_ERROR) if _IMPORT_ERROR else None,
        "cutlass_dsl_version": (
            getattr(cutlass, "__version__", "unknown") if cutlass else None
        ),
    }


def _compiled(
    x: torch.Tensor,
    weight: torch.Tensor,
    topk_ids: torch.Tensor,
    mapping: torch.Tensor,
    output: torch.Tensor,
    *,
    experts: int,
    adapters: int,
    input_pair_major: bool,
):
    if _IMPORT_ERROR is not None or cute is None or from_dlpack is None:
        raise RuntimeError(
            f"CuTe DSL unavailable: {type(_IMPORT_ERROR).__name__}: {_IMPORT_ERROR}"
        )
    pairs, n_size = output.shape
    k_size = x.shape[1]
    top_k = topk_ids.shape[1]
    key = (
        pairs,
        n_size,
        k_size,
        experts,
        adapters,
        top_k,
        int(input_pair_major),
    )
    compiled = _COMPILED.get(key)
    if compiled is None:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled = cute.compile(
            _indexed_host,
            from_dlpack(x),
            from_dlpack(weight),
            from_dlpack(topk_ids.reshape(-1)),
            from_dlpack(mapping),
            from_dlpack(output),
            stream,
            pairs,
            n_size,
            k_size,
            experts,
            adapters,
            top_k,
            input_pair_major,
        )
        _COMPILED[key] = compiled
    return compiled


def cutedsl_indexed_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    topk_ids: torch.Tensor,
    mapping: torch.Tensor,
    output: torch.Tensor,
    *,
    experts: int,
    adapters: int,
    input_pair_major: bool,
) -> None:
    """Compile/cache and launch the raw indexed CuTe DSL probe."""
    compiled = _compiled(
        x,
        weight,
        topk_ids,
        mapping,
        output,
        experts=experts,
        adapters=adapters,
        input_pair_major=input_pair_major,
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled(
        from_dlpack(x),
        from_dlpack(weight),
        from_dlpack(topk_ids.reshape(-1)),
        from_dlpack(mapping),
        from_dlpack(output),
        stream,
    )


__all__ = ["capability", "cutedsl_indexed_gemm"]
