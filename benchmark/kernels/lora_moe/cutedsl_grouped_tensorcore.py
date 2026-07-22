"""Benchmark-only Blackwell CuTe DSL grouped BF16 GEMM adapter.

This module deliberately does *not* implement another scalar CuTe probe.  It
adapts NVIDIA's installed ``blackwell/grouped_gemm.py`` example, whose kernel is
a persistent ``tcgen05`` Tensor Core implementation with TMA operand movement,
TMEM accumulators, warp specialization, and runtime problem descriptors.

The adapter keeps the serving-facing contract small: each problem is

``C_i[M_i, N_i] = A_i[M_i, K_i] @ B_i[N_i, K_i].T``.

All tensors may have different ``M/N/K`` dimensions, while every tensor keeps
its reduction/output dimension contiguous and 16-byte aligned.  Shapes,
strides, and pointers live on the GPU.  A plan owns those descriptors and can
be replayed by a CUDA graph as long as the referenced allocations remain
stable.  Empty groups must be removed by the route plan before construction.

Nothing here is imported by production serving code.  If this provider wins a
matched boundary benchmark, the reusable kernel should be promoted behind a
stable SGLang-owned API rather than importing an example file at runtime.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Sequence

import torch


@dataclass(frozen=True, slots=True)
class GroupedTactic:
    """Compile-time schedule knobs exposed by NVIDIA's grouped kernel."""

    mma_m: int = 128
    mma_n: int = 64
    cluster_m: int = 1
    cluster_n: int = 1
    use_2cta: bool = False
    tensormap_update: str = "SMEM"
    host_problem_shapes: bool = False

    def __post_init__(self) -> None:
        valid_m = (128, 256) if self.use_2cta else (64, 128)
        if self.mma_m not in valid_m:
            raise ValueError(
                f"mma_m={self.mma_m} must be one of {valid_m} for "
                f"use_2cta={self.use_2cta}"
            )
        if self.mma_n < 32 or self.mma_n > 256 or self.mma_n % 32:
            raise ValueError("mma_n must be in [32, 256] and divisible by 32")
        if self.cluster_m <= 0 or self.cluster_n <= 0:
            raise ValueError("cluster dimensions must be positive")
        if self.cluster_m * self.cluster_n > 16:
            raise ValueError("cluster size must not exceed 16 CTAs")
        if self.use_2cta and self.cluster_m % 2:
            raise ValueError("2-CTA MMA requires an even cluster_m")
        if self.tensormap_update not in ("SMEM", "GMEM"):
            raise ValueError("tensormap_update must be SMEM or GMEM")

    @property
    def key(self) -> tuple[object, ...]:
        return (
            self.mma_m,
            self.mma_n,
            self.cluster_m,
            self.cluster_n,
            self.use_2cta,
            self.tensormap_update,
            self.host_problem_shapes,
        )


@dataclass(slots=True)
class _Compiled:
    fn: object
    initial_cute: tuple[object, object, object]
    initial_torch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    kernel: object
    max_active_clusters: int
    sm_count: int
    compile_ms: float
    source_sha256: str


_SOURCE: ModuleType | None = None
_SOURCE_PATH: Path | None = None
_IMPORT_ERROR: BaseException | None = None
_COMPILED: dict[tuple[object, ...], _Compiled] = {}


def _candidate_source_paths() -> tuple[Path, ...]:
    override = os.environ.get("SGL_CUTEDSL_GROUPED_GEMM_SOURCE")
    paths: list[Path] = []
    if override:
        paths.append(Path(override))
    try:
        import flashinfer

        root = Path(flashinfer.__file__).resolve().parent
        paths.append(
            root
            / "data"
            / "cutlass"
            / "examples"
            / "python"
            / "CuTeDSL"
            / "blackwell"
            / "grouped_gemm.py"
        )
    except Exception:
        pass
    return tuple(dict.fromkeys(paths))


def _load_source() -> ModuleType:
    global _SOURCE, _SOURCE_PATH, _IMPORT_ERROR
    if _SOURCE is not None:
        return _SOURCE
    if _IMPORT_ERROR is not None:
        raise RuntimeError("Blackwell CuTe DSL grouped GEMM is unavailable") from _IMPORT_ERROR
    try:
        import cutlass  # noqa: F401
        import cutlass.cute  # noqa: F401

        for path in _candidate_source_paths():
            if not path.is_file():
                continue
            spec = importlib.util.spec_from_file_location(
                "_sgl_benchmark_cutlass_blackwell_grouped_gemm", path
            )
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _SOURCE = module
            _SOURCE_PATH = path
            return module
        raise FileNotFoundError(
            "could not locate flashinfer's bundled CuTeDSL/blackwell/grouped_gemm.py"
        )
    except BaseException as exc:
        _IMPORT_ERROR = exc
        raise RuntimeError("Blackwell CuTe DSL grouped GEMM is unavailable") from exc


def capability() -> dict[str, object]:
    """Report the exact compiler/kernel source capability without compiling."""

    try:
        module = _load_source()
        import cutlass

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        major, minor = torch.cuda.get_device_capability()
        if (major, minor) not in ((10, 0), (10, 3)):
            raise RuntimeError(f"requires SM100/SM103, found SM{major}{minor}")
        assert _SOURCE_PATH is not None
        digest = hashlib.sha256(_SOURCE_PATH.read_bytes()).hexdigest()
        return {
            "available": True,
            "device_capability": f"{major}.{minor}",
            "cutlass_version": getattr(cutlass, "__version__", "unknown"),
            "source_path": str(_SOURCE_PATH),
            "source_sha256": digest,
            "kernel_class": module.GroupedGemmKernel.__name__,
            "mechanism": "persistent_tcgen05_tma_tmem_warp_specialized",
        }
    except BaseException as exc:
        root = exc.__cause__ if exc.__cause__ is not None else exc
        return {
            "available": False,
            "error_type": type(root).__name__,
            "error": str(root),
        }


def clear_compile_cache() -> None:
    """Drop Python references to compiled variants (CUDA modules stay driver-owned)."""

    _COMPILED.clear()


def _cluster_tile_shape(tactic: GroupedTactic) -> tuple[int, int]:
    cta_m = tactic.mma_m // (2 if tactic.use_2cta else 1)
    return cta_m * tactic.cluster_m, tactic.mma_n * tactic.cluster_n


def _total_clusters(
    shapes: Sequence[tuple[int, int, int, int]], tactic: GroupedTactic
) -> int:
    tile_m, tile_n = _cluster_tile_shape(tactic)
    return sum(
        ((m + tile_m - 1) // tile_m) * ((n + tile_n - 1) // tile_n)
        for m, n, _k, _l in shapes
    )


def _compile(
    group_count: int,
    tactic: GroupedTactic,
    *,
    total_clusters: int,
    device: torch.device,
) -> tuple[_Compiled, bool]:
    module = _load_source()
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    import cutlass.utils as utils

    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    key = (
        device_index,
        group_count,
        *tactic.key,
        total_clusters if tactic.host_problem_shapes else -1,
    )
    cached = _COMPILED.get(key)
    if cached is not None:
        return cached, True

    update_mode = (
        utils.TensorMapUpdateMode.SMEM
        if tactic.tensormap_update == "SMEM"
        else utils.TensorMapUpdateMode.GMEM
    )
    kernel = module.GroupedGemmKernel(
        cutlass.Float32,
        tactic.use_2cta,
        (tactic.mma_m, tactic.mma_n),
        (tactic.cluster_m, tactic.cluster_n),
        update_mode,
    )

    # The grouped kernel uses these tensors only to establish BF16 dtype and
    # K-major A/B + N-major C layouts.  Real addresses are descriptor entries.
    initial_torch: list[torch.Tensor] = []
    initial_cute: list[object] = []
    for is_mode0_major in (False, False, False):
        _ptr, torch_tensor, cute_tensor, _cpu, _stride = (
            module.create_tensor_and_stride(
                1,
                8,
                8,
                is_mode0_major,
                cutlass.BFloat16,
            )
        )
        initial_torch.append(torch_tensor)
        initial_cute.append(cute_tensor)

    # Metadata layouts are compile-time; values remain runtime GPU data.
    shape_cute, _shape_torch = cutlass_torch.cute_tensor_like(
        torch.empty((group_count, 4), dtype=torch.int32),
        cutlass.Int32,
        is_dynamic_layout=False,
        assumed_align=16,
    )
    stride_cute, _stride_torch = cutlass_torch.cute_tensor_like(
        torch.empty((group_count, 3, 2), dtype=torch.int32),
        cutlass.Int32,
        is_dynamic_layout=False,
        assumed_align=16,
    )
    pointer_cute, _pointer_torch = cutlass_torch.cute_tensor_like(
        torch.empty((group_count, 3), dtype=torch.int64),
        cutlass.Int64,
        is_dynamic_layout=False,
        assumed_align=16,
    )
    hardware = utils.HardwareInfo()
    sm_count = int(hardware.get_max_active_clusters(1))
    cluster_ctas = tactic.cluster_m * tactic.cluster_n
    max_active_clusters = int(hardware.get_max_active_clusters(cluster_ctas))
    tensormap_shape = (
        sm_count,
        module.GroupedGemmKernel.num_tensormaps,
        module.GroupedGemmKernel.bytes_per_tensormap // 8,
    )
    tensormap_cute, _tensormap_torch = cutlass_torch.cute_tensor_like(
        torch.empty(tensormap_shape, dtype=torch.int64),
        cutlass.Int64,
        is_dynamic_layout=False,
    )
    launch_clusters = (
        total_clusters if tactic.host_problem_shapes else max_active_clusters
    )
    stream = cuda.CUstream(torch.cuda.current_stream(device_index).cuda_stream)
    try:
        from cutlass import CUDA_VERSION

        opt_level = (
            3
            if CUDA_VERSION.major < 13
            or (CUDA_VERSION.major == 13 and CUDA_VERSION.minor < 1)
            else 2
        )
    except ImportError:
        opt_level = 3
    started = time.perf_counter()
    fn = cute.compile(
        kernel,
        initial_cute[0],
        initial_cute[1],
        initial_cute[2],
        group_count,
        shape_cute,
        stride_cute,
        pointer_cute,
        launch_clusters,
        tensormap_cute,
        max_active_clusters,
        stream,
        options=f"--opt-level {opt_level}",
    )
    compile_ms = (time.perf_counter() - started) * 1e3
    assert _SOURCE_PATH is not None
    result = _Compiled(
        fn=fn,
        initial_cute=(initial_cute[0], initial_cute[1], initial_cute[2]),
        initial_torch=(initial_torch[0], initial_torch[1], initial_torch[2]),
        kernel=kernel,
        max_active_clusters=max_active_clusters,
        sm_count=sm_count,
        compile_ms=compile_ms,
        source_sha256=hashlib.sha256(_SOURCE_PATH.read_bytes()).hexdigest(),
    )
    _COMPILED[key] = result
    return result, False


class BlackwellGroupedGemmPlan:
    """Static allocation/descriptor plan for one graph-replayable grouped GEMM."""

    def __init__(
        self,
        a_tensors: Sequence[torch.Tensor],
        b_tensors: Sequence[torch.Tensor],
        c_tensors: Sequence[torch.Tensor],
        tactic: GroupedTactic = GroupedTactic(),
    ) -> None:
        if not a_tensors or not (
            len(a_tensors) == len(b_tensors) == len(c_tensors)
        ):
            raise ValueError("A/B/C must be nonempty sequences of equal length")
        self.a_tensors = tuple(a_tensors)
        self.b_tensors = tuple(b_tensors)
        self.c_tensors = tuple(c_tensors)
        self.tactic = tactic
        self.device = self.a_tensors[0].device
        if self.device.type != "cuda":
            raise ValueError("grouped GEMM tensors must be CUDA tensors")

        shapes: list[tuple[int, int, int, int]] = []
        strides: list[list[tuple[int, int]]] = []
        pointers: list[list[int]] = []
        for index, (a, b, c) in enumerate(
            zip(self.a_tensors, self.b_tensors, self.c_tensors)
        ):
            if a.ndim != 2 or b.ndim != 2 or c.ndim != 2:
                raise ValueError(f"group {index}: A/B/C must all be 2D")
            m, k = a.shape
            n, bk = b.shape
            if m <= 0 or n <= 0 or k <= 0:
                raise ValueError(f"group {index}: empty dimensions are unsupported")
            if bk != k or c.shape != (m, n):
                raise ValueError(
                    f"group {index}: expected B[N,{k}] and C[{m},{n}], got "
                    f"B{tuple(b.shape)} C{tuple(c.shape)}"
                )
            if any(t.device != self.device for t in (a, b, c)):
                raise ValueError(f"group {index}: all tensors must share a device")
            if any(t.dtype != torch.bfloat16 for t in (a, b, c)):
                raise ValueError(f"group {index}: only BF16 is supported")
            if a.stride(1) != 1 or b.stride(1) != 1 or c.stride(1) != 1:
                raise ValueError(
                    f"group {index}: reduction/output dimensions must be contiguous"
                )
            # TMA requires every contiguous row and base address to be 16-byte
            # aligned.  Non-unit row strides are legal and used for gate/up
            # slices packed into one allocation.
            if k % 8 or n % 8:
                raise ValueError(
                    f"group {index}: BF16 K={k} and N={n} must be divisible by 8"
                )
            if any(t.data_ptr() % 16 for t in (a, b, c)):
                raise ValueError(f"group {index}: tensor base is not 16-byte aligned")
            if any(value >= 2**31 for t in (a, b, c) for value in t.stride()):
                raise ValueError(f"group {index}: stride exceeds int32 metadata")
            shapes.append((m, n, k, 1))
            strides.append(
                [
                    (a.stride(0), a.stride(1)),
                    (b.stride(0), b.stride(1)),
                    (c.stride(0), c.stride(1)),
                ]
            )
            pointers.append([a.data_ptr(), b.data_ptr(), c.data_ptr()])

        self.shapes = tuple(shapes)
        self.total_clusters = _total_clusters(self.shapes, tactic)
        started = time.perf_counter()
        self.compiled, self.compile_cache_hit = _compile(
            len(self.shapes),
            tactic,
            total_clusters=self.total_clusters,
            device=self.device,
        )

        import cutlass
        import cutlass.torch as cutlass_torch

        self.shape_cute, self.shape_torch = cutlass_torch.cute_tensor_like(
            torch.tensor(shapes, dtype=torch.int32),
            cutlass.Int32,
            is_dynamic_layout=False,
            assumed_align=16,
        )
        self.stride_cute, self.stride_torch = cutlass_torch.cute_tensor_like(
            torch.tensor(strides, dtype=torch.int32),
            cutlass.Int32,
            is_dynamic_layout=False,
            assumed_align=16,
        )
        self.pointer_cute, self.pointer_torch = cutlass_torch.cute_tensor_like(
            torch.tensor(pointers, dtype=torch.int64),
            cutlass.Int64,
            is_dynamic_layout=False,
            assumed_align=16,
        )
        source = _load_source()
        tensormap_shape = (
            self.compiled.sm_count,
            source.GroupedGemmKernel.num_tensormaps,
            source.GroupedGemmKernel.bytes_per_tensormap // 8,
        )
        self.tensormap_cute, self.tensormap_torch = cutlass_torch.cute_tensor_like(
            torch.empty(tensormap_shape, dtype=torch.int64),
            cutlass.Int64,
            is_dynamic_layout=False,
        )
        self.plan_build_ms = (time.perf_counter() - started) * 1e3

    def __call__(self) -> None:
        import cuda.bindings.driver as cuda

        device_index = self.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        stream = cuda.CUstream(torch.cuda.current_stream(device_index).cuda_stream)
        self.compiled.fn(
            self.compiled.initial_cute[0],
            self.compiled.initial_cute[1],
            self.compiled.initial_cute[2],
            self.shape_cute,
            self.stride_cute,
            self.pointer_cute,
            self.tensormap_cute,
            stream,
        )

    def metadata(self) -> dict[str, object]:
        kernel = self.compiled.kernel
        return {
            "groups": len(self.shapes),
            "problem_shapes_mnkl": [list(shape) for shape in self.shapes],
            "total_clusters": self.total_clusters,
            "max_active_clusters": self.compiled.max_active_clusters,
            "tactic": asdict(self.tactic),
            "automatic_stages": {
                key: getattr(kernel, key, None)
                for key in ("num_ab_stage", "num_acc_stage", "num_epi_stage")
            },
            "compile_cache_hit": self.compile_cache_hit,
            "compile_ms": self.compiled.compile_ms,
            "plan_build_ms": self.plan_build_ms,
            "source_sha256": self.compiled.source_sha256,
            "mechanism": "persistent_tcgen05_tma_tmem_warp_specialized",
        }


__all__ = [
    "BlackwellGroupedGemmPlan",
    "GroupedTactic",
    "capability",
    "clear_compile_cache",
]
