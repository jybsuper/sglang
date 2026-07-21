"""Execution and profiling primitives for local MoE-LoRA benchmarks."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from math import floor
from typing import Any, Callable, Iterator, Literal, Sequence

ExecutionMode = Literal["eager", "cuda_graph"]
RunMode = Literal["time", "nsys", "ncu"]

_EXECUTION_MODES = frozenset(("eager", "cuda_graph"))
_RUN_MODES = frozenset(("time", "nsys", "ncu"))


def _load_torch() -> Any:
    # Keep CPU-only case discovery from eagerly initializing the CUDA surface.
    import torch

    return torch


def _require_positive_int(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


@dataclass(frozen=True, slots=True, kw_only=True)
class RunConfig:
    """Resolved controls for one timing or profiling invocation."""

    mode: RunMode = "time"
    execution: ExecutionMode = "eager"
    warmup: int = 20
    samples: int = 100
    inner_iterations: int = 1
    profile_iterations: int = 1

    def __post_init__(self) -> None:
        if self.mode not in _RUN_MODES:
            raise ValueError(f"unsupported run mode {self.mode!r}")
        if self.execution not in _EXECUTION_MODES:
            raise ValueError(f"unsupported execution mode {self.execution!r}")
        for name in (
            "warmup",
            "samples",
            "inner_iterations",
            "profile_iterations",
        ):
            _require_positive_int(name, getattr(self, name))


@dataclass(slots=True)
class PreparedBatch:
    """Callable batch plus a strong reference to its captured graph, if any."""

    run: Callable[[], None]
    launches_per_batch: int
    graph: Any | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class TimingStats:
    """CUDA-event latency summary in microseconds per logical invocation."""

    p20_us: float
    p50_us: float
    p80_us: float
    min_us: float
    max_us: float
    num_samples: int
    launches_per_batch: int


def make_batch(
    fn: Callable[[], None],
    *,
    execution: ExecutionMode,
    inner_iterations: int,
) -> PreparedBatch:
    """Prepare an eager loop or one CUDA graph containing the same loop.

    Compile kernels, initialize side streams, and allocate stable inputs before
    selecting ``cuda_graph`` because ``fn`` runs during capture.
    """

    if execution not in _EXECUTION_MODES:
        raise ValueError(f"unsupported execution mode {execution!r}")
    _require_positive_int("inner_iterations", inner_iterations)

    if execution == "eager":

        def eager_batch() -> None:
            for _ in range(inner_iterations):
                fn()

        return PreparedBatch(eager_batch, inner_iterations)

    torch = _load_torch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(inner_iterations):
            fn()
    return PreparedBatch(graph.replay, inner_iterations, graph)


def _linear_quantile(sorted_values: Sequence[float], q: float) -> float:
    position = (len(sorted_values) - 1) * q
    lower = floor(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def summarize_timings_us(
    samples_us: Sequence[float], *, launches_per_batch: int
) -> TimingStats:
    """Summarize already-normalized latency samples without GPU dependencies."""

    _require_positive_int("launches_per_batch", launches_per_batch)
    if not samples_us:
        raise ValueError("samples_us must not be empty")
    ordered = sorted(float(value) for value in samples_us)
    return TimingStats(
        p20_us=_linear_quantile(ordered, 0.2),
        p50_us=_linear_quantile(ordered, 0.5),
        p80_us=_linear_quantile(ordered, 0.8),
        min_us=ordered[0],
        max_us=ordered[-1],
        num_samples=len(ordered),
        launches_per_batch=launches_per_batch,
    )


def time_cuda_events(
    batch_fn: Callable[[], None],
    *,
    launches_per_batch: int,
    warmup: int,
    samples: int,
    before_sample: Callable[[], None] | None = None,
) -> TimingStats:
    """Measure unprofiled latency using CUDA events.

    ``before_sample`` runs on the current stream before every warmup batch and,
    for measured batches, before ``start.record()``. It can reset atomic
    destinations without adding that reset to reported latency. The measured
    callable must join side-stream work before it returns.
    """

    for name, value in (
        ("launches_per_batch", launches_per_batch),
        ("warmup", warmup),
        ("samples", samples),
    ):
        _require_positive_int(name, value)

    torch = _load_torch()
    for _ in range(warmup):
        if before_sample is not None:
            before_sample()
        batch_fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    for start, end in zip(starts, ends):
        if before_sample is not None:
            before_sample()
        start.record()
        batch_fn()
        end.record()
    torch.cuda.synchronize()

    return summarize_timings_us(
        [
            start.elapsed_time(end) * 1000.0 / launches_per_batch
            for start, end in zip(starts, ends)
        ],
        launches_per_batch=launches_per_batch,
    )


@contextmanager
def cuda_profile_range(
    label: str,
    *,
    emit_nvtx: bool = True,
    use_cuda_profiler_api: bool = True,
) -> Iterator[None]:
    """Delimit an already-warmed Nsight Systems or Compute capture region."""

    if not label or "\n" in label:
        raise ValueError("label must be a non-empty single line")

    torch = _load_torch()
    profiler_started = False
    nvtx_pushed = False
    torch.cuda.synchronize()
    try:
        if use_cuda_profiler_api:
            torch.cuda.cudart().cudaProfilerStart()
            profiler_started = True
        if emit_nvtx:
            torch.cuda.nvtx.range_push(label)
            nvtx_pushed = True
        yield
    finally:
        try:
            if nvtx_pushed:
                torch.cuda.nvtx.range_pop()
        finally:
            try:
                torch.cuda.synchronize()
            finally:
                if profiler_started:
                    torch.cuda.cudart().cudaProfilerStop()


__all__ = [
    "ExecutionMode",
    "PreparedBatch",
    "RunConfig",
    "RunMode",
    "TimingStats",
    "cuda_profile_range",
    "make_batch",
    "summarize_timings_us",
    "time_cuda_events",
]
