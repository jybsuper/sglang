from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmark.kernels.lora_moe import profiling


class _FakeGraph:
    def __init__(self, log: list[str]):
        self._log = log

    def replay(self) -> None:
        self._log.append("replay")


class _FakeGraphContext:
    def __init__(self, log: list[str]):
        self._log = log

    def __enter__(self):
        self._log.append("capture_enter")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._log.append("capture_exit")
        return False


class _FakeEvent:
    def __init__(self, cuda, *, enable_timing: bool):
        assert enable_timing
        self._cuda = cuda
        self._time_ms: float | None = None

    def record(self) -> None:
        self._time_ms = self._cuda.clock_ms

    def elapsed_time(self, end: _FakeEvent) -> float:
        assert self._time_ms is not None
        assert end._time_ms is not None
        return end._time_ms - self._time_ms


class _FakeCudart:
    def __init__(self, log: list[str]):
        self._log = log

    def cudaProfilerStart(self) -> None:
        self._log.append("profiler_start")

    def cudaProfilerStop(self) -> None:
        self._log.append("profiler_stop")


class _FakeNvtx:
    def __init__(self, log: list[str]):
        self._log = log

    def range_push(self, label: str) -> None:
        self._log.append(f"push:{label}")

    def range_pop(self) -> None:
        self._log.append("pop")


class _FakeCuda:
    def __init__(self):
        self.log: list[str] = []
        self.clock_ms = 0.0
        self.nvtx = _FakeNvtx(self.log)
        self._cudart = _FakeCudart(self.log)

    def CUDAGraph(self) -> _FakeGraph:
        self.log.append("graph_create")
        return _FakeGraph(self.log)

    def graph(self, graph: _FakeGraph) -> _FakeGraphContext:
        assert isinstance(graph, _FakeGraph)
        return _FakeGraphContext(self.log)

    def Event(self, *, enable_timing: bool) -> _FakeEvent:
        return _FakeEvent(self, enable_timing=enable_timing)

    def synchronize(self) -> None:
        self.log.append("synchronize")

    def cudart(self) -> _FakeCudart:
        return self._cudart


def _fake_torch() -> tuple[SimpleNamespace, _FakeCuda]:
    cuda = _FakeCuda()
    return SimpleNamespace(cuda=cuda), cuda


def test_eager_and_graph_batches_have_the_same_logical_batch_size(monkeypatch):
    eager_calls: list[str] = []
    eager = profiling.make_batch(
        lambda: eager_calls.append("launch"),
        execution="eager",
        inner_iterations=3,
    )
    eager.run()
    assert eager_calls == ["launch"] * 3
    assert eager.launches_per_batch == 3
    assert eager.graph is None

    fake_torch, cuda = _fake_torch()
    monkeypatch.setattr(profiling, "_load_torch", lambda: fake_torch)
    capture_calls: list[str] = []
    captured = profiling.make_batch(
        lambda: capture_calls.append("launch"),
        execution="cuda_graph",
        inner_iterations=3,
    )
    assert capture_calls == ["launch"] * 3
    assert cuda.log == ["graph_create", "capture_enter", "capture_exit"]
    assert captured.launches_per_batch == 3
    assert captured.graph is not None

    captured.run()
    assert cuda.log[-1] == "replay"


def test_cuda_event_timing_normalizes_and_summarizes_samples(monkeypatch):
    fake_torch, cuda = _fake_torch()
    monkeypatch.setattr(profiling, "_load_torch", lambda: fake_torch)
    durations_ms = iter((99.0, 99.0, 1.0, 2.0, 3.0, 4.0, 5.0))

    def batch() -> None:
        cuda.clock_ms += next(durations_ms)

    stats = profiling.time_cuda_events(
        batch,
        launches_per_batch=2,
        warmup=2,
        samples=5,
    )

    assert stats.min_us == 500.0
    assert stats.p20_us == 900.0
    assert stats.p50_us == 1500.0
    assert stats.p80_us == 2100.0
    assert stats.max_us == 2500.0
    assert stats.num_samples == 5
    assert stats.launches_per_batch == 2
    assert cuda.log.count("synchronize") == 2


def test_cuda_event_timing_runs_reset_before_unmeasured_boundary(monkeypatch):
    fake_torch, cuda = _fake_torch()
    monkeypatch.setattr(profiling, "_load_torch", lambda: fake_torch)
    actions: list[str] = []

    def reset() -> None:
        actions.append("reset")
        cuda.clock_ms += 10.0

    def batch() -> None:
        actions.append("batch")
        cuda.clock_ms += 1.0

    stats = profiling.time_cuda_events(
        batch,
        launches_per_batch=1,
        warmup=1,
        samples=2,
        before_sample=reset,
    )

    assert actions == ["reset", "batch"] * 3
    assert stats.min_us == stats.max_us == 1000.0


def test_cuda_profile_range_orders_markers_and_profiler_api(monkeypatch):
    fake_torch, cuda = _fake_torch()
    monkeypatch.setattr(profiling, "_load_torch", lambda: fake_torch)

    with profiling.cuda_profile_range("sgl_lora_moe::a1"):
        cuda.log.append("body")

    assert cuda.log == [
        "synchronize",
        "profiler_start",
        "push:sgl_lora_moe::a1",
        "body",
        "pop",
        "synchronize",
        "profiler_stop",
    ]


def test_cuda_profile_range_cleans_up_after_body_failure(monkeypatch):
    fake_torch, cuda = _fake_torch()
    monkeypatch.setattr(profiling, "_load_torch", lambda: fake_torch)

    with pytest.raises(RuntimeError, match="kernel failed"):
        with profiling.cuda_profile_range("sgl_lora_moe::b1"):
            raise RuntimeError("kernel failed")

    assert cuda.log[-3:] == ["pop", "synchronize", "profiler_stop"]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("mode", "torch_profiler"),
        ("execution", "piecewise"),
        ("warmup", 0),
        ("samples", True),
        ("inner_iterations", -1),
        ("profile_iterations", 0),
    ),
)
def test_run_config_rejects_invalid_controls(field, value):
    kwargs = {field: value}
    with pytest.raises(ValueError):
        profiling.RunConfig(**kwargs)


def test_helpers_reject_empty_samples_and_invalid_profile_label(monkeypatch):
    with pytest.raises(ValueError, match="must not be empty"):
        profiling.summarize_timings_us([], launches_per_batch=1)

    fake_torch, _ = _fake_torch()
    monkeypatch.setattr(profiling, "_load_torch", lambda: fake_torch)
    with pytest.raises(ValueError, match="single line"):
        with profiling.cuda_profile_range("bad\nlabel"):
            pass
