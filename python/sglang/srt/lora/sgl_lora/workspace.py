"""Runner-owned buffers and stream state for the SGL MoE-LoRA backend.

The workspace is deliberately policy-free.  A validated execution plan tells
the runner which buffers and overlap windows it needs; this object only makes
their addresses stable across CUDA-graph replay and owns the stream/event
objects used by those windows.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

import torch

_T = TypeVar("_T")


class MoeLoraWorkspace:
    """Address-stable tensors plus reusable side streams and events.

    Buffers are cached by semantic name *and exact tensor contract*.  Capture
    buckets may therefore coexist without reusing a tensor with the wrong
    shape.  Missing state is never created from inside a CUDA capture: callers
    must warm the selected plan and bucket first.
    """

    def __init__(self) -> None:
        self._graph_buffers: dict[
            tuple[str, tuple[int, ...], torch.dtype, torch.device], torch.Tensor
        ] = {}
        self._eager_buffers: dict[
            tuple[str, torch.dtype, torch.device], torch.Tensor
        ] = {}
        self._streams: dict[tuple[torch.device, int], torch.cuda.Stream] = {}
        self._events: dict[tuple[torch.device, str], torch.cuda.Event] = {}
        self._graph_mode = False

    def begin_forward(self, *, graph_mode: bool) -> None:
        """Select bounded eager reuse or exact graph-bucket retention.

        Eager forwards keep only the largest flat allocation per semantic
        buffer name, so changing prefill lengths cannot grow a server-lifetime
        shape cache. CUDA graphs retain exact warmed shapes because replacing
        an allocation would invalidate an older captured graph.
        """
        self._graph_mode = bool(graph_mode)

    @staticmethod
    def _capturing(device: torch.device) -> bool:
        return device.type == "cuda" and torch.cuda.is_current_stream_capturing()

    def tensor(
        self,
        name: str,
        shape: Sequence[int],
        *,
        dtype: torch.dtype,
        device: torch.device | str,
        zero: bool = False,
        initialize_zero: bool = False,
        zero_on_first_allocation: bool = False,
    ) -> torch.Tensor:
        """Return reusable storage under one of two explicit zeroing contracts.

        ``initialize_zero`` rezeros a reused eager view on every lookup.
        ``zero_on_first_allocation`` zeros only newly allocated or grown
        storage, for buffers whose owning kernel restores its own invariant.
        """
        if initialize_zero and zero_on_first_allocation:
            raise ValueError(
                "initialize_zero and zero_on_first_allocation are mutually exclusive"
            )
        resolved_device = torch.device(device)
        resolved_shape = tuple(int(dim) for dim in shape)
        zero_new_storage = initialize_zero or zero_on_first_allocation
        if self._graph_mode:
            key = (name, resolved_shape, dtype, resolved_device)
            tensor = self._graph_buffers.get(key)
            if tensor is None:
                if self._capturing(resolved_device):
                    raise RuntimeError(
                        "SGL MoE-LoRA workspace was not warmed before CUDA capture: "
                        f"missing {name!r} {resolved_shape} {dtype} on "
                        f"{resolved_device}"
                    )
                factory = torch.zeros if zero_new_storage else torch.empty
                tensor = factory(resolved_shape, dtype=dtype, device=resolved_device)
                self._graph_buffers[key] = tensor
        else:
            key = (name, dtype, resolved_device)
            elements = 1
            for dimension in resolved_shape:
                elements *= dimension
            storage = self._eager_buffers.get(key)
            if storage is None or storage.numel() < elements:
                if self._capturing(resolved_device):
                    raise RuntimeError(
                        "an eager SGL MoE-LoRA workspace cannot grow inside "
                        "CUDA capture"
                    )
                factory = torch.zeros if zero_new_storage else torch.empty
                storage = factory((elements,), dtype=dtype, device=resolved_device)
                self._eager_buffers[key] = storage
            tensor = storage[:elements].view(resolved_shape)
            if initialize_zero:
                tensor.zero_()
        if zero:
            tensor.zero_()
        return tensor

    def side_stream(
        self, device: torch.device | str, *, priority: int = 0
    ) -> torch.cuda.Stream:
        resolved_device = torch.device(device)
        if resolved_device.type != "cuda":
            raise ValueError("a CUDA side stream requires a CUDA device")
        key = (resolved_device, int(priority))
        stream = self._streams.get(key)
        if stream is None:
            if self._capturing(resolved_device):
                raise RuntimeError(
                    "SGL MoE-LoRA side stream was not created before CUDA capture"
                )
            stream = torch.cuda.Stream(device=resolved_device, priority=priority)
            self._streams[key] = stream
        return stream

    def event(self, device: torch.device | str, name: str) -> torch.cuda.Event:
        resolved_device = torch.device(device)
        if resolved_device.type != "cuda":
            raise ValueError("a CUDA event requires a CUDA device")
        key = (resolved_device, name)
        event = self._events.get(key)
        if event is None:
            if self._capturing(resolved_device):
                raise RuntimeError(
                    f"SGL MoE-LoRA event was not created before CUDA capture: {name}"
                )
            event = torch.cuda.Event()
            self._events[key] = event
        return event

    def warm_parallel_region(
        self,
        device: torch.device | str,
        name: str,
        *,
        priority: int = 0,
    ) -> None:
        """Create and materialize resources required by one overlap window."""
        resolved_device = torch.device(device)
        self.side_stream(resolved_device, priority=priority)
        current = torch.cuda.current_stream(resolved_device)
        for event_name in (f"{name}:ready", f"{name}:done"):
            event = self.event(resolved_device, event_name)
            # torch.cuda.Event allocates its CUDA handle lazily on first
            # record. Merely constructing the Python object before capture is
            # insufficient: first-record allocation inside capture can fail.
            if event.cuda_event == 0:
                event.record(current)

    @property
    def allocated_bytes(self) -> int:
        graph_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in self._graph_buffers.values()
        )
        eager_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in self._eager_buffers.values()
        )
        return graph_bytes + eager_bytes


def run_parallel(
    workspace: MoeLoraWorkspace,
    *,
    name: str,
    device: torch.device,
    compute: Callable[[], _T],
    side: Callable[[], object],
    side_priority: int = 0,
) -> _T:
    """Run one explicit fork/join region without host synchronization.

    CPU fixtures execute the same dependency order sequentially.  CUDA uses a
    runner-owned side stream and two reusable events; the event operations are
    capture-safe once :meth:`MoeLoraWorkspace.warm_parallel_region` has run.
    Both closures and their tensors remain strongly referenced through the
    final current-stream wait. That complete join is the lifetime contract:
    side-stream work never escapes this helper, so allocator ``record_stream``
    calls are neither required nor hidden here.
    """
    if device.type != "cuda":
        side()
        return compute()

    current = torch.cuda.current_stream(device)
    side_stream = workspace.side_stream(device, priority=side_priority)
    ready = workspace.event(device, f"{name}:ready")
    done = workspace.event(device, f"{name}:done")

    ready.record(current)
    side_stream.wait_event(ready)
    with torch.cuda.stream(side_stream):
        side()
        done.record(side_stream)
    result = compute()
    current.wait_event(done)
    return result
