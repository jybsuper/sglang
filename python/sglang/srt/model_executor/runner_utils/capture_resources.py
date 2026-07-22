"""Lifetime ownership for Python objects referenced by captured CUDA work."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_ACTIVE_CAPTURE_RESOURCES: ContextVar[list[Any] | None] = ContextVar(
    "active_cuda_graph_capture_resources", default=None
)


@contextmanager
def cuda_graph_capture_resource_scope() -> Iterator[list[Any]]:
    """Collect resources whose lifetime must match one captured graph."""
    resources: list[Any] = []
    token = _ACTIVE_CAPTURE_RESOURCES.set(resources)
    try:
        yield resources
    finally:
        _ACTIVE_CAPTURE_RESOURCES.reset(token)


def keep_cuda_graph_capture_resource(resource: Any) -> None:
    """Bind ``resource`` to the currently recording graph's owner."""
    resources = _ACTIVE_CAPTURE_RESOURCES.get()
    if resources is None:
        raise RuntimeError(
            "CUDA graph capture resource has no active owner; record through a "
            "CUDA graph backend or cuda_graph_capture_resource_scope()"
        )
    resources.append(resource)
