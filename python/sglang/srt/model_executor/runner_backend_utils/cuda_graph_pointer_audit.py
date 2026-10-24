# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Debug audit of CUDA graph operands against PyTorch allocation lifetimes.

SGLANG_DEBUG_CUDA_GRAPH_POINTERS retains raw graphs and inspects their
kernel/memcpy/memset operands via cuda-python. Allocations made during a
captured forward from its graphs' pools are graph-owned temporaries; other
tracked operands must remain alive. Incomplete inspection fails closed.

Device-side pointer tables, non-PyTorch allocations and stale-but-live metadata
are not inspected, so a clean audit is not numerical validation. Tracing
continues between sessions without stacks; later frees are checked by the
next session or checkpoint().
"""

from __future__ import annotations

import ctypes
import gc
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

from sglang.srt.environ import envs

try:
    from cuda.bindings import driver as cuda_drv
except ImportError:  # pragma: no cover - audit is unavailable without cuda-python
    cuda_drv = None

logger = logging.getLogger(__name__)

_MARKER_BYTES = 512
_MAX_TRACE_ENTRIES = 2_000_000
_REPORT_FRAMES = 5
_REPORT_LIMIT = 16
# CU_LAUNCH_PARAM_* tags of the ``extra`` launch-argument array.
_LAUNCH_PARAM_END = 0
_LAUNCH_PARAM_BUFFER_POINTER = 1
_LAUNCH_PARAM_BUFFER_SIZE = 2
_CUDA_ERROR_INVALID_VALUE = 1


def enabled() -> bool:
    return envs.SGLANG_DEBUG_CUDA_GRAPH_POINTERS.get()


class InspectionError(RuntimeError):
    """A captured graph could not be inspected completely."""


def _check(result):
    err = result[0]
    if int(err) != 0:
        raise InspectionError(f"CUDA driver call failed: {err}")
    return result[1] if len(result) == 2 else result[1:]


def _kernel_name(func) -> str:
    try:
        return _check(cuda_drv.cuFuncGetName(func)).decode(errors="replace")
    except Exception:
        return f"func@{int(func):#x}"


def _words(raw: bytes) -> Iterator[Tuple[int, int]]:
    for offset in range(0, len(raw) - 7, 8):
        yield offset, int.from_bytes(raw[offset : offset + 8], "little")


def _kernel_node_operands(node) -> Tuple[str, List[Tuple[str, int]]]:
    """Kernel name and every 8-byte word of its launch arguments."""
    params = _check(cuda_drv.cuGraphKernelNodeGetParams(node))
    func = params.func
    if int(func) == 0 and int(params.kern) != 0:
        func = _check(cuda_drv.cuKernelGetFunction(params.kern))
    if int(func) == 0:
        raise InspectionError("kernel node without a function handle")
    words: List[Tuple[str, int]] = []
    kernel_params = int(params.kernelParams)
    if kernel_params:
        index = 0
        while True:
            info = cuda_drv.cuFuncGetParamInfo(func, index)
            if int(info[0]) == _CUDA_ERROR_INVALID_VALUE:
                break  # past the last parameter
            if int(info[0]) != 0:
                raise InspectionError(
                    f"cuFuncGetParamInfo({_kernel_name(func)}, {index}) failed: "
                    f"{info[0]}"
                )
            size = int(info[2])
            arg = ctypes.c_void_p.from_address(kernel_params + 8 * index).value
            if arg and size >= 8:
                for offset, word in _words(ctypes.string_at(arg, size)):
                    words.append((f"arg{index}+{offset}", word))
            index += 1
    elif int(params.extra):
        extra = int(params.extra)
        buffer = size = 0
        for slot in range(16):
            tag = ctypes.c_void_p.from_address(extra + 16 * slot).value or 0
            value = ctypes.c_void_p.from_address(extra + 16 * slot + 8).value or 0
            if tag == _LAUNCH_PARAM_END:
                break
            if tag == _LAUNCH_PARAM_BUFFER_POINTER:
                buffer = value
            elif tag == _LAUNCH_PARAM_BUFFER_SIZE and value:
                size = ctypes.c_size_t.from_address(value).value
        if not (buffer and size):
            raise InspectionError(
                f"kernel {_kernel_name(func)} uses an 'extra' argument buffer "
                "the audit could not locate"
            )
        for offset, word in _words(ctypes.string_at(buffer, size)):
            words.append((f"extra+{offset}", word))
    return _kernel_name(func), words


def _is_device_memory(memory_type) -> bool:
    return memory_type in (
        cuda_drv.CUmemorytype.CU_MEMORYTYPE_DEVICE,
        cuda_drv.CUmemorytype.CU_MEMORYTYPE_UNIFIED,
    )


def _graph_operands(raw_graph: int, operands: Dict[int, List[str]]) -> int:
    """Collect operands recursively and count nodes; reject uninspectable types."""
    _, count = _check(cuda_drv.cuGraphGetNodes(raw_graph, 0))
    nodes, _ = _check(cuda_drv.cuGraphGetNodes(raw_graph, count))
    node_type = cuda_drv.CUgraphNodeType
    total = len(nodes)
    for node in nodes:
        kind = _check(cuda_drv.cuGraphNodeGetType(node))
        if kind == node_type.CU_GRAPH_NODE_TYPE_KERNEL:
            name, words = _kernel_node_operands(node)
            for slot, word in words:
                operands.setdefault(word, []).append(f"{name} {slot}")
        elif kind == node_type.CU_GRAPH_NODE_TYPE_MEMCPY:
            params = _check(cuda_drv.cuGraphMemcpyNodeGetParams(node))
            if _is_device_memory(params.srcMemoryType):
                operands.setdefault(int(params.srcDevice), []).append("memcpy src")
            if _is_device_memory(params.dstMemoryType):
                operands.setdefault(int(params.dstDevice), []).append("memcpy dst")
        elif kind == node_type.CU_GRAPH_NODE_TYPE_MEMSET:
            params = _check(cuda_drv.cuGraphMemsetNodeGetParams(node))
            operands.setdefault(int(params.dst), []).append("memset dst")
        elif kind == node_type.CU_GRAPH_NODE_TYPE_GRAPH:
            child = _check(cuda_drv.cuGraphChildGraphNodeGetGraph(node))
            total += _graph_operands(int(child), operands)
        elif kind in (
            node_type.CU_GRAPH_NODE_TYPE_EMPTY,
            node_type.CU_GRAPH_NODE_TYPE_WAIT_EVENT,
            node_type.CU_GRAPH_NODE_TYPE_EVENT_RECORD,
        ):
            continue
        else:
            raise InspectionError(f"node type {kind} is not inspected")
    return total


def _pool_of(entry: dict) -> Optional[tuple]:
    pool = entry.get("pool_id")
    return None if pool is None else tuple(pool)


class _Lifetimes:
    """Per-block trace intervals: alloc=-1 predates tracing; free=None is live."""

    def __init__(self, traces: List[dict]):
        self.last_alloc: Dict[int, int] = {}
        open_blocks: Dict[int, int] = {}
        lifetimes: Dict[int, List[Tuple[int, Optional[int], int, Optional[tuple]]]] = {}
        for index, entry in enumerate(traces):
            action = entry["action"]
            if action == "alloc":
                open_blocks[entry["addr"]] = index
                self.last_alloc[entry["addr"]] = index
            elif action == "free_requested":
                addr = entry["addr"]
                start = open_blocks.pop(addr, -1)
                lifetimes.setdefault(addr, []).append(
                    (start, index, entry["size"], _pool_of(entry))
                )
        for addr, start in open_blocks.items():
            entry = traces[start]
            lifetimes.setdefault(addr, []).append(
                (start, None, entry["size"], _pool_of(entry))
            )
        self.lifetimes = lifetimes
        self.bases = np.array(sorted(lifetimes), dtype=np.uint64)
        self.max_end = np.array(
            [
                int(base) + max(size for _, _, size, _ in lifetimes[int(base)])
                for base in self.bases
            ],
            dtype=np.uint64,
        )
        self.prefix_max_end = (
            np.maximum.accumulate(self.max_end) if len(self.max_end) else self.max_end
        )

    def covering(
        self, pointer: int
    ) -> List[Tuple[int, int, Optional[int], int, Optional[tuple]]]:
        """Return (base, alloc, free, size, pool) intervals containing pointer."""
        if not len(self.bases):
            return []
        key = np.uint64(pointer)
        last = int(np.searchsorted(self.bases, key, side="right")) - 1
        if last < 0:
            return []
        first = int(np.searchsorted(self.prefix_max_end, key, side="right"))
        if first > last:
            return []
        candidates = np.nonzero(self.max_end[first : last + 1] > key)[0] + first
        out = []
        for position in candidates:
            base = int(self.bases[position])
            for start, end, size, pool in self.lifetimes[base]:
                if base + size > pointer:
                    out.append((base, start, end, size, pool))
        return out


@dataclass
class _GraphRecord:
    label: str
    owner: int
    graph: Any  # torch.cuda.CUDAGraph captured with keep_graph=True
    forward_begin: torch.Tensor  # markers: their alloc events bound the windows
    begin: torch.Tensor
    end: torch.Tensor
    pools: Optional[frozenset]  # allocator pools the captured graphs own
    nodes: int = 0
    operands: Dict[int, List[str]] = field(default_factory=dict)
    scanned: bool = False
    failure: Optional[str] = None


@dataclass
class Finding:
    graph: str
    kind: str
    pointer: int
    base: int
    size: int
    alloc_idx: int
    free_idx: int
    capture_end: int
    slots: List[str]


@dataclass
class Findings:
    graphs: int = 0
    nodes: int = 0
    pointers: int = 0
    static: int = 0
    temporaries: int = 0
    intermediates: int = 0
    outlived: int = 0
    untracked: int = 0
    incomplete: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    dangling: List[Finding] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.failed:
            return "failed"
        if self.dangling:
            return "dangling"
        if self.incomplete:
            return "incomplete"
        return "clean"


def _format_frames(entry: Optional[dict]) -> str:
    frames = (entry or {}).get("frames") or []
    shown = []
    for frame in frames[:_REPORT_FRAMES]:
        filename = str(frame.get("filename", "")).split("/")
        shown.append(
            f"{'/'.join(filename[-2:])}:{frame.get('line')} {frame.get('name')}"
        )
    return " <- ".join(shown) if shown else "(no stack recorded)"


class CudaGraphPointerAudit:
    """Per-device graph records, retained until their owner retires them."""

    def __init__(self, device: torch.device):
        if cuda_drv is None:
            raise RuntimeError(
                "SGLANG_DEBUG_CUDA_GRAPH_POINTERS needs cuda-python (cuda.bindings)"
            )
        try:
            torch.cuda.CUDAGraph(keep_graph=True)
        except TypeError as e:
            raise RuntimeError(
                "SGLANG_DEBUG_CUDA_GRAPH_POINTERS needs torch.cuda.CUDAGraph(keep_graph=True)"
            ) from e
        self.device = device
        self.records: List[_GraphRecord] = []
        self._recording = False
        self._session_start = 0
        self._capture: Optional[dict] = None
        self._traces: List[dict] = []

    # -- allocator trace -----------------------------------------------------

    def _record_history(self, context: Optional[str]) -> None:
        torch.cuda.memory._record_memory_history(
            enabled="all",
            context=context,
            stacks="python",
            max_entries=_MAX_TRACE_ENTRIES,
            device=self.device,
        )
        self._recording = True

    def begin_session(self) -> None:
        self._session_start = len(self.records)
        self._record_history(context="all")

    def end_session(self) -> Findings:
        """Analyze every live record; raise unless the inspection is clean."""
        findings = self.analyze()
        if self.records:
            # Keep tracing between sessions, without stacks, for checkpoint().
            self._record_history(context=None)
        else:
            self.close()
        self._log(findings)
        if findings.status != "clean":
            raise RuntimeError(self.report(findings))
        return findings

    def abandon_session(self) -> None:
        """Discard this failed session; keep tracing any earlier graphs."""
        self._capture = None
        del self.records[self._session_start :]
        if self.records:
            self._record_history(context=None)
        else:
            self.close()

    def close(self) -> None:
        self.records.clear()
        self._capture = None
        if self._recording:
            torch.cuda.memory._record_memory_history(enabled=None, device=self.device)
            self._recording = False

    def retire(self, owner: Any) -> None:
        """Drop the records of a backend that reset its graphs."""
        self.records = [r for r in self.records if r.owner != id(owner)]
        if not self.records:
            self.close()

    # -- capture bookkeeping ---------------------------------------------------

    def mark(self) -> torch.Tensor:
        """A kept allocation whose trace entry pins one end of a capture window."""
        return torch.empty(_MARKER_BYTES, dtype=torch.uint8, device=self.device)

    def new_graph(self) -> torch.cuda.CUDAGraph:
        return torch.cuda.CUDAGraph(keep_graph=True)

    @property
    def capturing(self) -> bool:
        return self._capture is not None

    def begin_capture(self, owner: Any, label: str) -> None:
        """Start a forward containing begin_graph()/add_graph() segments."""
        assert self._capture is None, "a captured forward is already open"
        self._capture = {
            "owner": id(owner),
            "label": label,
            "forward_begin": self.mark(),
            "begin": None,
            "pending": [],
        }

    def begin_graph(self) -> None:
        assert self._capture is not None, "begin_graph() outside begin_capture()"
        self._capture["begin"] = self.mark()

    def add_graph(self, graph: torch.cuda.CUDAGraph, instantiate: bool = True) -> None:
        """Register a captured graph; disable instantiate if the caller does it."""
        capture = self._capture
        assert capture is not None, "add_graph() outside begin_capture()"
        begin = capture["begin"]
        if begin is None:
            begin = capture["forward_begin"]
        capture["begin"] = None
        capture["pending"].append((graph, begin, self.mark(), instantiate))

    def cancel_capture(self) -> None:
        """The captured forward failed; forget its graphs."""
        self._capture = None

    def end_capture(self) -> None:
        capture = self._capture
        assert capture is not None, "end_capture() without begin_capture()"
        self._capture = None
        pools: Optional[frozenset] = None
        try:
            pools = frozenset(
                tuple(graph.pool()) for graph, _, _, _ in capture["pending"]
            )
        except Exception:  # a graph whose pool cannot be read: nothing is exempt
            pass
        for index, (graph, begin, end, instantiate) in enumerate(capture["pending"]):
            if instantiate:
                graph.instantiate()
            label = capture["label"]
            if len(capture["pending"]) > 1:
                label = f"{label} segment={index}"
            self.records.append(
                _GraphRecord(
                    label,
                    capture["owner"],
                    graph,
                    capture["forward_begin"],
                    begin,
                    end,
                    pools,
                )
            )

    # -- analysis --------------------------------------------------------------

    def analyze(self) -> Findings:
        torch.cuda.synchronize(self.device)
        # Capture-time garbage (objects dropped by the last capture) must reach
        # the trace before it is read.
        gc.collect()
        torch.cuda.synchronize(self.device)
        traces = torch.cuda.memory._snapshot(self.device)["device_traces"][
            self.device.index
        ]
        self._traces = traces
        lifetimes = _Lifetimes(traces)
        findings = Findings()
        if len(traces) >= _MAX_TRACE_ENTRIES - 1:
            findings.incomplete.append(
                f"allocator trace ring is full ({len(traces)} entries); "
                "older captures may be misattributed"
            )
        seen: set = set()
        unknown_pools: set = set()
        for record in self.records:
            if not record.scanned:
                record.scanned = True
                try:
                    record.nodes = _graph_operands(
                        record.graph.raw_cuda_graph(), record.operands
                    )
                except InspectionError as e:
                    record.failure = str(e)
            if record.failure is not None:
                findings.failed.append(f"{record.label}: {record.failure}")
                continue
            forward_begin = lifetimes.last_alloc.get(record.forward_begin.data_ptr())
            begin = lifetimes.last_alloc.get(record.begin.data_ptr())
            end = lifetimes.last_alloc.get(record.end.data_ptr())
            if forward_begin is None or begin is None or end is None:
                findings.incomplete.append(
                    f"{record.label}: capture markers are no longer in the allocator trace"
                )
                continue
            findings.graphs += 1
            findings.nodes += record.nodes
            for pointer, slots in record.operands.items():
                findings.pointers += 1
                overlapping = 0
                for base, start, free, size, pool in lifetimes.covering(pointer):
                    if start > end or (free is not None and free < begin):
                        continue  # not alive during this graph's capture
                    overlapping += 1
                    if start >= forward_begin:
                        # Allocated during the captured forward: only the pool
                        # the captured graphs own holds graph-owned memory.
                        if pool is None or record.pools is None:
                            unknown_pools.add((record.label, base))
                            continue
                        if pool in record.pools:
                            if start < begin:
                                findings.intermediates += 1
                            elif free is None or free <= end:
                                findings.temporaries += 1
                            else:
                                findings.outlived += 1
                            continue
                        kind = (
                            "allocated outside the graph pool during capture"
                            if start >= begin
                            else "eager bridge allocated during the captured forward"
                        )
                    else:
                        kind = "allocated before capture"
                    if free is None:
                        findings.static += 1
                        continue
                    key = (record.label, base, start, free)
                    if key in seen:
                        continue
                    seen.add(key)
                    findings.dangling.append(
                        Finding(
                            record.label,
                            kind,
                            pointer,
                            base,
                            size,
                            start,
                            free,
                            end,
                            slots,
                        )
                    )
                if not overlapping:
                    findings.untracked += 1
        for label, base in sorted(unknown_pools):
            findings.incomplete.append(
                f"{label}: operand lifetime at {base:#x} has no allocator pool information"
            )
        return findings

    def _log(self, findings: Findings) -> None:
        logger.info(
            "CUDA graph pointer audit (%s): %d graph(s), %d node(s), %d pointer(s): "
            "%d static, %d temporaries, %d earlier-segment intermediates, "
            "%d outlived capture, %d untracked, %d dangling, %d incomplete, %d failed",
            findings.status,
            findings.graphs,
            findings.nodes,
            findings.pointers,
            findings.static,
            findings.temporaries,
            findings.intermediates,
            findings.outlived,
            findings.untracked,
            len(findings.dangling),
            len(findings.incomplete),
            len(findings.failed),
        )

    def report(self, findings: Findings) -> str:
        traces = self._traces
        lines = [
            f"CUDA graph pointer audit {findings.status}: {len(findings.dangling)} "
            "captured graph operand(s) point at memory freed while the graph is "
            f"alive, {len(findings.incomplete)} incomplete and {len(findings.failed)} "
            f"failed inspection(s) (graphs={findings.graphs}, nodes={findings.nodes}, "
            f"pointers={findings.pointers})."
        ]
        for finding in findings.dangling[:_REPORT_LIMIT]:
            when = "after" if finding.free_idx > finding.capture_end else "during"
            alloc = traces[finding.alloc_idx] if finding.alloc_idx >= 0 else None
            free = traces[finding.free_idx]
            origin = (
                f"{finding.kind} (trace #{finding.alloc_idx}, pool {alloc.get('pool_id')})"
                if alloc is not None
                else "allocated before the audit started recording"
            )
            lines.append(
                f"- graph {finding.graph!r}: {finding.pointer:#x} in block "
                f"{finding.base:#x} ({finding.size} B), {origin}, freed {when} "
                f"capture (trace #{finding.free_idx}); read by {len(finding.slots)} "
                f"operand(s), e.g. {finding.slots[0][:160]}"
            )
            lines.append(f"    allocated at: {_format_frames(alloc)}")
            lines.append(f"    freed at: {_format_frames(free)}")
        if len(findings.dangling) > _REPORT_LIMIT:
            lines.append(f"- ... {len(findings.dangling) - _REPORT_LIMIT} more")
        for reason in findings.incomplete[:_REPORT_LIMIT]:
            lines.append(f"- incomplete: {reason}")
        for reason in findings.failed[:_REPORT_LIMIT]:
            lines.append(f"- failed: {reason}")
        return "\n".join(lines)


_audits: Dict[int, CudaGraphPointerAudit] = {}
_active: Optional[CudaGraphPointerAudit] = None


def active() -> Optional[CudaGraphPointerAudit]:
    """Current capture-session audit, or None."""
    return _active


def shape_label(runner: Any, shape_key: Any) -> str:
    parts = [type(runner).__name__, f"size={getattr(shape_key, 'size', shape_key)}"]
    for name in ("variant_label", "attention_variant"):
        value = getattr(shape_key, name, None)
        if value is not None:
            parts.append(f"{name}={value}")
    return " ".join(parts)


@contextmanager
def session() -> Iterator[Optional[CudaGraphPointerAudit]]:
    """Audit a capture loop, failing closed or discarding an aborted session."""
    global _active
    if not enabled():
        yield None
        return
    if not torch.cuda.is_available():
        logger.warning("SGLANG_DEBUG_CUDA_GRAPH_POINTERS is CUDA-only; audit skipped")
        yield None
        return
    if _active is not None:
        yield _active  # nested capture loop on the same device
        return
    index = torch.cuda.current_device()
    audit = _audits.get(index)
    if audit is None:
        audit = _audits[index] = CudaGraphPointerAudit(torch.device("cuda", index))
    audit.begin_session()
    _active = audit
    try:
        yield audit
    except BaseException:
        audit.abandon_session()
        raise
    finally:
        _active = None
    audit.end_session()


def retire(owner: Any) -> None:
    """A graph backend reset its graphs: forget their records."""
    for audit in _audits.values():
        audit.retire(owner)


def shutdown() -> None:
    """Drop every audit and stop tracing (tests)."""
    global _active
    for audit in _audits.values():
        audit.close()
    _audits.clear()
    _active = None


def checkpoint(reason: str) -> Optional[Findings]:
    """Recheck this device's live graphs, e.g. after eager workspace growth."""
    if not _audits:
        return None
    audit = _audits.get(torch.cuda.current_device())
    if audit is None or not audit.records or audit.capturing:
        return None
    findings = audit.analyze()
    audit._log(findings)
    if findings.status != "clean":
        raise RuntimeError(f"{reason}: {audit.report(findings)}")
    return findings
