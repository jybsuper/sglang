"""Small, dependency-free records for MoE-LoRA benchmarks.

One :class:`MoeLoraBenchCase` is one fully resolved run. Candidate sets live in
``matrix.py``; timing/reporting controls live in the future benchmark drivers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Phase = Literal["decode", "prefill"]
Device = Literal["h200", "gb300"]
Scope = Literal["K0", "O0", "M0"]
Pipeline = Literal["N0", "C0", "C1", "C2", "C3", "C4", "C5"]


def _positive(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelShape:
    key: str
    h_model: int
    h_moe: int
    intermediate_size: int
    num_experts: int
    top_k: int
    num_slices: int
    activation: str
    moe_layers: int

    def __post_init__(self) -> None:
        for name in (
            "h_model",
            "h_moe",
            "intermediate_size",
            "num_experts",
            "top_k",
            "num_slices",
            "moe_layers",
        ):
            _positive(name, getattr(self, name))
        if self.top_k > self.num_experts:
            raise ValueError("top_k cannot exceed num_experts")


@dataclass(frozen=True, slots=True, kw_only=True)
class AdapterBatch:
    """Adapter occupancy and rank for one run.

    ``l_active`` excludes base-only rows. ``b_base`` is exactly 0 or 1, and
    ``l_capacity`` is the configured ``max_loras_per_batch`` including base.
    """

    l_active: int
    b_base: int
    l_capacity: int
    rank: int
    max_rank: int
    physical_rank: int
    shared_outer: bool = False

    def __post_init__(self) -> None:
        if type(self.l_active) is not int or self.l_active < 0:
            raise ValueError("l_active must be a nonnegative integer")
        if type(self.b_base) is not int or self.b_base not in (0, 1):
            raise ValueError("b_base must be exactly 0 or 1")
        _positive("l_capacity", self.l_capacity)
        if self.l_active + self.b_base == 0:
            raise ValueError("a case needs LoRA rows or base-only rows")
        if self.l_active + self.b_base > self.l_capacity:
            raise ValueError("l_active + b_base cannot exceed l_capacity")
        for name in ("rank", "max_rank", "physical_rank"):
            _positive(name, getattr(self, name))
        if not self.rank <= self.physical_rank <= self.max_rank:
            raise ValueError("rank <= physical_rank <= max_rank is required")

    @property
    def l_groups(self) -> int:
        return self.l_active + self.b_base


@dataclass(frozen=True, slots=True, kw_only=True)
class FactorShapes:
    """Local allocated factor shapes: ``[capacity, experts, rows, cols]``."""

    gate_up_a: tuple[int, int, int, int]
    gate_up_b: tuple[int, int, int, int]
    down_a: tuple[int, int, int, int]
    down_b: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True, kw_only=True)
class MoeLoraBenchCase:
    case_id: str
    model: ModelShape
    adapters: AdapterBatch
    t_local: int
    phase: Phase
    device: Device
    provider: str
    scope: Scope
    stage: str
    pipeline: Pipeline
    graph_mode: Literal["eager", "cuda_graph"]
    routing: str
    cache_state: Literal["cold", "hot", "producer"]
    tp_size: int = 1
    ep_size: int = 1
    moe_dp_size: int = 1
    ep_rank: int = 0
    intermediate_alignment: int = 1

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must not be empty")
        for name in (
            "t_local",
            "tp_size",
            "ep_size",
            "moe_dp_size",
            "intermediate_alignment",
        ):
            _positive(name, getattr(self, name))
        if self.tp_size % (self.ep_size * self.moe_dp_size):
            raise ValueError("tp_size must be divisible by ep_size * moe_dp_size")
        if not 0 <= self.ep_rank < self.ep_size:
            raise ValueError("ep_rank must be in [0, ep_size)")
        if self.model.num_experts % self.ep_size:
            raise ValueError("num_experts must be divisible by ep_size")
        if self.model.intermediate_size % self.moe_tp_size:
            raise ValueError("intermediate_size must be divisible by moe_tp_size")

    @property
    def moe_tp_size(self) -> int:
        return self.tp_size // (self.ep_size * self.moe_dp_size)

    @property
    def e_local(self) -> int:
        return self.model.num_experts // self.ep_size

    @property
    def i_local(self) -> int:
        return self.model.intermediate_size // self.moe_tp_size

    @property
    def i_physical(self) -> int:
        alignment = self.intermediate_alignment
        return ((self.i_local + alignment - 1) // alignment) * alignment

    @property
    def pair_capacity(self) -> int:
        return self.t_local * self.model.top_k

    @property
    def global_expert_offset(self) -> int:
        return self.ep_rank * self.e_local

    @property
    def factor_shapes(self) -> FactorShapes:
        a_experts = 1 if self.adapters.shared_outer else self.e_local
        b_experts = 1 if self.adapters.shared_outer else self.e_local
        capacity = self.adapters.l_capacity
        rank = self.adapters.max_rank
        slices = self.model.num_slices
        return FactorShapes(
            gate_up_a=(capacity, a_experts, slices * rank, self.model.h_moe),
            gate_up_b=(capacity, self.e_local, slices * self.i_physical, rank),
            down_a=(capacity, self.e_local, rank, self.i_physical),
            down_b=(capacity, b_experts, self.model.h_moe, rank),
        )
