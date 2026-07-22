"""Static rank plans for MoE LoRA factor storage and CUDA-graph families.

The serving pool allocates adapter slots at ``R_max``, while each adapter has a
logical rank ``R``.  These are different contracts:

* ``padded_rmax`` keeps one resident tensor family at ``R_max`` and requires
  every slot's inactive rank tail to be zero;
* ``packed_bucket`` packs factors at adapter publication time into one resident
  tensor family per physical-rank bucket.

This module only resolves immutable policy metadata.  It deliberately does not
allocate CUDA tensors, inspect a live request, or select a policy from benchmark
results.  A graph owner can therefore use :attr:`MoeLoraRankPlan.graph_key`
before capture and replay the selected topology without per-forward Python
objects or rank-metadata kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

MoeLoraRankPolicy = Literal["padded_rmax", "packed_bucket"]


def resolve_moe_lora_physical_rank(
    logical_rank: int,
    *,
    alignment: int = 16,
    minimum: int = 16,
) -> int:
    """Resolve tensor-core/provider padding without changing logical rank."""

    if type(logical_rank) is not int or logical_rank <= 0:
        raise ValueError("logical_rank must be a positive integer")
    if type(alignment) is not int or alignment <= 0:
        raise ValueError("alignment must be a positive integer")
    if type(minimum) is not int or minimum <= 0:
        raise ValueError("minimum must be a positive integer")
    required = max(logical_rank, minimum)
    return ((required + alignment - 1) // alignment) * alignment


@dataclass(frozen=True, slots=True)
class MoeLoraRankBucket:
    """One load-time resident factor family with one physical rank."""

    physical_rank: int
    adapter_slots: tuple[int, ...]
    logical_ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.physical_rank <= 0:
            raise ValueError("physical_rank must be positive")
        if not self.adapter_slots:
            raise ValueError("a rank bucket must contain at least one adapter slot")
        if len(self.adapter_slots) != len(self.logical_ranks):
            raise ValueError("adapter_slots and logical_ranks must have equal length")
        if tuple(sorted(self.adapter_slots)) != self.adapter_slots:
            raise ValueError("adapter_slots must be sorted")
        if len(set(self.adapter_slots)) != len(self.adapter_slots):
            raise ValueError("adapter_slots must be unique")
        if any(rank <= 0 or rank > self.physical_rank for rank in self.logical_ranks):
            raise ValueError("logical ranks must be in (0, physical_rank]")


@dataclass(frozen=True, slots=True)
class MoeLoraRankPlan:
    """Immutable adapter-publish and graph-family rank plan.

    ``slot_ranks`` uses zero for an empty adapter slot.  Base-only requests are
    represented by adapter id ``-1`` elsewhere and never occupy a rank bucket.
    ``slot_to_bucket`` is a precomputed tuple lookup; ``-1`` means an empty
    slot.  All tuple metadata is built once with the plan, not during replay.
    """

    policy: MoeLoraRankPolicy
    allocated_rank: int
    slot_ranks: tuple[int, ...]
    buckets: tuple[MoeLoraRankBucket, ...]
    slot_to_bucket: tuple[int, ...]
    graph_key: tuple[object, ...]
    rank_alignment: int
    minimum_physical_rank: int
    runtime_metadata_allocations: int = 0

    @property
    def has_active_adapters(self) -> bool:
        return bool(self.buckets)

    @property
    def active_adapter_count(self) -> int:
        return sum(rank > 0 for rank in self.slot_ranks)

    @property
    def physical_ranks(self) -> tuple[int, ...]:
        return tuple(bucket.physical_rank for bucket in self.buckets)


def build_moe_lora_rank_plan(
    slot_ranks: Sequence[int],
    *,
    allocated_rank: int,
    policy: MoeLoraRankPolicy,
    rank_alignment: int = 16,
    minimum_physical_rank: int = 16,
) -> MoeLoraRankPlan:
    """Build a static rank plan at adapter publication or graph setup time.

    A packed plan groups slots by *physical* rank, so logical ranks 8 and 12
    may share an R16 resident family while retaining their distinct logical
    tails.  A padded plan always has at most one R_max family.
    """

    if policy not in ("padded_rmax", "packed_bucket"):
        raise ValueError(f"unsupported MoE LoRA rank policy {policy!r}")
    if type(allocated_rank) is not int or allocated_rank <= 0:
        raise ValueError("allocated_rank must be a positive integer")
    if type(rank_alignment) is not int or rank_alignment <= 0:
        raise ValueError("rank_alignment must be a positive integer")
    if type(minimum_physical_rank) is not int or minimum_physical_rank <= 0:
        raise ValueError("minimum_physical_rank must be a positive integer")
    if not slot_ranks:
        raise ValueError("slot_ranks must contain the configured adapter slots")

    ranks = tuple(slot_ranks)
    for rank in ranks:
        if type(rank) is not int or rank < 0 or rank > allocated_rank:
            raise ValueError("each slot rank must be an integer in [0, allocated_rank]")

    active_slots = tuple(index for index, rank in enumerate(ranks) if rank > 0)
    buckets: list[MoeLoraRankBucket] = []
    if active_slots and policy == "padded_rmax":
        physical_rank = resolve_moe_lora_physical_rank(
            allocated_rank,
            alignment=rank_alignment,
            minimum=minimum_physical_rank,
        )
        buckets.append(
            MoeLoraRankBucket(
                physical_rank=physical_rank,
                adapter_slots=active_slots,
                logical_ranks=tuple(ranks[index] for index in active_slots),
            )
        )
    elif active_slots:
        slots_by_physical_rank: dict[int, list[int]] = {}
        for index in active_slots:
            physical_rank = resolve_moe_lora_physical_rank(
                ranks[index],
                alignment=rank_alignment,
                minimum=minimum_physical_rank,
            )
            slots_by_physical_rank.setdefault(physical_rank, []).append(index)
        for physical_rank in sorted(slots_by_physical_rank):
            slots = tuple(slots_by_physical_rank[physical_rank])
            buckets.append(
                MoeLoraRankBucket(
                    physical_rank=physical_rank,
                    adapter_slots=slots,
                    logical_ranks=tuple(ranks[index] for index in slots),
                )
            )

    slot_to_bucket = [-1] * len(ranks)
    for bucket_index, bucket in enumerate(buckets):
        for slot in bucket.adapter_slots:
            slot_to_bucket[slot] = bucket_index
    frozen_buckets = tuple(buckets)
    # Include resident slots as well as ranks: a captured route plan indexes the
    # physical pool by slot and cannot silently reuse a graph after eviction
    # publishes a different slot/rank assignment.
    if frozen_buckets:
        graph_key: tuple[object, ...] = (
            "moe_lora_rank_plan_v1",
            policy,
            allocated_rank,
            ranks,
            tuple(
                (bucket.physical_rank, bucket.adapter_slots, bucket.logical_ranks)
                for bucket in frozen_buckets
            ),
        )
    else:
        # No-LoRA execution bypasses rank-shaped factor kernels entirely. Do
        # not multiply graph families by an irrelevant storage policy/R_max.
        graph_key = ("moe_lora_rank_plan_v1", "no_lora")
    return MoeLoraRankPlan(
        policy=policy,
        allocated_rank=allocated_rank,
        slot_ranks=ranks,
        buckets=frozen_buckets,
        slot_to_bucket=tuple(slot_to_bucket),
        graph_key=graph_key,
        rank_alignment=rank_alignment,
        minimum_physical_rank=minimum_physical_rank,
    )


__all__ = [
    "MoeLoraRankBucket",
    "MoeLoraRankPlan",
    "MoeLoraRankPolicy",
    "build_moe_lora_rank_plan",
    "resolve_moe_lora_physical_rank",
]
