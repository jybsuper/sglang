"""Load-time factor transforms for the MoE-LoRA mixed-rank policy benchmark.

The source bundle models a canonical adapter artifact whose inactive rank tails
and unoccupied slots are poisoned.  Both candidates are then constructed at
load time from the same valid prefixes:

* padded: capacity-sized R_max tensors, active tails explicitly zeroed;
* packed: capacity-sized R_phys tensors for one rank bucket.

Nothing in this module is imported by serving code.  The production-neutral
static planner lives in ``sglang.srt.lora.sgl_lora.rank_policy``.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch


@dataclass(frozen=True, slots=True)
class MoeLoraFactorBundle:
    gate_up_a: torch.Tensor
    gate_up_b: torch.Tensor
    down_a: torch.Tensor
    down_b: torch.Tensor
    physical_rank: int
    setup_ms: float
    representation: str

    @property
    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (self.gate_up_a, self.gate_up_b, self.down_a, self.down_b)

    @property
    def resident_bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self.tensors)


def _validate_source(
    source: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    allocated_rank: int,
    num_slices: int,
    slot_ranks: tuple[int, ...],
) -> None:
    gate_a, gate_b, down_a, down_b = source
    if any(tensor.ndim != 4 for tensor in source):
        raise ValueError("all MoE LoRA factors must be four-dimensional")
    capacity = gate_a.shape[0]
    if len(slot_ranks) != capacity or any(
        tensor.shape[0] != capacity for tensor in source
    ):
        raise ValueError("slot_ranks and every factor must have the same capacity")
    if gate_a.shape[2] != num_slices * allocated_rank:
        raise ValueError("gate_up_a rank rows do not match slices * allocated_rank")
    if gate_b.shape[3] != allocated_rank:
        raise ValueError("gate_up_b rank columns do not match allocated_rank")
    if down_a.shape[2] != allocated_rank or down_b.shape[3] != allocated_rank:
        raise ValueError("down factor rank dimensions do not match allocated_rank")
    if (
        gate_a.device != gate_b.device
        or gate_a.device != down_a.device
        or gate_a.device != down_b.device
    ):
        raise ValueError("all factors must live on one device")
    if len({tensor.dtype for tensor in source}) != 1:
        raise ValueError("all factors must use one dtype")
    if any(rank < 0 or rank > allocated_rank for rank in slot_ranks):
        raise ValueError("slot ranks must be in [0, allocated_rank]")


def poison_canonical_factor_source(
    source: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    allocated_rank: int,
    num_slices: int,
    slot_ranks: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Retain valid logical prefixes and poison every inactive source element."""

    _validate_source(
        source,
        allocated_rank=allocated_rank,
        num_slices=num_slices,
        slot_ranks=slot_ranks,
    )
    gate_a, gate_b, down_a, down_b = source
    poisoned = tuple(torch.full_like(tensor, float("nan")) for tensor in source)
    dst_gate_a, dst_gate_b, dst_down_a, dst_down_b = poisoned
    for slot, logical_rank in enumerate(slot_ranks):
        if logical_rank == 0:
            continue
        for slice_index in range(num_slices):
            source_start = slice_index * allocated_rank
            source_end = source_start + logical_rank
            dst_gate_a[slot, :, source_start:source_end].copy_(
                gate_a[slot, :, source_start:source_end]
            )
        dst_gate_b[slot, :, :, :logical_rank].copy_(gate_b[slot, :, :, :logical_rank])
        dst_down_a[slot, :, :logical_rank].copy_(down_a[slot, :, :logical_rank])
        dst_down_b[slot, :, :, :logical_rank].copy_(down_b[slot, :, :, :logical_rank])
    return poisoned


def load_factor_rank_representation(
    canonical_source: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    allocated_rank: int,
    physical_rank: int,
    num_slices: int,
    slot_ranks: tuple[int, ...],
    representation: str,
) -> MoeLoraFactorBundle:
    """Build one zero-tailed resident representation outside forward execution."""

    _validate_source(
        canonical_source,
        allocated_rank=allocated_rank,
        num_slices=num_slices,
        slot_ranks=slot_ranks,
    )
    if physical_rank <= 0:
        raise ValueError("physical_rank must be positive")
    if any(rank > physical_rank for rank in slot_ranks):
        raise ValueError("physical_rank cannot be smaller than an active logical rank")

    gate_a, gate_b, down_a, down_b = canonical_source
    # Do not charge pending canonical-source poison/copy work or a preceding
    # candidate transform to this one-shot descriptive setup measurement.
    if gate_a.is_cuda:
        torch.cuda.synchronize(gate_a.device)
    started = perf_counter()
    # Keep inactive slots poisoned.  Any accidental route to an unoccupied slot
    # then contaminates the output instead of silently looking like zero LoRA.
    packed_gate_a = torch.full(
        (*gate_a.shape[:2], num_slices * physical_rank, gate_a.shape[3]),
        float("nan"),
        dtype=gate_a.dtype,
        device=gate_a.device,
    )
    packed_gate_b = torch.full(
        (*gate_b.shape[:3], physical_rank),
        float("nan"),
        dtype=gate_b.dtype,
        device=gate_b.device,
    )
    packed_down_a = torch.full(
        (*down_a.shape[:2], physical_rank, down_a.shape[3]),
        float("nan"),
        dtype=down_a.dtype,
        device=down_a.device,
    )
    packed_down_b = torch.full(
        (*down_b.shape[:3], physical_rank),
        float("nan"),
        dtype=down_b.dtype,
        device=down_b.device,
    )
    for slot, logical_rank in enumerate(slot_ranks):
        if logical_rank == 0:
            continue
        # Zero physical tails before copying the logical prefixes.  This is the
        # only valid representation for kernels that execute R_phys rather than
        # consume per-slot logical ranks.
        packed_gate_a[slot].zero_()
        packed_gate_b[slot].zero_()
        packed_down_a[slot].zero_()
        packed_down_b[slot].zero_()
        for slice_index in range(num_slices):
            source_start = slice_index * allocated_rank
            destination_start = slice_index * physical_rank
            packed_gate_a[
                slot,
                :,
                destination_start : destination_start + logical_rank,
            ].copy_(
                gate_a[
                    slot,
                    :,
                    source_start : source_start + logical_rank,
                ]
            )
        packed_gate_b[slot, :, :, :logical_rank].copy_(
            gate_b[slot, :, :, :logical_rank]
        )
        packed_down_a[slot, :, :logical_rank].copy_(down_a[slot, :, :logical_rank])
        packed_down_b[slot, :, :, :logical_rank].copy_(
            down_b[slot, :, :, :logical_rank]
        )
    if gate_a.is_cuda:
        torch.cuda.synchronize(gate_a.device)
    setup_ms = (perf_counter() - started) * 1e3
    return MoeLoraFactorBundle(
        gate_up_a=packed_gate_a,
        gate_up_b=packed_gate_b,
        down_a=packed_down_a,
        down_b=packed_down_b,
        physical_rank=physical_rank,
        setup_ms=setup_ms,
        representation=representation,
    )


def bind_factor_bundle(fixture, bundle: MoeLoraFactorBundle) -> None:
    """Bind a load-time bundle to a benchmark fixture before warmup/capture."""

    if fixture.lora_info is None:
        raise ValueError("cannot bind LoRA factors to a base-only fixture")
    fixture.lora_weights = bundle.tensors
    fixture.lora_info.gate_up_lora_a_weights = bundle.gate_up_a
    fixture.lora_info.gate_up_lora_b_weights = bundle.gate_up_b
    fixture.lora_info.down_lora_a_weights = bundle.down_a
    fixture.lora_info.down_lora_b_weights = bundle.down_b
    fixture.lora_info.max_lora_rank = bundle.physical_rank


def logical_factor_bytes(
    canonical_source: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    allocated_rank: int,
    num_slices: int,
    slot_ranks: tuple[int, ...],
) -> int:
    """Count valid factor bytes, excluding slot and rank padding."""

    _validate_source(
        canonical_source,
        allocated_rank=allocated_rank,
        num_slices=num_slices,
        slot_ranks=slot_ranks,
    )
    gate_a, gate_b, down_a, down_b = canonical_source
    elements = 0
    for rank in slot_ranks:
        if rank == 0:
            continue
        elements += gate_a.shape[1] * num_slices * rank * gate_a.shape[3]
        elements += gate_b.shape[1] * gate_b.shape[2] * rank
        elements += down_a.shape[1] * rank * down_a.shape[3]
        elements += down_b.shape[1] * down_b.shape[2] * rank
    return elements * gate_a.element_size()


__all__ = [
    "MoeLoraFactorBundle",
    "bind_factor_bundle",
    "load_factor_rank_representation",
    "logical_factor_bytes",
    "poison_canonical_factor_source",
]
