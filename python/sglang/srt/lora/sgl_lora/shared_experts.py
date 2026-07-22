"""Explicit expert-ID contracts for model shared experts and routed LoRA.

Model shared experts are base-model experts that are always selected. They are
not LoRA factors and must never be folded into the virtual-expert ID space used
by routed-expert LoRA. Most local standard-dispatch paths place shared slots
after the local routed experts, so the existing range check is sufficient.
DeepEP/MegaMOE global physical IDs are different: shared slots are interleaved
per EP rank and the routed IDs after the first rank contain gaps.

This module builds the one load/attach-time lookup table needed by those
non-contiguous layouts. The table maps the provider's incoming physical ID to
the routed LoRA factor index, with ``-1`` for shared or non-local slots. The
virtual-ID kernel consumes it directly, avoiding a layout-conversion launch on
every forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

ExpertIdLayout = Literal[
    "global_contiguous",
    "global_per_rank_shared",
    "local_contiguous",
]
FactorDomain = Literal["global", "local"]


@dataclass(frozen=True, slots=True, kw_only=True)
class MoeLoraExpertTopology:
    """Resolved model/provider expert identity for one MoE layer.

    ``num_routed_experts`` is always the global logical routed count. A local
    factor buffer contains ``num_routed_experts / ep_size`` entries. Shared
    slots are deliberately excluded from both factor domains.
    """

    num_routed_experts: int
    num_fused_shared_experts: int
    ep_size: int = 1
    ep_rank: int = 0
    id_layout: ExpertIdLayout = "global_contiguous"
    factor_domain: FactorDomain = "global"

    def __post_init__(self) -> None:
        if self.num_routed_experts <= 0:
            raise ValueError("num_routed_experts must be positive")
        if self.num_fused_shared_experts < 0:
            raise ValueError("num_fused_shared_experts must be non-negative")
        if self.ep_size <= 0:
            raise ValueError("ep_size must be positive")
        if not 0 <= self.ep_rank < self.ep_size:
            raise ValueError("ep_rank must be in [0, ep_size)")
        if self.num_routed_experts % self.ep_size:
            raise ValueError("num_routed_experts must be divisible by ep_size")
        if self.id_layout == "local_contiguous" and self.factor_domain != "local":
            raise ValueError("local_contiguous IDs require local routed factors")

    @property
    def num_local_routed_experts(self) -> int:
        return self.num_routed_experts // self.ep_size

    @property
    def num_factor_experts(self) -> int:
        if self.factor_domain == "local":
            return self.num_local_routed_experts
        return self.num_routed_experts

    @property
    def num_incoming_experts(self) -> int:
        if self.id_layout == "local_contiguous":
            return self.num_local_routed_experts + self.num_fused_shared_experts
        if self.id_layout == "global_per_rank_shared":
            return (
                self.num_routed_experts + self.ep_size * self.num_fused_shared_experts
            )
        return self.num_routed_experts + self.num_fused_shared_experts


def build_routed_expert_id_map(
    topology: MoeLoraExpertTopology,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Map provider physical expert IDs to routed-only LoRA factor IDs.

    The result is static for a layer/provider topology and should be retained
    with the layer, not rebuilt in ``forward``. ``-1`` entries identify model
    shared experts or routed experts that are not present in a local factor
    buffer.
    """

    result = torch.full(
        (topology.num_incoming_experts,),
        -1,
        dtype=torch.int32,
        device=device,
    )
    local_routed = topology.num_local_routed_experts

    if topology.id_layout == "local_contiguous":
        result[:local_routed] = torch.arange(
            local_routed, dtype=torch.int32, device=device
        )
        return result

    if topology.id_layout == "global_contiguous":
        routed = torch.arange(
            topology.num_routed_experts, dtype=torch.int32, device=device
        )
        if topology.factor_domain == "global":
            result[: topology.num_routed_experts] = routed
        else:
            start = topology.ep_rank * local_routed
            result[start : start + local_routed] = torch.arange(
                local_routed, dtype=torch.int32, device=device
            )
        return result

    # DeepEP/MegaMOE physical order:
    # [rank0 routed..., rank0 shared..., rank1 routed..., rank1 shared..., ...].
    physical_per_rank = local_routed + topology.num_fused_shared_experts
    for rank in range(topology.ep_size):
        physical_start = rank * physical_per_rank
        if topology.factor_domain == "global":
            factor_start = rank * local_routed
        elif rank == topology.ep_rank:
            factor_start = 0
        else:
            continue
        result[physical_start : physical_start + local_routed] = torch.arange(
            factor_start,
            factor_start + local_routed,
            dtype=torch.int32,
            device=device,
        )
    return result


__all__ = [
    "ExpertIdLayout",
    "FactorDomain",
    "MoeLoraExpertTopology",
    "build_routed_expert_id_map",
]
