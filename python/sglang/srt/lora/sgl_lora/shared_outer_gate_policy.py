"""Host-only selection policy for shared-outer gate/up LoRA-A."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SharedOuterGateAKernel(str, Enum):
    GENERIC_VIRTUAL_EXPERT = "generic_virtual_expert"
    TOKEN_DEDUP_PAIR_OUTPUT = "token_dedup_pair_output"


@dataclass(frozen=True, slots=True)
class SharedOuterGateAKey:
    """Host-only selector key for one gate/up A site."""

    site: str
    device_family: str
    phase: str
    graph_mode: bool
    num_tokens: int
    hidden_size: int
    rank: int
    top_k: int
    has_base_rows: bool
    num_segments: int
    max_segment_len: int


@dataclass(frozen=True, slots=True)
class SharedOuterGateAPlan:
    key: SharedOuterGateAKey
    kernel: SharedOuterGateAKernel
    config_key: str
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int
    reason: str

    @property
    def uses_token_dedup(self) -> bool:
        return self.kernel is SharedOuterGateAKernel.TOKEN_DEDUP_PAIR_OUTPUT


_TOKEN_DEDUP_CONFIG = dict(
    config_key="bm16-bn32-bk64-w4",
    block_m=16,
    block_n=32,
    block_k=64,
    num_warps=4,
    num_stages=3,
)


def _device_family(device_capability: tuple[int, int] | None) -> str:
    if device_capability is None:
        return "unknown"
    major, _minor = device_capability
    if major == 9:
        return "hopper"
    if major == 10:
        return "blackwell"
    return f"sm{major}"


def build_shared_outer_gate_a_plan(
    *,
    shared_outer: bool,
    device_capability: tuple[int, int] | None,
    phase: str,
    graph_mode: bool,
    num_tokens: int,
    hidden_size: int,
    rank: int,
    top_k: int,
    has_base_rows: bool,
    num_segments: int,
    max_segment_len: int,
) -> SharedOuterGateAPlan:
    """Select the measured gate-A implementation from immutable host metadata."""

    key = SharedOuterGateAKey(
        site="gate_a",
        device_family=_device_family(device_capability),
        phase=phase,
        graph_mode=graph_mode,
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        rank=rank,
        top_k=top_k,
        has_base_rows=has_base_rows,
        num_segments=num_segments,
        max_segment_len=max_segment_len,
    )

    def generic(reason: str) -> SharedOuterGateAPlan:
        return SharedOuterGateAPlan(
            key=key,
            kernel=SharedOuterGateAKernel.GENERIC_VIRTUAL_EXPERT,
            config_key="generic",
            block_m=0,
            block_n=0,
            block_k=0,
            num_warps=0,
            num_stages=0,
            reason=reason,
        )

    if not shared_outer:
        return generic("gate/up A is not shared across routed experts")
    if key.device_family not in {"hopper", "blackwell"}:
        return generic("token dedup is measured only on Hopper and Blackwell")
    if phase != "decode":
        return generic("only decode is in the measured gate-A matrix")
    if num_tokens <= 0 or num_segments <= 0 or max_segment_len <= 0:
        return generic("empty or incomplete segment metadata")
    if top_k <= 1:
        return generic("top-k one has no repeated gate-A work to remove")
    if hidden_size not in {2048, 4096} or rank not in {64, 128}:
        return generic("hidden size or rank is outside the measured matrix")

    # The segmented grid is rectangular in [segment, max_segment_tiles].  A
    # highly skewed request mix can therefore launch far more masked rows than
    # either equal-span or all-one fragmentation.  The 16x bound retains the
    # measured one-token fragmentation case while keeping unmeasured, more
    # pathological skew on the generic route.
    grid_rows = num_segments * ((max_segment_len + 15) // 16) * 16
    if grid_rows > 16 * num_tokens:
        return generic("segment skew exceeds the measured masked-grid envelope")

    # These are the exact portable-win cells measured on both H200 and GB300.
    measured_high_rank = (
        num_tokens == 32 and hidden_size == 2048 and rank == 128 and top_k == 8
    )
    measured_large_domain = (
        num_tokens == 256
        and hidden_size == 2048
        and rank == 64
        and top_k == 8
        and not has_base_rows
    )
    measured_wide = (
        num_tokens == 32
        and hidden_size == 4096
        and rank == 64
        and top_k == 10
        and not has_base_rows
    )
    if not (measured_high_rank or measured_large_domain or measured_wide):
        return generic("shape is outside the exact cross-device win cells")

    return SharedOuterGateAPlan(
        key=key,
        kernel=SharedOuterGateAKernel.TOKEN_DEDUP_PAIR_OUTPUT,
        reason=(
            "shared gate/up A removes repeated top-k GEMMs while preserving "
            "the pair-major consumer contract in one launch"
        ),
        **_TOKEN_DEDUP_CONFIG,
    )


__all__ = [
    "SharedOuterGateAKey",
    "SharedOuterGateAKernel",
    "SharedOuterGateAPlan",
    "build_shared_outer_gate_a_plan",
]
