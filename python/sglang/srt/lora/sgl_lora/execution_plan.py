"""Host-side execution planning for the SGL-LoRA MoE pipeline.

The planner consumes explicit orchestration metadata.  In particular, a token
count never decides whether a forward is decode or prefill: that distinction is
resolved from :class:`ForwardMode` before the model runs.  The returned plan is
immutable and therefore safe to close over while recording a CUDA graph.

The first production policy is deliberately evidence-bounded:

* captured decode through 128 tokens uses complete C2, optionally with the
  measured C3 gate-A overlap when the two-stream feature is requested;
* larger captured decode and every prefill/extend use the scalable C2 partial
  tail rather than the large-T C2F/C3 finalizer;
* eager decode uses C2F only through the portable measured range, then C2P;
* unsupported phases/ranks retain the established C0 serial fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from sglang.srt.lora.sgl_lora.shared_outer_gate_policy import (
    SharedOuterGateAPlan,
    build_shared_outer_gate_a_plan,
)


class MoeLoraExecutionPath(str, Enum):
    C0_SERIAL = "c0_serial"
    C2_PARTIAL = "c2_partial"
    C2_FULL = "c2_full"
    C3_OVERLAP = "c3_overlap"


@dataclass(frozen=True, slots=True)
class MoeLoraExecutionPlan:
    path: MoeLoraExecutionPath
    phase: str
    graph_mode: bool
    has_base_rows: bool
    consumer_schedule: str = "aligned"
    consumer_block_size_n: int = 64
    consumer_num_warps: int = 4
    finalize_block_size_h: int = 32
    finalize_num_warps: int = 4
    provider_key: str = "deepgemm_bf16"
    shared_outer_gate_a_plan: SharedOuterGateAPlan | None = None
    reason: str = ""

    @property
    def uses_side_stream(self) -> bool:
        return self.path is MoeLoraExecutionPath.C3_OVERLAP


def classify_forward_phase(forward_mode) -> str:
    """Normalize a ForwardMode-like object without importing model code.

    ``is_decode`` and ``is_extend`` are the orchestration contract.  Keeping
    this helper duck-typed also makes the phase boundary independently testable.
    """
    if forward_mode is not None and forward_mode.is_decode():
        return "decode"
    if forward_mode is not None and forward_mode.is_extend(
        include_draft_extend_v2=True
    ):
        return "prefill"
    return "other"


def resolve_static_has_base_rows(
    *,
    weight_indices: Sequence[int],
    lora_ranks: Sequence[int],
    graph_mode: bool,
    capture_variant: str | None,
) -> bool:
    """Resolve the optional base-row activation work without a device scan.

    A single production ``lora`` graph currently serves both all-adapter and
    mixed adapter/base batches.  Its capture must therefore retain the base-row
    fill.  Eager calls use the exact host-side rank assignment.  A future graph
    key that distinguishes all-active and mixed variants can pass the exact
    value here without changing any kernel ABI.
    """
    if graph_mode and capture_variant == "lora":
        return True
    if not weight_indices:
        return True
    return any(lora_ranks[index] <= 0 for index in weight_indices)


def _consumer_schedule(
    *, phase: str, graph_mode: bool, num_tokens: int, rank: int, has_base_rows: bool
) -> str:
    # Captured C3/C2F is kept on the exact aligned schedule used by the final
    # whole-pipeline matrix.  For eager/prefill mixed rows, pair ownership avoids
    # the charged aligned base-only prepass where cross-model measurements found
    # it materially slower.  All-active prefill keeps aligned route reuse.
    if graph_mode and phase == "decode" and num_tokens <= 128:
        return "aligned"
    if has_base_rows and (num_tokens >= 8 or (rank <= 16 and num_tokens > 1)):
        return "pair"
    return "aligned"


def _finalizer_config(rank: int) -> tuple[int, int]:
    # Portable rank tiers from the H200/GB300 fused-finalizer sweep.
    if rank <= 32:
        return 64, 4
    if rank <= 64:
        return 64, 2
    return 32, 4


def build_moe_lora_execution_plan(
    *,
    phase: str,
    graph_mode: bool,
    num_tokens: int,
    rank: int,
    has_base_rows: bool,
    two_stream_requested: bool,
    fused_supported: bool = True,
    base_lora_expert_domains_match: bool = True,
    provider_key: str = "deepgemm_bf16",
    shared_outer: bool = False,
    device_capability: tuple[int, int] | None = None,
    hidden_size: int = 0,
    top_k: int = 0,
    num_segments: int = 0,
    max_segment_len: int = 0,
) -> MoeLoraExecutionPlan:
    """Return one immutable production plan for a resolved forward shape."""
    if phase not in {"decode", "prefill", "other"}:
        raise ValueError(f"unknown SGL-LoRA forward phase {phase!r}")
    if num_tokens < 0:
        raise ValueError("num_tokens must be nonnegative")

    schedule = _consumer_schedule(
        phase=phase,
        graph_mode=graph_mode,
        num_tokens=num_tokens,
        rank=rank,
        has_base_rows=has_base_rows,
    )
    finalizer_block_h, finalizer_warps = _finalizer_config(rank)
    shared_outer_gate_a_plan = build_shared_outer_gate_a_plan(
        shared_outer=shared_outer,
        device_capability=device_capability,
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

    common = dict(
        phase=phase,
        graph_mode=graph_mode,
        has_base_rows=has_base_rows,
        consumer_schedule=schedule,
        consumer_block_size_n=(16 if num_tokens <= 1 and rank <= 16 else 64),
        consumer_num_warps=4,
        finalize_block_size_h=finalizer_block_h,
        finalize_num_warps=finalizer_warps,
        provider_key=provider_key,
        shared_outer_gate_a_plan=shared_outer_gate_a_plan,
    )

    if not base_lora_expert_domains_match:
        return MoeLoraExecutionPlan(
            path=MoeLoraExecutionPath.C0_SERIAL,
            reason=(
                "physical base-expert IDs include shared slots while routed LoRA "
                "factors exclude them; mapped C2/C3 consumers are not implemented"
            ),
            **common,
        )

    if not fused_supported:
        return MoeLoraExecutionPlan(
            path=MoeLoraExecutionPath.C0_SERIAL,
            reason=(
                f"{provider_key} uses the provider-neutral serial topology; "
                "the fused tail is BF16-specific"
            ),
            **common,
        )

    if phase == "other" or rank <= 0 or rank > 128:
        return MoeLoraExecutionPlan(
            path=MoeLoraExecutionPath.C0_SERIAL,
            reason="outside the measured fused BF16 phase/rank envelope",
            **common,
        )

    if phase == "prefill":
        return MoeLoraExecutionPlan(
            path=MoeLoraExecutionPath.C2_PARTIAL,
            reason="prefill/extend uses the scalable serial C2 partial tail",
            **common,
        )

    if graph_mode:
        if num_tokens <= 128:
            path = (
                MoeLoraExecutionPath.C3_OVERLAP
                if two_stream_requested
                else MoeLoraExecutionPath.C2_FULL
            )
            reason = (
                "captured decode in the measured C3 overlap range"
                if two_stream_requested
                else "captured decode uses portable complete serial C2"
            )
            return MoeLoraExecutionPlan(path=path, reason=reason, **common)
        return MoeLoraExecutionPlan(
            path=MoeLoraExecutionPath.C2_PARTIAL,
            reason="captured decode above 128 avoids the large-T fused finalizer",
            **common,
        )

    if num_tokens <= 256:
        return MoeLoraExecutionPlan(
            path=MoeLoraExecutionPath.C2_FULL,
            reason="eager decode uses complete C2 only in the portable range",
            **common,
        )
    return MoeLoraExecutionPlan(
        path=MoeLoraExecutionPath.C2_PARTIAL,
        reason="large eager decode uses the scalable serial C2 partial tail",
        **common,
    )


__all__ = [
    "MoeLoraExecutionPath",
    "MoeLoraExecutionPlan",
    "build_moe_lora_execution_plan",
    "classify_forward_phase",
    "resolve_static_has_base_rows",
]
