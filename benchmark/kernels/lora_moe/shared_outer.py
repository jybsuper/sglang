"""Reference algebra for MoE adapters with shared outer LoRA factors.

This module is deliberately benchmark-only.  It records the two identities
that make ``experts_shared_outer_loras`` interesting without prescribing a
production kernel or route representation:

* gate/up A is shared across experts, so ``x_t @ A_l.T`` is token/adapter
  owned and may be computed once before being consumed by every top-k slot;
* down B is shared across experts, so routed rank vectors may be weighted and
  reduced before the single ``B_l`` multiplication.

The repeated-pair functions are controls matching today's logical work.  The
factorized functions are FP32 oracles for future kernels.  Expert IDs are not
arguments because a shared outer factor is selected only by adapter; local-EP
ownership has already been reflected in the pair inputs to down B.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class AdapterSpan:
    """One contiguous token span owned by an adapter.

    ``adapter_id=None`` represents base-only rows.  Runtime requests are
    already segmented by adapter in the normal LoRA batch metadata, so spans
    let the benchmark express adapter grouping without measuring a synthetic
    sort.
    """

    start: int
    stop: int
    adapter_id: int | None

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("AdapterSpan requires 0 <= start < stop")
        if self.adapter_id is not None and self.adapter_id < 0:
            raise ValueError("adapter_id must be nonnegative or None")


def contiguous_adapter_spans(
    *,
    num_tokens: int,
    active_adapters: int,
    include_base: bool,
) -> tuple[AdapterSpan, ...]:
    """Partition tokens into deterministic contiguous adapter/base spans."""

    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    if active_adapters < 0:
        raise ValueError("active_adapters must be nonnegative")
    group_count = active_adapters + int(include_base)
    if group_count <= 0:
        raise ValueError("at least one active adapter or a base span is required")
    if group_count > num_tokens:
        raise ValueError("every requested adapter/base group needs at least one token")

    quotient, remainder = divmod(num_tokens, group_count)
    spans: list[AdapterSpan] = []
    start = 0
    for group_id in range(group_count):
        width = quotient + int(group_id < remainder)
        stop = start + width
        adapter_id = group_id if group_id < active_adapters else None
        spans.append(AdapterSpan(start, stop, adapter_id))
        start = stop
    return tuple(spans)


def token_lora_mapping_from_spans(
    spans: tuple[AdapterSpan, ...], *, device: torch.device | str | None = None
) -> torch.Tensor:
    """Build the production ``-1``-for-base token mapping for ``spans``."""

    if not spans or spans[0].start != 0:
        raise ValueError("spans must be nonempty and start at token zero")
    for left, right in zip(spans, spans[1:]):
        if left.stop != right.start:
            raise ValueError("spans must be contiguous and non-overlapping")
    mapping = torch.full((spans[-1].stop,), -1, dtype=torch.int32, device=device)
    for span in spans:
        if span.adapter_id is not None:
            mapping[span.start : span.stop] = span.adapter_id
    return mapping


def _logical_shared_factor(
    factor: torch.Tensor, *, name: str, expected_rank: int
) -> torch.Tensor:
    """Accept logical ``[L,...]`` or stored ``[L,1,...]`` shared factors."""

    if factor.ndim == expected_rank + 1:
        if factor.shape[1] != 1:
            raise ValueError(f"{name} stored expert dimension must equal 1")
        factor = factor[:, 0]
    if factor.ndim != expected_rank:
        raise ValueError(f"{name} has unexpected rank {factor.ndim}")
    return factor


def _validate_spans(
    spans: tuple[AdapterSpan, ...], *, num_tokens: int, capacity: int
) -> None:
    if not spans or spans[0].start != 0 or spans[-1].stop != num_tokens:
        raise ValueError("spans must cover every token exactly")
    for index, span in enumerate(spans):
        if index and spans[index - 1].stop != span.start:
            raise ValueError("spans must be contiguous and non-overlapping")
        if span.adapter_id is not None and span.adapter_id >= capacity:
            raise ValueError("span adapter_id exceeds factor capacity")


def gate_up_a_repeated_pair_reference(
    hidden_states: torch.Tensor,
    shared_a: torch.Tensor,
    spans: tuple[AdapterSpan, ...],
    *,
    top_k: int,
) -> torch.Tensor:
    """FP32 control that recomputes shared gate/up A for every top-k slot."""

    shared_a = _logical_shared_factor(shared_a, name="shared_a", expected_rank=3)
    if hidden_states.ndim != 2 or shared_a.shape[-1] != hidden_states.shape[-1]:
        raise ValueError("hidden_states/shared_a must be [T,H] and [L,N,H]")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    _validate_spans(
        spans, num_tokens=hidden_states.shape[0], capacity=shared_a.shape[0]
    )

    output = torch.zeros(
        hidden_states.shape[0],
        top_k,
        shared_a.shape[-2],
        dtype=torch.float32,
        device=hidden_states.device,
    )
    x = hidden_states.float()
    a = shared_a.float()
    for span in spans:
        if span.adapter_id is None:
            continue
        for topk_slot in range(top_k):
            output[span.start : span.stop, topk_slot] = (
                x[span.start : span.stop] @ a[span.adapter_id].T
            )
    return output


def gate_up_a_deduplicated_reference(
    hidden_states: torch.Tensor,
    shared_a: torch.Tensor,
    spans: tuple[AdapterSpan, ...],
    *,
    top_k: int,
    materialize_pairs: bool = True,
) -> torch.Tensor:
    """FP32 token/adapter-owned gate/up A oracle.

    With ``materialize_pairs=False`` the result is ``[T,N]``.  A future gate-B
    consumer can use that token-owned representation directly.  ``True``
    broadcasts it to today's ``[T,K,N]`` pair contract.
    """

    shared_a = _logical_shared_factor(shared_a, name="shared_a", expected_rank=3)
    if hidden_states.ndim != 2 or shared_a.shape[-1] != hidden_states.shape[-1]:
        raise ValueError("hidden_states/shared_a must be [T,H] and [L,N,H]")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    _validate_spans(
        spans, num_tokens=hidden_states.shape[0], capacity=shared_a.shape[0]
    )

    token_output = torch.zeros(
        hidden_states.shape[0],
        shared_a.shape[-2],
        dtype=torch.float32,
        device=hidden_states.device,
    )
    x = hidden_states.float()
    a = shared_a.float()
    for span in spans:
        if span.adapter_id is None:
            continue
        token_output[span.start : span.stop] = (
            x[span.start : span.stop] @ a[span.adapter_id].T
        )
    if not materialize_pairs:
        return token_output
    return token_output[:, None, :].expand(-1, top_k, -1).clone()


def down_b_repeated_pair_reference(
    pair_rank: torch.Tensor,
    shared_b: torch.Tensor,
    topk_weights: torch.Tensor,
    spans: tuple[AdapterSpan, ...],
) -> torch.Tensor:
    """FP32 control: apply shared down B to every pair, then weighted-sum."""

    shared_b = _logical_shared_factor(shared_b, name="shared_b", expected_rank=3)
    if pair_rank.ndim != 3 or topk_weights.shape != pair_rank.shape[:2]:
        raise ValueError("pair_rank/topk_weights must be [T,K,R] and [T,K]")
    if shared_b.shape[-1] != pair_rank.shape[-1]:
        raise ValueError("shared_b rank does not match pair_rank")
    _validate_spans(spans, num_tokens=pair_rank.shape[0], capacity=shared_b.shape[0])

    output = torch.zeros(
        pair_rank.shape[0],
        shared_b.shape[-2],
        dtype=torch.float32,
        device=pair_rank.device,
    )
    rank = pair_rank.float()
    weights = topk_weights.float()
    b = shared_b.float()
    for span in spans:
        if span.adapter_id is None:
            continue
        for topk_slot in range(pair_rank.shape[1]):
            contribution = (
                rank[span.start : span.stop, topk_slot] @ b[span.adapter_id].T
            )
            output[span.start : span.stop] += (
                contribution * weights[span.start : span.stop, topk_slot, None]
            )
    return output


def down_b_weighted_rank_reduction_reference(
    pair_rank: torch.Tensor,
    shared_b: torch.Tensor,
    topk_weights: torch.Tensor,
    spans: tuple[AdapterSpan, ...],
) -> torch.Tensor:
    """FP32 oracle: weighted-reduce rank vectors, then apply shared down B."""

    shared_b = _logical_shared_factor(shared_b, name="shared_b", expected_rank=3)
    if pair_rank.ndim != 3 or topk_weights.shape != pair_rank.shape[:2]:
        raise ValueError("pair_rank/topk_weights must be [T,K,R] and [T,K]")
    if shared_b.shape[-1] != pair_rank.shape[-1]:
        raise ValueError("shared_b rank does not match pair_rank")
    _validate_spans(spans, num_tokens=pair_rank.shape[0], capacity=shared_b.shape[0])

    reduced_rank = torch.sum(
        pair_rank.float() * topk_weights.float().unsqueeze(-1), dim=1
    )
    output = torch.zeros(
        pair_rank.shape[0],
        shared_b.shape[-2],
        dtype=torch.float32,
        device=pair_rank.device,
    )
    b = shared_b.float()
    for span in spans:
        if span.adapter_id is None:
            continue
        output[span.start : span.stop] = (
            reduced_rank[span.start : span.stop] @ b[span.adapter_id].T
        )
    return output


def arithmetic_work(
    *, active_tokens: int, top_k: int, hidden_size: int, rank: int, slices: int = 2
) -> dict[str, int | float]:
    """Return exact multiply counts for the two shared-outer identities."""

    for name, value in (
        ("active_tokens", active_tokens),
        ("top_k", top_k),
        ("hidden_size", hidden_size),
        ("rank", rank),
        ("slices", slices),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    gate_token = active_tokens * slices * rank * hidden_size
    gate_pair = gate_token * top_k
    down_b_once = active_tokens * hidden_size * rank
    down_b_pair = down_b_once * top_k
    down_rank_reduce = active_tokens * top_k * rank
    return {
        "gate_a_repeated_pair_multiplies": gate_pair,
        "gate_a_deduplicated_multiplies": gate_token,
        "gate_a_gemm_reduction_ratio": float(top_k),
        "down_b_repeated_pair_multiplies": down_b_pair,
        "down_b_weighted_rank_reduce_multiplies": down_b_once + down_rank_reduce,
        "down_b_gemm_reduction_ratio": float(top_k),
    }


__all__ = [
    "AdapterSpan",
    "arithmetic_work",
    "contiguous_adapter_spans",
    "down_b_repeated_pair_reference",
    "down_b_weighted_rank_reduction_reference",
    "gate_up_a_deduplicated_reference",
    "gate_up_a_repeated_pair_reference",
    "token_lora_mapping_from_spans",
]
