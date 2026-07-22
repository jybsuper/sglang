"""Workspace estimation and admission for the SGL LoRA MoE pipeline.

The BF16 provider uses masked DeepGEMM tensors whose leading shape is
``[E_local, m_max]``.  That is inexpensive at decode sizes but can consume many
GiB for a long prefill.  Keep the accounting in one provider-facing planner so
the runner does not grow token/rank assertions and future quant providers can
replace the estimate with their own workspace contract.
"""

from __future__ import annotations

from collections.abc import Callable

import msgspec
import torch

_MIB = 1 << 20


class MoeLoraWorkspaceEstimate(msgspec.Struct, frozen=True, kw_only=True):
    """Incremental bytes owned by one local MoE-LoRA invocation."""

    num_tokens: int
    m_max: int
    eager_peak_bytes: int
    capture_peak_bytes: int
    routing_bytes: int

    def peak_bytes(self, *, capture: bool) -> int:
        return self.capture_peak_bytes if capture else self.eager_peak_bytes


def _tensor_bytes(*shape: int, element_size: int) -> int:
    elements = 1
    for dim in shape:
        elements *= dim
    return elements * element_size


def _routing_plan_bytes(
    *,
    num_tokens: int,
    top_k: int,
    num_local_experts: int,
    max_loras: int,
    block_m: int,
) -> int:
    """Conservative bytes for one materialized aligned virtual-expert plan."""
    pairs = num_tokens * top_k
    buckets = max(1, num_local_experts * max_loras)
    nonempty = min(pairs, buckets)
    padded_pairs = pairs + nonempty * (block_m - 1)
    padded_pairs = ((padded_pairs + block_m - 1) // block_m) * block_m

    # virtual_topk_ids + sorted_token_ids + expert_ids + per-token bool mask,
    # plus the count/cumsum scratch used by the native align implementation.
    return (
        _tensor_bytes(pairs, element_size=4)
        + _tensor_bytes(padded_pairs, element_size=4)
        + _tensor_bytes(padded_pairs // block_m, element_size=4)
        + _tensor_bytes(num_tokens, element_size=1)
        + _tensor_bytes(buckets + 2, element_size=4)
        + 4
    )


def estimate_bf16_moe_lora_workspace(
    *,
    num_tokens: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    rank: int,
    num_local_experts: int,
    max_loras: int,
    element_size: int = 2,
) -> MoeLoraWorkspaceEstimate:
    """Estimate the live incremental storage of the current BF16 pipeline.

    ``capture_peak_bytes`` intentionally assumes that disposal is disabled by
    the graph allocator and therefore sums all invocation-owned buffers.
    ``eager_peak_bytes`` follows the explicit disposal points in the runner.
    The estimate excludes caller-owned inputs, base/LoRA weights, and outputs
    that already existed before entering the runner.
    """
    if (
        min(
            num_tokens,
            top_k,
            hidden_size,
            intermediate_size,
            rank,
            num_local_experts,
            max_loras,
        )
        <= 0
    ):
        raise ValueError("workspace dimensions must all be positive")

    # This is the exact policy in moe_ep_deepgemm_preprocess.  It deliberately
    # advances to the next 256-row bucket even when T is already aligned.
    m_max = (num_tokens // 256 + 1) * 256
    pair_rows = num_tokens * top_k
    expert_rows = num_local_experts * m_max

    gate_delta = _tensor_bytes(
        pair_rows, 2 * intermediate_size, element_size=element_size
    )
    gate_rank = _tensor_bytes(pair_rows, 2 * rank, element_size=element_size)
    down_rank = _tensor_bytes(pair_rows, rank, element_size=element_size)
    activation_bridge = _tensor_bytes(
        pair_rows, intermediate_size, element_size=element_size
    )
    hidden_permuted = _tensor_bytes(expert_rows, hidden_size, element_size=element_size)
    gateup_out = _tensor_bytes(
        expert_rows, 2 * intermediate_size, element_size=element_size
    )
    act_out = _tensor_bytes(expert_rows, intermediate_size, element_size=element_size)
    down_out = _tensor_bytes(expert_rows, hidden_size, element_size=element_size)
    output = _tensor_bytes(num_tokens, hidden_size, element_size=element_size)

    # Gate-A, gate-B, down-A and down-B can use distinct cache keys.  The
    # block-16 plan is the larger padding case and safely bounds the current
    # block-16/32/64 schedules.  Shared-outer plans have fewer buckets.
    routing_bytes = 4 * _routing_plan_bytes(
        num_tokens=num_tokens,
        top_k=top_k,
        num_local_experts=num_local_experts,
        max_loras=max_loras,
        block_m=16,
    )

    persistent_gate = routing_bytes + gate_delta + gate_rank
    eager_peak = max(
        # S1 + S2: permuted input and W13 output coexist.
        persistent_gate + hidden_permuted + gateup_out,
        # S3: W13 output, activation output and down-A source coexist.
        persistent_gate + gateup_out + act_out + activation_bridge,
        # S4: gate buffers have been disposed before W2 output is allocated.
        routing_bytes + act_out + activation_bridge + down_out,
        # S5 and the down-LoRA tail.
        routing_bytes + activation_bridge + down_out + output,
        routing_bytes + activation_bridge + down_rank + output,
    )

    capture_peak = (
        routing_bytes
        + gate_delta
        + gate_rank
        + down_rank
        + activation_bridge
        + hidden_permuted
        + gateup_out
        + act_out
        + down_out
        + output
    )
    return MoeLoraWorkspaceEstimate(
        num_tokens=num_tokens,
        m_max=m_max,
        eager_peak_bytes=eager_peak,
        capture_peak_bytes=capture_peak,
        routing_bytes=routing_bytes,
    )


class MoeLoraWorkspacePlanner:
    """Shared device-aware admission cache for sequential LoRA MoE layers."""

    def __init__(
        self,
        *,
        memory_info: Callable[[torch.device], tuple[int, int]] | None = None,
        reserve_fraction: float = 0.02,
        minimum_reserve_bytes: int = 512 * _MIB,
    ) -> None:
        if not 0 <= reserve_fraction < 1:
            raise ValueError("reserve_fraction must be in [0, 1)")
        if minimum_reserve_bytes < 0:
            raise ValueError("minimum_reserve_bytes must be nonnegative")
        self._memory_info = memory_info or self._cuda_memory_info
        self._reserve_fraction = reserve_fraction
        self._minimum_reserve_bytes = minimum_reserve_bytes
        self._admitted: set[tuple] = set()
        self._memory_snapshots: dict[tuple[str, int | None], tuple[int, int]] = {}

    @staticmethod
    def _cuda_memory_info(device: torch.device) -> tuple[int, int]:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        # CUDA reports allocator-reserved blocks as unavailable even though
        # PyTorch can immediately reuse their unallocated portion.
        reclaimable = max(
            0,
            torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device),
        )
        return min(total_bytes, free_bytes + reclaimable), total_bytes

    def admit(
        self,
        *,
        estimate: MoeLoraWorkspaceEstimate,
        device: torch.device,
        capture: bool,
        geometry_key: tuple,
        memory_query_safe: bool = True,
    ) -> None:
        """Admit one shape or fail before any large pipeline buffer is allocated."""
        key = (device.type, device.index, capture, geometry_key, estimate.num_tokens)
        if key in self._admitted:
            return

        device_key = (device.type, device.index)
        if memory_query_safe:
            available_bytes, total_bytes = self._memory_info(device)
            self._memory_snapshots[device_key] = (available_bytes, total_bytes)
        else:
            snapshot = self._memory_snapshots.get(device_key)
            if snapshot is None:
                raise RuntimeError(
                    "sgl_lora workspace admission must be preflighted before "
                    "CUDA stream capture; run the normal warmup for this shape "
                    "before recording the graph"
                )
            available_bytes, total_bytes = snapshot
        reserve_bytes = max(
            self._minimum_reserve_bytes,
            int(total_bytes * self._reserve_fraction),
        )
        usable_bytes = max(0, available_bytes - reserve_bytes)
        required_bytes = estimate.peak_bytes(capture=capture)
        if required_bytes > usable_bytes:
            mode = "CUDA-graph capture" if capture else "eager execution"
            raise MemoryError(
                "sgl_lora cannot admit the masked-MoE workspace for "
                f"{estimate.num_tokens} local tokens during {mode}: estimated "
                f"incremental peak {required_bytes / (1 << 30):.2f} GiB exceeds "
                f"the device-aware budget {usable_bytes / (1 << 30):.2f} GiB "
                f"({available_bytes / (1 << 30):.2f} GiB reusable, "
                f"{reserve_bytes / (1 << 30):.2f} GiB reserved). Reduce "
                "--chunked-prefill-size or select a lower-workspace MoE provider."
            )
        self._admitted.add(key)
