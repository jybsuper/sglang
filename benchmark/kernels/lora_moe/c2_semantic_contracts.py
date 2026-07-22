"""Logical contracts and strict PyTorch oracles for experimental C2 consumers.

The C2 kernels consume a provider-private, expert-masked GEMM1 output.  Model
semantics (which projections exist and which of them LoRA targets) must not be
inferred from a physical tensor width.  This module makes those two domains
explicit for benchmark and correctness work without changing serving dispatch.

``logical_slices`` names the projections used by the activation.  Each physical
provider slice has ``physical_intermediate_size`` columns, of which only the
first ``logical_intermediate_size`` are model values.  ``lora_target_slices``
is an ordered subset of the logical slices; LoRA-A/B pack only that subset and
therefore never require materialized zero factors for untargeted slices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple

import torch
import torch.nn.functional as F

Activation = Literal["swiglu", "relu2"]


@dataclass(frozen=True, slots=True, kw_only=True)
class C2SemanticContract:
    """Logical activation/slice semantics plus the provider output layout."""

    activation: Activation
    logical_slices: tuple[str, ...]
    lora_target_slices: tuple[str, ...]
    logical_intermediate_size: int
    physical_intermediate_size: int
    provider_slice_order: tuple[str, ...] | None = None
    provider_row_domain: Literal["masked_destination"] = "masked_destination"
    provider_padding_policy: Literal["zero"] = "zero"

    def __post_init__(self) -> None:
        if self.logical_intermediate_size <= 0:
            raise ValueError("logical_intermediate_size must be positive")
        if self.physical_intermediate_size < self.logical_intermediate_size:
            raise ValueError("physical width cannot be smaller than logical width")
        if len(set(self.logical_slices)) != len(self.logical_slices):
            raise ValueError("logical slice names must be unique")
        expected = ("gate", "up") if self.activation == "swiglu" else ("value",)
        if self.logical_slices != expected:
            raise ValueError(
                f"{self.activation} requires logical slices {expected}, got "
                f"{self.logical_slices}"
            )
        if len(set(self.lora_target_slices)) != len(self.lora_target_slices):
            raise ValueError("LoRA target slices must be unique")
        unknown = set(self.lora_target_slices).difference(self.logical_slices)
        if unknown:
            raise ValueError(f"LoRA targets unknown logical slices: {sorted(unknown)}")
        provider_order = self.resolved_provider_slice_order
        if sorted(provider_order) != sorted(self.logical_slices):
            raise ValueError(
                "provider_slice_order must contain every logical slice exactly once"
            )

    @property
    def resolved_provider_slice_order(self) -> tuple[str, ...]:
        return self.provider_slice_order or self.logical_slices

    @property
    def num_logical_slices(self) -> int:
        return len(self.logical_slices)

    @property
    def num_lora_slices(self) -> int:
        return len(self.lora_target_slices)

    @property
    def provider_output_width(self) -> int:
        return self.num_logical_slices * self.physical_intermediate_size

    def provider_slice_offset(self, name: str) -> int:
        return (
            self.resolved_provider_slice_order.index(name)
            * self.physical_intermediate_size
        )

    def lora_slice_offset(self, name: str, width: int) -> int:
        return self.lora_target_slices.index(name) * width

    def benchmark_kernel_family(self) -> str:
        """Return the narrow kernel family compatible with this ABI.

        This deliberately does not select on a model name.  Activation and
        physical layout are the semantic/provider keys; model presets merely
        resolve their dimensions.
        """

        if (
            self.activation == "swiglu"
            and self.lora_target_slices == ("gate", "up")
            and self.resolved_provider_slice_order == ("gate", "up")
            and self.logical_intermediate_size
            == self.physical_intermediate_size
        ):
            return "gated_swiglu_v1"
        if (
            self.activation == "relu2"
            and self.lora_target_slices == ("value",)
            and self.resolved_provider_slice_order == ("value",)
        ):
            return "nongated_relu2_v1"
        return "oracle_only_requires_provider_specialization"


class ConsumerReference(NamedTuple):
    activation_output: torch.Tensor
    down_rank_input: torch.Tensor


def reference_c2_consumer(
    contract: C2SemanticContract,
    gateup_output: torch.Tensor,
    gate_intermediate: torch.Tensor,
    gate_lora_b: torch.Tensor,
    down_lora_a: torch.Tensor,
    src2dst: torch.Tensor,
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    *,
    local_expert_offset: int = 0,
    block_size_n: int | None = None,
    activation_output_dtype: torch.dtype | None = None,
    padding_value: float = 0.0,
    invalid_destination_fill: float = -123.0,
) -> ConsumerReference:
    """Materialize the C2 semantic boundaries with an intentionally strict oracle.

    Gate/value-B deltas and activation outputs round through the destination
    dtype.  Down-A partials optionally round once per ``block_size_n`` tile to
    reproduce the current Triton split-I accumulation contract.
    """

    if topk_ids.ndim != 2:
        raise ValueError("topk_ids must be [token, top-k]")
    num_tokens, top_k = topk_ids.shape
    if token_lora_mapping.shape != (num_tokens,):
        raise ValueError("token_lora_mapping must have one entry per token")
    num_pairs = num_tokens * top_k
    if src2dst.numel() != num_pairs:
        raise ValueError("src2dst must have one entry per routed pair")
    if gateup_output.shape[-1] != contract.provider_output_width:
        raise ValueError("gateup_output does not match provider output width")
    if gate_lora_b.shape[2] != (
        contract.num_lora_slices * contract.physical_intermediate_size
    ):
        raise ValueError("gate_lora_b does not match packed LoRA slice width")

    gate_rank = gate_lora_b.shape[-1]
    down_rank = down_lora_a.shape[-2]
    if gate_intermediate.shape != (num_tokens, top_k, contract.num_lora_slices * gate_rank):
        raise ValueError("gate_intermediate does not match packed LoRA slices")
    if down_lora_a.shape[-1] != contract.physical_intermediate_size:
        raise ValueError("down_lora_a physical width does not match provider width")

    out_dtype = activation_output_dtype or gateup_output.dtype
    destination_rows = gateup_output.shape[0]
    act_out = torch.full(
        (destination_rows, contract.physical_intermediate_size),
        invalid_destination_fill,
        dtype=out_dtype,
        device=gateup_output.device,
    )
    down_rank_input = torch.zeros(
        (num_tokens, top_k, down_rank),
        dtype=out_dtype,
        device=gateup_output.device,
    )
    flat_ids = topk_ids.reshape(-1)
    flat_gate_intermediate = gate_intermediate.reshape(num_pairs, -1)
    flat_down = down_rank_input.reshape(num_pairs, down_rank)

    num_loras, num_local_experts = gate_lora_b.shape[:2]
    logical_i = contract.logical_intermediate_size
    tile = block_size_n or logical_i
    for pair_idx in range(num_pairs):
        global_expert = int(flat_ids[pair_idx])
        local_expert = global_expert - local_expert_offset
        if not 0 <= local_expert < num_local_experts:
            continue
        destination = int(src2dst.reshape(-1)[pair_idx])
        if not 0 <= destination < destination_rows:
            raise ValueError("a valid pair has an out-of-range provider destination")
        adapter = int(token_lora_mapping[pair_idx // top_k])
        has_lora = 0 <= adapter < num_loras

        slices: dict[str, torch.Tensor] = {}
        for name in contract.logical_slices:
            start = contract.provider_slice_offset(name)
            slices[name] = gateup_output[
                destination, start : start + logical_i
            ].float()

        if has_lora:
            for name in contract.lora_target_slices:
                a_start = contract.lora_slice_offset(name, gate_rank)
                b_start = contract.lora_slice_offset(
                    name, contract.physical_intermediate_size
                )
                delta = (
                    gate_lora_b[
                        adapter,
                        local_expert,
                        b_start : b_start + logical_i,
                    ].float()
                    @ flat_gate_intermediate[
                        pair_idx, a_start : a_start + gate_rank
                    ].float()
                ).to(out_dtype)
                slices[name] = slices[name] + delta.float()

        if contract.activation == "swiglu":
            activated = F.silu(slices["gate"]) * slices["up"]
        else:
            activated = torch.relu(slices["value"]).square()
        activated = activated.to(out_dtype)
        act_out[destination].fill_(padding_value)
        act_out[destination, :logical_i] = activated

        if has_lora:
            accumulated = torch.zeros(
                down_rank, dtype=out_dtype, device=gateup_output.device
            )
            for start in range(0, logical_i, tile):
                stop = min(start + tile, logical_i)
                partial = (
                    down_lora_a[
                        adapter, local_expert, :, start:stop
                    ].float()
                    @ activated[start:stop].float()
                ).to(out_dtype)
                accumulated = (accumulated.float() + partial.float()).to(out_dtype)
            flat_down[pair_idx] = accumulated

    return ConsumerReference(act_out, down_rank_input)


def reference_weighted_down_delta(
    down_rank_input: torch.Tensor,
    down_lora_b: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    *,
    local_expert_offset: int = 0,
    routed_scaling_factor: float = 1.0,
    destination_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Strict token-domain down-B oracle with routed scaling applied once."""

    num_tokens, top_k = topk_ids.shape
    if down_rank_input.shape[:2] != (num_tokens, top_k):
        raise ValueError("down_rank_input and top-k domains differ")
    if topk_weights.shape != topk_ids.shape:
        raise ValueError("topk_weights and topk_ids must have the same shape")
    hidden = down_lora_b.shape[-2]
    output = torch.zeros(
        (num_tokens, hidden),
        dtype=torch.float32,
        device=down_rank_input.device,
    )
    num_loras, num_local_experts = down_lora_b.shape[:2]
    for token in range(num_tokens):
        adapter = int(token_lora_mapping[token])
        if not 0 <= adapter < num_loras:
            continue
        for slot in range(top_k):
            local_expert = int(topk_ids[token, slot]) - local_expert_offset
            if not 0 <= local_expert < num_local_experts:
                continue
            delta = (
                down_lora_b[adapter, local_expert].float()
                @ down_rank_input[token, slot].float()
            )
            output[token] += (
                delta
                * topk_weights[token, slot].float()
                * float(routed_scaling_factor)
            )
    return output.to(destination_dtype)


__all__ = [
    "C2SemanticContract",
    "ConsumerReference",
    "reference_c2_consumer",
    "reference_weighted_down_delta",
]
