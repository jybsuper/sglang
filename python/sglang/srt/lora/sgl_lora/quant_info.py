"""Provider payloads for the SGL LoRA MoE pipeline.

The payloads deliberately describe both the model dimensions and the resident
provider representation.  Packed weights do not have a common physical shape:
for example Marlin stores K-major tiles while NVFP4 stores two FP4 values per
byte.  Consumers must therefore use ``num_local_experts``,
``intermediate_size``, and ``hidden_size`` for semantic sizing instead of
reverse-engineering dimensions from a packed tensor.

All providers keep the conventional ``w13_weight`` / ``w2_weight`` names.  The
tensors are borrowed from the base MoE layer and remain owned by that layer;
``sgl_lora`` never copies or reorders weights on a forward.
"""

from __future__ import annotations

import msgspec
import torch


class SglLoraBf16QuantInfo(msgspec.Struct, kw_only=True):
    """Unquantized standard-layout MoE weights.

    ``w13_weight`` is ``[E_local, 2 * I, H]`` (gate first) and ``w2_weight`` is
    ``[E_local, H, I]``.
    """

    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    num_local_experts: int
    intermediate_size: int
    hidden_size: int


class SglLoraFp8QuantInfo(msgspec.Struct, kw_only=True):
    """DeepGEMM FP8 base weights in canonical ``[E, N, K]`` layout."""

    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    w13_scale: torch.Tensor
    w2_scale: torch.Tensor
    block_shape: tuple[int, int]
    num_local_experts: int
    intermediate_size: int
    hidden_size: int
    use_mxfp8: bool = False


class SglLoraNvFp4QuantInfo(msgspec.Struct, kw_only=True):
    """Native NVFP4 W4A4 payload for the masked CuteDSL primitives.

    The injectable primitive consumes canonical, non-interleaved gate-first
    W13 weights and swizzled block scales.  The fused standard CuteDSL/TRT-LLM
    representations are intentionally not accepted: those kernels own the
    activation and second activation quantization, so they expose no correct
    gate/up LoRA injection point.
    """

    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    w13_blockscale: torch.Tensor
    w2_blockscale: torch.Tensor
    w13_alpha: torch.Tensor
    w2_alpha: torch.Tensor
    w13_input_scale: torch.Tensor
    w2_input_scale: torch.Tensor
    num_local_experts: int
    intermediate_size: int
    hidden_size: int
    gate_first: bool = True
    interleaved: bool = False


class SglLoraMarlinQuantInfo(msgspec.Struct, kw_only=True):
    """Marlin W4A16 payload.

    ``w13_weight`` and ``w2_weight`` are the provider's packed qweights.  Their
    shapes are not semantic; use the explicit dimensions below.
    """

    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    w13_scales: torch.Tensor
    w2_scales: torch.Tensor
    weight_bits: int
    num_local_experts: int
    intermediate_size: int
    hidden_size: int
    w13_g_idx_sort_indices: torch.Tensor | None = None
    w2_g_idx_sort_indices: torch.Tensor | None = None
    w13_g_idx: torch.Tensor | None = None
    w2_g_idx: torch.Tensor | None = None
    w13_qzeros: torch.Tensor | None = None
    w2_qzeros: torch.Tensor | None = None
    w13_global_scale: torch.Tensor | None = None
    w2_global_scale: torch.Tensor | None = None
    w13_bias: torch.Tensor | None = None
    w2_bias: torch.Tensor | None = None
    expert_map: torch.Tensor | None = None
    global_num_experts: int = -1
    is_k_full: bool = True


SglLoraQuantInfo = (
    SglLoraBf16QuantInfo
    | SglLoraFp8QuantInfo
    | SglLoraNvFp4QuantInfo
    | SglLoraMarlinQuantInfo
)


__all__ = [
    "SglLoraBf16QuantInfo",
    "SglLoraFp8QuantInfo",
    "SglLoraMarlinQuantInfo",
    "SglLoraNvFp4QuantInfo",
    "SglLoraQuantInfo",
]
