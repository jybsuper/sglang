"""Static planning and Triton execution for shared-outer gate/up LoRA-A.

Shared-outer adapters store one gate/up A factor per adapter rather than one
per routed expert.  The generic virtual-expert shrink nevertheless repeats the
same ``hidden @ A.T`` work for every top-k slot.  The token-deduplicated kernel
below computes that product once for each request token and broadcasts the
result into the existing pair-major ``[T, K, 2R]`` consumer contract while the
accumulator is still resident.  No runtime descriptor build, token sort, or
extra materialization launch is required.

Selection is deliberately static and evidence-bounded.  Every field in
:class:`SharedOuterGateAKey` is host metadata fixed before a kernel launches,
so the decision is safe to close over during CUDA-graph capture.  Shapes that
were wide, tiny, mixed/noisy, or outside the H200/GB300 measurements keep the
generic virtual-expert implementation.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.lora.sgl_lora.shared_outer_gate_policy import (
    SharedOuterGateAKernel,
    SharedOuterGateAKey,
    SharedOuterGateAPlan,
    build_shared_outer_gate_a_plan,
)


@triton.jit
def _shared_outer_gate_a_token_dedup_kernel(
    hidden_ptr,
    factor_ptr,
    output_ptr,
    segment_indptr_ptr,
    segment_lora_ids_ptr,
    stride_xt,
    stride_xh,
    stride_fl,
    stride_fn,
    stride_fh,
    stride_ot,
    stride_ok,
    stride_on,
    N: tl.constexpr,
    H: tl.constexpr,
    TOP_K: tl.constexpr,
    TILES_M_PER_SEGMENT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SIGNAL_EXPAND_PDL: tl.constexpr,
):
    segment_tile = tl.program_id(0)
    segment_id = segment_tile // TILES_M_PER_SEGMENT
    segment_m_tile = segment_tile % TILES_M_PER_SEGMENT
    n_tile = tl.program_id(1)

    segment_start = tl.load(segment_indptr_ptr + segment_id)
    segment_stop = tl.load(segment_indptr_ptr + segment_id + 1)
    segment_length = segment_stop - segment_start
    adapter_id = tl.load(segment_lora_ids_ptr + segment_id).to(tl.int64)

    local_m = segment_m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_m = segment_start + local_m
    offsets_n = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    mask_m = local_m < segment_length
    mask_n = offsets_n < N

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_block in range(0, tl.cdiv(H, BLOCK_K)):
        current_k = k_block * BLOCK_K + offsets_k
        mask_k = current_k < H
        hidden = tl.load(
            hidden_ptr
            + offsets_m[:, None] * stride_xt
            + current_k[None, :] * stride_xh,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        factor = tl.load(
            factor_ptr
            + adapter_id * stride_fl
            + offsets_n[:, None] * stride_fn
            + current_k[None, :] * stride_fh,
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0,
        )
        accumulator += tl.dot(hidden, tl.trans(factor))

    value = accumulator.to(output_ptr.dtype.element_ty)
    for topk_slot in range(0, TOP_K):
        tl.store(
            output_ptr
            + offsets_m[:, None] * stride_ot
            + topk_slot * stride_ok
            + offsets_n[None, :] * stride_on,
            value,
            mask=mask_m[:, None] & mask_n[None, :],
        )

    if SIGNAL_EXPAND_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def invoke_shared_outer_gate_a_token_dedup(
    hidden_states: torch.Tensor,
    shared_gate_a: torch.Tensor,
    pair_output: torch.Tensor,
    segment_indptr: torch.Tensor,
    segment_lora_ids: torch.Tensor,
    max_segment_len: int,
    plan: SharedOuterGateAPlan,
    *,
    signal_expand_pdl: bool = False,
) -> None:
    """Compute shared gate/up A once per token and emit ``[T,K,2R]``."""

    if not plan.uses_token_dedup:
        raise ValueError("token-dedup launcher requires a selected token-dedup plan")
    if hidden_states.ndim != 2 or shared_gate_a.ndim != 4:
        raise ValueError("expected hidden [T,H] and gate A [L,1,2R,H]")
    if shared_gate_a.shape[1] != 1:
        raise ValueError("shared-outer gate A must have one stored expert factor")
    if pair_output.ndim != 3:
        raise ValueError("shared-outer gate A output must be [T,K,2R]")
    tokens, hidden_size = hidden_states.shape
    output_width = shared_gate_a.shape[2]
    if shared_gate_a.shape[3] != hidden_size:
        raise ValueError("shared-outer gate A hidden dimension does not match input")
    if pair_output.shape[0] != tokens or pair_output.shape[2] != output_width:
        raise ValueError("shared-outer gate A pair output shape does not match")
    if segment_indptr.ndim != 1 or segment_lora_ids.ndim != 1:
        raise ValueError("segment metadata must be one-dimensional")
    if segment_indptr.numel() != segment_lora_ids.numel() + 1:
        raise ValueError("segment indptr and adapter IDs have inconsistent lengths")
    if max_segment_len <= 0:
        raise ValueError("max_segment_len must be positive")

    tiles_m = triton.cdiv(max_segment_len, plan.block_m)
    tiles_n = triton.cdiv(output_width, plan.block_n)
    grid = (segment_lora_ids.numel() * tiles_m, tiles_n)

    from sglang.srt.lora.sgl_lora.triton_ops.virtual_experts import (
        _get_pdl_launch_metadata,
    )

    enable_pdl, pdl_kwargs = _get_pdl_launch_metadata()
    signal_expand_pdl = signal_expand_pdl and enable_pdl
    _shared_outer_gate_a_token_dedup_kernel[grid](
        hidden_states,
        shared_gate_a,
        pair_output,
        segment_indptr,
        segment_lora_ids,
        hidden_states.stride(0),
        hidden_states.stride(1),
        shared_gate_a.stride(0),
        shared_gate_a.stride(2),
        shared_gate_a.stride(3),
        pair_output.stride(0),
        pair_output.stride(1),
        pair_output.stride(2),
        N=output_width,
        H=hidden_size,
        TOP_K=pair_output.shape[1],
        TILES_M_PER_SEGMENT=tiles_m,
        BLOCK_M=plan.block_m,
        BLOCK_N=plan.block_n,
        BLOCK_K=plan.block_k,
        SIGNAL_EXPAND_PDL=signal_expand_pdl,
        num_warps=plan.num_warps,
        num_stages=plan.num_stages,
        **(pdl_kwargs if signal_expand_pdl else {}),
    )


__all__ = [
    "SharedOuterGateAKey",
    "SharedOuterGateAKernel",
    "SharedOuterGateAPlan",
    "build_shared_outer_gate_a_plan",
    "invoke_shared_outer_gate_a_token_dedup",
]
