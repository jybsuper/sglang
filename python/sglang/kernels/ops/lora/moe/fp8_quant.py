"""Match upstream FP8 scale arithmetic, with zero codes for zero-scale rows.

Whole-row division uses fast math only on SM90; group quantization always does.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

_WHOLE_ROW_FAST_DIV = tl.constexpr(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
)


@triton.jit
def quantize_fp8_groups(grouped, WHOLE_ROW: tl.constexpr):
    """Return FP32 codes and per-group scales for [groups, group_size] input."""
    amax = tl.max(tl.abs(grouped), axis=1)
    if WHOLE_ROW:
        if _WHOLE_ROW_FAST_DIV:
            scale = amax / 448.0
            inv = tl.where(scale == 0.0, 0.0, 1.0 / scale)
        else:
            scale = libdevice.div_rn(amax, 448.0)
            inv = tl.where(scale == 0.0, 0.0, libdevice.div_rn(1.0, scale))
        q = grouped * inv[:, None]
    else:
        amax = tl.maximum(amax, 1e-10)
        scale = amax * (1.0 / 448.0)
        q = grouped * (448.0 / amax)[:, None]
    return tl.clamp(q, -448.0, 448.0), scale


@triton.jit
def _quantize_rows_per_token_kernel(
    x_ptr,  # bf16 rows, viewed flat [rows, K]
    q_ptr,  # fp8e4m3 [rows, K]
    s_ptr,  # fp32 [rows]
    masked_m_ptr,  # [E] int32 live rows per slab (MASKED only)
    rows_per_slab,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MASKED: tl.constexpr,
):
    row = tl.program_id(0)
    if MASKED:
        if row % rows_per_slab >= tl.load(masked_m_ptr + row // rows_per_slab):
            return
    offs = tl.arange(0, BLOCK_K)
    base = row.to(tl.int64) * K
    x = tl.load(x_ptr + base + offs, mask=offs < K, other=0).to(tl.float32)
    q, scale = quantize_fp8_groups(tl.reshape(x, (1, BLOCK_K)), True)
    tl.store(
        q_ptr + base + offs,
        tl.reshape(q, (BLOCK_K,)).to(q_ptr.dtype.element_ty),
        mask=offs < K,
    )
    tl.store(s_ptr + row + tl.arange(0, 1), scale)


def quantize_rows_per_token_fp8(
    x: torch.Tensor,
    *,
    q_out: torch.Tensor,
    scale_out: torch.Tensor,
    masked_m: torch.Tensor | None = None,
) -> None:
    """One FP8 scale per row of [rows, K] or [E, m_max, K]; masked slabs skip
    the rows past masked_m[e]."""
    k = x.shape[-1]
    rows = x.numel() // k
    if rows == 0:
        return
    masked = masked_m is not None
    _quantize_rows_per_token_kernel[(rows,)](
        x,
        q_out,
        scale_out,
        masked_m if masked else scale_out,
        x.shape[1] if masked else 1,
        K=k,
        BLOCK_K=triton.next_power_of_2(k),
        MASKED=masked,
        num_warps=4 if k <= 4096 else 8,
    )
