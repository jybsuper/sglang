"""Row quantization into block-scaled operands for the CuTeDSL base GEMMs.

Rows are addressed through ``pair_to_row`` (routing pair -> permuted row, -1
for pairs without a row), so decode batches touch only their live rows.
Every ``sf_vec_size`` consecutive K elements of a row share one E8M0 scale,
the smallest power of two ``2 ** e`` with ``amax <= elem_max * 2 ** e``, so no
element clips. Elements round to nearest even into E4M3 (MXFP8) or E2M1 pairs
with the even index in the low nibble (MXFP4).
Scales land in the tcgen05 atom order the kernel reads: 512-byte blocks of
128 rows by 4 scale groups, ``((row % 32) * 16 + (row // 32 % 4) * 4 + group % 4``,
blocks ordered by (slab, row block, group block).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# Four E2M1 pairs per 32-bit register; the first source lands in the high nibble.
_E2M1X2_ASM = tl.constexpr(
    """{
.reg .b8 b<4>;
cvt.rn.satfinite.e2m1x2.f32 b0, $5, $1;
cvt.rn.satfinite.e2m1x2.f32 b1, $6, $2;
cvt.rn.satfinite.e2m1x2.f32 b2, $7, $3;
cvt.rn.satfinite.e2m1x2.f32 b3, $8, $4;
mov.b32 $0, {b0, b1, b2, b3};
}"""
)

# Pairs per program; decode batches route a few hundred, prefill tens of thousands.
_ROWS_PER_PROGRAM = 16


@triton.jit
def _pow2(exponent):
    """Exact ``2 ** exponent`` for int32 exponents in the normal range."""
    return ((exponent + 127) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _pack_e2m1_pairs(scaled):
    """[rows, cols] fp32 -> [rows, cols // 2] uint8 with the even column low."""
    rows: tl.constexpr = scaled.shape[0]
    cols: tl.constexpr = scaled.shape[1]
    even, odd = tl.split(tl.reshape(scaled, (rows, cols // 2, 2)))
    return tl.inline_asm_elementwise(
        _E2M1X2_ASM,
        "=r,r,r,r,r,r,r,r,r",
        [even, odd],
        dtype=tl.uint8,
        is_pure=True,
        pack=4,
    )


@triton.jit
def _sf_atom_offsets(rows, group_idx, rows_per_slab, row_atoms, k_atoms):
    """[rows] permuted rows and [groups] scale groups -> [rows, groups] bytes."""
    slab = rows // rows_per_slab
    slab_row = rows % rows_per_slab
    row_block = (slab * row_atoms + slab_row // 128)[:, None]
    row_byte = ((slab_row % 32) * 16 + (slab_row // 32 % 4) * 4)[:, None]
    return (
        (row_block * k_atoms + (group_idx // 4)[None, :]) * 512
        + row_byte
        + (group_idx % 4)[None, :]
    )


@triton.jit
def _quant_mx_rows_kernel(
    x_ptr,
    out_ptr,
    sf_ptr,
    pair_to_row_ptr,
    num_pairs,
    stride_x_row,
    stride_out_row,
    rows_per_slab,
    valid_cols,
    row_atoms,
    k_atoms,
    VEC: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ROWS: tl.constexpr,
    ELEM_MAX: tl.constexpr,
    ELEM_MAX_LOG2: tl.constexpr,
    PACK_E2M1: tl.constexpr,
):
    pairs = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    rows = tl.load(pair_to_row_ptr + pairs, mask=pairs < num_pairs, other=-1)
    live = rows >= 0
    if tl.max(live.to(tl.int32)) == 0:
        return
    rows = tl.where(live, rows, 0)
    col0 = tl.program_id(1) * BLOCK_K
    cols = col0 + tl.arange(0, BLOCK_K)
    col_mask = live[:, None] & (cols < valid_cols)[None, :]
    x = tl.load(
        x_ptr + rows[:, None] * stride_x_row + cols[None, :], mask=col_mask, other=0.0
    )
    groups = tl.reshape(x.to(tl.float32), (ROWS, BLOCK_K // VEC, VEC))
    amax = tl.max(tl.abs(groups), axis=2)
    # floor(log2(amax)) from the float bits, then one comparison for the ceiling.
    exponent = (amax.to(tl.int32, bitcast=True) >> 23) - 127 - ELEM_MAX_LOG2
    exponent = tl.maximum(exponent, -126)  # sub-normal scales flush to zero
    exponent += (amax > ELEM_MAX * _pow2(exponent)).to(tl.int32)
    exponent = tl.where(amax > 0, exponent, 0)
    scaled = groups * _pow2(-exponent)[:, :, None]

    group_idx = col0 // VEC + tl.arange(0, BLOCK_K // VEC)
    group_mask = live[:, None] & (group_idx < tl.cdiv(valid_cols, VEC))[None, :]
    sf_offs = _sf_atom_offsets(rows, group_idx, rows_per_slab, row_atoms, k_atoms)
    tl.store(sf_ptr + sf_offs, (exponent + 127).to(tl.uint8), mask=group_mask)

    if PACK_E2M1:
        packed = _pack_e2m1_pairs(tl.reshape(scaled, (ROWS, BLOCK_K)))
        pair_cols = col0 // 2 + tl.arange(0, BLOCK_K // 2)
        tl.store(
            out_ptr + rows[:, None] * stride_out_row + pair_cols[None, :],
            packed,
            mask=live[:, None] & (pair_cols < (valid_cols + 1) // 2)[None, :],
        )
    else:
        tl.store(
            out_ptr + rows[:, None] * stride_out_row + cols[None, :],
            tl.reshape(scaled, (ROWS, BLOCK_K)).to(out_ptr.dtype.element_ty),
            mask=col_mask,
        )


@triton.jit
def _quant_nvfp4_rows_kernel(
    x_ptr,
    out_ptr,
    sf_ptr,
    pair_to_row_ptr,
    num_pairs,
    global_scale,
    stride_x_row,
    stride_out_row,
    rows_per_slab,
    valid_cols,
    row_atoms,
    k_atoms,
    BLOCK_K: tl.constexpr,
    ROWS: tl.constexpr,
):
    VEC: tl.constexpr = 16
    pairs = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    rows = tl.load(pair_to_row_ptr + pairs, mask=pairs < num_pairs, other=-1)
    live = rows >= 0
    if tl.max(live.to(tl.int32)) == 0:
        return
    rows = tl.where(live, rows, 0)
    col0 = tl.program_id(1) * BLOCK_K
    cols = col0 + tl.arange(0, BLOCK_K)
    x = tl.load(
        x_ptr + rows[:, None] * stride_x_row + cols[None, :],
        mask=live[:, None] & (cols < valid_cols)[None, :],
        other=0.0,
    )
    groups = tl.reshape(x.to(tl.float32), (ROWS, BLOCK_K // VEC, VEC))
    amax = tl.max(tl.abs(groups), axis=2)
    # E4M3 block scales relative to the global scale; normal range only. The
    # per-group divisions are correctly rounded and elements multiply by the
    # reciprocal, so host references reproduce the codes.
    block_scale = libdevice.div_rn(amax, 6.0 * global_scale)
    block_scale = tl.minimum(tl.maximum(block_scale, 0.015625), 448.0)
    block_scale = block_scale.to(tl.float8e4nv)
    inv_denom = libdevice.div_rn(1.0, block_scale.to(tl.float32) * global_scale)
    scaled = groups * inv_denom[:, :, None]

    group_idx = col0 // VEC + tl.arange(0, BLOCK_K // VEC)
    group_mask = live[:, None] & (group_idx < tl.cdiv(valid_cols, VEC))[None, :]
    sf_offs = _sf_atom_offsets(rows, group_idx, rows_per_slab, row_atoms, k_atoms)
    tl.store(sf_ptr + sf_offs, block_scale.to(tl.uint8, bitcast=True), mask=group_mask)

    packed = _pack_e2m1_pairs(tl.reshape(scaled, (ROWS, BLOCK_K)))
    pair_cols = col0 // 2 + tl.arange(0, BLOCK_K // 2)
    tl.store(
        out_ptr + rows[:, None] * stride_out_row + pair_cols[None, :],
        packed,
        mask=live[:, None] & (pair_cols < (valid_cols + 1) // 2)[None, :],
    )


def _grid(num_pairs: int, k: int, block_k: int):
    return (triton.cdiv(num_pairs, _ROWS_PER_PROGRAM), triton.cdiv(k, block_k))


def _block_k(k: int) -> int:
    return 256 if k % 256 == 0 else 128


def quant_mx_rows(
    x: torch.Tensor,
    *,
    out: torch.Tensor,
    sf_out: torch.Tensor,
    sf_vec_size: int,
    rows_per_slab: int,
    pair_to_row: torch.Tensor,
) -> None:
    """Quantize ``x`` rows into ``out`` (E4M3, or uint8 E2M1 pairs) and ``sf_out``.

    ``x`` is [rows, K] (a [slabs, rows_per_slab, K] view of expert slabs is
    flattened by the caller); ``out`` has the same rows with K or K // 2
    columns; ``sf_out`` holds ``scale_atom_bytes(rows_per_slab, K, vec, slabs)``.
    ``pair_to_row`` (int32) lists the rows to quantize, -1 entries skipped.
    """
    rows, k = x.shape
    assert k % sf_vec_size == 0 and rows % rows_per_slab == 0
    pack_e2m1 = out.dtype == torch.uint8
    if pack_e2m1:
        assert out.shape == (rows, k // 2)
    else:
        assert out.dtype == torch.float8_e4m3fn and out.shape == (rows, k)
    assert pair_to_row.dtype == torch.int32
    block_k = _block_k(k)
    assert block_k % sf_vec_size == 0
    _quant_mx_rows_kernel[_grid(pair_to_row.numel(), k, block_k)](
        x,
        out,
        sf_out,
        pair_to_row,
        pair_to_row.numel(),
        x.stride(0),
        out.stride(0),
        rows_per_slab,
        k,
        -(-rows_per_slab // 128),
        -(-(k // sf_vec_size) // 4),
        VEC=sf_vec_size,
        BLOCK_K=block_k,
        ROWS=_ROWS_PER_PROGRAM,
        ELEM_MAX=6.0 if pack_e2m1 else 448.0,
        ELEM_MAX_LOG2=2 if pack_e2m1 else 8,
        PACK_E2M1=pack_e2m1,
        num_warps=4,
    )


def quant_nvfp4_rows(
    x: torch.Tensor,
    *,
    out: torch.Tensor,
    sf_out: torch.Tensor,
    global_scale: float,
    rows_per_slab: int,
    pair_to_row: torch.Tensor,
) -> None:
    """NVFP4 rows: E2M1 pairs with E4M3 scales per 16 under one global scale.

    ``pair = round(x * (1 / (block_scale * global_scale)))``; the GEMM epilogue
    applies the global scales. Layouts follow :func:`quant_mx_rows`.
    """
    rows, k = x.shape
    assert k % 16 == 0 and rows % rows_per_slab == 0
    assert out.dtype == torch.uint8 and out.shape == (rows, k // 2)
    assert pair_to_row.dtype == torch.int32
    block_k = _block_k(k)
    _quant_nvfp4_rows_kernel[_grid(pair_to_row.numel(), k, block_k)](
        x,
        out,
        sf_out,
        pair_to_row,
        pair_to_row.numel(),
        float(global_scale),
        x.stride(0),
        out.stride(0),
        rows_per_slab,
        k,
        -(-rows_per_slab // 128),
        -(-(k // 16) // 4),
        BLOCK_K=block_k,
        ROWS=_ROWS_PER_PROGRAM,
        num_warps=4,
    )


def pack_scale_atoms(scales: torch.Tensor) -> torch.Tensor:
    """[slabs, rows, groups] scale bytes -> flat tcgen05 atoms (host-side packing)."""
    slabs, rows, groups = scales.shape
    row_atoms, k_atoms = -(-rows // 128), -(-groups // 4)
    padded = torch.zeros(
        (slabs, row_atoms * 128, k_atoms * 4), dtype=scales.dtype, device=scales.device
    )
    padded[:, :rows, :groups] = scales
    return (
        padded.view(slabs, row_atoms, 4, 32, k_atoms, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .view(-1)
    )
