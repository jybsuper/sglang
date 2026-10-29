"""Row quantization into MX block-scaled operands for the CuTeDSL base GEMMs.

Every ``sf_vec_size`` consecutive K elements of a row share one E8M0 scale,
the smallest power of two ``2 ** e`` with ``amax <= elem_max * 2 ** e``, so no
element clips. Elements become
E4M3 (MXFP8) or E2M1 pairs with the even index in the low nibble (MXFP4).
Scales land in the tcgen05 atom order the kernel reads: 512-byte blocks of
128 rows by 4 scale groups, ``((row % 32) * 16 + (row // 32 % 4) * 4 + group % 4``,
blocks ordered by (slab, row block, group block).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _quant_mx_rows_kernel(
    x_ptr,
    out_ptr,
    sf_ptr,
    stride_x_row,
    stride_out_row,
    rows_per_slab,
    valid_cols,
    row_atoms,
    k_atoms,
    VEC: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ELEM_MAX: tl.constexpr,
    ELEM_MAX_LOG2: tl.constexpr,
    PACK_E2M1: tl.constexpr,
):
    row = tl.program_id(0)
    col0 = tl.program_id(1) * BLOCK_K
    offs = col0 + tl.arange(0, BLOCK_K)
    x = tl.load(x_ptr + row * stride_x_row + offs, mask=offs < valid_cols, other=0.0)
    groups = tl.reshape(x.to(tl.float32), (BLOCK_K // VEC, VEC))
    amax = tl.max(tl.abs(groups), axis=1)
    # floor(log2(amax)) from the float bits, then one comparison for the ceiling.
    floor_log2 = (amax.to(tl.int32, bitcast=True) >> 23) - 127
    exponent = floor_log2 - ELEM_MAX_LOG2
    exponent += (amax > ELEM_MAX * tl.exp2(exponent.to(tl.float32))).to(tl.int32)
    # Keep 2**exponent a normal float: exp2 flushes sub-normals to zero.
    exponent = tl.where(amax > 0, exponent, 0)
    exponent = tl.minimum(tl.maximum(exponent, -126), 127)
    scaled = groups / tl.exp2(exponent.to(tl.float32))[:, None]

    group_idx = col0 // VEC + tl.arange(0, BLOCK_K // VEC)
    slab = row // rows_per_slab
    slab_row = row % rows_per_slab
    sf_offs = (
        ((slab * row_atoms + slab_row // 128) * k_atoms + group_idx // 4) * 512
        + (slab_row % 32) * 16
        + ((slab_row // 32) % 4) * 4
        + (group_idx % 4)
    )
    tl.store(
        sf_ptr + sf_offs,
        (exponent + 127).to(tl.uint8),
        mask=group_idx < (valid_cols + VEC - 1) // VEC,
    )

    if PACK_E2M1:
        flat = tl.reshape(scaled, (BLOCK_K,))
        magnitude = tl.abs(flat)
        code = (
            (magnitude > 0.25).to(tl.int32)
            + (magnitude > 0.75).to(tl.int32)
            + (magnitude > 1.25).to(tl.int32)
            + (magnitude > 1.75).to(tl.int32)
            + (magnitude > 2.5).to(tl.int32)
            + (magnitude > 3.5).to(tl.int32)
            + (magnitude > 5.0).to(tl.int32)
        )
        code = code | ((flat < 0).to(tl.int32) << 3)
        shift = (tl.arange(0, BLOCK_K) % 2) * 4
        packed = tl.sum(tl.reshape(code << shift, (BLOCK_K // 2, 2)), axis=1)
        pair_offs = col0 // 2 + tl.arange(0, BLOCK_K // 2)
        tl.store(
            out_ptr + row * stride_out_row + pair_offs,
            packed.to(tl.uint8),
            mask=pair_offs < (valid_cols + 1) // 2,
        )
    else:
        tl.store(
            out_ptr + row * stride_out_row + offs,
            tl.reshape(scaled, (BLOCK_K,)).to(out_ptr.dtype.element_ty),
            mask=offs < valid_cols,
        )


def quant_mx_rows(
    x: torch.Tensor,
    *,
    out: torch.Tensor,
    sf_out: torch.Tensor,
    sf_vec_size: int,
    rows_per_slab: int,
) -> None:
    """Quantize ``x`` rows into ``out`` (E4M3, or uint8 E2M1 pairs) and ``sf_out``.

    ``x`` is [rows, K] (a [slabs, rows_per_slab, K] view of expert slabs is
    flattened by the caller); ``out`` has the same rows with K or K // 2
    columns; ``sf_out`` holds ``scale_atom_bytes(rows_per_slab, K, vec, slabs)``.
    """
    rows, k = x.shape
    assert k % sf_vec_size == 0 and rows % rows_per_slab == 0
    pack_e2m1 = out.dtype == torch.uint8
    if pack_e2m1:
        assert out.shape == (rows, k // 2)
    else:
        assert out.dtype == torch.float8_e4m3fn and out.shape == (rows, k)
    block_k = min(1024, triton.next_power_of_2(k))
    assert block_k % sf_vec_size == 0
    row_atoms = -(-rows_per_slab // 128)
    k_atoms = -(-(k // sf_vec_size) // 4)
    grid = (rows, triton.cdiv(k, block_k))
    _quant_mx_rows_kernel[grid](
        x,
        out,
        sf_out,
        x.stride(0),
        out.stride(0),
        rows_per_slab,
        k,
        row_atoms,
        k_atoms,
        VEC=sf_vec_size,
        BLOCK_K=block_k,
        ELEM_MAX=6.0 if pack_e2m1 else 448.0,
        ELEM_MAX_LOG2=2 if pack_e2m1 else 8,
        PACK_E2M1=pack_e2m1,
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
