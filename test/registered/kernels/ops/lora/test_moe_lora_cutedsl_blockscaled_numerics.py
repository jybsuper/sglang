"""Block-scaled CuTeDSL grouped GEMMs against exact dequantized references.

MXFP8 (E4M3 x E8M0/32), MXFP4 (E2M1 x E8M0/32) and NVFP4 (E2M1 x E4M3/16)
operands are quantized on the host with the same rules as the Triton row
quantizer; the scale atoms follow the tcgen05 128x4 block order.
"""

from __future__ import annotations

import time

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_E2M1_MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def _e2m1_codes(magnitude: torch.Tensor) -> torch.Tensor:
    """Nearest E2M1 code; ties go to the even code, as the hardware cvt.rn does."""
    mid = _E2M1_MID.to(magnitude.device)
    idx = torch.bucketize(magnitude, mid)  # a tie lands on the lower code
    tie = (idx < 7) & (magnitude == mid[idx.clamp(max=6)]) & (idx % 2 == 1)
    return idx + tie.long()


def _skip_unless_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("block-scaled tcgen05 MMA needs SM100")


def _quant_mx_ref(x: torch.Tensor, vec: int, elem: str):
    """[..., K] -> (data, scale bytes [..., K/vec] uint8, exact dequant fp32)."""
    *lead, k = x.shape
    groups = x.float().reshape(*lead, k // vec, vec)
    amax = groups.abs().amax(dim=-1)
    elem_max, elem_max_log2 = (6.0, 2) if elem == "e2m1" else (448.0, 8)
    # Smallest power of two with amax <= elem_max * 2**e, as the row quantizer.
    floor_log2 = torch.frexp(amax)[1].float() - 1
    exponent = floor_log2 - elem_max_log2
    exponent = exponent + (amax > elem_max * torch.exp2(exponent)).float()
    exponent = torch.where(amax > 0, exponent, torch.zeros_like(amax))
    exponent = exponent.clamp(-126.0, 127.0)
    scale = torch.exp2(exponent)
    scaled = groups / scale[..., None]
    if elem == "e2m1":
        idx = _e2m1_codes(scaled.abs())
        sign = scaled < 0
        table = _E2M1.to(x.device)
        values = torch.where(sign, -table[idx], table[idx])
        codes = (idx | (sign.long() << 3)).to(torch.uint8).reshape(*lead, k)
        data = codes[..., 0::2] | (codes[..., 1::2] << 4)
    else:
        q = scaled.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        values = q.float()
        data = q.reshape(*lead, k)
    dequant = (values * scale[..., None]).reshape(*lead, k)
    return data, (exponent + 127).to(torch.uint8), dequant


def _quant_nvfp4_ref(x: torch.Tensor, global_scale: torch.Tensor):
    """NVFP4 with E4M3 group-16 scales under a per-slab global scale."""
    *lead, k = x.shape
    groups = x.float().reshape(*lead, k // 16, 16)
    amax = groups.abs().amax(dim=-1)
    gs = global_scale.view(-1, *([1] * (len(lead) - 1)), 1).float()
    block_scale = (
        (amax / (6.0 * gs)).clamp(min=2.0**-6, max=448.0).to(torch.float8_e4m3fn)
    )
    denom = block_scale.float() * gs
    scaled = (groups * (1.0 / denom)[..., None]).clamp(-6.0, 6.0)
    idx = _e2m1_codes(scaled.abs())
    sign = scaled < 0
    table = _E2M1.to(x.device)
    values = torch.where(sign, -table[idx], table[idx])
    codes = (idx | (sign.long() << 3)).to(torch.uint8).reshape(*lead, k)
    data = codes[..., 0::2] | (codes[..., 1::2] << 4)
    dequant = (values * denom[..., None]).reshape(*lead, k)
    return data, block_scale.view(torch.uint8), dequant


FORMATS = {
    "mxfp8": dict(elem="e4m3", vec=32),
    "mxfp4": dict(elem="e2m1", vec=32),
    "nvfp4": dict(elem="e2m1", vec=16),
}


def _quantize(fmt: str, x: torch.Tensor, global_scale=None):
    if fmt == "nvfp4":
        return _quant_nvfp4_ref(x, global_scale)
    return _quant_mx_ref(x, FORMATS[fmt]["vec"], FORMATS[fmt]["elem"])


def _rel_l2(a, b):
    return (a - b).norm() / b.norm().clamp(min=1e-12)


@pytest.mark.parametrize("fmt", sorted(FORMATS))
def test_triton_row_quantizer_matches_reference(fmt):
    """The Triton quantizer reproduces the host rule and atom order exactly."""
    _skip_unless_sm100()
    from sglang.kernels.ops.lora.moe.cutedsl.api import scale_atom_bytes
    from sglang.kernels.ops.lora.moe.quant_blockscaled import (
        BlockScaledRows,
        pack_scale_atoms,
        quantize_rows,
    )

    vec, elem = FORMATS[fmt]["vec"], FORMATS[fmt]["elem"]
    device = torch.device("cuda")
    slabs, rows, k = 3, 200, 512
    x = (torch.randn(slabs, rows, k, device=device) * 0.3).to(torch.bfloat16)
    x[0, 5] = 0  # an all-zero row keeps a finite scale
    # One group of exact E2M1 ties (amax = 6 * 2**-3) rounds to the even codes.
    x[1, 7, :32] = 0
    x[1, 7, :8] = (
        torch.tensor([0.75, -0.25, 1.25, -1.75, 2.5, -3.5, 5.0, 6.0]) * 2.0**-3
    )
    pair_to_row = torch.arange(slabs * rows, dtype=torch.int32, device=device)
    global_scale = 0.3
    data_ref, sf_ref, _ = _quantize(
        fmt, x, torch.full((slabs,), global_scale, device=device)
    )
    out = torch.empty(
        (slabs * rows, k // 2 if elem == "e2m1" else k),
        dtype=torch.uint8 if elem == "e2m1" else torch.float8_e4m3fn,
        device=device,
    )
    sf_out = torch.zeros(
        scale_atom_bytes(rows, k, vec, slabs), dtype=torch.uint8, device=device
    )
    target = BlockScaledRows(
        data=out,
        scales=sf_out,
        sf_vec_size=vec,
        rows_per_slab=rows,
        global_scale=global_scale if fmt == "nvfp4" else None,
    )
    quantize_rows(x.view(slabs * rows, k), target, pair_to_row)
    assert torch.equal(
        out.view(torch.uint8), data_ref.view(torch.uint8).view(slabs * rows, -1)
    )
    assert torch.equal(sf_out, pack_scale_atoms(sf_ref))


@pytest.mark.parametrize("fmt", sorted(FORMATS))
@pytest.mark.parametrize("k", [640, 5120])
def test_blockscaled_row_movers_match_row_quantizer(fmt, k):
    """Both dispatch fills (expert slabs and contiguous segments) write the row
    quantizer's codes and scale atoms bitwise over whole column chunks (K=5120
    is five chunks of 1024, K=640 five of 128); other rows stay untouched."""
    if FORMATS[fmt]["elem"] == "e2m1":
        _skip_unless_sm100()  # the E2M1 conversion needs the sm_100 cvt
    from sglang.kernels.ops.lora.moe.cutedsl.api import scale_atom_bytes
    from sglang.kernels.ops.lora.moe.dispatch_contiguous import (
        contiguous_m_pad_ceiling,
        dispatch_fill_rows_contiguous_blockscaled,
        dispatch_layout_contiguous,
    )
    from sglang.kernels.ops.lora.moe.dispatch_masked import (
        dispatch_fill_masked_blockscaled,
    )
    from sglang.kernels.ops.lora.moe.quant_blockscaled import (
        BlockScaledRows,
        quantize_rows,
    )

    vec, elem = FORMATS[fmt]["vec"], FORMATS[fmt]["elem"]
    device = torch.device("cuda")
    num_tokens, top_k, num_experts = 24, 4, 8
    generator = torch.Generator().manual_seed(0x5B1F)
    scores = torch.rand((num_tokens, num_experts), generator=generator)
    topk_ids = torch.topk(scores, top_k, dim=1).indices.to(torch.int32)
    topk_ids[0] = -1  # a token without experts
    topk_ids[1, 2] = -1  # one dropped pair
    topk_ids = topk_ids.to(device)
    x = (torch.randn((num_tokens, k), generator=generator) * 0.3).to(torch.bfloat16)
    x[2].zero_()
    x[3, ::128] = 32
    x = x.to(device)
    global_scale = 0.3 if fmt == "nvfp4" else None
    pair_to_row = torch.empty(num_tokens * top_k, dtype=torch.int32, device=device)
    valid = topk_ids.view(-1) >= 0
    tokens = torch.arange(num_tokens, device=device).repeat_interleave(top_k)[valid]

    def rows(num_rows, rows_per_slab):
        data = torch.full(
            (num_rows, k // 2 if elem == "e2m1" else k), 0x55, dtype=torch.uint8
        ).to(device)
        if elem != "e2m1":
            data = data.view(torch.float8_e4m3fn)
        scales = torch.full(
            (scale_atom_bytes(rows_per_slab, k, vec, num_rows // rows_per_slab),),
            0x55,
            dtype=torch.uint8,
        ).to(device)
        return BlockScaledRows(
            data=data,
            scales=scales,
            sf_vec_size=vec,
            rows_per_slab=rows_per_slab,
            global_scale=global_scale,
        )

    for domain in ("masked", "contiguous"):
        if domain == "masked":
            m_max = 32
            masked_m = torch.empty(num_experts, dtype=torch.int32, device=device)
            target = rows(num_experts * m_max, m_max)
            dispatch_fill_masked_blockscaled(
                x,
                topk_ids,
                top_k,
                masked_m_out=masked_m,
                pair_to_row_out=pair_to_row,
                rows_out=target,
            )
            assert torch.equal(
                masked_m,
                torch.bincount(
                    topk_ids.view(-1)[valid].long(), minlength=num_experts
                ).to(torch.int32),
            )
        else:
            counts = torch.empty(num_experts, dtype=torch.int32, device=device)
            offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
            m_pad = contiguous_m_pad_ceiling(pair_to_row.numel(), num_experts, 128)
            dispatch_layout_contiguous(
                x,
                topk_ids,
                num_experts,
                top_k,
                128,
                seg_counts_out=counts,
                seg_offsets_out=offsets,
                pair_to_row_out=pair_to_row,
            )
            target = rows(m_pad, m_pad)
            dispatch_fill_rows_contiguous_blockscaled(
                x, topk_ids, pair_to_row, rows_out=target
            )
        assert bool(torch.all(pair_to_row[~valid] == -1))
        # The row quantizer over each token's row at its destination.
        reference = rows(target.data.shape[0], target.rows_per_slab)
        source = torch.zeros(
            (target.data.shape[0], k), dtype=torch.bfloat16, device=device
        )
        source[pair_to_row[valid].long()] = x[tokens]
        quantize_rows(source, reference, pair_to_row)
        assert torch.equal(
            target.data.view(torch.uint8), reference.data.view(torch.uint8)
        ), domain
        assert torch.equal(target.scales, reference.scales), domain


@pytest.mark.parametrize("fmt", sorted(FORMATS))
@pytest.mark.parametrize("rows_mode", ["masked", "contiguous"])
def test_blockscaled_grouped_gemm_matches_dequantized_reference(fmt, rows_mode):
    _skip_unless_sm100()
    from sglang.kernels.ops.lora.moe.cutedsl.api import (
        GroupedGemmConfig,
        prepare_contiguous_blockscaled,
        prepare_masked_blockscaled,
        scale_atom_bytes,
    )
    from sglang.kernels.ops.lora.moe.cutedsl.schedule_builder import (
        build_dual_stage_schedules_contiguous,
        build_dual_stage_schedules_masked,
    )
    from sglang.kernels.ops.lora.moe.quant_blockscaled import pack_scale_atoms

    vec = FORMATS[fmt]["vec"]
    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(3)
    experts, n, k, m_max = 4, 256, 512, 256
    weights = (torch.randn(experts, n, k, generator=g) * 0.05).to(device)
    tokens = (torch.randn(experts, m_max, k, generator=g) * 0.5).to(device)
    w_gs = torch.full((experts,), 0.01, device=device)
    t_gs = torch.full((experts,), 0.1, device=device)
    w_data, w_sf, w_deq = _quantize(fmt, weights, w_gs)
    t_data, t_sf, t_deq = _quantize(fmt, tokens, t_gs)
    alpha, alpha_rows_per_half = None, 1
    if fmt == "nvfp4":
        # Global scales reach the output through the per-expert alpha epilogue;
        # the second half of the weight rows doubles its alpha to prove the
        # gate/up selection.
        alpha = torch.stack([w_gs * t_gs, 2 * w_gs * t_gs], dim=1).contiguous()
        alpha_rows_per_half = n // 2
        w_deq = w_deq.clone()
        w_deq[:, n // 2 :] *= 2
    sf_weights = pack_scale_atoms(w_sf)
    assert sf_weights.numel() == scale_atom_bytes(n, k, vec, experts)
    config = GroupedGemmConfig(mma_tiler_mn=(128, 128), persistent_clusters=8)
    masked_m = torch.tensor([200, 0, 128, 7], dtype=torch.int32, device=device)

    if rows_mode == "masked":
        sf_tokens = pack_scale_atoms(t_sf)
        c = torch.zeros((experts, m_max, n), dtype=torch.bfloat16, device=device)
        schedule, tiles, _, _ = build_dual_stage_schedules_masked(
            masked_m,
            m_max=m_max,
            token_width=128,
            n_gemm1=n,
            n_gemm2=n,
            output_width=128,
            cluster_shape_mn=(1, 1),
            use_2cta_instrs=False,
        )
        prepared = prepare_masked_blockscaled(
            t_data,
            w_data,
            sf_tokens,
            sf_weights,
            c,
            masked_m,
            config=config,
            sf_vec_size=vec,
            alpha=alpha,
            alpha_rows_per_half=alpha_rows_per_half,
            direct_schedule=schedule,
            schedule_tiles=tiles,
        )
        prepared.launch()
        torch.cuda.synchronize()
        for e in range(experts):
            rows = int(masked_m[e])
            if rows == 0:
                continue
            ref = t_deq[e, :rows] @ w_deq[e].T
            assert _rel_l2(c[e, :rows].float(), ref) < 1.5e-2, (fmt, e)
    else:
        seg_counts = masked_m
        seg_offsets = torch.zeros(experts + 1, dtype=torch.int32, device=device)
        seg_offsets[1:] = torch.cumsum(((seg_counts + 127) // 128) * 128, 0)
        m_ceil = int(seg_offsets[-1])
        flat = torch.zeros((m_ceil, k), device=device)
        for e in range(experts):
            start, rows = int(seg_offsets[e]), int(seg_counts[e])
            flat[start : start + rows] = tokens[e, :rows]
        f_data, f_sf, f_deq = _quantize(fmt, flat.unsqueeze(0), t_gs[:1])
        f_data = f_data[0]
        sf_tokens = pack_scale_atoms(f_sf)
        assert sf_tokens.numel() == scale_atom_bytes(m_ceil, k, vec, 1)
        c = torch.zeros((m_ceil, n), dtype=torch.bfloat16, device=device)
        schedule, tiles, _, _ = build_dual_stage_schedules_contiguous(
            seg_counts,
            m_pad_ceiling=m_ceil,
            max_expert_rows=m_max,
            m_alignment=128,
            token_width=128,
            n_gemm1=n,
            n_gemm2=n,
            output_width=128,
            cluster_shape_mn=(1, 1),
            use_2cta_instrs=False,
        )
        prepared = prepare_contiguous_blockscaled(
            f_data,
            w_data,
            sf_tokens,
            sf_weights,
            c,
            seg_offsets,
            config=config,
            sf_vec_size=vec,
            alpha=alpha,
            alpha_rows_per_half=alpha_rows_per_half,
            direct_schedule=schedule,
            schedule_tiles=tiles,
        )
        prepared.launch()
        torch.cuda.synchronize()
        for e in range(experts):
            start, rows = int(seg_offsets[e]), int(seg_counts[e])
            if rows == 0:
                continue
            ref = f_deq[0, start : start + rows] @ w_deq[e].T
            assert _rel_l2(c[start : start + rows].float(), ref) < 1.5e-2, (fmt, e)


def test_blockscaled_alpha_split_must_align_with_output_tiles():
    """The epilogue applies one alpha per output tile, so a gate/up boundary
    inside a tile, or more halves than alpha columns, is refused before compiling."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    from sglang.kernels.ops.lora.moe.cutedsl.api import (
        GroupedGemmConfig,
        prepare_contiguous_blockscaled,
        scale_atom_bytes,
    )

    device = torch.device("cuda")
    experts, n, k, m_ceil = 1, 384, 256, 128
    a = torch.zeros((m_ceil, k // 2), dtype=torch.uint8, device=device)
    b = torch.zeros((experts, n, k // 2), dtype=torch.uint8, device=device)
    sf_tokens = torch.zeros(
        scale_atom_bytes(m_ceil, k, 16), dtype=torch.uint8, device=device
    )
    sf_weights = torch.zeros(
        scale_atom_bytes(n, k, 16, experts), dtype=torch.uint8, device=device
    )
    c = torch.zeros((m_ceil, n), dtype=torch.bfloat16, device=device)
    seg_offsets = torch.zeros((experts + 1,), dtype=torch.int32, device=device)
    alpha = torch.ones((experts, 2), dtype=torch.float32, device=device)
    # A boundary at row 192 falls inside the tile covering rows 128-255; a
    # boundary every 128 rows needs three columns.
    for rows_per_half in (192, 128):
        with pytest.raises(ValueError, match="alpha changes every"):
            prepare_contiguous_blockscaled(
                a,
                b,
                sf_tokens,
                sf_weights,
                c,
                seg_offsets,
                config=GroupedGemmConfig(mma_tiler_mn=(128, 128)),
                sf_vec_size=16,
                alpha=alpha,
                alpha_rows_per_half=rows_per_half,
            )


@pytest.mark.parametrize("fmt", sorted(FORMATS))
def test_fused_activation_quant_matches_row_quantizer(fmt):
    """act_delta's block-scaled output equals the row quantizer on its BF16 rows."""
    _skip_unless_sm100()
    from sglang.kernels.ops.lora.moe.activation_delta import act_delta
    from sglang.kernels.ops.lora.moe.cutedsl.api import scale_atom_bytes
    from sglang.kernels.ops.lora.moe.quant_blockscaled import (
        BlockScaledRows,
        quantize_rows,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    num_tokens, top_k, num_experts, inter = 7, 3, 8, 384
    num_pairs = num_tokens * top_k
    vec, elem = FORMATS[fmt]["vec"], FORMATS[fmt]["elem"]
    gateup = torch.randn(num_pairs, 2 * inter, dtype=torch.bfloat16, device=device)
    delta = 0.1 * torch.randn(
        num_tokens, top_k, 2 * inter, dtype=torch.bfloat16, device=device
    )
    topk_ids = torch.randint(
        0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device
    )
    topk_ids[0, 0] = -1
    pair_to_row = torch.arange(num_pairs, dtype=torch.int32, device=device)
    pair_to_row[0] = -1

    def rows_spec():
        data = torch.zeros(
            (num_pairs, inter // 2 if elem == "e2m1" else inter),
            dtype=torch.uint8 if elem == "e2m1" else torch.float8_e4m3fn,
            device=device,
        )
        scales = torch.zeros(
            scale_atom_bytes(num_pairs, inter, vec), dtype=torch.uint8, device=device
        )
        return BlockScaledRows(
            data=data,
            scales=scales,
            sf_vec_size=vec,
            rows_per_slab=num_pairs,
            global_scale=0.3 if fmt == "nvfp4" else None,
        )

    def run(act_quant):
        act = torch.zeros(num_pairs, inter, dtype=torch.bfloat16, device=device)
        pairs = torch.zeros(
            num_tokens, top_k, inter, dtype=torch.bfloat16, device=device
        )
        act_delta(
            gateup,
            delta,
            act,
            pairs,
            pair_to_row,
            topk_ids,
            num_local_experts=num_experts,
            act_quant=act_quant,
        )
        return act

    ref_act = run(None)
    fused = rows_spec()
    assert torch.equal(run(fused), ref_act)
    separate = rows_spec()
    quantize_rows(ref_act, separate, pair_to_row)
    assert torch.equal(fused.data, separate.data)
    assert torch.equal(fused.scales, separate.scales)


def test_triton_row_quantizer_addresses_rows_past_2_31_elements():
    """Row offsets are int64. With int32 ones, a Beam TP4 decode batch of 16k
    tokens (256 slabs x 16384 rows x 640) quantized experts 205-255 wrong."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    from sglang.kernels.ops.lora.moe.quant_blockscaled import (
        BlockScaledRows,
        quantize_rows,
    )

    device = torch.device("cuda")
    k, vec = 128, 32
    rows = 2**31 // k + 2048  # the last rows' offsets pass 2**31 elements
    if torch.cuda.mem_get_info(device)[0] < 8 * 2**30:
        pytest.skip("needs 8 GiB of free device memory")
    x = torch.zeros((rows, k), dtype=torch.bfloat16, device=device)
    generator = torch.Generator(device="cpu").manual_seed(3)
    pattern = (torch.randn(k, generator=generator) * 0.3).to(torch.bfloat16)
    probes = torch.tensor(
        [0, 2**31 // k - 1, 2**31 // k, rows - 1], dtype=torch.int32, device=device
    )
    x[probes.long()] = pattern.to(device)
    out = torch.empty((rows, k), dtype=torch.float8_e4m3fn, device=device)
    row_atoms = -(-rows // 128)  # k // vec = 4 groups: one atom column
    sf = torch.zeros(row_atoms * 512, dtype=torch.uint8, device=device)
    target = BlockScaledRows(data=out, scales=sf, sf_vec_size=vec, rows_per_slab=rows)
    quantize_rows(x, target, probes)
    torch.cuda.synchronize()

    codes = out.view(torch.uint8)[probes.long()]
    assert codes[0].any()
    assert torch.equal(codes, codes[:1].expand_as(codes))

    def scale_bytes(row: int) -> torch.Tensor:
        base = (row // 128) * 512 + (row % 32) * 16 + (row // 32 % 4) * 4
        return sf[base : base + 4]

    reference = scale_bytes(0)
    assert reference.any()
    for row in probes.tolist()[1:]:
        assert torch.equal(scale_bytes(row), reference), row


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


@pytest.mark.parametrize("fmt", sorted(FORMATS))
@pytest.mark.parametrize("rows_mode", ["masked", "contiguous"])
def test_blockscaled_epilogue_lora_matches_references(fmt, rows_mode):
    """The down GEMM with the epilogue LoRA: every row of a tile gets its own
    slot's ``bridge @ B_slot[expert]^T`` added in FP32 before the alpha
    epilogue and one BF16 rounding. Against an FP64 target it is at least as
    close as the separate path (BF16 base plus BF16 delta), zero planes or
    zero slots reproduce the plain kernel bitwise, and rows without an
    adapter stay base rows."""
    _skip_unless_sm100()
    from sglang.kernels.ops.lora.moe.cutedsl.api import (
        GroupedGemmConfig,
        prepare_contiguous_blockscaled,
        prepare_masked_blockscaled,
    )
    from sglang.kernels.ops.lora.moe.cutedsl.schedule_builder import (
        build_dual_stage_schedules_contiguous,
        build_dual_stage_schedules_masked,
    )
    from sglang.kernels.ops.lora.moe.quant_blockscaled import pack_scale_atoms

    vec = FORMATS[fmt]["vec"]
    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(11)
    experts, n, k, m_max, slots, rank = 4, 256, 512, 256, 2, 16
    weights = (torch.randn(experts, n, k, generator=g) * 0.05).to(device)
    tokens = (torch.randn(experts, m_max, k, generator=g) * 0.5).to(device)
    w_gs = torch.full((experts,), 0.01, device=device)
    t_gs = torch.full((experts,), 0.1, device=device)
    w_data, w_sf, w_deq = _quantize(fmt, weights, w_gs)
    sf_weights = pack_scale_atoms(w_sf)
    alpha = None
    alpha_values = torch.ones(experts, device=device, dtype=torch.float64)
    if fmt == "nvfp4":
        # One alpha per expert over every output row, as the down GEMM uses it.
        alpha = (w_gs * t_gs).reshape(experts, 1).contiguous()
        alpha_values = alpha.reshape(-1).double()
    config = GroupedGemmConfig(mma_tiler_mn=(128, 128), persistent_clusters=8)
    masked_m = torch.tensor([200, 0, 128, 7], dtype=torch.int32, device=device)
    # LoRA operands: per (slot, expert) down-B weights, a bridge per row. The
    # bridge is the A kernel's FP32 accumulator: each path rounds it to BF16
    # once (the separate path as is, the epilogue path divided by alpha).
    lora_w = (torch.randn(slots, experts, n, rank, generator=g) * 0.2).to(
        device=device, dtype=torch.bfloat16
    )
    bridge = (torch.randn(experts, m_max, rank, generator=g) * 0.5).to(device)
    # Rows alternate slots 0, 1 and no adapter (-1) so tiles mix all three.
    row_slot = (torch.arange(m_max) % 3 - 1).to(device)
    slots_t = torch.tensor([slots], dtype=torch.int32, device=device)

    def planes_for(expert_rows, bridge_rows, slot_rows, expert_of_row):
        """Zeroed planes with each row's bridge on its slot's plane, divided by
        the row's expert alpha before its one BF16 rounding, as the down A
        kernel stores it."""
        planes = torch.zeros(
            (slots, expert_rows, rank), dtype=torch.bfloat16, device=device
        )
        for s in range(slots):
            mask = slot_rows == s
            scaled = bridge_rows[mask] / alpha_values[expert_of_row[mask]].float()[:, None]
            planes[s, mask] = scaled.to(torch.bfloat16)
        return planes

    def target_and_separate(rows_deq, expert_of_row, slot_rows, bridge_rows, c_base):
        """FP64 target A @ W^T + bridge @ B_slot[e]^T per row (the dequantized
        operands carry the global scales the kernel applies as alpha), and the
        separate path's composition BF16(base) + BF16(BF16(bridge) @ B^T)."""
        base = torch.einsum(
            "rk,rnk->rn", rows_deq.double(), w_deq[expert_of_row].double()
        )
        adapted = (slot_rows >= 0).double()[:, None]
        w_rows = lora_w[slot_rows.clamp(min=0), expert_of_row].double()
        delta = torch.einsum("rq,rnq->rn", bridge_rows.double(), w_rows) * adapted
        delta_bf16_bridge = (
            torch.einsum(
                "rq,rnq->rn", bridge_rows.to(torch.bfloat16).double(), w_rows
            )
            * adapted
        )
        separate = c_base.double() + delta_bf16_bridge.to(torch.bfloat16).double()
        return base + delta, separate

    if rows_mode == "masked":
        t_data, t_sf, t_deq = _quantize(fmt, tokens, t_gs)
        sf_tokens = pack_scale_atoms(t_sf)
        schedule, tiles, _, _ = build_dual_stage_schedules_masked(
            masked_m, m_max=m_max, token_width=128, n_gemm1=n, n_gemm2=n,
            output_width=128, cluster_shape_mn=(1, 1), use_2cta_instrs=False,
        )
        expert_of_row = torch.arange(experts, device=device).repeat_interleave(m_max)
        slot_rows = row_slot.repeat(experts)
        bridge_rows = bridge.reshape(-1, rank)
        planes = planes_for(experts * m_max, bridge_rows, slot_rows, expert_of_row)
        planes = planes.view(slots, experts, m_max, rank).transpose(0, 1).reshape(
            slots * experts, m_max, rank
        ).contiguous()
        # The kernel indexes planes by slot * experts + expert.
        planes = planes.view(experts, slots, m_max, rank).transpose(0, 1).reshape(
            slots * experts, m_max, rank
        ).contiguous()
        live = torch.cat(
            [torch.arange(e * m_max, e * m_max + int(masked_m[e]), device=device) for e in range(experts)]
        )

        def run(lora):
            c = torch.zeros((experts, m_max, n), dtype=torch.bfloat16, device=device)
            prepared = prepare_masked_blockscaled(
                t_data, w_data, sf_tokens, sf_weights, c, masked_m, config=config,
                sf_vec_size=vec, alpha=alpha, alpha_rows_per_half=1 << 30,
                direct_schedule=schedule, schedule_tiles=tiles,
                **(lora or {}),
            )
            prepared.launch()
            torch.cuda.synchronize()
            return c.reshape(-1, n)

        rows_deq = t_deq.reshape(-1, k)
        lora_args = dict(
            lora_w=lora_w.reshape(slots * experts, n, rank), lora_planes=planes, lora_slots=slots_t
        )
    else:
        seg_counts = masked_m
        seg_offsets = torch.zeros(experts + 1, dtype=torch.int32, device=device)
        seg_offsets[1:] = torch.cumsum(((seg_counts + 127) // 128) * 128, 0)
        m_ceil = int(seg_offsets[-1])
        flat = torch.zeros((m_ceil, k), device=device)
        expert_of_row = torch.zeros(m_ceil, dtype=torch.long, device=device)
        slot_rows = torch.full((m_ceil,), -1, dtype=torch.long, device=device)
        bridge_rows = torch.zeros((m_ceil, rank), device=device)
        live = []
        for e in range(experts):
            start, rows = int(seg_offsets[e]), int(seg_counts[e])
            flat[start : start + rows] = tokens[e, :rows]
            expert_of_row[start : start + rows] = e
            slot_rows[start : start + rows] = row_slot[:rows]
            bridge_rows[start : start + rows] = bridge[e, :rows]
            live.append(torch.arange(start, start + rows, device=device))
        live = torch.cat(live)
        f_data, f_sf, f_deq = _quantize(fmt, flat.unsqueeze(0), t_gs[:1])
        f_data, rows_deq = f_data[0], f_deq[0]
        sf_tokens = pack_scale_atoms(f_sf)
        schedule, tiles, _, _ = build_dual_stage_schedules_contiguous(
            seg_counts, m_pad_ceiling=m_ceil, max_expert_rows=m_max, m_alignment=128,
            token_width=128, n_gemm1=n, n_gemm2=n, output_width=128,
            cluster_shape_mn=(1, 1), use_2cta_instrs=False,
        )
        planes = planes_for(m_ceil, bridge_rows, slot_rows, expert_of_row)

        def run(lora):
            c = torch.zeros((m_ceil, n), dtype=torch.bfloat16, device=device)
            prepared = prepare_contiguous_blockscaled(
                f_data, w_data, sf_tokens, sf_weights, c, seg_offsets, config=config,
                sf_vec_size=vec, alpha=alpha, alpha_rows_per_half=1 << 30,
                direct_schedule=schedule, schedule_tiles=tiles,
                **(lora or {}),
            )
            prepared.launch()
            torch.cuda.synchronize()
            return c

        lora_args = dict(
            lora_w=lora_w.reshape(slots * experts, n, rank), lora_planes=planes, lora_slots=slots_t
        )

    c_base = run(None)
    c_fused = run(lora_args)
    target, separate = target_and_separate(rows_deq, expert_of_row, slot_rows, bridge_rows, c_base)
    err_fused = _rel_l2(c_fused.double()[live], target[live])
    err_separate = _rel_l2(separate[live], target[live])
    assert err_fused < 1.5e-2 and err_separate < 1.5e-2, (fmt, rows_mode, err_fused, err_separate)
    # One rounding after the FP32 sum is never worse than two.
    assert err_fused <= err_separate * 1.05, (fmt, rows_mode, err_fused, err_separate)
    # Rows without an adapter are base rows, bitwise.
    base_rows = live[slot_rows[live] < 0]
    assert torch.equal(c_fused[base_rows], c_base[base_rows])
    # Adapted rows actually moved.
    adapted_rows = live[slot_rows[live] >= 0]
    assert not torch.equal(c_fused[adapted_rows], c_base[adapted_rows])
    # Zero planes, or no slots, add exact zeros.
    zero = dict(lora_args, lora_planes=torch.zeros_like(lora_args["lora_planes"]))
    assert torch.equal(run(zero), c_base)
    none = dict(lora_args, lora_slots=torch.zeros_like(slots_t))
    assert torch.equal(run(none), c_base)
    # Only the first slot: slot-1 rows become base rows.
    first = dict(lora_args, lora_slots=torch.ones_like(slots_t))
    c_first = run(first)
    slot1 = live[slot_rows[live] == 1]
    slot0 = live[slot_rows[live] == 0]
    assert torch.equal(c_first[slot1], c_base[slot1])
    assert torch.equal(c_first[slot0], c_fused[slot0])


def _wait_bounded(seconds: float) -> None:
    """Wait for the current stream with a deadline: a deadlock between the
    kernel's warps would otherwise hang the process."""
    done = torch.cuda.Event()
    done.record()
    deadline = time.monotonic() + seconds
    while not done.query():
        if time.monotonic() > deadline:
            pytest.fail(
                f"the block-scaled GEMM did not finish within {seconds:.0f} s "
                "(LoRA pipeline deadlock?)"
            )
        time.sleep(0.005)


class _EpilogueDownProblem:
    """A down GEMM over four experts (one empty, one partial tile) on two
    persistent CTAs, every row carrying one of eight adapter slots or none,
    with the operands the down A kernel would place for any set of resident
    slots."""

    capacity = 8

    def __init__(self, fmt: str, rows_mode: str) -> None:
        from sglang.kernels.ops.lora.moe.cutedsl.api import (
            GroupedGemmConfig,
            prepare_contiguous_blockscaled,
            prepare_masked_blockscaled,
        )
        from sglang.kernels.ops.lora.moe.cutedsl.schedule_builder import (
            build_dual_stage_schedules_contiguous,
            build_dual_stage_schedules_masked,
        )
        from sglang.kernels.ops.lora.moe.quant_blockscaled import pack_scale_atoms

        vec = FORMATS[fmt]["vec"]
        device = torch.device("cuda")
        g = torch.Generator(device="cpu").manual_seed(13)
        experts, n, k, m_max, rank = 4, 256, 512, 256, 16
        self.experts, self.n, self.rank = experts, n, rank
        weights = (torch.randn(experts, n, k, generator=g) * 0.05).to(device)
        tokens = (torch.randn(experts, m_max, k, generator=g) * 0.5).to(device)
        w_gs = torch.full((experts,), 0.01, device=device)
        t_gs = torch.full((experts,), 0.1, device=device)
        w_data, w_sf, self.w_deq = _quantize(fmt, weights, w_gs)
        sf_weights = pack_scale_atoms(w_sf)
        alpha = None
        self.alpha_values = torch.ones(experts, device=device, dtype=torch.float64)
        if fmt == "nvfp4":
            alpha = (w_gs * t_gs).reshape(experts, 1).contiguous()
            self.alpha_values = alpha.reshape(-1).double()
        # Eight output tiles over two CTAs: each reuses its LoRA stages across tiles.
        config = GroupedGemmConfig(mma_tiler_mn=(128, 128), persistent_clusters=2)
        masked_m = torch.tensor([200, 0, 128, 7], dtype=torch.int32, device=device)
        self.lora_w = (
            torch.randn(self.capacity, experts, n, rank, generator=g) * 0.2
        ).to(device=device, dtype=torch.bfloat16)
        bridge = (torch.randn(experts, m_max, rank, generator=g) * 0.5).to(device)
        # Rows cycle through the eight slots and the base path.
        row_slot = (torch.arange(m_max) % (self.capacity + 1) - 1).to(device)
        common = dict(
            config=config, sf_vec_size=vec, alpha=alpha, alpha_rows_per_half=1 << 30
        )
        if rows_mode == "masked":
            t_data, t_sf, t_deq = _quantize(fmt, tokens, t_gs)
            sf_tokens = pack_scale_atoms(t_sf)
            schedule, tiles, _, _ = build_dual_stage_schedules_masked(
                masked_m,
                m_max=m_max,
                token_width=128,
                n_gemm1=n,
                n_gemm2=n,
                output_width=128,
                cluster_shape_mn=(1, 1),
                use_2cta_instrs=False,
            )
            self.expert_of_row = torch.arange(experts, device=device).repeat_interleave(
                m_max
            )
            self.slot_rows = row_slot.repeat(experts)
            self.bridge_rows = bridge.reshape(-1, rank)
            self.rows_deq = t_deq.reshape(-1, k)
            self.live = torch.cat(
                [
                    torch.arange(e * m_max, e * m_max + int(masked_m[e]), device=device)
                    for e in range(experts)
                ]
            )
            # One plane per (slot, expert) slab of m_max rows.
            self.planes_per_slot, self.rows_per_plane = experts, m_max

            def prepare(lora):
                c = torch.zeros(
                    (experts, m_max, n), dtype=torch.bfloat16, device=device
                )
                prepared = prepare_masked_blockscaled(
                    t_data,
                    w_data,
                    sf_tokens,
                    sf_weights,
                    c,
                    masked_m,
                    direct_schedule=schedule,
                    schedule_tiles=tiles,
                    **common,
                    **lora,
                )
                return c.reshape(-1, n), prepared

        else:
            seg_offsets = torch.zeros(experts + 1, dtype=torch.int32, device=device)
            seg_offsets[1:] = torch.cumsum(((masked_m + 127) // 128) * 128, 0)
            m_ceil = int(seg_offsets[-1])
            flat = torch.zeros((m_ceil, k), device=device)
            self.expert_of_row = torch.zeros(m_ceil, dtype=torch.long, device=device)
            self.slot_rows = torch.full((m_ceil,), -1, dtype=torch.long, device=device)
            self.bridge_rows = torch.zeros((m_ceil, rank), device=device)
            live = []
            for e in range(experts):
                start, rows = int(seg_offsets[e]), int(masked_m[e])
                flat[start : start + rows] = tokens[e, :rows]
                self.expert_of_row[start : start + rows] = e
                self.slot_rows[start : start + rows] = row_slot[:rows]
                self.bridge_rows[start : start + rows] = bridge[e, :rows]
                live.append(torch.arange(start, start + rows, device=device))
            self.live = torch.cat(live)
            f_data, f_sf, f_deq = _quantize(fmt, flat.unsqueeze(0), t_gs[:1])
            f_data, self.rows_deq = f_data[0], f_deq[0]
            sf_tokens = pack_scale_atoms(f_sf)
            schedule, tiles, _, _ = build_dual_stage_schedules_contiguous(
                masked_m,
                m_pad_ceiling=m_ceil,
                max_expert_rows=m_max,
                m_alignment=128,
                token_width=128,
                n_gemm1=n,
                n_gemm2=n,
                output_width=128,
                cluster_shape_mn=(1, 1),
                use_2cta_instrs=False,
            )
            # One plane per slot over the flat rows.
            self.planes_per_slot, self.rows_per_plane = 1, m_ceil

            def prepare(lora):
                c = torch.zeros((m_ceil, n), dtype=torch.bfloat16, device=device)
                prepared = prepare_contiguous_blockscaled(
                    f_data,
                    w_data,
                    sf_tokens,
                    sf_weights,
                    c,
                    seg_offsets,
                    direct_schedule=schedule,
                    schedule_tiles=tiles,
                    **common,
                    **lora,
                )
                return c, prepared

        self._prepare = prepare
        self.device = device

    def operands(self, placement: list) -> dict:
        """Operands for ``len(placement)`` resident slots: position ``p``
        carries slot ``placement[p]``'s rows (bridge over the row's expert
        alpha, rounded once) and weights, or slot ``p``'s weights and no
        rows for ``None``."""
        resident = len(placement)
        weights = torch.empty(
            (resident, self.experts, self.n, self.rank),
            dtype=torch.bfloat16,
            device=self.device,
        )
        planes = torch.zeros(
            (resident, self.planes_per_slot * self.rows_per_plane, self.rank),
            dtype=torch.bfloat16,
            device=self.device,
        )
        for p, s in enumerate(placement):
            weights[p] = self.lora_w[p if s is None else s]
            if s is None:
                continue
            mask = self.slot_rows == s
            scaled = (
                self.bridge_rows[mask]
                / self.alpha_values[self.expert_of_row[mask]].float()[:, None]
            )
            planes[p, mask] = scaled.to(torch.bfloat16)
        return dict(
            lora_w=weights.reshape(-1, self.n, self.rank),
            lora_planes=planes.reshape(-1, self.rows_per_plane, self.rank),
            lora_slots=torch.tensor([resident], dtype=torch.int32, device=self.device),
        )

    def run(self, lora: dict | None = None) -> torch.Tensor:
        c, prepared = self._prepare(lora or {})
        prepared.launch()
        _wait_bounded(60.0)
        return c

    def target(self, placement: list) -> torch.Tensor:
        """FP64 ``A @ W^T`` plus each placed row's ``bridge @ B_slot[e]^T``."""
        base = torch.einsum(
            "rk,rnk->rn",
            self.rows_deq.double(),
            self.w_deq[self.expert_of_row].double(),
        )
        placed = torch.zeros_like(self.slot_rows, dtype=torch.bool)
        for s in placement:
            if s is not None:
                placed |= self.slot_rows == s
        w_rows = self.lora_w[self.slot_rows.clamp(min=0), self.expert_of_row].double()
        delta = torch.einsum("rq,rnq->rn", self.bridge_rows.double(), w_rows)
        return base + delta * placed.double()[:, None]


# Resident slots as the pool presents them (position -> slot of the rows it
# carries, None = a resident adapter without rows): bounds below, at and
# above the kernel's four TMA stages, one adapter alone in slot 4 or 7,
# mixed slots, all eight, unused trailing slots, and residents without rows.
_SLOT_PLACEMENTS = {
    "one": [0],
    "three": [0, 1, 2],
    "four": [0, 1, 2, 3],
    "five": [0, 1, 2, 3, 4],
    "eight": [0, 1, 2, 3, 4, 5, 6, 7],
    "slot4_alone": [None, None, None, None, 4],
    "slot7_alone": [None, None, None, None, None, None, None, 7],
    "mixed": [None, 1, None, None, 4, None, None, 7],
    "trailing": [0, 1, None, None, None, None, None, None],
    "no_rows": [None] * 8,
}


@pytest.mark.parametrize("fmt", sorted(FORMATS))
@pytest.mark.parametrize("rows_mode", ["masked", "contiguous"])
def test_blockscaled_epilogue_lora_slot_bounds(fmt, rows_mode):
    """The epilogue LoRA over every resident-slot bound of an eight-slot pool
    (the kernel streams the slots through four TMA stages, so bounds past
    four reuse a stage within the tile). A row's result depends only on its
    own slot: bitwise the result of the same adapter served alone from slot
    0, base rows and rows of residents without rows stay base rows, and every
    bound lands on the FP64 target."""
    _skip_unless_sm100()
    problem = _EpilogueDownProblem(fmt, rows_mode)
    live, slot_rows = problem.live, problem.slot_rows
    c_base = problem.run()
    alone = {}
    for s in range(problem.capacity):
        alone[s] = problem.run(problem.operands([s]))
        assert _rel_l2(alone[s].double()[live], problem.target([s])[live]) < 1.5e-2, (
            fmt,
            rows_mode,
            s,
        )
        rows = live[slot_rows[live] == s]
        assert not torch.equal(alone[s][rows], c_base[rows]), (fmt, rows_mode, s)
    for name, placement in _SLOT_PLACEMENTS.items():
        c = problem.run(problem.operands(placement))
        expected = c_base.clone()
        for s in placement:
            if s is not None:
                rows = slot_rows == s
                expected[rows] = alone[s][rows]
        assert torch.equal(c[live], expected[live]), (fmt, rows_mode, name)
        assert _rel_l2(c.double()[live], problem.target(placement)[live]) < 1.5e-2, (
            fmt,
            rows_mode,
            name,
        )
