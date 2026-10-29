"""MX block-scaled CuTeDSL providers against references built from the same
quantization rule as the row quantizer, so the GEMMs are compared exactly.
"""

from __future__ import annotations

import pytest
import torch

from sglang.srt.lora.moe.base_gemm_provider import select_provider_cls
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_E = 4
_H = 512
_I = 256
_TOPK = 2
_T = 40
_VEC = 32

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


def _quant_mx(x: torch.Tensor, elem: str):
    """[..., K] -> (data, E8M0 bytes [..., K/32], exact dequant fp32)."""
    *lead, k = x.shape
    groups = x.float().reshape(*lead, k // _VEC, _VEC)
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
        table = _E2M1.to(x.device)
        idx = _e2m1_codes(scaled.abs())
        sign = scaled < 0
        values = torch.where(sign, -table[idx], table[idx])
        codes = (idx | (sign.long() << 3)).to(torch.uint8).reshape(*lead, k)
        data = codes[..., 0::2] | (codes[..., 1::2] << 4)
    else:
        q = scaled.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        values = q.float()
        data = q.reshape(*lead, k)
    return (
        data,
        (exponent + 127).to(torch.uint8),
        (values * scale[..., None]).reshape(*lead, k),
    )


def _quant_nvfp4(x: torch.Tensor, global_scale: float):
    """E2M1 pairs with E4M3 scales per 16 under one global scale (the row rule)."""
    *lead, k = x.shape
    groups = x.float().reshape(*lead, k // 16, 16)
    amax = groups.abs().amax(dim=-1)
    block = (amax / (6.0 * global_scale)).clamp(2.0**-6, 448.0).to(torch.float8_e4m3fn)
    denom = block.float() * global_scale
    scaled = (groups * (1.0 / denom)[..., None]).clamp(-6.0, 6.0)
    table = _E2M1.to(x.device)
    idx = _e2m1_codes(scaled.abs())
    sign = scaled < 0
    values = torch.where(sign, -table[idx], table[idx])
    codes = (idx | (sign.long() << 3)).to(torch.uint8).reshape(*lead, k)
    data = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return data, block.view(torch.uint8), (values * denom[..., None]).reshape(*lead, k)


_W13_GLOBAL = 0.02
_W2_GLOBAL = 0.01
_A13_GLOBAL = 0.25
_A2_GLOBAL = 0.05


def _quant_info(family: str, w13: torch.Tensor, w2: torch.Tensor):
    from sglang.srt.lora.moe.quant_info import (
        MoeLoraMxfp4QuantInfo,
        MoeLoraMxfp8QuantInfo,
        MoeLoraNvFp4QuantInfo,
    )

    if family == "nvfp4":
        # Distinct gate/up global scales exercise the alpha half selection.
        w13_gs = torch.tensor([[_W13_GLOBAL, 2 * _W13_GLOBAL]] * _E, device=w13.device)
        w13_q, w13_s, w13_eff = [], [], []
        for half in range(2):
            q, s, eff = _quant_nvfp4(
                w13[:, half * _I : (half + 1) * _I], w13_gs[0, half].item()
            )
            w13_q.append(q), w13_s.append(s), w13_eff.append(eff)
        w2_q, w2_s, w2_eff = _quant_nvfp4(w2, _W2_GLOBAL)
        info = MoeLoraNvFp4QuantInfo(
            w13_weight=torch.cat(w13_q, dim=1),
            w13_scale=torch.cat(w13_s, dim=1),
            w2_weight=w2_q,
            w2_scale=w2_s,
            elements_per_byte=2,
            num_local_experts=_E,
            intermediate_size=_I,
            hidden_size=_H,
            w13_global_scale=w13_gs,
            w2_global_scale=torch.full((_E,), _W2_GLOBAL, device=w2.device),
            w13_input_scale=_A13_GLOBAL,
            w2_input_scale=_A2_GLOBAL,
        )
        return info, torch.cat(w13_eff, dim=1), w2_eff, "nvfp4"

    elem = "e2m1" if family == "mxfp4" else "e4m3"
    w13_q, w13_s, w13_eff = _quant_mx(w13, elem)
    w2_q, w2_s, w2_eff = _quant_mx(w2, elem)
    cls = MoeLoraMxfp4QuantInfo if family == "mxfp4" else MoeLoraMxfp8QuantInfo
    if family == "mxfp4":
        # The loader keeps FP4 pairs as int8.
        w13_q, w2_q = w13_q.view(torch.int8), w2_q.view(torch.int8)
    info = cls(
        w13_weight=w13_q,
        w13_scale=w13_s,
        w2_weight=w2_q,
        w2_scale=w2_s,
        elements_per_byte=2 if family == "mxfp4" else 1,
        num_local_experts=_E,
        intermediate_size=_I,
        hidden_size=_H,
    )
    return info, w13_eff, w2_eff, elem


def _gather_pairs(slab_or_rows, pair_to_row, pair_idx):
    rows = (
        slab_or_rows.reshape(-1, slab_or_rows.shape[-1])
        if slab_or_rows.ndim == 3
        else slab_or_rows
    )
    return rows[pair_to_row[pair_idx].long()]


def _rel_l2(a, b):
    return (a - b).norm() / b.norm().clamp(min=1e-12)


def _quant_activation(x: torch.Tensor, elem: str, stage: str):
    """Dequantized activation rows as the provider quantizes them."""
    if elem == "nvfp4":
        gs = _A13_GLOBAL if stage == "gemm1" else _A2_GLOBAL
        return _quant_nvfp4(x, gs)[2]
    return _quant_mx(x, elem)[2]


@pytest.mark.parametrize("family", ["mxfp8", "mxfp4", "nvfp4"])
@pytest.mark.parametrize("rows", ["expert_major", "route_major"])
def test_blockscaled_gateup_and_down_match_reference(family, rows):
    _skip_unless_sm100()
    from sglang.srt.runtime_context import get_context

    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(7)

    def rand(*shape, scale=0.05):
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16).to(device)

    w13, w2 = rand(_E, 2 * _I, _H), rand(_E, _H, _I)
    hidden = rand(_T, _H, scale=0.5)
    topk_ids = torch.stack(
        [torch.randperm(_E, generator=g)[:_TOPK] for _ in range(_T)]
    ).to(device=device, dtype=torch.int32)
    topk_ids[1, 0] = -1  # an EP hole
    quant_info, w13_eff, w2_eff, elem = _quant_info(family, w13, w2)

    provider = select_provider_cls(rows, family, "cutedsl")(quant_info)
    with get_context().override_server_args():
        state = provider.prepare(hidden, topk_ids, _TOPK)
        gateup = torch.zeros(
            provider.gateup_out_shape(state), dtype=torch.bfloat16, device=device
        )
        provider.gateup(state, gateup)

    flat_ids = topk_ids.flatten()
    pair_idx = torch.nonzero(flat_ids >= 0, as_tuple=True)[0]
    experts = flat_ids[pair_idx].long()
    tokens = (pair_idx // _TOPK).long()
    pair_to_row = state.pair_to_row

    # The provider quantizes its rows with the same rule, so references are exact.
    hidden_eff = _quant_activation(hidden[tokens], elem, "gemm1")
    ref_gateup = torch.einsum("ph,pnh->pn", hidden_eff, w13_eff[experts])
    got_gateup = _gather_pairs(gateup, pair_to_row, pair_idx).float()
    assert _rel_l2(got_gateup, ref_gateup) < 1.5e-2

    act_pairs = (torch.randn_like(ref_gateup[:, :_I]) * 0.1).to(torch.bfloat16)
    act = torch.zeros(
        provider.act_out_shape(state), dtype=torch.bfloat16, device=device
    )
    act.reshape(-1, _I)[pair_to_row[pair_idx].long()] = act_pairs
    down = torch.zeros(
        provider.down_out_shape(state), dtype=torch.bfloat16, device=device
    )
    with get_context().override_server_args():
        provider.down(state, act, down)
    act_eff = _quant_activation(act_pairs, elem, "gemm2")
    ref_down = torch.einsum("pi,phi->ph", act_eff, w2_eff[experts])
    got_down = _gather_pairs(down, pair_to_row, pair_idx).float()
    assert _rel_l2(got_down, ref_down) < 1.5e-2


def test_block_scaled_admission():
    from types import SimpleNamespace

    from sglang.srt.lora.moe.base_gemm_provider import resolve_vendor
    from sglang.srt.lora.moe.quant_info import MoeLoraMxfp8QuantInfo

    assert resolve_vendor("mxfp8", "triton") == "cutedsl"
    assert resolve_vendor("mxfp4", "cutedsl") == "cutedsl"
    assert resolve_vendor("mxfp4", None) == "marlin"
    layer = SimpleNamespace(
        params_dtype=torch.bfloat16,
        w13_weight=torch.zeros((2, 128, 192), dtype=torch.float8_e4m3fn),
        w13_weight_scale_inv=torch.zeros((2, 128, 6), dtype=torch.uint8),
        w2_weight=torch.zeros((2, 192, 64), dtype=torch.float8_e4m3fn),
        w2_weight_scale_inv=torch.zeros((2, 192, 2), dtype=torch.uint8),
        num_local_experts=2,
        intermediate_size_per_partition=64,
        hidden_size=192,
    )
    with pytest.raises(NotImplementedError, match="hidden_size % 128"):
        MoeLoraMxfp8QuantInfo.from_layer(layer)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
