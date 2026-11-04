"""Per-tensor FP8 provider against FP32 references from dequantized checkpoint
bytes, with static and dynamic per-tensor activation scales.
"""

from __future__ import annotations

import pytest
import torch

from sglang.srt.lora.moe.base_gemm_provider import select_provider_cls
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_E = 4
_H = 512
_I = 512
_TOPK = 2
_T = 32
_FMAX = torch.finfo(torch.float8_e4m3fn).max


def _tensor_quant(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-expert fp8 quant; returns (fp8 weight, fp32 scale [E])."""
    scale = weight.float().abs().amax(dim=(1, 2)).clamp(min=1e-6) / _FMAX
    return (weight.float() / scale[:, None, None]).to(torch.float8_e4m3fn), scale


def _case(device, hidden_size=_H):
    g = torch.Generator(device="cpu").manual_seed(7)

    def rand(*shape):
        return (torch.randn(*shape, generator=g) * 0.05).to(torch.bfloat16).to(device)

    w13 = rand(_E, 2 * _I, hidden_size)
    w2 = rand(_E, hidden_size, _I)
    hidden = rand(_T, hidden_size)
    topk_ids = torch.stack(
        [torch.randperm(_E, generator=g)[:_TOPK] for _ in range(_T)]
    ).to(device=device, dtype=torch.int32)
    return w13, w2, hidden, topk_ids


def _quant_info(w13, w2, hidden, act_pairs, static: bool, hidden_size=_H):
    from sglang.srt.lora.moe.quant_info import MoeLoraFp8TensorQuantInfo

    w13_q, w13_s = _tensor_quant(w13)
    w2_q, w2_s = _tensor_quant(w2)
    # A static checkpoint scale covers the activations it was calibrated on.
    a13 = (hidden.float().abs().amax() / _FMAX).reshape(()) if static else None
    a2 = (act_pairs.float().abs().amax() / _FMAX).reshape(()) if static else None
    info = MoeLoraFp8TensorQuantInfo(
        w13_weight=w13_q,
        w13_scale=w13_s,
        w2_weight=w2_q,
        w2_scale=w2_s,
        w13_input_scale=a13,
        w2_input_scale=a2,
        num_local_experts=_E,
        intermediate_size=_I,
        hidden_size=hidden_size,
    )
    return (
        info,
        w13_q.float() * w13_s[:, None, None],
        w2_q.float() * w2_s[:, None, None],
    )


def _gather_pairs(rows, pair_to_row, pair_idx):
    return rows.reshape(-1, rows.shape[-1])[pair_to_row[pair_idx].long()]


def _rel_l2(a, b):
    return (a - b).norm() / b.norm().clamp(min=1e-12)


@pytest.mark.parametrize("hidden_size", [_H, 7168])
@pytest.mark.parametrize("static", [True, False])
def test_fp8_tensor_gateup_and_down_match_reference(static, hidden_size):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    w13, w2, hidden, topk_ids = _case(device, hidden_size)
    flat_ids = topk_ids.flatten()
    pair_idx = torch.nonzero(flat_ids >= 0, as_tuple=True)[0]
    experts = flat_ids[pair_idx].long()
    tokens = (pair_idx // _TOPK).long()
    g = torch.Generator(device="cpu").manual_seed(11)
    act_pairs = (torch.randn(len(pair_idx), _I, generator=g) * 0.1).to(
        device=device, dtype=torch.bfloat16
    )
    quant_info, w13_eff, w2_eff = _quant_info(
        w13, w2, hidden, act_pairs, static, hidden_size=hidden_size
    )

    from sglang.srt.runtime_context import get_context

    provider = select_provider_cls("route_major", "fp8_tensor", "triton")(quant_info)
    with get_context().override_server_args():
        state = provider.prepare(hidden, topk_ids, _TOPK)
        gateup = torch.empty(
            provider.gateup_out_shape(state), dtype=torch.bfloat16, device=device
        )
        provider.gateup(state, gateup)
    pair_to_row = state.pair_to_row

    # One activation scale per tensor: the coarsest FP8 quantization.
    ref_gateup = torch.einsum("ph,pnh->pn", hidden[tokens].float(), w13_eff[experts])
    got_gateup = _gather_pairs(gateup, pair_to_row, pair_idx).float()
    assert _rel_l2(got_gateup, ref_gateup) < 8e-2

    act = torch.zeros(
        provider.act_out_shape(state), dtype=torch.bfloat16, device=device
    )
    act.reshape(-1, _I)[pair_to_row[pair_idx].long()] = act_pairs
    down = torch.empty(
        provider.down_out_shape(state), dtype=torch.bfloat16, device=device
    )
    with get_context().override_server_args():
        provider.down(state, act, down)
    ref_down = torch.einsum("pi,phi->ph", act_pairs.float(), w2_eff[experts])
    got_down = _gather_pairs(down, pair_to_row, pair_idx).float()
    assert _rel_l2(got_down, ref_down) < 8e-2


def test_fp8_tensor_rejects_unfolded_scales():
    from sglang.srt.lora.moe.quant_info import MoeLoraFp8TensorQuantInfo

    w13, w2, _hidden, _topk_ids = _case(torch.device("cpu"))
    w13_q, w13_s = _tensor_quant(w13)
    w2_q, w2_s = _tensor_quant(w2)
    layer = type(
        "Layer",
        (),
        dict(
            w13_weight=w13_q,
            w13_weight_scale=torch.stack([w13_s, w13_s], dim=1),
            w2_weight=w2_q,
            w2_weight_scale=w2_s,
            w13_input_scale=None,
            w2_input_scale=None,
            num_local_experts=_E,
        ),
    )
    with pytest.raises(ValueError, match=r"w13_scale must be fp32 \[4\]"):
        MoeLoraFp8TensorQuantInfo.from_layer(layer)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
