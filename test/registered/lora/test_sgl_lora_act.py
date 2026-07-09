import sys

import pytest
import torch

from sglang.srt.lora.sgl_lora.triton_ops import silu_mul_delta_masked
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


def _split_gate_up(row: torch.Tensor, *, gate_first: bool, interleaved: bool):
    if interleaved:
        first, second = row[0::2], row[1::2]
    else:
        inter = row.numel() // 2
        first, second = row[:inter], row[inter:]
    return (first, second) if gate_first else (second, first)


@pytest.mark.parametrize("gate_first", [True, False])
@pytest.mark.parametrize("interleaved", [True, False])
@pytest.mark.parametrize("with_delta", [True, False])
@pytest.mark.parametrize("inter", [37, 768])
def test_silu_mul_delta_masked_matches_reference(
    gate_first: bool,
    interleaved: bool,
    with_delta: bool,
    inter: int,
):
    torch.manual_seed(7)
    device = "cuda"
    num_experts, m_max = 2, 4
    num_tokens, top_k = 3, 2

    gate_up = torch.randn(
        num_experts,
        m_max,
        2 * inter,
        dtype=torch.bfloat16,
        device=device,
    )
    delta = torch.randn(
        num_tokens,
        top_k,
        2 * inter,
        dtype=torch.bfloat16,
        device=device,
    )
    topk_ids = torch.tensor(
        [[0, 1], [-1, 0], [1, -1]], dtype=torch.int32, device=device
    )
    # Each valid routed pair owns a distinct row in the masked expert layout.
    src2dst = torch.tensor([0, 4, 0, 1, 5, 0], dtype=torch.int32, device=device)

    act_out = torch.full(
        (num_experts, m_max, inter),
        -123.0,
        dtype=torch.bfloat16,
        device=device,
    )
    activation_lora_input = torch.full(
        (num_tokens, top_k, inter),
        -123.0,
        dtype=torch.bfloat16,
        device=device,
    )
    expected_act_out = act_out.clone()
    expected_lora_input = torch.zeros_like(activation_lora_input)

    for pair_idx in range(num_tokens * top_k):
        if int(topk_ids.view(-1)[pair_idx]) < 0:
            continue
        dst = int(src2dst[pair_idx])
        row = gate_up.view(-1, 2 * inter)[dst].float()
        gate, up = _split_gate_up(row, gate_first=gate_first, interleaved=interleaved)
        if with_delta:
            gate = gate + delta.view(-1, 2 * inter)[pair_idx, :inter].float()
            up = up + delta.view(-1, 2 * inter)[pair_idx, inter:].float()
        activated = torch.nn.functional.silu(gate) * up
        expected_act_out.view(-1, inter)[dst] = activated.to(torch.bfloat16)
        expected_lora_input.view(-1, inter)[pair_idx] = activated.to(torch.bfloat16)

    silu_mul_delta_masked(
        gate_up,
        delta if with_delta else None,
        act_out,
        activation_lora_input,
        src2dst,
        topk_ids,
        gate_first=gate_first,
        interleaved=interleaved,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(act_out, expected_act_out, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(
        activation_lora_input, expected_lora_input, rtol=1e-2, atol=1e-2
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
