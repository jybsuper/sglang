import sys

import pytest
import torch

from sglang.srt.lora.sgl_lora.triton_ops.fused_down_finalize import (
    fused_down_b_finalize,
    indexed_down_b_add,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


def _reference(
    down_output,
    down_intermediate,
    down_b,
    src2dst,
    topk_ids,
    topk_weights,
    token_lora_mapping,
    *,
    scale,
    local_expert_offset,
):
    num_tokens, top_k = topk_ids.shape
    num_experts = down_output.shape[0]
    hidden = down_output.shape[-1]
    result = torch.zeros(num_tokens, hidden, dtype=torch.float32, device="cuda")
    shared_outer = down_b.shape[1] == 1
    for token in range(num_tokens):
        adapter = int(token_lora_mapping[token])
        for k_idx in range(top_k):
            global_expert = int(topk_ids[token, k_idx])
            expert = global_expert - local_expert_offset
            if not 0 <= expert < num_experts:
                continue
            pair = token * top_k + k_idx
            routed_weight = topk_weights[token, k_idx].float() * scale
            dst = int(src2dst[pair])
            value = down_output.view(-1, hidden)[dst].float()
            if 0 <= adapter < down_b.shape[0]:
                b_expert = 0 if shared_outer else expert
                value = value + (
                    down_b[adapter, b_expert].float()
                    @ down_intermediate[token, k_idx].float()
                )
            result[token] += routed_weight * value
    return result


@pytest.mark.parametrize("rank", [32, 64, 128])
@pytest.mark.parametrize("shared_outer", [False, True])
def test_fused_down_finalize_matches_independent_fp32_oracle_and_graph(
    rank: int, shared_outer: bool
):
    torch.manual_seed(20260722 + rank + shared_outer)
    device = "cuda"
    num_tokens, top_k = 4, 4
    num_experts, m_max, hidden, max_loras = 3, 6, 96, 2
    local_expert_offset = 10
    scale = 1.75

    down_output = 0.1 * torch.randn(
        num_experts, m_max, hidden, dtype=torch.bfloat16, device=device
    )
    down_intermediate = 0.05 * torch.randn(
        num_tokens, top_k, rank, dtype=torch.bfloat16, device=device
    )
    b_experts = 1 if shared_outer else num_experts
    down_b = 0.05 * torch.randn(
        max_loras,
        b_experts,
        hidden,
        rank,
        dtype=torch.bfloat16,
        device=device,
    )
    # Covers valid global IDs, invalid sentinel, and non-local IDs on both sides.
    topk_ids = torch.tensor(
        [
            [10, 11, 12, -1],
            [11, 12, 10, 9],
            [12, 10, 13, 11],
            [10, -1, 11, 12],
        ],
        dtype=torch.int32,
        device=device,
    )
    topk_weights = torch.rand(num_tokens, top_k, dtype=torch.float32, device=device)
    # Active adapter, base-only row, second adapter, invalid positive slot.
    token_lora_mapping = torch.tensor(
        [0, -1, 1, max_loras], dtype=torch.int32, device=device
    )
    src2dst = torch.zeros(num_tokens * top_k, dtype=torch.int32, device=device)
    expert_counts = [0] * num_experts
    for pair, global_expert in enumerate(topk_ids.flatten().tolist()):
        expert = global_expert - local_expert_offset
        if 0 <= expert < num_experts:
            src2dst[pair] = expert * m_max + expert_counts[expert]
            expert_counts[expert] += 1

    expected = _reference(
        down_output,
        down_intermediate,
        down_b,
        src2dst,
        topk_ids,
        topk_weights,
        token_lora_mapping,
        scale=scale,
        local_expert_offset=local_expert_offset,
    )
    output = torch.empty_like(expected)

    def launch() -> None:
        fused_down_b_finalize(
            down_output,
            down_intermediate,
            down_b,
            output,
            src2dst,
            topk_ids,
            topk_weights,
            token_lora_mapping,
            routed_scaling_factor=scale,
            local_expert_offset=local_expert_offset,
            shared_outer=shared_outer,
            block_size_h=32,
        )

    launch()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-5)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-5)

    # Requested BF16 conversion is explicit and preserves the same FP32 sum.
    output_bf16 = torch.empty_like(expected, dtype=torch.bfloat16)
    fused_down_b_finalize(
        down_output,
        down_intermediate,
        down_b,
        output_bf16,
        src2dst,
        topk_ids,
        topk_weights,
        token_lora_mapping,
        routed_scaling_factor=scale,
        local_expert_offset=local_expert_offset,
        shared_outer=shared_outer,
        block_size_h=32,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(output_bf16.float(), expected, rtol=8e-3, atol=8e-3)

    # Fairness control: retain a separate base finalize, but remove B routing.
    base_only_mapping = torch.full_like(token_lora_mapping, -1)
    base_only = _reference(
        down_output,
        down_intermediate,
        down_b,
        src2dst,
        topk_ids,
        topk_weights,
        base_only_mapping,
        scale=scale,
        local_expert_offset=local_expert_offset,
    ).to(torch.bfloat16)
    indexed_output = base_only.clone()

    def indexed_launch() -> None:
        indexed_down_b_add(
            down_intermediate,
            down_b,
            indexed_output,
            topk_ids,
            topk_weights,
            token_lora_mapping,
            routed_scaling_factor=scale,
            local_expert_offset=local_expert_offset,
            num_local_experts=num_experts,
            shared_outer=shared_outer,
            block_size_h=32,
        )

    indexed_launch()
    torch.cuda.synchronize()
    torch.testing.assert_close(indexed_output.float(), expected, rtol=2e-2, atol=2e-2)

    indexed_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(indexed_graph):
        indexed_output.copy_(base_only)
        indexed_launch()
    indexed_output.fill_(float("nan"))
    indexed_graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(indexed_output.float(), expected, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
