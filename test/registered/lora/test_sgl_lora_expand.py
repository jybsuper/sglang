import sys

import pytest
import torch

from sglang.srt.lora.sgl_lora.triton_ops.expand import (
    invoke_moe_lora_expand_add,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@pytest.mark.parametrize("rank", [16, 64])
@pytest.mark.parametrize("num_output_slices", [1, 2])
def test_direct_expand_uses_the_requested_input_slice(
    rank: int,
    num_output_slices: int,
):
    """Each output slice must consume its corresponding LoRA-A result."""
    torch.manual_seed(11)
    device = "cuda"
    num_tokens = 3
    # 193 is intentionally not divisible by a tensor-core tile.  The two-slice
    # grid must mask each slice tail without crossing the gate/up boundary.
    half_output = 193
    output_size = 2 * half_output if num_output_slices == 2 else 384
    block_size_m = 16

    # Gate/up shrink is canonical [gate-A result | up-A result]. Keeping the
    # halves independent makes an accidental reuse of gate-A immediately visible.
    intermediate = torch.randn(
        num_tokens,
        num_output_slices * rank,
        dtype=torch.bfloat16,
        device=device,
    )
    weight = torch.randn(1, output_size, rank, dtype=torch.bfloat16, device=device)
    output = torch.empty(num_tokens, output_size, dtype=torch.bfloat16, device=device)
    topk_ids = torch.zeros((num_tokens, 1), dtype=torch.int32, device=device)
    topk_weights = torch.ones((num_tokens, 1), dtype=torch.float32, device=device)

    # One aligned expert block. Padded entries use num_valid_tokens as the
    # sentinel, matching moe_align_block_size's contract.
    sorted_token_ids = torch.full(
        (block_size_m,), num_tokens, dtype=torch.int32, device=device
    )
    sorted_token_ids[:num_tokens] = torch.arange(
        num_tokens, dtype=torch.int32, device=device
    )
    expert_ids = torch.zeros((1,), dtype=torch.int32, device=device)
    num_tokens_post_padded = torch.tensor(
        [block_size_m], dtype=torch.int32, device=device
    )
    config = {
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
    }

    invoke_moe_lora_expand_add(
        intermediate,
        weight,
        output,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        config,
        mul_routed_weight=False,
        fuse_sum_all_reduce=False,
        num_output_slices=num_output_slices,
    )
    torch.cuda.synchronize()

    if num_output_slices == 2:
        expected = torch.cat(
            (
                intermediate[:, :rank] @ weight[0, :half_output].T,
                intermediate[:, rank:] @ weight[0, half_output:].T,
            ),
            dim=1,
        )
    else:
        expected = intermediate[:, :rank] @ weight[0].T

    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("rank", [16, 64])
def test_direct_expand_matches_production_down_projection(rank: int):
    """Cover routed weighting plus top-k collapse used by the Phase-1a runner."""
    torch.manual_seed(29)
    device = "cuda"
    num_tokens, top_k = 3, 2
    num_pairs = num_tokens * top_k
    num_virtual_experts = 3
    output_size = 128
    block_size_m = 16

    intermediate = torch.randn(num_pairs, rank, dtype=torch.bfloat16, device=device)
    weight = torch.randn(
        num_virtual_experts,
        output_size,
        rank,
        dtype=torch.bfloat16,
        device=device,
    )
    topk_ids = torch.tensor([[0, 1], [1, 2], [0, -1]], dtype=torch.int32, device=device)
    topk_weights = torch.tensor(
        [[0.7, 0.3], [0.2, 0.8], [1.0, 0.0]],
        dtype=torch.float32,
        device=device,
    )

    # The first three blocks route real virtual experts.  The final -1 block
    # contains the invalid last pair and mirrors align's sentinel bucket.
    pair_experts = [0, 1, 1, 2, 0, -1]
    expert_to_pairs = ([0, 4], [1, 2], [3], [5])
    sorted_token_ids = torch.full(
        (4 * block_size_m,), num_pairs, dtype=torch.int32, device=device
    )
    for block, pair_ids in enumerate(expert_to_pairs):
        sorted_token_ids[
            block * block_size_m : block * block_size_m + len(pair_ids)
        ] = torch.tensor(pair_ids, dtype=torch.int32, device=device)
    expert_ids = torch.tensor([0, 1, 2, -1], dtype=torch.int32, device=device)
    num_tokens_post_padded = torch.tensor(
        [sorted_token_ids.numel()], dtype=torch.int32, device=device
    )

    base_output = torch.randn(
        num_tokens, output_size, dtype=torch.bfloat16, device=device
    )
    output = base_output.clone()
    expected = base_output.float()
    flat_weights = topk_weights.flatten()
    for pair_idx, expert_id in enumerate(pair_experts):
        if expert_id < 0:
            continue
        token_idx = pair_idx // top_k
        delta = intermediate[pair_idx].float() @ weight[expert_id].float().T
        expected[token_idx] += delta * flat_weights[pair_idx]

    config = {
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
    }
    invoke_moe_lora_expand_add(
        intermediate,
        weight,
        output,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        config,
        mul_routed_weight=True,
        fuse_sum_all_reduce=True,
        num_output_slices=1,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        output, expected.to(torch.bfloat16), rtol=5e-2, atol=5e-2
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
