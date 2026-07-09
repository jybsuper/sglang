import sys

import pytest
import torch

from sglang.srt.lora.sgl_lora.triton_ops.virtual_experts import (
    merged_experts_fused_moe_lora_add,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


@pytest.mark.parametrize("rank", [16, 64])
def test_gate_up_a_b_matches_reference_and_reuses_routing(rank: int):
    """Exercise the complete virtual-expert gate/up A+B path used by sgl_lora."""
    torch.manual_seed(41)
    device = "cuda"
    num_tokens, top_k = 4, 2
    num_experts, max_loras = 3, 2
    hidden_size, intermediate_size = 64, 192

    hidden_states = 0.1 * torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )
    lora_a = 0.1 * torch.randn(
        max_loras,
        num_experts,
        2 * rank,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    lora_b = 0.1 * torch.randn(
        max_loras,
        num_experts,
        2 * intermediate_size,
        rank,
        dtype=torch.bfloat16,
        device=device,
    )
    topk_ids = torch.tensor(
        [[0, 1], [2, 0], [1, 2], [0, 2]], dtype=torch.int32, device=device
    )
    topk_weights = torch.ones((num_tokens, top_k), dtype=torch.float32, device=device)
    token_lora_mapping = torch.tensor([0, 1, -1, 1], dtype=torch.int32, device=device)

    output = torch.zeros(
        num_tokens,
        top_k,
        2 * intermediate_size,
        dtype=torch.bfloat16,
        device=device,
    )
    intermediate = torch.empty(
        num_tokens,
        top_k,
        2 * rank,
        dtype=torch.bfloat16,
        device=device,
    )
    routing_cache = {}

    # This is the exact pre-warm sequence used before the optional side-stream
    # fork.  The full call below must consume the same cached routing tensors.
    merged_experts_fused_moe_lora_add(
        output=output,
        hidden_states=hidden_states,
        lora_a=lora_a,
        lora_b=lora_b,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        token_lora_mapping=token_lora_mapping,
        mul_routed_weight=False,
        experts_shared_outer_loras_a=False,
        experts_shared_outer_loras_b=False,
        routing_cache=routing_cache,
        fuse_add_to_output=False,
        use_direct_expand_add=True,
        num_output_slices=2,
        stage="routing",
    )
    assert routing_cache
    cached_ptrs = {
        key: tuple(t.data_ptr() for t in tensors)
        for key, tensors in routing_cache.items()
    }

    merged_experts_fused_moe_lora_add(
        output=output,
        hidden_states=hidden_states,
        lora_a=lora_a,
        lora_b=lora_b,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        token_lora_mapping=token_lora_mapping,
        mul_routed_weight=False,
        experts_shared_outer_loras_a=False,
        experts_shared_outer_loras_b=False,
        routing_cache=routing_cache,
        fuse_add_to_output=False,
        use_direct_expand_add=True,
        num_output_slices=2,
        intermediate_buffer=intermediate,
    )
    torch.cuda.synchronize()

    assert set(routing_cache) == set(cached_ptrs)
    assert {
        key: tuple(t.data_ptr() for t in tensors)
        for key, tensors in routing_cache.items()
    } == cached_ptrs

    expected = torch.zeros_like(output)
    for token_idx in range(num_tokens):
        lora_idx = int(token_lora_mapping[token_idx])
        if lora_idx < 0:
            continue
        for route_idx in range(top_k):
            expert_idx = int(topk_ids[token_idx, route_idx])
            shrink = hidden_states[token_idx] @ lora_a[lora_idx, expert_idx].T
            expected[token_idx, route_idx, :intermediate_size] = (
                shrink[:rank] @ lora_b[lora_idx, expert_idx, :intermediate_size].T
            )
            expected[token_idx, route_idx, intermediate_size:] = (
                shrink[rank:] @ lora_b[lora_idx, expert_idx, intermediate_size:].T
            )

    torch.testing.assert_close(output, expected, rtol=3e-2, atol=3e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
