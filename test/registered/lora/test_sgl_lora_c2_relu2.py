import sys

import pytest
import torch

from benchmark.kernels.lora_moe.c2_semantic_contracts import (
    C2SemanticContract,
    reference_c2_consumer,
    reference_weighted_down_delta,
)
from sglang.srt.lora.sgl_lora.triton_ops.fused_c2_relu2 import (
    fused_value_b_relu2_down_a,
    fused_value_b_relu2_down_a_aligned,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


def _aligned_route(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    *,
    num_local_experts: int,
    local_expert_offset: int,
    block_m: int,
):
    num_pairs = topk_ids.numel()
    top_k = topk_ids.shape[1]
    buckets: dict[int, list[int]] = {}
    for pair_idx in range(num_pairs):
        expert = int(topk_ids.reshape(-1)[pair_idx]) - local_expert_offset
        adapter = int(token_lora_mapping[pair_idx // top_k])
        virtual = (
            adapter * num_local_experts + expert
            if adapter >= 0 and 0 <= expert < num_local_experts
            else -1
        )
        if virtual >= 0:
            buckets.setdefault(virtual, []).append(pair_idx)
    sorted_pairs: list[int] = []
    route_experts: list[int] = []
    for virtual in sorted(buckets):
        pairs = buckets[virtual]
        for start in range(0, len(pairs), block_m):
            block = pairs[start : start + block_m]
            sorted_pairs.extend(block + [num_pairs] * (block_m - len(block)))
            route_experts.append(virtual)
    device = topk_ids.device
    return (
        torch.tensor(sorted_pairs, dtype=torch.int32, device=device),
        torch.tensor(route_experts, dtype=torch.int32, device=device),
        torch.tensor([len(sorted_pairs)], dtype=torch.int32, device=device),
    )


@pytest.mark.parametrize(
    ("logical_i", "physical_i", "rank", "block_n", "schedule"),
    [
        (37, 48, 16, 16, "pair"),
        (64, 64, 32, 32, "pair"),
        (65, 80, 64, 32, "aligned"),
        (192, 192, 64, 64, "aligned"),
    ],
)
def test_fused_relu2_consumer_matches_logical_provider_contract(
    logical_i: int,
    physical_i: int,
    rank: int,
    block_n: int,
    schedule: str,
):
    torch.manual_seed(20260722)
    device = "cuda"
    num_tokens, top_k, loras, experts = 4, 2, 2, 3
    local_offset = 5
    num_pairs = num_tokens * top_k
    destination_rows = num_pairs + 3
    contract = C2SemanticContract(
        activation="relu2",
        logical_slices=("value",),
        lora_target_slices=("value",),
        logical_intermediate_size=logical_i,
        physical_intermediate_size=physical_i,
    )
    value = 0.1 * torch.randn(
        destination_rows, physical_i, dtype=torch.bfloat16, device=device
    )
    value_a = 0.1 * torch.randn(
        num_tokens, top_k, rank, dtype=torch.bfloat16, device=device
    )
    value_b = 0.1 * torch.randn(
        loras, experts, physical_i, rank, dtype=torch.bfloat16, device=device
    )
    down_a = 0.1 * torch.randn(
        loras, experts, rank, physical_i, dtype=torch.bfloat16, device=device
    )
    topk_ids = torch.tensor(
        [[5, 6], [7, 5], [4, 8], [6, 7]], dtype=torch.int32, device=device
    )
    mapping = torch.tensor([0, -1, 1, 1], dtype=torch.int32, device=device)
    # Valid pairs own distinct, non-dense provider destinations. Invalid pairs
    # intentionally carry nonsense destinations that must never be consumed.
    src2dst = torch.tensor(
        [0, 3, 6, 1, 999, 999, 8, 10], dtype=torch.int32, device=device
    )
    expected = reference_c2_consumer(
        contract,
        value,
        value_a,
        value_b,
        down_a,
        src2dst,
        topk_ids,
        mapping,
        local_expert_offset=local_offset,
        block_size_n=block_n,
        activation_output_dtype=torch.bfloat16,
    )
    act_out = torch.full(
        (destination_rows, physical_i),
        -123.0,
        dtype=torch.bfloat16,
        device=device,
    )
    down_intermediate = torch.zeros(
        num_tokens, top_k, rank, dtype=torch.bfloat16, device=device
    )
    if schedule == "pair":
        fused_value_b_relu2_down_a(
            value,
            value_a,
            value_b,
            down_a,
            act_out,
            down_intermediate,
            src2dst,
            topk_ids,
            mapping,
            logical_intermediate_size=logical_i,
            local_expert_offset=local_offset,
            block_size_n=block_n,
        )
    else:
        block_m = 16
        route = _aligned_route(
            topk_ids,
            mapping,
            num_local_experts=experts,
            local_expert_offset=local_offset,
            block_m=block_m,
        )
        fused_value_b_relu2_down_a_aligned(
            value,
            value_a,
            value_b,
            down_a,
            act_out,
            down_intermediate,
            src2dst,
            topk_ids,
            *route,
            logical_intermediate_size=logical_i,
            route_block_size_m=block_m,
            token_lora_mapping=mapping,
            local_expert_offset=local_offset,
            block_size_n=block_n,
        )
    torch.cuda.synchronize()

    # Errors are measured against the LoRA/activation signal, not hidden by a
    # large base output. The small absolute floor covers BF16 dot order only.
    act_error = float((act_out.float() - expected.activation_output.float()).abs().max())
    act_signal = float(expected.activation_output.float().abs().max())
    down_error = float(
        (down_intermediate.float() - expected.down_rank_input.float()).abs().max()
    )
    down_signal = float(expected.down_rank_input.float().abs().max())
    assert act_error <= max(2e-4, 0.03 * act_signal)
    assert down_error <= max(5e-4, 0.05 * down_signal)
    # Base-only rows produce activation values but no down-LoRA rank input.
    assert bool((down_intermediate[1] == 0).all())
    # Provider padding is always explicit zero for every valid destination.
    if logical_i < physical_i:
        assert bool((act_out[[0, 1, 3, 6, 8, 10], logical_i:] == 0).all())

    # The consumer itself is routing-weight invariant. The downstream contract
    # applies both top-k and non-unit routed scaling exactly once and preserves
    # a caller-selected destination dtype.
    down_b = 0.1 * torch.randn(
        loras, experts, 11, rank, dtype=torch.bfloat16, device=device
    )
    topk_weights = torch.tensor(
        [[0.4, 0.6], [0.25, 0.75], [0.5, 0.5], [0.7, 0.3]],
        dtype=torch.float32,
        device=device,
    )
    for destination_dtype in (torch.bfloat16, torch.float32):
        final_delta = reference_weighted_down_delta(
            down_intermediate,
            down_b,
            topk_ids,
            topk_weights,
            mapping,
            local_expert_offset=local_offset,
            routed_scaling_factor=1.75,
            destination_dtype=destination_dtype,
        )
        assert final_delta.dtype == destination_dtype
        assert bool(torch.isfinite(final_delta).all())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
