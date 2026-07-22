import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.lora.sgl_lora.triton_ops.fused_c2 import (
    fused_gate_up_b_swiglu_down_a,
    fused_gate_up_b_swiglu_down_a_aligned,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@pytest.mark.parametrize(
    ("inter", "rank", "block_size_n", "local_expert_offset", "schedule"),
    [
        (37, 16, 16, 0, "pair"),
        (192, 32, 64, 0, "pair"),
        (192, 64, 32, 0, "pair"),
        (192, 128, 16, 0, "pair"),
        (64, 16, 32, 5, "pair"),
        (37, 16, 16, 0, "aligned"),
        (192, 64, 64, 0, "aligned"),
        (192, 128, 16, 0, "aligned"),
        (64, 16, 32, 5, "aligned"),
    ],
)
def test_fused_c2_consumer_matches_materialized_bf16_contract(
    inter: int,
    rank: int,
    block_size_n: int,
    local_expert_offset: int,
    schedule: str,
):
    """C2 removes two tensors, but preserves their BF16 semantic boundaries."""
    torch.manual_seed(20260722)
    device = "cuda"
    num_tokens, top_k = 3, 2
    num_experts, max_loras, m_max = 3, 2, 3

    gateup = 0.1 * torch.randn(
        num_experts, m_max, 2 * inter, dtype=torch.bfloat16, device=device
    )
    gate_intermediate = 0.1 * torch.randn(
        num_tokens, top_k, 2 * rank, dtype=torch.bfloat16, device=device
    )
    gate_b = 0.1 * torch.randn(
        max_loras,
        num_experts,
        2 * inter,
        rank,
        dtype=torch.bfloat16,
        device=device,
    )
    down_a = 0.1 * torch.randn(
        max_loras,
        num_experts,
        rank,
        inter,
        dtype=torch.bfloat16,
        device=device,
    )

    # Pair 2 is invalid; pair 3 is a valid base-only row.  All valid pairs own
    # distinct destinations in the masked expert layout.
    local_topk_ids = torch.tensor(
        [[0, 1], [-1, 0], [1, 2]], dtype=torch.int32, device=device
    )
    topk_ids = torch.where(
        local_topk_ids >= 0,
        local_topk_ids + local_expert_offset,
        local_topk_ids,
    )
    token_lora_mapping = torch.tensor([0, -1, 1], dtype=torch.int32, device=device)
    src2dst = torch.tensor([0, 3, 0, 1, 4, 6], dtype=torch.int32, device=device)

    sentinel = -123.0
    act_out = torch.full_like(gateup[..., :inter], sentinel)
    down_intermediate = torch.zeros(
        num_tokens, top_k, rank, dtype=torch.bfloat16, device=device
    )
    expected_act = act_out.clone()
    expected_down = torch.zeros_like(down_intermediate)

    gateup_flat = gateup.view(-1, 2 * inter)
    expected_act_flat = expected_act.view(-1, inter)
    for pair_idx in range(num_tokens * top_k):
        global_expert = int(topk_ids.view(-1)[pair_idx])
        if global_expert < 0:
            continue
        expert = global_expert - local_expert_offset
        adapter = int(token_lora_mapping[pair_idx // top_k])
        dst = int(src2dst[pair_idx])
        gate = gateup_flat[dst, :inter].float()
        up = gateup_flat[dst, inter:].float()
        if adapter >= 0:
            gate_delta = (
                gate_b[adapter, expert, :inter].float()
                @ gate_intermediate.view(-1, 2 * rank)[pair_idx, :rank].float()
            ).to(torch.bfloat16)
            up_delta = (
                gate_b[adapter, expert, inter:].float()
                @ gate_intermediate.view(-1, 2 * rank)[pair_idx, rank:].float()
            ).to(torch.bfloat16)
            gate = gate + gate_delta.float()
            up = up + up_delta.float()
        activated = (F.silu(gate) * up).to(torch.bfloat16)
        expected_act_flat[dst] = activated

        if adapter >= 0:
            # The kernel splits I and atomically accumulates BF16 partials.
            # Reproduce those explicit rounding points in the oracle.
            accumulated = torch.zeros(rank, dtype=torch.bfloat16, device=device)
            for start in range(0, inter, block_size_n):
                stop = min(start + block_size_n, inter)
                partial = (
                    down_a[adapter, expert, :, start:stop].float()
                    @ activated[start:stop].float()
                ).to(torch.bfloat16)
                accumulated = (accumulated.float() + partial.float()).to(torch.bfloat16)
            expected_down.view(-1, rank)[pair_idx] = accumulated

    if schedule == "pair":
        fused_gate_up_b_swiglu_down_a(
            gateup,
            gate_intermediate,
            gate_b,
            down_a,
            act_out,
            down_intermediate,
            src2dst,
            topk_ids,
            token_lora_mapping,
            local_expert_offset=local_expert_offset,
            block_size_n=block_size_n,
        )
    else:
        route_block_m = 16
        buckets: dict[int, list[int]] = {}
        for pair_idx in range(num_tokens * top_k):
            global_expert = int(topk_ids.view(-1)[pair_idx])
            adapter = int(token_lora_mapping[pair_idx // top_k])
            virtual_expert = (
                adapter * num_experts + global_expert - local_expert_offset
                if adapter >= 0 and global_expert >= 0
                else -1
            )
            buckets.setdefault(virtual_expert, []).append(pair_idx)

        sorted_pairs: list[int] = []
        route_experts: list[int] = []
        for virtual_expert in sorted(buckets):
            pairs = buckets[virtual_expert]
            for start in range(0, len(pairs), route_block_m):
                block = pairs[start : start + route_block_m]
                sorted_pairs.extend(
                    block + [num_tokens * top_k] * (route_block_m - len(block))
                )
                route_experts.append(virtual_expert)

        sorted_pair_ids = torch.tensor(sorted_pairs, dtype=torch.int32, device=device)
        virtual_expert_ids = torch.tensor(
            route_experts, dtype=torch.int32, device=device
        )
        num_pairs_post_padded = torch.tensor(
            [len(sorted_pairs)], dtype=torch.int32, device=device
        )
        fused_gate_up_b_swiglu_down_a_aligned(
            gateup,
            gate_intermediate,
            gate_b,
            down_a,
            act_out,
            down_intermediate,
            src2dst,
            topk_ids,
            sorted_pair_ids,
            virtual_expert_ids,
            num_pairs_post_padded,
            route_block_size_m=route_block_m,
            block_size_n=block_size_n,
        )
    torch.cuda.synchronize()

    torch.testing.assert_close(act_out, expected_act, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(down_intermediate, expected_down, rtol=5e-2, atol=5e-2)
    # Base-only rows still feed W2, but must not produce a down-LoRA rank input.
    assert bool((down_intermediate[1] == 0).all())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
