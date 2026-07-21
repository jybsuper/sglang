import sys

import pytest
import torch

from sglang.srt.lora.sgl_lora.triton_ops.shrink import (
    IndexedLoraAKernelConfig,
    IndexedLoraARowPlan,
    LoraAInputRowDomain,
    LoraASplitKAccumulation,
    invoke_indexed_lora_a_shrink,
    select_indexed_lora_a_kernel_config,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")


def _routing(
    pair_groups: list[int],
    block_m: int,
    *,
    extra_capacity_blocks: int = 0,
) -> IndexedLoraARowPlan:
    num_pairs = len(pair_groups)
    chunks = []
    group_ids = []
    for group in sorted(set(pair_groups)):
        pair_ids = [
            idx for idx, pair_group in enumerate(pair_groups) if pair_group == group
        ]
        for begin in range(0, len(pair_ids), block_m):
            padded = torch.full((block_m,), num_pairs, dtype=torch.int32)
            chunk = pair_ids[begin : begin + block_m]
            padded[: len(chunk)] = torch.tensor(chunk, dtype=torch.int32)
            chunks.append(padded)
            group_ids.append(group)
    num_active_blocks = len(chunks)
    for _ in range(extra_capacity_blocks):
        chunks.append(torch.full((block_m,), num_pairs, dtype=torch.int32))
        group_ids.append(-1)
    return IndexedLoraARowPlan(
        sorted_pair_ids=torch.cat(chunks).cuda(),
        block_group_ids=torch.tensor(group_ids, dtype=torch.int32, device="cuda"),
        num_pairs_post_padded=torch.tensor(
            [num_active_blocks * block_m], dtype=torch.int32, device="cuda"
        ),
        block_m=block_m,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "input_row_domain",
    [LoraAInputRowDomain.TOKEN, LoraAInputRowDomain.ROUTED_PAIR],
)
def test_indexed_lora_a_looped_k_matches_packed_factor_reference(
    dtype: torch.dtype,
    input_row_domain: LoraAInputRowDomain,
):
    torch.manual_seed(101)
    num_tokens, topk = 3, 2
    num_pairs, hidden_size, packed_rank = num_tokens * topk, 65, 48
    pair_groups = [0, 1, 0, 1, 0, -1]
    block_m = 16
    row_plan = _routing(pair_groups, block_m, extra_capacity_blocks=3)
    input_rows = (
        num_tokens if input_row_domain == LoraAInputRowDomain.TOKEN else num_pairs
    )
    hidden_states = torch.randn(input_rows, hidden_size, dtype=dtype, device="cuda")
    weight = torch.randn(2, packed_rank, hidden_size, dtype=dtype, device="cuda")
    output = torch.zeros(num_pairs, packed_rank, dtype=dtype, device="cuda")
    kernel_config = IndexedLoraAKernelConfig(
        block_n=32,
        block_k=32,
        split_k=1,
        num_warps=4,
    )
    invoke_indexed_lora_a_shrink(
        hidden_states,
        weight,
        output,
        row_plan,
        num_valid_pairs=num_pairs,
        router_topk=topk,
        input_row_domain=input_row_domain,
        kernel_config=kernel_config,
    )
    torch.cuda.synchronize()

    expected = torch.zeros_like(output)
    for pair_idx, group in enumerate(pair_groups):
        if group < 0:
            continue
        input_idx = (
            pair_idx // topk
            if input_row_domain == LoraAInputRowDomain.TOKEN
            else pair_idx
        )
        expected[pair_idx] = hidden_states[input_idx] @ weight[group].T
    torch.testing.assert_close(output, expected, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("output_dtype", [torch.float32, torch.bfloat16])
def test_indexed_lora_a_split_k_clears_output_and_replays_graph(
    output_dtype: torch.dtype,
):
    torch.manual_seed(211)
    num_tokens, topk = 4, 2
    num_pairs, hidden_size, packed_rank = num_tokens * topk, 257, 96
    pair_groups = [0, 1, 0, 1, 1, 0, 1, 0]
    block_m = 16
    row_plan = _routing(pair_groups, block_m)
    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    weight = torch.randn(
        2, packed_rank, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    output = torch.full(
        (num_pairs, packed_rank), 31.0, dtype=output_dtype, device="cuda"
    )
    kernel_config = IndexedLoraAKernelConfig(
        block_n=32,
        block_k=64,
        split_k=4,
        num_warps=4,
        split_k_accumulation=(
            LoraASplitKAccumulation.FP32
            if output_dtype == torch.float32
            else LoraASplitKAccumulation.OUTPUT_DTYPE
        ),
    )

    def launch() -> None:
        invoke_indexed_lora_a_shrink(
            hidden_states,
            weight,
            output,
            row_plan,
            num_valid_pairs=num_pairs,
            router_topk=topk,
            input_row_domain=LoraAInputRowDomain.TOKEN,
            kernel_config=kernel_config,
        )

    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    graph.replay()
    torch.cuda.synchronize()

    expected = torch.empty_like(output)
    for pair_idx, group in enumerate(pair_groups):
        expected[pair_idx] = (
            hidden_states[pair_idx // topk].float() @ weight[group].float().T
        )
    if output_dtype == torch.float32:
        torch.testing.assert_close(output, expected, rtol=3e-3, atol=3e-3)
    else:
        torch.testing.assert_close(output, expected, rtol=3e-2, atol=1.25e-1)


def test_indexed_lora_a_can_clear_rows_omitted_by_its_routing_domain():
    block_m = 16
    num_pairs, hidden_size, packed_rank = 4, 32, 16
    sorted_pair_ids = torch.arange(num_pairs, dtype=torch.int32, device="cuda")
    sorted_pair_ids = torch.nn.functional.pad(
        sorted_pair_ids, (0, block_m - num_pairs), value=num_pairs
    )
    row_plan = IndexedLoraARowPlan(
        sorted_pair_ids=sorted_pair_ids,
        block_group_ids=torch.tensor([-1], dtype=torch.int32, device="cuda"),
        num_pairs_post_padded=torch.tensor([block_m], dtype=torch.int32, device="cuda"),
        block_m=block_m,
    )
    hidden_states = torch.randn(
        num_pairs, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    weight = torch.randn(
        1, packed_rank, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    output = torch.full(
        (num_pairs, packed_rank), 31.0, dtype=torch.bfloat16, device="cuda"
    )

    invoke_indexed_lora_a_shrink(
        hidden_states,
        weight,
        output,
        row_plan,
        num_valid_pairs=num_pairs,
        router_topk=1,
        input_row_domain=LoraAInputRowDomain.ROUTED_PAIR,
        kernel_config=IndexedLoraAKernelConfig(
            block_n=16,
            block_k=32,
            split_k=1,
            num_warps=4,
        ),
        clear_output=True,
    )
    torch.testing.assert_close(output, torch.zeros_like(output))


@pytest.mark.parametrize("packed_rank", [8, 257])
def test_indexed_lora_a_handles_small_and_multi_tile_packed_rank(packed_rank: int):
    torch.manual_seed(307 + packed_rank)
    num_pairs, hidden_size, block_m = 3, 33, 16
    row_plan = _routing([0, 1, 0], block_m, extra_capacity_blocks=1)
    hidden_states = torch.randn(
        num_pairs, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    weight = torch.randn(
        2, packed_rank, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    output = torch.empty(num_pairs, packed_rank, dtype=torch.bfloat16, device="cuda")
    invoke_indexed_lora_a_shrink(
        hidden_states,
        weight,
        output,
        row_plan,
        num_valid_pairs=num_pairs,
        router_topk=1,
        input_row_domain=LoraAInputRowDomain.ROUTED_PAIR,
        kernel_config=IndexedLoraAKernelConfig(
            block_n=16 if packed_rank == 8 else 64,
            block_k=32,
            split_k=1,
            num_warps=4,
        ),
    )
    expected = torch.stack(
        [
            hidden_states[0] @ weight[0].T,
            hidden_states[1] @ weight[1].T,
            hidden_states[2] @ weight[0].T,
        ]
    )
    torch.testing.assert_close(output, expected, rtol=3e-2, atol=3e-2)


def test_indexed_lora_a_selector_bounds_packed_rank_tile():
    weight = torch.empty(7, 256, 7168, device="meta")
    row_plan = IndexedLoraARowPlan(
        sorted_pair_ids=torch.empty(1024, dtype=torch.int32, device="meta"),
        block_group_ids=torch.empty(64, dtype=torch.int32, device="meta"),
        num_pairs_post_padded=torch.empty(1, dtype=torch.int32, device="meta"),
        block_m=16,
    )
    config = select_indexed_lora_a_kernel_config(
        weight,
        row_plan,
        {"num_warps": 4},
    )
    assert config.block_n == 64
    assert config.block_k == 256
    assert 1 <= config.split_k <= 8


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
