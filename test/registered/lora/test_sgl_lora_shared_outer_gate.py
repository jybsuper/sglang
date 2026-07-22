import sys

import pytest
import torch

from sglang.srt.lora.sgl_lora.shared_outer_gate import (
    build_shared_outer_gate_a_plan,
    invoke_shared_outer_gate_a_token_dedup,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-small")


def _fixture(*, mixed: bool):
    if torch.cuda.get_device_capability()[0] not in {9, 10}:
        pytest.skip("shared-outer token dedup is promoted on Hopper and Blackwell")
    torch.manual_seed(20260722)
    tokens, hidden_size, rank, top_k = 32, 2048, 128, 8
    if mixed:
        lengths = (6, 5, 7, 4, 10)
        adapter_ids = (1, 2, 0, 3, 1)
        capacity = 4
    else:
        lengths = (4,) * 8
        adapter_ids = tuple(range(8))
        capacity = 8
    indptr = [0]
    for length in lengths:
        indptr.append(indptr[-1] + length)
    assert indptr[-1] == tokens

    hidden = 0.05 * torch.randn(
        tokens, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    factor = 0.05 * torch.randn(
        capacity,
        1,
        2 * rank,
        hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    if mixed:
        # Slot zero models the resident no-LoRA factor buffer.
        factor[0].zero_()
    segment_indptr = torch.tensor(indptr, dtype=torch.int32, device="cuda")
    segment_lora_ids = torch.tensor(adapter_ids, dtype=torch.int32, device="cuda")
    plan = build_shared_outer_gate_a_plan(
        shared_outer=True,
        device_capability=torch.cuda.get_device_capability(),
        phase="decode",
        graph_mode=True,
        num_tokens=tokens,
        hidden_size=hidden_size,
        rank=rank,
        top_k=top_k,
        has_base_rows=mixed,
        num_segments=len(lengths),
        max_segment_len=max(lengths),
    )
    assert plan.uses_token_dedup
    return hidden, factor, segment_indptr, segment_lora_ids, max(lengths), top_k, plan


def _reference(hidden, factor, indptr, adapter_ids, top_k):
    output = torch.empty(
        hidden.shape[0],
        top_k,
        factor.shape[2],
        dtype=torch.float32,
        device=hidden.device,
    )
    boundaries = indptr.cpu().tolist()
    ids = adapter_ids.cpu().tolist()
    for segment_id, adapter_id in enumerate(ids):
        start, stop = boundaries[segment_id : segment_id + 2]
        token = hidden[start:stop].float() @ factor[adapter_id, 0].float().T
        output[start:stop] = token[:, None, :]
    return output


def _token_mapping(indptr, adapter_ids):
    boundaries = indptr.cpu().tolist()
    ids = adapter_ids.cpu().tolist()
    mapping = torch.empty(boundaries[-1], dtype=torch.int32, device="cuda")
    for segment_id, adapter_id in enumerate(ids):
        mapping[boundaries[segment_id] : boundaries[segment_id + 1]] = adapter_id
    return mapping


@pytest.mark.parametrize("mixed", (False, True))
@pytest.mark.parametrize("execution", ("eager", "cuda_graph"))
def test_shared_outer_gate_token_dedup_matches_reference(mixed, execution):
    hidden, factor, indptr, ids, max_len, top_k, plan = _fixture(mixed=mixed)
    output = torch.empty(
        hidden.shape[0],
        top_k,
        factor.shape[2],
        dtype=torch.bfloat16,
        device="cuda",
    )

    def invoke():
        invoke_shared_outer_gate_a_token_dedup(
            hidden, factor, output, indptr, ids, max_len, plan
        )

    if execution == "cuda_graph":
        invoke()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            invoke()
        graph.replay()
    else:
        invoke()
    torch.cuda.synchronize()

    reference = _reference(hidden, factor, indptr, ids, top_k)
    torch.testing.assert_close(output.float(), reference, rtol=2e-2, atol=5e-2)
    torch.testing.assert_close(
        output[:, 1:].float(),
        output[:, :1].expand_as(output[:, 1:]).float(),
        rtol=0.0,
        atol=0.0,
    )


def test_selected_shared_a_composes_with_physical_shared_expert_ids():
    from sglang.srt.lora.sgl_lora.triton_ops.virtual_experts import (
        merged_experts_fused_moe_lora_add,
    )

    hidden, factor, indptr, ids, max_len, top_k, plan = _fixture(mixed=False)
    tokens = hidden.shape[0]
    num_experts = 8
    rank = factor.shape[2] // 2
    topk_ids = (
        torch.arange(tokens * top_k, dtype=torch.int32, device="cuda")
        .view(tokens, top_k)
        .remainder(num_experts + 2)
    )
    expert_id_map = torch.tensor(
        [0, 1, 2, 3, -1, 4, 5, 6, 7, -1],
        dtype=torch.int32,
        device="cuda",
    )
    topk_weights = torch.full(
        (tokens, top_k), 1.0 / top_k, dtype=torch.float32, device="cuda"
    )
    mapping = _token_mapping(indptr, ids)
    dummy_b = torch.empty(
        factor.shape[0],
        num_experts,
        2,
        rank,
        dtype=torch.bfloat16,
        device="cuda",
    )

    def shrink(output, cache, selected_plan):
        common = dict(
            output=output,
            hidden_states=hidden,
            lora_a=factor,
            lora_b=dummy_b,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            token_lora_mapping=mapping,
            mul_routed_weight=False,
            experts_shared_outer_loras_a=True,
            experts_shared_outer_loras_b=False,
            routing_cache=cache,
            fuse_add_to_output=False,
            num_output_slices=2,
            local_expert_offset=0,
            local_num_experts=num_experts,
            expert_id_map=expert_id_map,
            shared_outer_gate_a_plan=selected_plan,
            segment_indptr=indptr,
            segment_lora_ids=ids,
            max_segment_len=max_len,
        )
        merged_experts_fused_moe_lora_add(stage="routing", **common)
        merged_experts_fused_moe_lora_add(
            stage="shrink", intermediate_buffer=output, **common
        )

    selected = torch.empty(
        tokens, top_k, factor.shape[2], dtype=torch.bfloat16, device="cuda"
    )
    generic = torch.empty_like(selected)
    selected_cache = {}
    generic_cache = {}
    shrink(selected, selected_cache, plan)
    shrink(generic, generic_cache, None)
    torch.cuda.synchronize()

    assert all(not key[1] for key in selected_cache)
    assert any(key[1] for key in generic_cache)
    assert all(key[3] == expert_id_map.data_ptr() for key in selected_cache)
    torch.testing.assert_close(selected.float(), generic.float(), rtol=2e-2, atol=5e-2)


@pytest.mark.parametrize("rank,tokens", ((64, 256), (128, 32)))
@pytest.mark.parametrize("execution", ("eager", "cuda_graph"))
def test_selected_full_chain_obeys_pdl_expand_contract(rank, tokens, execution):
    from sglang.srt.lora.sgl_lora.triton_ops.virtual_experts import (
        lora_pdl_policy,
        merged_experts_fused_moe_lora_add,
    )

    if torch.cuda.get_device_capability()[0] not in {9, 10}:
        pytest.skip("shared-outer token dedup is promoted on Hopper and Blackwell")
    torch.manual_seed(20260723 + rank)
    hidden_size, top_k, num_experts, output_slice = 2048, 8, 8, 192
    hidden = 0.05 * torch.randn(
        tokens, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    factor_a = 0.05 * torch.randn(
        4, 1, 2 * rank, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    factor_b = 0.05 * torch.randn(
        4,
        num_experts,
        2 * output_slice,
        rank,
        dtype=torch.bfloat16,
        device="cuda",
    )
    topk_ids = (
        torch.arange(tokens * top_k, dtype=torch.int32, device="cuda")
        .view(tokens, top_k)
        .remainder(num_experts)
    )
    weights = torch.ones((tokens, top_k), dtype=torch.float32, device="cuda")
    segment_len = 4
    num_segments = tokens // segment_len
    indptr = torch.arange(0, tokens + 1, segment_len, dtype=torch.int32, device="cuda")
    ids = torch.arange(num_segments, dtype=torch.int32, device="cuda").remainder(4)
    mapping = ids.repeat_interleave(segment_len)
    plan = build_shared_outer_gate_a_plan(
        shared_outer=True,
        device_capability=torch.cuda.get_device_capability(),
        phase="decode",
        graph_mode=True,
        num_tokens=tokens,
        hidden_size=hidden_size,
        rank=rank,
        top_k=top_k,
        has_base_rows=False,
        num_segments=num_segments,
        max_segment_len=segment_len,
    )
    assert plan.uses_token_dedup

    def run(output, intermediate, selected_plan, enable_pdl, cache):
        with lora_pdl_policy(enable_pdl):
            merged_experts_fused_moe_lora_add(
                output=output,
                hidden_states=hidden,
                lora_a=factor_a,
                lora_b=factor_b,
                topk_ids=topk_ids,
                topk_weights=weights,
                token_lora_mapping=mapping,
                mul_routed_weight=False,
                experts_shared_outer_loras_a=True,
                experts_shared_outer_loras_b=False,
                routing_cache=cache,
                fuse_add_to_output=False,
                use_direct_expand_add=rank <= 64,
                num_output_slices=2,
                shared_outer_gate_a_plan=selected_plan,
                segment_indptr=indptr,
                segment_lora_ids=ids,
                max_segment_len=segment_len,
                intermediate_buffer=intermediate,
            )

    generic = torch.empty(
        tokens, top_k, 2 * output_slice, dtype=torch.bfloat16, device="cuda"
    )
    selected = torch.empty_like(generic)
    generic_intermediate = torch.empty(
        tokens, top_k, 2 * rank, dtype=torch.bfloat16, device="cuda"
    )
    selected_intermediate = torch.empty_like(generic_intermediate)
    run(generic, generic_intermediate, None, False, {})
    selected_cache = {}
    run(selected, selected_intermediate, plan, True, selected_cache)
    if execution == "cuda_graph":
        capture_cache = {}
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(selected, selected_intermediate, plan, True, capture_cache)
        graph.replay()
    else:
        run(selected, selected_intermediate, plan, True, selected_cache)
    torch.cuda.synchronize()
    torch.testing.assert_close(selected.float(), generic.float(), rtol=3e-2, atol=6e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
