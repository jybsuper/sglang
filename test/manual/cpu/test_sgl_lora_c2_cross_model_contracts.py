import pytest
import torch

from benchmark.kernels.lora_moe.c2_semantic_contracts import (
    C2SemanticContract,
    reference_c2_consumer,
    reference_weighted_down_delta,
)


def _tiny_inputs(contract: C2SemanticContract, *, rank: int = 2):
    torch.manual_seed(20260722)
    tokens, top_k, loras, experts, rows = 3, 2, 2, 2, 7
    value = 0.1 * torch.randn(
        rows, contract.provider_output_width, dtype=torch.bfloat16
    )
    value_a = 0.1 * torch.randn(
        tokens,
        top_k,
        contract.num_lora_slices * rank,
        dtype=torch.bfloat16,
    )
    value_b = 0.1 * torch.randn(
        loras,
        experts,
        contract.num_lora_slices * contract.physical_intermediate_size,
        rank,
        dtype=torch.bfloat16,
    )
    down_a = 0.1 * torch.randn(
        loras,
        experts,
        rank,
        contract.physical_intermediate_size,
        dtype=torch.bfloat16,
    )
    topk_ids = torch.tensor([[5, 6], [4, 7], [6, 5]], dtype=torch.int32)
    mapping = torch.tensor([0, -1, 1], dtype=torch.int32)
    src2dst = torch.tensor([0, 2, 99, 99, 4, 6], dtype=torch.int32)
    return value, value_a, value_b, down_a, src2dst, topk_ids, mapping


@pytest.mark.parametrize("target", [("gate",), ("up",), ()])
def test_partial_swiglu_slice_targets_need_no_zero_factors(target: tuple[str, ...]):
    contract = C2SemanticContract(
        activation="swiglu",
        logical_slices=("gate", "up"),
        lora_target_slices=target,
        logical_intermediate_size=3,
        physical_intermediate_size=5,
    )
    args = _tiny_inputs(contract)
    result = reference_c2_consumer(
        contract, *args, local_expert_offset=5, block_size_n=2
    )
    assert result.activation_output.shape == (7, 5)
    # Valid destinations zero their physical provider padding.
    assert bool((result.activation_output[[0, 2, 4, 6], 3:] == 0).all())
    # IDs below/above the local [5, 7) range never consume src2dst=99.
    assert bool((result.down_rank_input[1] == 0).all())
    # No target factors are allocated when no base slice is targeted.
    if not target:
        assert args[1].shape[-1] == 0
        assert args[2].shape[-2] == 0
    assert contract.benchmark_kernel_family() == "oracle_only_requires_provider_specialization"


def test_relu2_padding_invalid_ids_base_rows_and_destination_dtype():
    contract = C2SemanticContract(
        activation="relu2",
        logical_slices=("value",),
        lora_target_slices=("value",),
        logical_intermediate_size=3,
        physical_intermediate_size=5,
    )
    args = _tiny_inputs(contract)
    result = reference_c2_consumer(
        contract,
        *args,
        local_expert_offset=5,
        block_size_n=2,
        activation_output_dtype=torch.bfloat16,
    )
    assert result.activation_output.dtype == torch.bfloat16
    assert contract.benchmark_kernel_family() == "nongated_relu2_v1"
    # Invalid local IDs leave every destination row they would have named alone.
    assert bool((result.activation_output[[1, 3, 5]] == -123).all())
    # Token 1 is base-only (and also has invalid IDs), hence no down rank output.
    assert bool((result.down_rank_input[1] == 0).all())
    assert bool((result.activation_output[[0, 2, 4, 6], 3:] == 0).all())

    down_b = 0.1 * torch.randn(2, 2, 4, 2, dtype=torch.bfloat16)
    weights = torch.tensor(
        [[0.25, 0.75], [0.5, 0.5], [0.6, 0.4]], dtype=torch.float32
    )
    delta1 = reference_weighted_down_delta(
        result.down_rank_input,
        down_b,
        args[-2],
        weights,
        args[-1],
        local_expert_offset=5,
        routed_scaling_factor=1.0,
        destination_dtype=torch.float32,
    )
    delta175 = reference_weighted_down_delta(
        result.down_rank_input,
        down_b,
        args[-2],
        weights,
        args[-1],
        local_expert_offset=5,
        routed_scaling_factor=1.75,
        destination_dtype=torch.float32,
    )
    assert torch.allclose(delta175, 1.75 * delta1, rtol=2e-6, atol=2e-6)
    assert delta175.dtype == torch.float32
    delta_bf16 = reference_weighted_down_delta(
        result.down_rank_input,
        down_b,
        args[-2],
        weights,
        args[-1],
        local_expert_offset=5,
        routed_scaling_factor=1.75,
        destination_dtype=torch.bfloat16,
    )
    assert delta_bf16.dtype == torch.bfloat16


def test_provider_order_is_explicit_not_inferred_from_width():
    canonical = C2SemanticContract(
        activation="swiglu",
        logical_slices=("gate", "up"),
        lora_target_slices=("gate", "up"),
        logical_intermediate_size=8,
        physical_intermediate_size=8,
    )
    reordered = C2SemanticContract(
        activation="swiglu",
        logical_slices=("gate", "up"),
        lora_target_slices=("gate", "up"),
        logical_intermediate_size=8,
        physical_intermediate_size=8,
        provider_slice_order=("up", "gate"),
    )
    assert canonical.provider_output_width == reordered.provider_output_width == 16
    assert canonical.provider_slice_offset("gate") == 0
    assert reordered.provider_slice_offset("gate") == 8
    assert canonical.benchmark_kernel_family() == "gated_swiglu_v1"
    assert (
        reordered.benchmark_kernel_family()
        == "oracle_only_requires_provider_specialization"
    )


@pytest.mark.parametrize(
    ("activation", "slices"),
    [("swiglu", ("gate",)), ("relu2", ("gate", "up"))],
)
def test_invalid_activation_slice_contract_rejected(activation, slices):
    with pytest.raises(ValueError):
        C2SemanticContract(
            activation=activation,
            logical_slices=slices,
            lora_target_slices=slices,
            logical_intermediate_size=8,
            physical_intermediate_size=8,
        )
