import json
from dataclasses import replace

import pytest
import torch

from benchmark.kernels.lora_moe.bench_shared_outer import (
    DOWN_VARIANTS,
    GATE_VARIANTS,
    SHARED_OUTER_CASES,
    _case_metadata,
    _json_environment,
    _resource_limit_result,
    _select_cases,
    _site_variants,
    parse_args,
)
from benchmark.kernels.lora_moe.matrix import model_shape_cases
from benchmark.kernels.lora_moe.shared_outer import (
    AdapterSpan,
    arithmetic_work,
    contiguous_adapter_spans,
    down_b_repeated_pair_reference,
    down_b_weighted_rank_reduction_reference,
    gate_up_a_deduplicated_reference,
    gate_up_a_repeated_pair_reference,
    token_lora_mapping_from_spans,
)
from benchmark.kernels.lora_moe.shared_outer_triton import (
    GATE_A_CONFIGS_BY_KEY,
    build_shared_outer_tile_plan,
)


def test_shared_outer_factor_shapes_only_share_the_outer_factors():
    case = replace(
        model_shape_cases("h200")[0],
        adapters=replace(model_shape_cases("h200")[0].adapters, shared_outer=True),
    )
    shapes = case.factor_shapes

    assert shapes.gate_up_a[1] == 1
    assert shapes.down_b[1] == 1
    assert shapes.gate_up_b[1] == case.e_local
    assert shapes.down_a[1] == case.e_local


def test_contiguous_adapter_spans_encode_base_as_minus_one():
    spans = contiguous_adapter_spans(
        num_tokens=11, active_adapters=3, include_base=True
    )
    assert spans == (
        AdapterSpan(0, 3, 0),
        AdapterSpan(3, 6, 1),
        AdapterSpan(6, 9, 2),
        AdapterSpan(9, 11, None),
    )
    assert token_lora_mapping_from_spans(spans).tolist() == [
        0,
        0,
        0,
        1,
        1,
        1,
        2,
        2,
        2,
        -1,
        -1,
    ]


def test_gate_up_shared_a_token_dedup_matches_repeated_pairs():
    generator = torch.Generator().manual_seed(20260722)
    hidden = torch.randn(9, 13, generator=generator)
    shared_a = torch.randn(3, 1, 10, 13, generator=generator)
    spans = contiguous_adapter_spans(num_tokens=9, active_adapters=2, include_base=True)

    repeated = gate_up_a_repeated_pair_reference(hidden, shared_a, spans, top_k=4)
    deduplicated = gate_up_a_deduplicated_reference(hidden, shared_a, spans, top_k=4)
    token_owned = gate_up_a_deduplicated_reference(
        hidden, shared_a, spans, top_k=4, materialize_pairs=False
    )

    assert torch.allclose(deduplicated, repeated)
    assert torch.allclose(token_owned[:, None, :].expand_as(repeated), repeated)
    assert torch.count_nonzero(repeated[6:]) == 0


@pytest.mark.parametrize("top_k", (1, 2, 8))
def test_down_shared_b_weighted_rank_reduction_matches_repeated_pairs(top_k: int):
    generator = torch.Generator().manual_seed(20260722 + top_k)
    pair_rank = torch.randn(12, top_k, 7, generator=generator)
    shared_b = torch.randn(4, 1, 17, 7, generator=generator)
    topk_weights = torch.rand(12, top_k, generator=generator)
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)
    spans = contiguous_adapter_spans(
        num_tokens=12, active_adapters=3, include_base=True
    )

    repeated = down_b_repeated_pair_reference(pair_rank, shared_b, topk_weights, spans)
    reduced = down_b_weighted_rank_reduction_reference(
        pair_rank, shared_b, topk_weights, spans
    )

    assert torch.allclose(reduced, repeated, rtol=2e-5, atol=2e-5)
    assert torch.count_nonzero(repeated[9:]) == 0


def test_shared_outer_arithmetic_counts_expose_topk_redundancy():
    work = arithmetic_work(
        active_tokens=32,
        top_k=8,
        hidden_size=2048,
        rank=64,
        slices=2,
    )
    assert work["gate_a_gemm_reduction_ratio"] == 8.0
    assert work["down_b_gemm_reduction_ratio"] == 8.0
    assert (
        work["gate_a_repeated_pair_multiplies"]
        == 8 * work["gate_a_deduplicated_multiplies"]
    )
    assert (
        work["down_b_weighted_rank_reduce_multiplies"]
        < work["down_b_repeated_pair_multiplies"]
    )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"num_tokens": 0, "active_adapters": 1, "include_base": False},
        {"num_tokens": 2, "active_adapters": 3, "include_base": False},
        {"num_tokens": 2, "active_adapters": 0, "include_base": False},
    ),
)
def test_invalid_span_requests_are_rejected(kwargs):
    with pytest.raises(ValueError):
        contiguous_adapter_spans(**kwargs)


def test_shared_outer_benchmark_matrix_covers_rank_adapter_and_phase_axes():
    assert {case.rank for case in SHARED_OUTER_CASES} == {16, 32, 64, 128}
    assert {case.active_adapters for case in SHARED_OUTER_CASES} >= {1, 3, 4, 8}
    assert {case.phase for case in SHARED_OUTER_CASES} == {"decode", "prefill"}
    assert any(case.include_base for case in SHARED_OUTER_CASES)
    assert max(case.hidden_size for case in SHARED_OUTER_CASES) == 7168

    metadata = _case_metadata(_select_cases("qwen35-t32-r64-l4-base", False)[0])
    assert metadata["active_tokens"] < metadata["tokens"]
    assert metadata["factor_shapes"]["gate_up_a"][1] == 1
    assert metadata["factor_shapes"]["down_b"][1] == 1
    assert metadata["arithmetic_work"]["gate_a_gemm_reduction_ratio"] == 8.0


def test_shared_outer_cli_keeps_site_specific_variants_explicit():
    args = parse_args(
        [
            "--case-id",
            "qwen35-t32-r64-l4",
            "--site",
            "down-b",
            "--variant",
            "weighted-rank-reduce",
            "--execution",
            "cuda_graph",
        ]
    )
    assert args.variant == "weighted-rank-reduce"
    assert _site_variants("gate-a", "all") == GATE_VARIANTS
    assert _site_variants("down-b", "all") == DOWN_VARIANTS
    with pytest.raises(ValueError, match="invalid for gate-a"):
        _site_variants("gate-a", "weighted-rank-reduce")


def test_shared_outer_environment_normalizes_path_metadata(monkeypatch, tmp_path):
    args = parse_args(["--json-output", str(tmp_path / "result.json")])
    monkeypatch.setattr(
        "benchmark.kernels.lora_moe.bench_shared_outer._environment",
        lambda parsed: {"cli": vars(parsed)},
    )
    environment = _json_environment(args)
    assert environment["cli"]["json_output"] == str(tmp_path / "result.json")
    json.dumps(environment)


def test_shared_outer_tile_plan_keeps_tensor_core_tiles_inside_adapter_spans():
    spans = contiguous_adapter_spans(
        num_tokens=32, active_adapters=4, include_base=False
    )
    config = GATE_A_CONFIGS_BY_KEY["bm16-bn64-bk64-w4"]
    plan = build_shared_outer_tile_plan(
        spans,
        output_width=128,
        block_m=config.block_m,
        block_n=config.block_n,
        device=torch.device("cpu"),
    )
    # Four eight-token adapter spans, each with two N tiles.  A tile never
    # crosses an adapter boundary even though BM=16.
    assert plan.num_tiles == 8
    assert plan.adapter_ids.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert plan.token_starts.tolist() == [0, 0, 8, 8, 16, 16, 24, 24]
    assert plan.token_counts.tolist() == [8] * 8
    assert plan.n_block_ids.tolist() == [0, 1] * 4


def test_resource_limit_result_preserves_matrix_evidence():
    from benchmark.kernels.lora_moe.shared_outer_triton import (
        DOWN_B_CONFIGS_BY_KEY,
    )

    gate = GATE_A_CONFIGS_BY_KEY["bm16-bn32-bk64-w4"]
    down = DOWN_B_CONFIGS_BY_KEY["bm16-bn128-br64-w4"]
    result = _resource_limit_result(
        site="gate-a",
        variant="production",
        scope="K0",
        gate_config=gate,
        down_config=down,
        error=RuntimeError("shared memory"),
    )
    assert result["status"] == "unsupported_resource_limit"
    assert result["candidate_config"] is None
    assert result["error"] == "shared memory"
