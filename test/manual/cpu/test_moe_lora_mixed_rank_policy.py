import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from benchmark.kernels.lora_moe import summarize_mixed_rank_policy as summarizer
from benchmark.kernels.lora_moe.mixed_rank_policy import (
    load_factor_rank_representation,
    logical_factor_bytes,
    poison_canonical_factor_source,
)


def _source():
    return (
        torch.arange(2 * 1 * 8 * 3, dtype=torch.float32).reshape(2, 1, 8, 3),
        torch.arange(2 * 1 * 10 * 4, dtype=torch.float32).reshape(2, 1, 10, 4),
        torch.arange(2 * 1 * 4 * 5, dtype=torch.float32).reshape(2, 1, 4, 5),
        torch.arange(2 * 1 * 3 * 4, dtype=torch.float32).reshape(2, 1, 3, 4),
    )


def test_rank_packing_is_slice_aware_and_zeroes_only_active_physical_tails():
    source = _source()
    poisoned = poison_canonical_factor_source(
        source,
        allocated_rank=4,
        num_slices=2,
        slot_ranks=(2, 0),
    )
    packed = load_factor_rank_representation(
        poisoned,
        allocated_rank=4,
        physical_rank=2,
        num_slices=2,
        slot_ranks=(2, 0),
        representation="test",
    )
    assert torch.equal(packed.gate_up_a[0, :, :2], source[0][0, :, :2])
    assert torch.equal(packed.gate_up_a[0, :, 2:4], source[0][0, :, 4:6])
    assert torch.equal(packed.gate_up_b[0, :, :, :2], source[1][0, :, :, :2])
    assert torch.equal(packed.down_a[0, :, :2], source[2][0, :, :2])
    assert torch.equal(packed.down_b[0, :, :, :2], source[3][0, :, :, :2])
    assert torch.isnan(packed.gate_up_a[1]).all()
    assert torch.isnan(packed.gate_up_b[1]).all()


def test_padded_representation_zeros_active_tails_but_keeps_inactive_slots_poisoned():
    source = _source()
    poisoned = poison_canonical_factor_source(
        source,
        allocated_rank=4,
        num_slices=2,
        slot_ranks=(2, 0),
    )
    padded = load_factor_rank_representation(
        poisoned,
        allocated_rank=4,
        physical_rank=4,
        num_slices=2,
        slot_ranks=(2, 0),
        representation="test",
    )
    assert torch.equal(padded.gate_up_a[0, :, :2], source[0][0, :, :2])
    assert torch.count_nonzero(padded.gate_up_a[0, :, 2:4]) == 0
    assert torch.equal(padded.gate_up_a[0, :, 4:6], source[0][0, :, 4:6])
    assert torch.count_nonzero(padded.gate_up_a[0, :, 6:8]) == 0
    assert torch.count_nonzero(padded.gate_up_b[0, :, :, 2:]) == 0
    assert torch.isnan(padded.down_b[1]).all()


def test_logical_factor_bytes_excludes_rank_and_empty_slot_padding():
    source = _source()
    expected_elements = 1 * 2 * 2 * 3 + 1 * 10 * 2 + 1 * 2 * 5 + 1 * 3 * 2
    assert (
        logical_factor_bytes(
            source,
            allocated_rank=4,
            num_slices=2,
            slot_ranks=(2, 0),
        )
        == expected_elements * source[0].element_size()
    )


def test_physical_representation_can_pad_beyond_allocated_source_rank():
    source = _source()
    poisoned = poison_canonical_factor_source(
        source,
        allocated_rank=4,
        num_slices=2,
        slot_ranks=(3, 0),
    )
    physical = load_factor_rank_representation(
        poisoned,
        allocated_rank=4,
        physical_rank=8,
        num_slices=2,
        slot_ranks=(3, 0),
        representation="test",
    )
    assert physical.gate_up_a.shape[2] == 16
    assert torch.equal(physical.gate_up_a[0, :, :3], source[0][0, :, :3])
    assert torch.count_nonzero(physical.gate_up_a[0, :, 3:8]) == 0
    assert torch.equal(physical.gate_up_a[0, :, 8:11], source[0][0, :, 4:7])
    assert torch.count_nonzero(physical.gate_up_a[0, :, 11:16]) == 0


def _summary_artifact(execution: str, order: str, *, samples: int = 2):
    data = {
        "case": {"device": "h200", "case_id": "tiny"},
        "environment": {
            "cli": {"execution": execution, "samples": samples},
        },
        "order": order,
        "policies": {
            policy: {"timing": {"raw_samples_us": [1.0] * samples}}
            for policy in summarizer.POLICIES
        },
    }
    return Path(f"raw/tiny__{execution}__{order}.json"), data


def _tiny_complete_matrix():
    return [
        _summary_artifact(execution, order)
        for execution in ("eager", "cuda_graph")
        for order in ("forward", "reverse")
    ]


@pytest.mark.parametrize("corruption", ("missing", "duplicate", "sample_count"))
def test_summary_matrix_validation_rejects_incomplete_or_inconsistent_data(
    monkeypatch, corruption
):
    monkeypatch.setattr(
        summarizer,
        "mixed_rank_cases",
        lambda _device: (SimpleNamespace(case_id="tiny"),),
    )
    results = _tiny_complete_matrix()
    if corruption == "missing":
        results.pop()
    elif corruption == "duplicate":
        results.append(copy.deepcopy(results[0]))
    else:
        results[0][1]["policies"]["packed_bucket"]["timing"]["raw_samples_us"].pop()
    with pytest.raises(ValueError):
        summarizer._validate_complete_matrix(results)


def _policy_cell(
    *,
    rank: int,
    active: int,
    occupancy: str,
    effect: float,
    order_effects: tuple[float, float] | None = None,
):
    raw_winner = "padded_rmax" if effect >= 0.0 else "packed_bucket"
    if order_effects is None:
        order_effects = (effect, effect)
    return {
        "device": "h200",
        "case_id": f"{occupancy}-r{rank}",
        "execution": "cuda_graph",
        "model": "test-model",
        "phase": "decode",
        "T": 32,
        "K": 8,
        "H_moe": 2048,
        "I": 512,
        "E_local": 256,
        "occupancy": occupancy,
        "R": rank,
        "R_max": 128,
        "L_active": active,
        "B_base": int(occupancy in ("mixed", "all-base")),
        "L_capacity": 8,
        "packed_vs_padded_percent": effect,
        "order_effect_percent": dict(zip(summarizer.ORDERS, order_effects)),
        "raw_p50_winner": raw_winner,
        "regret_percent": {
            "padded_rmax": max(-effect, 0.0),
            "packed_bucket": max(effect, 0.0),
        },
    }


def test_noise_bounds_and_regret_exclude_base_and_null_control_winners():
    cells = [
        _policy_cell(
            rank=128,
            active=1,
            occupancy="mixed",
            effect=2.0,
        ),
        _policy_cell(rank=32, active=1, occupancy="mixed", effect=1.0),
        _policy_cell(rank=64, active=1, occupancy="mixed", effect=-5.0),
        _policy_cell(rank=16, active=1, occupancy="mixed", effect=5.0),
        _policy_cell(
            rank=128,
            active=0,
            occupancy="all-base",
            effect=9.0,
        ),
    ]
    summarizer._apply_noise_bounds(cells)

    assert cells[0]["evidence_class"] == "matched-null-control"
    assert cells[1]["evidence_class"] == "inconclusive-observed-bound"
    assert cells[1]["evidence_winner"] is None
    assert cells[2]["evidence_winner"] == "packed_bucket"
    assert cells[3]["evidence_winner"] == "padded_rmax"
    assert cells[4]["evidence_class"] == "all-base-bypass"

    regret = summarizer._static_regret(cells)
    global_row = next(
        row for row in regret if row["scope"] == "device_execution_global"
    )
    assert global_row["num_cells"] == 3
    assert global_row["evidence_winner_counts"] == {
        "padded_rmax": 1,
        "packed_bucket": 1,
    }
    assert global_row["inconclusive_cells"] == 1
    assert global_row["evidence_policy_recommendation"] is None
    assert (
        global_row["evidence_policy_recommendation_status"]
        == "beyond_bound_cells_disagree"
    )


def test_noise_bound_rejects_order_sign_flip_and_one_weak_order():
    cells = [
        _policy_cell(rank=128, active=1, occupancy="mixed", effect=1.0),
        _policy_cell(
            rank=32,
            active=1,
            occupancy="mixed",
            effect=-4.0,
            order_effects=(-6.0, 2.0),
        ),
        _policy_cell(
            rank=64,
            active=1,
            occupancy="mixed",
            effect=-3.0,
            order_effects=(-5.0, -0.5),
        ),
    ]
    summarizer._apply_noise_bounds(cells)

    assert cells[1]["evidence_class"] == "inconclusive-observed-bound"
    assert not cells[1]["evidence_consistent_order_winner"]
    assert cells[2]["evidence_class"] == "inconclusive-observed-bound"
    assert cells[2]["evidence_consistent_order_winner"]
    assert not cells[2]["evidence_each_order_exceeds_null"]


@pytest.mark.parametrize("corruption", ("duplicate_null", "missing_null"))
def test_noise_bound_rejects_duplicate_or_missing_shape_matched_null(corruption):
    null = _policy_cell(rank=128, active=1, occupancy="mixed", effect=1.0)
    lower = _policy_cell(rank=32, active=1, occupancy="mixed", effect=-5.0)
    cells = (
        [lower] if corruption == "missing_null" else [null, copy.deepcopy(null), lower]
    )

    with pytest.raises(ValueError):
        summarizer._apply_noise_bounds(cells)


def _correctness_artifact(
    *,
    rank: int,
    error: float,
    signal: float = 0.00325,
    relative_l2: float = 0.01,
    cosine: float = 0.9999,
):
    data = {
        "case": {
            "device": "h200",
            "model": "test-model",
            "phase": "prefill",
            "T": 2048,
            "K": 8,
            "H_moe": 2048,
            "I": 512,
            "E_local": 256,
            "R": rank,
            "R_max": 128,
            "L_active": 8,
            "B_base": 0,
            "L_capacity": 8,
        },
        "environment": {"cli": {"execution": "eager"}},
        "order": "forward",
        "correctness": {
            "padded_vs_packed_max_abs": error,
            "output_max_abs": 0.03,
            "active_delta": {
                "max_abs": signal,
                "difference": {"relative_l2": relative_l2},
                "cosine_similarity": cosine,
                "base_reference_max_abs_difference": 0.0,
            },
        },
    }
    return Path(f"raw/r{rank}.json"), data


def test_active_correctness_accepts_bf16_error_but_rejects_dropped_delta():
    null = _correctness_artifact(rank=128, error=0.00024, signal=0.0068)
    valid = _correctness_artifact(rank=32, error=0.00043)
    validations = summarizer._validate_active_correctness([null, valid])
    assert len(validations) == 2
    assert validations[1]["output_error"] < validations[1]["allowed_output_error"]

    dropped = _correctness_artifact(rank=32, error=0.00325)
    with pytest.raises(ValueError, match="output error"):
        summarizer._validate_active_correctness([null, dropped])


@pytest.mark.parametrize(
    ("relative_l2", "cosine"),
    ((0.11, 0.9999), (0.01, 0.994)),
)
def test_active_correctness_rejects_wrong_delta_vector(relative_l2, cosine):
    null = _correctness_artifact(rank=128, error=0.00024, signal=0.0068)
    wrong = _correctness_artifact(
        rank=32,
        error=0.00043,
        relative_l2=relative_l2,
        cosine=cosine,
    )
    with pytest.raises(ValueError, match="delta-vector"):
        summarizer._validate_active_correctness([null, wrong])
