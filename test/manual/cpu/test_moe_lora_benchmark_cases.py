from dataclasses import FrozenInstanceError, replace

import pytest

from benchmark.kernels.lora_moe.cases import AdapterBatch
from benchmark.kernels.lora_moe.matrix import (
    MODEL_PRESETS,
    get_model_shape,
    model_shape_cases,
    p0_cases,
)


def test_model_presets_match_the_design():
    assert tuple(MODEL_PRESETS) == (
        "qwen3.5-35b-a3b",
        "qwen3.5-397b-a17b",
        "kimi-k2.5",
        "glm-5.2",
        "nemotron-3-super",
        "nemotron-3-nano",
    )
    super_shape = get_model_shape("nemotron-3-super")
    assert (super_shape.h_model, super_shape.h_moe) == (4096, 1024)
    assert (
        super_shape.intermediate_size,
        super_shape.num_experts,
        super_shape.top_k,
    ) == (
        2688,
        512,
        22,
    )
    assert (super_shape.num_slices, super_shape.activation) == (1, "relu2")


@pytest.mark.parametrize(
    "updates",
    (
        {"l_active": -1},
        {"b_base": 2},
        {"b_base": True},
        {"l_active": 4, "b_base": 1, "l_capacity": 4},
        {"rank": 64, "max_rank": 32},
    ),
)
def test_adapter_batch_rejects_invalid_occupancy_or_rank(updates):
    values = dict(
        l_active=1,
        b_base=0,
        l_capacity=8,
        rank=32,
        max_rank=32,
        physical_rank=32,
    )
    values.update(updates)
    with pytest.raises(ValueError):
        AdapterBatch(**values)


def test_p0_cells_are_exact_resolved_runs():
    cases = p0_cases("h200")
    actual = tuple(
        (
            case.t_local,
            case.adapters.l_active,
            case.adapters.b_base,
            case.adapters.l_capacity,
            case.adapters.rank,
        )
        for case in cases
    )
    assert actual == (
        (1, 1, 0, 1, 32),
        (32, 0, 1, 8, 64),
        (32, 1, 0, 1, 64),
        (32, 1, 0, 8, 64),
        (32, 1, 1, 8, 64),
        (32, 4, 1, 5, 64),
        (32, 7, 1, 8, 128),
        (32, 8, 0, 8, 128),
        (256, 1, 0, 1, 32),
        (128, 3, 0, 8, 64),
        (256, 3, 0, 8, 64),
        (257, 3, 0, 8, 64),
        (2048, 3, 0, 8, 64),
        (256, 8, 0, 8, 128),
    )
    assert len({case.case_id for case in cases}) == 14
    assert cases[1].pipeline == "N0"
    assert all(case.pipeline == "C0" for i, case in enumerate(cases) if i != 1)
    assert cases[-1].cache_state == "hot"
    assert all(case.phase == "prefill" for case in cases[9:13])


def test_local_geometry_and_factor_shapes_are_derived():
    qwen = model_shape_cases("gb300")[1]
    tp4 = replace(qwen, tp_size=4)
    assert (tp4.e_local, tp4.i_local, tp4.pair_capacity) == (512, 256, 320)
    assert tp4.factor_shapes.gate_up_a == (8, 512, 64, 4096)
    assert tp4.factor_shapes.gate_up_b == (8, 512, 512, 32)
    assert tp4.factor_shapes.down_a == (8, 512, 32, 256)
    assert tp4.factor_shapes.down_b == (8, 512, 4096, 32)

    ep4 = replace(qwen, tp_size=4, ep_size=4, ep_rank=3)
    assert (ep4.e_local, ep4.i_local, ep4.global_expert_offset) == (128, 1024, 384)

    nemotron = model_shape_cases("gb300")[4]
    assert nemotron.factor_shapes.gate_up_a[-1] == 1024
    assert nemotron.factor_shapes.gate_up_b[-2] == 2688


def test_case_records_are_frozen_and_scalar_resolved():
    case = p0_cases("h200")[0]
    assert hash(case)
    assert case.provider == "deepgemm_bf16"
    assert case.scope == case.stage == "M0"
    with pytest.raises(FrozenInstanceError):
        case.t_local = 2


def test_unknown_model_has_a_clear_error():
    with pytest.raises(ValueError, match="unknown model preset"):
        get_model_shape("missing")
