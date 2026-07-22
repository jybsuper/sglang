import importlib.util
import sys
from pathlib import Path

import pytest


def _load_rank_policy_module():
    path = (
        Path(__file__).resolve().parents[3]
        / "python/sglang/srt/lora/sgl_lora/rank_policy.py"
    )
    spec = importlib.util.spec_from_file_location("_test_sgl_lora_rank_policy", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_rank_policy = _load_rank_policy_module()
build_moe_lora_rank_plan = _rank_policy.build_moe_lora_rank_plan
resolve_moe_lora_physical_rank = _rank_policy.resolve_moe_lora_physical_rank


def test_uniform_lower_rank_separates_logical_allocated_and_physical_rank():
    padded = build_moe_lora_rank_plan(
        (32, 32, 0, 0), allocated_rank=128, policy="padded_rmax"
    )
    packed = build_moe_lora_rank_plan(
        (32, 32, 0, 0), allocated_rank=128, policy="packed_bucket"
    )

    assert padded.slot_ranks == packed.slot_ranks == (32, 32, 0, 0)
    assert padded.physical_ranks == (128,)
    assert packed.physical_ranks == (32,)
    assert padded.buckets[0].logical_ranks == (32, 32)
    assert packed.buckets[0].adapter_slots == (0, 1)
    assert padded.slot_to_bucket == packed.slot_to_bucket == (0, 0, -1, -1)
    assert padded.runtime_metadata_allocations == 0
    assert packed.runtime_metadata_allocations == 0
    rebuilt = build_moe_lora_rank_plan(
        (32, 32, 0, 0), allocated_rank=128, policy="packed_bucket"
    )
    changed = build_moe_lora_rank_plan(
        (32, 0, 32, 0), allocated_rank=128, policy="packed_bucket"
    )
    assert packed.graph_key == rebuilt.graph_key
    assert hash(packed.graph_key) == hash(rebuilt.graph_key)
    assert packed.graph_key != changed.graph_key


def test_packed_plan_groups_heterogeneous_slots_by_physical_rank():
    plan = build_moe_lora_rank_plan(
        (32, 0, 64, 12, 8, 32),
        allocated_rank=128,
        policy="packed_bucket",
    )
    assert plan.physical_ranks == (16, 32, 64)
    assert plan.buckets[0].adapter_slots == (3, 4)
    assert plan.buckets[0].logical_ranks == (12, 8)
    assert plan.buckets[1].adapter_slots == (0, 5)
    assert plan.buckets[2].adapter_slots == (2,)
    assert plan.slot_to_bucket == (1, -1, 2, 0, 0, 1)


def test_all_base_plan_has_no_rank_bucket_or_runtime_metadata():
    graph_keys = []
    for policy in ("padded_rmax", "packed_bucket"):
        plan = build_moe_lora_rank_plan((0,) * 8, allocated_rank=128, policy=policy)
        assert not plan.has_active_adapters
        assert plan.active_adapter_count == 0
        assert plan.buckets == ()
        assert plan.slot_to_bucket == (-1,) * 8
        assert plan.runtime_metadata_allocations == 0
        graph_keys.append(plan.graph_key)
    assert (
        graph_keys[0]
        == graph_keys[1]
        == (
            "moe_lora_rank_plan_v1",
            "no_lora",
        )
    )


def test_non_aligned_allocated_rank_resolves_larger_physical_storage():
    for policy in ("padded_rmax", "packed_bucket"):
        plan = build_moe_lora_rank_plan((33,), allocated_rank=33, policy=policy)
        assert plan.physical_ranks == (48,)


@pytest.mark.parametrize("logical,physical", ((8, 16), (12, 16), (16, 16), (17, 32)))
def test_physical_rank_padding_preserves_small_logical_ranks(logical, physical):
    assert resolve_moe_lora_physical_rank(logical) == physical


def test_non_aligned_minimum_still_resolves_an_aligned_physical_rank():
    assert resolve_moe_lora_physical_rank(8, alignment=16, minimum=20) == 32


@pytest.mark.parametrize(
    "kwargs",
    (
        {"slot_ranks": (), "allocated_rank": 128, "policy": "padded_rmax"},
        {"slot_ranks": (129,), "allocated_rank": 128, "policy": "padded_rmax"},
        {"slot_ranks": (32,), "allocated_rank": 0, "policy": "padded_rmax"},
        {"slot_ranks": (32,), "allocated_rank": 128, "policy": "unknown"},
        {
            "slot_ranks": (0,),
            "allocated_rank": 128,
            "policy": "packed_bucket",
            "rank_alignment": 0,
        },
        {
            "slot_ranks": (0,),
            "allocated_rank": 128,
            "policy": "padded_rmax",
            "minimum_physical_rank": -1,
        },
    ),
)
def test_rank_plan_rejects_invalid_static_metadata(kwargs):
    with pytest.raises(ValueError):
        build_moe_lora_rank_plan(**kwargs)
