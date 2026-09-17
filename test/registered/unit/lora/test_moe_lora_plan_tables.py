"""Shipped MoE plans, selector boundaries, and consumer-specific tile contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sglang.srt.lora.moe import plan as ep
from sglang.srt.lora.moe.plan import (
    ActFamily,
    ActivationFn,
    AFamily,
    BFamily,
    DownOverlap,
    FinalizeFamily,
    GateUpOverlap,
    MoeLoraLaunchConfig,
    RouteBuilderFamily,
)
from sglang.srt.lora.utils import Phase
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def _packaged_plans(monkeypatch):
    monkeypatch.delenv("SGLANG_LORA_MOE_CONFIG_DIR", raising=False)
    ep._load_plans.cache_clear()
    try:
        yield
    finally:
        ep._load_plans.cache_clear()


@pytest.fixture
def override_table(tmp_path, monkeypatch):
    path = tmp_path / "sm100.plans.json"
    data = json.loads((Path(ep._CONFIG_DIR) / path.name).read_text())
    monkeypatch.setenv("SGLANG_LORA_MOE_CONFIG_DIR", str(tmp_path))

    def write():
        path.write_text(json.dumps(data))
        ep._load_plans.cache_clear()

    return data, write


def _resolve(
    architecture="sm100",
    layout=False,
    rank=16,
    act=ActivationFn.SILU,
    hidden=4096,
    experts=256,
    quant="bf16",
):
    return ep.resolve_plans(
        quant_family=quant,
        architecture=architecture,
        is_shared_outer=layout,
        physical_rank=rank,
        activation=act,
        hidden_size=hidden,
        num_local_experts=experts,
    )


@pytest.mark.parametrize("architecture", ("sm90", "sm100"))
@pytest.mark.parametrize("shared", (False, True))
def test_shipped_decode_plan(architecture, shared):
    selected = _resolve(architecture, layout=shared, rank=32)[Phase.DECODE]
    plan = selected.plan
    indexed_b = architecture == "sm100" and not shared
    assert selected.base_gemm_rows == "expert_major"
    assert plan.gate_up_a.family is AFamily.GROUPED
    assert plan.down_a.family is (
        AFamily.PER_ROW if architecture == "sm90" and not shared else AFamily.GROUPED
    )
    assert plan.gate_up_b.family is (BFamily.PER_ROW if indexed_b else BFamily.GROUPED)
    assert (plan.down_b.family if plan.down_b else None) == (
        None if shared else BFamily.PER_ROW if indexed_b else BFamily.GROUPED
    )
    assert plan.act.family is ActFamily.MATERIALIZED
    assert plan.finalize.family is (
        FinalizeFamily.SHARED_ONE_PASS if shared else FinalizeFamily.MATERIALIZED
    )
    assert plan.gate_up_overlap is GateUpOverlap.GATE_UP_A_B
    assert plan.down_overlap is (DownOverlap.DOWN_A if shared else DownOverlap.DOWN_A_B)
    assert plan.route_builder is (
        RouteBuilderFamily.PARALLEL_SHARED_OUTER
        if shared
        else RouteBuilderFamily.STANDARD
    )


@pytest.mark.parametrize("architecture", ("sm90", "sm100"))
@pytest.mark.parametrize("shared", (False, True))
def test_shipped_prefill_plan(architecture, shared):
    selected = _resolve(architecture, layout=shared, rank=32)[Phase.PREFILL]
    plan = selected.plan
    assert selected.base_gemm_rows == "route_major"
    assert plan.gate_up_a.family is (
        AFamily.TOKEN_GROUPED if shared else AFamily.GROUPED
    )
    assert plan.act.family is ActFamily.B_ACTIVATION and plan.gate_up_b is None
    assert plan.finalize.family is (
        FinalizeFamily.SHARED_TOKEN_DELTA if shared else FinalizeFamily.MATERIALIZED
    )
    assert plan.down_b_into_base == (architecture == "sm90" and not shared)
    assert plan.gate_up_overlap is GateUpOverlap.NONE
    assert plan.down_overlap is DownOverlap.NONE
    assert plan.route_builder is (
        RouteBuilderFamily.PARALLEL_SHARED_OUTER
        if shared
        else RouteBuilderFamily.STANDARD
    )


@pytest.mark.parametrize("shared", (False, True))
def test_sm90_prefill_rank_changes_plan(shared):
    prefix = "prefill.shared" if shared else "prefill.per_expert"
    small = _resolve("sm90", layout=shared, rank=16)[Phase.PREFILL]
    wide = _resolve("sm90", layout=shared, rank=17)[Phase.PREFILL]
    assert small.name == prefix + ".rank_le16" and wide.name == prefix
    assert small.plan.gate_up_overlap is GateUpOverlap.GATE_UP_A
    assert wide.plan.gate_up_overlap is GateUpOverlap.NONE
    if shared:
        assert small.plan.route_builder is RouteBuilderFamily.STANDARD
        assert wide.plan.route_builder is RouteBuilderFamily.PARALLEL_SHARED_OUTER


@pytest.mark.parametrize("rank,rule_index", ((8, 0), (9, 1), (16, 1), (17, 2)))
def test_sm100_prefill_rank_selects_the_complete_launch_config(rank, rule_index):
    row = next(
        row
        for row in ep._load_plans("sm100").scenarios
        if row.name == "prefill.per_expert"
    )
    selected = _resolve(rank=rank)[Phase.PREFILL]
    config = selected.tiles.config_for(1)
    assert selected.name == "prefill.per_expert"
    assert config == MoeLoraLaunchConfig(**row.tiles[rule_index].sites)


@pytest.mark.parametrize(
    "rank,split_k,b_widths",
    (
        (16, (8, 4, 4, 1), (128, 128, 128, 128)),
        (17, (1, 1, 1, 1), (128, 512, 512, 512)),
        (32, (1, 1, 1, 1), (128, 512, 512, 512)),
        (33, (1, 1, 1, 1), (128, 512, 512, 256)),
    ),
)
def test_sm100_decode_rank_and_token_ladders(rank, split_k, b_widths):
    configs = [
        _resolve(rank=rank)[Phase.DECODE].tiles.config_for(t) for t in (4, 5, 16, 17)
    ]
    assert tuple(c.gate_up_a.get("SPLIT_K", 1) for c in configs) == split_k
    assert all(c.down_a.get("SPLIT_K", 1) == 1 for c in configs)
    assert tuple(c.gate_up_b["BLOCK_SIZE_N"] for c in configs) == b_widths


def test_rank_filtered_ladder_uses_launch_defaults():
    config = {**MoeLoraLaunchConfig().gate_up_a, "BLOCK_SIZE_N": 64}
    rules = [ep._TileRuleModel(max_rank=8, sites={"gate_up_a": config})]
    assert ep._tile_table(rules, 8).config_for(1).gate_up_a == config
    assert ep._tile_table(rules, 9).config_for(1) == MoeLoraLaunchConfig()


@pytest.mark.parametrize(
    "architecture,shared,phase",
    (
        ("sm90", False, Phase.DECODE),
        ("sm100", True, Phase.DECODE),
        ("sm100", False, Phase.PREFILL),
    ),
)
def test_split_k_is_not_enabled_for_other_paths(architecture, shared, phase):
    tiles = _resolve(architecture, layout=shared)[phase].tiles
    for tokens in (4, 5, 16, 17):
        config = tiles.config_for(tokens)
        assert (
            config.gate_up_a.get("SPLIT_K", 1) == config.down_a.get("SPLIT_K", 1) == 1
        )


@pytest.mark.parametrize(
    "quant,experts,expected",
    (
        ("bf16", 512, "decode.shared"),
        ("fp8", 511, "decode.shared"),
        ("fp8", 512, "decode.shared.fp8.e_ge512"),
        ("nvfp4", 512, "decode.shared.nvfp4"),
    ),
)
def test_sm100_shared_quant_selection(quant, experts, expected):
    selected = _resolve(layout=True, experts=experts, quant=quant)
    assert selected[Phase.DECODE].name == expected
    assert selected[Phase.PREFILL].name == "prefill.shared"
    assert (
        _resolve(experts=experts, quant=quant)[Phase.DECODE].name == "decode.per_expert"
    )
    if quant == "nvfp4":
        plan = selected[Phase.DECODE].plan
        assert plan.gate_up_a.family is AFamily.TOKEN_DENSE
        assert plan.gate_up_b.family is BFamily.PER_ROW
        assert plan.down_a.family is AFamily.PER_ROW
    elif quant == "fp8" and experts == 512:
        plan = selected[Phase.DECODE].plan
        assert plan.act.family is ActFamily.B_ACTIVATION
        assert plan.gate_up_overlap is GateUpOverlap.GATE_UP_A


@pytest.mark.parametrize("architecture", ("sm90", "sm100"))
@pytest.mark.parametrize("axis", ("hidden", "experts"))
def test_tuned_domain_boundary(architecture, axis):
    limit = 512 if axis == "experts" else 4096 if architecture == "sm90" else 8192
    assert (
        _resolve(architecture, **{axis: limit})[Phase.DECODE].name
        == "decode.per_expert"
    )
    outside = _resolve(architecture, **{axis: limit + 1})
    assert outside[Phase.DECODE].name == "fallback.decode"
    assert outside[Phase.PREFILL].name == "fallback.prefill.per_expert"
    shared = _resolve(architecture, layout=True, quant="fp8", **{axis: limit + 1})
    assert shared[Phase.DECODE].name == "fallback.decode.shared"


@pytest.mark.parametrize("architecture", ("default", "unknown"))
def test_missing_architecture_uses_serial_fallback(architecture):
    selected = _resolve(architecture)
    assert selected[Phase.DECODE].name == "fallback.decode"
    assert selected[Phase.PREFILL].name == "fallback.prefill.per_expert"
    assert selected[Phase.DECODE].plan.gate_up_overlap is GateUpOverlap.NONE
    assert selected[Phase.DECODE].plan.down_overlap is DownOverlap.NONE


@pytest.mark.parametrize("architecture", ("sm90", "sm100"))
@pytest.mark.parametrize("shared", (False, True))
def test_activation_is_injected_without_changing_selection(architecture, shared):
    original = _resolve(architecture, layout=shared)
    changed = _resolve(architecture, layout=shared, act=ActivationFn.RELU2)
    for phase in Phase:
        assert original[phase].name == changed[phase].name
        assert original[phase].plan.act.activation is ActivationFn.SILU
        assert changed[phase].plan.act.activation is ActivationFn.RELU2


@pytest.mark.parametrize("architecture", ("sm90", "sm100", "default"))
def test_shipped_rows_have_unique_names_and_the_selected_finalize_tiles(architecture):
    table = ep._load_plans(architecture)
    rows = (*table.scenarios, *table.fallback)
    assert len({row.name for row in rows}) == len(rows)
    sections = {
        FinalizeFamily.SHARED_TOKEN_DELTA: "shared_token_delta",
        FinalizeFamily.SHARED_ONE_PASS: "shared_one_pass",
    }
    for row in rows:
        section = sections.get(row.plan.finalize_family)
        for rule in row.tiles:
            # Selected shared finalizers must declare their own tuning section.
            assert set(rule.sites) & set(sections.values()) == (
                {section} if section else set()
            )


def test_override_wins_and_packaged_table_is_restored(override_table, monkeypatch):
    data, write = override_table
    data["scenarios"][0]["tiles"][0]["sites"]["gate_up_a"]["num_warps"] = 8
    write()
    assert _resolve()[Phase.DECODE].tiles.config_for(4).gate_up_a["num_warps"] == 8
    monkeypatch.delenv("SGLANG_LORA_MOE_CONFIG_DIR")
    ep._load_plans.cache_clear()
    assert _resolve()[Phase.DECODE].tiles.config_for(4).gate_up_a["num_warps"] == 4


@pytest.mark.parametrize("defect", ("tile", "fallback_plan", "wildcard"))
def test_invalid_unselected_rows_fail_at_load(override_table, defect):
    data, write = override_table
    if defect == "tile":
        row = data["scenarios"][0]
        row["layout"] = "shared"
        row["tiles"][0]["sites"]["gate_up_a"]["BLOCK_SIZE_N"] = 24
        error = "gate_up_a.BLOCK_SIZE_N"
    elif defect == "fallback_plan":
        data["fallback"][0]["plan"]["down_a_family"] = "token_grouped"
        error = "shared gate/up-A family"
    else:
        row = next(r for r in data["scenarios"] if r["name"] == "decode.shared")
        row["layout"] = None
        error = "requires shared-outer"
    write()
    with pytest.raises(ValueError, match=error):
        _resolve()


@pytest.mark.parametrize("defect", ("duplicate", "shadowed"))
def test_invalid_token_ladders_fail_at_load(override_table, defect):
    data, write = override_table
    rules = data["scenarios"][0]["tiles"]
    if defect == "duplicate":
        rules[1]["max_tokens"] = rules[0]["max_tokens"]
    else:
        rules.append({"sites": {}})
    write()
    with pytest.raises(ValueError, match="tile"):
        _resolve()


def test_unknown_domain_bound_does_not_widen_the_table(override_table):
    data, write = override_table
    data["domain"]["max_hidden_size"] = data["domain"].pop("max_hidden")
    write()
    with pytest.raises(ValueError, match="max_hidden_size"):
        _resolve()


def test_numeric_split_mode_is_not_a_launch_dimension():
    config = {**MoeLoraLaunchConfig().gate_up_a, "SPLIT_MODE": 1}
    with pytest.raises(ValueError, match="SPLIT_MODE"):
        MoeLoraLaunchConfig(gate_up_a=config)


@pytest.mark.parametrize("key,value", (("num_stages", 0), ("BLOCK_SIZE_N", None)))
def test_launch_config_requires_positive_values_and_required_keys(key, value):
    config = dict(MoeLoraLaunchConfig().gate_up_a)
    if value is None:
        del config[key]
    else:
        config[key] = value
    with pytest.raises(
        ValueError, match="missing launch keys" if value is None else "positive"
    ):
        MoeLoraLaunchConfig(gate_up_a=config)


def test_lora_backends_have_distinct_dispatch_predicates_and_cli_choices():
    from sglang.srt.arg_groups.choices import MOE_RUNNER_BACKEND_CHOICES
    from sglang.srt.layers.moe.utils import MoeRunnerBackend

    lora_names = {"lora_cutedsl", "lora_triton", "lora_marlin"}
    for backend in MoeRunnerBackend:
        lora = backend.value in lora_names
        assert backend.is_lora() == lora
        assert backend.is_lora_marlin() == (backend is MoeRunnerBackend.LORA_MARLIN)
        if lora:
            assert not backend.is_triton() and not backend.is_marlin()
            assert backend.value in MOE_RUNNER_BACKEND_CHOICES


def _load_consumer_plan(monkeypatch, layout, plan, sites):
    row = {
        "name": "consumer",
        "layout": layout,
        "base_gemm_rows": "expert_major",
        "plan": plan,
        "tiles": [{"sites": sites}],
    }
    monkeypatch.setattr(ep, "_read_table", lambda _: {"fallback": [row]})
    ep._load_plans.cache_clear()
    return ep._load_plans("consumer")


@pytest.mark.parametrize(
    "layout,plan,site,tile_keys",
    (
        ("per_expert", {}, "gate_up_a", ("BLOCK_SIZE_N", "BLOCK_SIZE_K")),
        ("per_expert", {}, "gate_up_b", ("BLOCK_SIZE_N",)),
        (
            "shared",
            {"gate_up_a_family": "token_grouped"},
            "gate_up_a",
            ("BLOCK_SIZE_N", "BLOCK_SIZE_K"),
        ),
        (
            "per_expert",
            {"act_family": "b_activation"},
            "b_activation",
            ("BLOCK_SIZE_W", "BLOCK_SIZE_K"),
        ),
        (
            "per_expert",
            {"down_b_family": "per_row", "down_b_into_base": True},
            "down_b",
            ("BLOCK_SIZE_N", "BLOCK_SIZE_K"),
        ),
        (
            "shared",
            {"finalize_family": "shared_token_delta"},
            "down_b",
            ("BLOCK_SIZE_N",),
        ),
    ),
)
def test_dot_consumers_require_grouping_and_dot_tiles(
    monkeypatch, layout, plan, site, tile_keys
):
    default = getattr(MoeLoraLaunchConfig(), site)
    config = {key: value for key, value in default.items() if key != "GROUP_SIZE_M"}
    with pytest.raises(ValueError, match=f"{site} requires GROUP_SIZE_M"):
        _load_consumer_plan(monkeypatch, layout, plan, {site: config})
    for key in tile_keys:
        config = {**default, key: 8}
        with pytest.raises(ValueError, match=f"{site}.{key}"):
            _load_consumer_plan(monkeypatch, layout, plan, {site: config})
        config[key] = 16
        _load_consumer_plan(monkeypatch, layout, plan, {site: config})


@pytest.mark.parametrize(
    "layout,plan,sites",
    (
        (
            "per_expert",
            {
                f"{site}_family": "per_row"
                for site in ("gate_up_a", "gate_up_b", "down_a", "down_b")
            },
            ("gate_up_a", "gate_up_b", "down_a", "down_b", "b_activation"),
        ),
        (
            "shared",
            {"gate_up_a_family": "token_dense", "gate_up_b_family": "per_row"},
            ("gate_up_a", "gate_up_b"),
        ),
        ("shared", {"finalize_family": "shared_one_pass"}, ("down_b",)),
    ),
)
def test_vector_and_unused_tiles_do_not_require_dot_geometry(
    monkeypatch, layout, plan, sites
):
    default = MoeLoraLaunchConfig()
    configs = {
        site: {
            key: 8 if key.startswith("BLOCK_SIZE_") and key != "BLOCK_SIZE_M" else value
            for key, value in getattr(default, site).items()
            if key != "GROUP_SIZE_M"
        }
        for site in sites
    }
    _load_consumer_plan(monkeypatch, layout, plan, configs)


@pytest.mark.parametrize("shared", (False, True))
def test_grouped_b_keeps_launcher_clamped_k(monkeypatch, shared):
    default = MoeLoraLaunchConfig()
    sites = {
        site: {**getattr(default, site), "BLOCK_SIZE_K": 8}
        for site in ("gate_up_b", "down_b")
    }
    plan = {"finalize_family": "shared_token_delta"} if shared else {}
    _load_consumer_plan(monkeypatch, "shared" if shared else "per_expert", plan, sites)


def test_token_grouped_does_not_accept_split_k(monkeypatch):
    sites = {"gate_up_a": {**MoeLoraLaunchConfig().gate_up_a, "SPLIT_K": 2}}
    with pytest.raises(
        ValueError, match="token_grouped does not provide split-K scratch"
    ):
        _load_consumer_plan(
            monkeypatch, "shared", {"gate_up_a_family": "token_grouped"}, sites
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
