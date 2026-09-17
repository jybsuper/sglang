from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.srt.lora.dense import plan as plan_module
from sglang.srt.lora.dense.plan import (
    AFamily,
    BFamily,
    DenseLoraKind,
    DensePlan,
    DensePlanTable,
    Overlap,
)
from sglang.srt.lora.utils import Phase
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.fixture
def plan_config(tmp_path, monkeypatch):
    monkeypatch.setattr(plan_module, "_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("SGLANG_LORA_DENSE_CONFIG_DIR", str(tmp_path))
    plan_module.load_plans.cache_clear()
    try:
        yield tmp_path
    finally:
        plan_module.load_plans.cache_clear()


@pytest.fixture
def shipped_plans(monkeypatch):
    monkeypatch.delenv("SGLANG_LORA_DENSE_CONFIG_DIR", raising=False)
    plan_module.load_plans.cache_clear()
    try:
        yield
    finally:
        plan_module.load_plans.cache_clear()


def test_plan_rejects_incompatible_families():
    with pytest.raises(ValueError, match="all_slots"):
        DensePlan(a_family=AFamily.ALL_SLOTS, b_family=BFamily.GROUPED)


def test_plan_rejects_unknown_split_mode():
    with pytest.raises(ValueError, match="SPLIT_MODE"):
        DensePlan(a_tiles={**DensePlan().a_tiles, "SPLIT_K": 4, "SPLIT_MODE": "atomic"})


@pytest.mark.parametrize(
    "site,change",
    [
        ("a_tiles", {"BLOCK_SIZE_K": 3}),
        ("b_tiles", {"num_warps": 3}),
        ("a_tiles", {"num_stages": 0}),
    ],
)
def test_dense_tiles_are_validated_before_launch(site, change):
    with pytest.raises(ValueError):
        DensePlan(**{site: {**getattr(DensePlan(), site), **change}})


@pytest.mark.parametrize("change", ["missing", "unknown"])
def test_dense_tile_keys_are_validated(change):
    tiles = dict(DensePlan().a_tiles)
    if change == "missing":
        tiles.pop("GROUP_SIZE_M")
    else:
        tiles["UNKNOWN_TILE"] = 1
    with pytest.raises(ValueError, match="a_tiles requires"):
        DensePlan(a_tiles=tiles)


@pytest.mark.parametrize(
    "site,key",
    [
        ("a_tiles", "BLOCK_SIZE_N"),
        ("a_tiles", "BLOCK_SIZE_K"),
        ("b_tiles", "BLOCK_SIZE_N"),
    ],
)
def test_grouped_dot_tiles_require_at_least_16(site, key):
    tiles = {**getattr(DensePlan(), site), key: 8}
    with pytest.raises(ValueError, match=key):
        DensePlan(**{site: tiles})
    tiles[key] = 16
    assert getattr(DensePlan(**{site: tiles}), site)[key] == 16


@pytest.mark.parametrize("a_family", [AFamily.PER_ROW, AFamily.ALL_SLOTS])
def test_nondot_tiles_may_be_smaller(a_family):
    plan = DensePlan(
        a_family=a_family,
        b_family=BFamily.PER_ROW,
        a_tiles={**DensePlan().a_tiles, "BLOCK_SIZE_N": 8, "BLOCK_SIZE_K": 8},
        b_tiles={**DensePlan().b_tiles, "BLOCK_SIZE_N": 8, "BLOCK_SIZE_K": 8},
    )
    assert plan.a_tiles["BLOCK_SIZE_N"] == plan.b_tiles["BLOCK_SIZE_K"] == 8
    assert not plan.needs_aligned_route
    grouped_b = DensePlan(b_tiles={**DensePlan().b_tiles, "BLOCK_SIZE_K": 8})
    assert grouped_b.b_tiles["BLOCK_SIZE_K"] == 8
    assert grouped_b.needs_aligned_route


def test_dense_table_validates_even_unmatched_rows(plan_config):
    (plan_config / "default.plans.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "name": "unmatched",
                        "min_in_features": 999999,
                        "spec": {"a_family": "all_slots"},
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="unmatched.*all_slots"):
        DensePlanTable("default", max_rank=64)


def test_defaults_without_a_table(plan_config):
    assert plan_module.load_plans("sm90") is None
    table = DensePlanTable("sm90", max_rank=64)
    decode = table.plan_for(DenseLoraKind.LINEAR, Phase.DECODE, 8)
    prefill = table.plan_for(DenseLoraKind.LINEAR, Phase.PREFILL, 2048)
    assert decode.a_tiles["SPLIT_K"] == 4
    assert decode.block_size == 16 and prefill.block_size == 64
    assert prefill.overlap is Overlap.NONE
    assert prefill.b_tiles["BLOCK_SIZE_N"] == 64


def test_table_rows_match_in_order(plan_config):
    rows = {
        "rows": [
            {
                "name": "tiny decode lm_head",
                "phase": "decode",
                "kinds": ["lm_head"],
                "max_tokens": 16,
                "spec": {
                    "a_family": "per_row",
                    "b_family": "per_row",
                    "overlap": "ab_delta",
                },
            },
            {
                "name": "small ranks decode",
                "phase": "decode",
                "max_tokens": 64,
                "max_rank": 32,
                "spec": {"a_family": "all_slots", "b_family": "per_row"},
            },
            {
                "name": "prefill",
                "phase": "prefill",
                "spec": {
                    "block_size": 32,
                    "a_tiles": {**DensePlan().a_tiles, "BLOCK_SIZE_N": 32},
                },
            },
        ],
    }
    (plan_config / "default.plans.json").write_text(json.dumps(rows))
    table = DensePlanTable("default", max_rank=32)
    # Both decode rows match at 16 tokens; the narrower first row wins.
    assert (
        table.plan_for(DenseLoraKind.LM_HEAD, Phase.DECODE, 16).a_family
        is AFamily.PER_ROW
    )
    assert (
        table.plan_for(DenseLoraKind.LM_HEAD, Phase.DECODE, 17).a_family
        is AFamily.ALL_SLOTS
    )
    assert (
        table.plan_for(DenseLoraKind.LINEAR, Phase.DECODE, 16).a_family
        is AFamily.ALL_SLOTS
    )
    assert (
        table.plan_for(DenseLoraKind.LINEAR, Phase.DECODE, 64).a_family
        is AFamily.ALL_SLOTS
    )
    assert (
        table.plan_for(DenseLoraKind.LINEAR, Phase.DECODE, 65).a_family
        is AFamily.GROUPED
    )
    prefill = table.plan_for(DenseLoraKind.LINEAR, Phase.PREFILL, 4096)
    assert prefill.block_size == 32 and prefill.a_tiles["BLOCK_SIZE_N"] == 32
    fallback = DensePlanTable("default", max_rank=33).plan_for(
        DenseLoraKind.LINEAR, Phase.DECODE, 16
    )
    assert fallback.a_family is AFamily.GROUPED and fallback.a_tiles["SPLIT_K"] == 4


def test_geometry_rows_do_not_leak_to_neighboring_sites(plan_config):
    rows = {
        "rows": [
            {
                "name": "bounded geometry",
                "phase": "decode",
                "max_tokens": 8,
                "min_in_features": 256,
                "max_in_features": 512,
                "min_out_features": 128,
                "max_out_features": 256,
                "spec": {"block_size": 32},
            }
        ]
    }
    (plan_config / "default.plans.json").write_text(json.dumps(rows))
    table = DensePlanTable("default", max_rank=64)
    for k, n, expected in (
        (256, 128, 32),
        (512, 256, 32),
        (255, 128, 16),
        (513, 128, 16),
        (256, 127, 16),
        (256, 257, 16),
    ):
        # Reusing one table also catches cache keys that omit K or N.
        assert (
            table.plan_for(DenseLoraKind.LINEAR, Phase.DECODE, 8, k, n).block_size
            == expected
        )


@pytest.mark.parametrize("architecture", ["default", "sm90", "sm100"])
@pytest.mark.parametrize("phase", [Phase.DECODE, Phase.PREFILL])
@pytest.mark.parametrize("kind", [DenseLoraKind.EMBEDDING, DenseLoraKind.LM_HEAD])
def test_shipped_vocab_plans_keep_the_supported_execution_contract(
    shipped_plans, architecture, phase, kind
):
    for rank, tokens, k, n in ((8, 1, 256, 131), (128, 1024, 2048, 65536)):
        table = DensePlanTable(architecture, max_rank=rank)
        plan = table.plan_for(kind, phase, tokens, k, n)
        assert plan.a_family is AFamily.GROUPED
        assert plan.b_family is BFamily.GROUPED
        assert plan.overlap is Overlap.NONE
    assert plan == table.plan_for(kind, phase, tokens, k, 0)
    assert plan == table.plan_for(kind, phase, tokens, 0, n)


def test_table_rejects_all_slots_rows_that_name_sink_down():
    row = {
        "name": "decode.sink_down",
        "kinds": ["sink_down"],
        "spec": {"a_family": "all_slots", "b_family": "per_row"},
    }
    with pytest.raises(ValueError, match="sink_down"):
        plan_module._PlanRowModel.model_validate(row)
    plan_module._PlanRowModel.model_validate({**row, "kinds": ["linear"]})
    plan_module._PlanRowModel.model_validate({**row, "spec": {"a_family": "grouped"}})


def test_windowed_sites_skip_all_slots_rows(plan_config):
    rows = {
        "rows": [
            {
                "name": "decode.tokens_le16",
                "phase": "decode",
                "max_tokens": 16,
                "spec": {
                    "a_family": "all_slots",
                    "b_family": "per_row",
                    "overlap": "a",
                },
            },
            {
                "name": "decode",
                "phase": "decode",
                "spec": {"block_size": 32, "overlap": "ab_delta"},
            },
        ],
    }
    (plan_config / "default.plans.json").write_text(json.dumps(rows))
    table = DensePlanTable("default", max_rank=64)
    linear = table.plan_for(DenseLoraKind.LINEAR, Phase.DECODE, 16)
    sink = table.plan_for(DenseLoraKind.SINK_DOWN, Phase.DECODE, 16)
    assert linear.a_family is AFamily.ALL_SLOTS and linear.overlap is Overlap.A
    assert sink.a_family is AFamily.GROUPED
    assert sink.block_size == 32 and sink.overlap is Overlap.AB_DELTA
    assert table.plan_for(DenseLoraKind.LINEAR, Phase.DECODE, 17) == sink


def test_shipped_h200_table_serial_in_proj_exceptions(shipped_plans):
    tables = {rank: DensePlanTable("sm90", max_rank=rank) for rank in (64, 65)}
    for rank, tokens, k, n, block_size in (
        (64, 2048, 2048, 12288, 16),
        (64, 2049, 2048, 12288, 64),
        (65, 2048, 2048, 12288, 32),
        (64, 2048, 2047, 12288, 64),
        (64, 2048, 2049, 12288, 64),
        (64, 2048, 2048, 12287, 64),
        (64, 2048, 2048, 12289, 64),
    ):
        plan = tables[rank].plan_for(DenseLoraKind.LINEAR, Phase.PREFILL, tokens, k, n)
        assert plan.block_size == block_size
        assert plan.overlap is Overlap.NONE


ROOT = Path(__file__).resolve().parents[4] / "python/sglang/srt"


def _tree(path):
    return ast.parse((ROOT / path).read_text())


def _runner_constructor(cuda):
    tree = _tree("lora/dense/runner.py")
    runner = next(
        node for node in tree.body if getattr(node, "name", None) == "DenseLoraRunner"
    )
    init = next(
        node for node in runner.body if getattr(node, "name", None) == "__init__"
    )
    mapping = _tree("lora/utils.py")
    capability = next(
        node
        for node in mapping.body
        if getattr(node, "name", None) == "architecture_for_capability"
    )
    namespace = {"torch": SimpleNamespace(cuda=cuda), "load_plans": Mock()}
    module = ast.Module(
        body=[*ast.parse("from __future__ import annotations").body, capability, init],
        type_ignores=[],
    )
    exec(compile(module, "runner_constructor", "exec"), namespace)
    return namespace["__init__"], namespace["load_plans"]


@pytest.mark.parametrize(
    "major,override,expected",
    [
        (8, None, "default"),
        (9, None, "sm90"),
        (10, None, "sm100"),
        (9, "sm100", "sm100"),
    ],
)
def test_runner_resolves_architecture_from_the_device_unless_overridden(
    major, override, expected
):
    cuda = SimpleNamespace(get_device_capability=Mock(return_value=(major, 0)))
    init, load_plans = _runner_constructor(cuda)
    device = SimpleNamespace(type="cuda", index=0)
    runner = SimpleNamespace(reset=Mock())
    init(runner, object(), max_loras=4, device=device, architecture=override)
    if override is None:
        cuda.get_device_capability.assert_called_once_with(device)
    else:
        cuda.get_device_capability.assert_not_called()
    assert runner.architecture == expected
    # The table (and any override) is validated when the runner is built.
    load_plans.assert_called_once_with(runner.architecture)


@pytest.mark.parametrize("override", [None, "sm90"])
def test_cpu_runner_does_not_query_cuda_metadata(override):
    cuda = SimpleNamespace(get_device_capability=Mock())
    init, _ = _runner_constructor(cuda)
    runner = SimpleNamespace(reset=Mock())
    init(
        runner,
        object(),
        max_loras=4,
        device=SimpleNamespace(type="cpu"),
        architecture=override,
    )
    cuda.get_device_capability.assert_not_called()
    assert runner.architecture == ("default" if override is None else override)


def _geometry_host_function(name, **scope):
    path = ROOT.parent / "kernels/ops/lora/common/lora_b.py"
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    module = ast.Module(
        body=[*ast.parse("from __future__ import annotations").body, node],
        type_ignores=[],
    )
    exec(compile(module, str(path), "exec"), scope)
    return scope[name]


def _geometry_host_namespace():
    from contextlib import nullcontext
    from typing import NamedTuple

    import torch

    path = ROOT.parent / "kernels/ops/lora/common/lora_b.py"
    names = {
        "SliceGeometry",
        "_geometry_inputs",
        "slice_geometry",
        "_build_geometry",
    }
    module = ast.Module(
        body=[
            *ast.parse("from __future__ import annotations").body,
            *(
                node
                for node in ast.parse(path.read_text()).body
                if getattr(node, "name", None) in names
            ),
        ],
        type_ignores=[],
    )
    cuda = SimpleNamespace(
        device=Mock(side_effect=lambda _: nullcontext()),
        current_device=Mock(return_value=0),
        is_current_stream_capturing=Mock(return_value=False),
    )
    scope = dict(
        NamedTuple=NamedTuple,
        torch=SimpleNamespace(
            cuda=cuda,
            device=torch.device,
            int32=torch.int32,
            tensor=Mock(
                side_effect=lambda values, **kw: torch.tensor(values, dtype=kw["dtype"])
            ),
        ),
        triton=SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d),
        _GEOMETRY={},
        _NOT_CUDA="the expand kernels read their offset tables from a CUDA device",
    )
    exec(compile(module, str(path), "exec"), scope)
    return scope


@pytest.mark.parametrize(
    "offsets,tile,columns",
    [
        ((), 64, None),
        ((1, 33), 64, None),
        ((0, 32, 16), 64, None),
        ((0, 2**31), 64, None),
        ((0, 32), 0, None),
        ((0, 32), 24, None),
        ((0, 32, 64), 64, (0,)),
        ((0, 32, 64), 64, (0, -1)),
        ((0, 32, 64), 64, (0, 2**31)),
    ],
)
def test_geometry_rejects_malformed_metadata(offsets, tile, columns):
    scope = _geometry_host_namespace()
    with pytest.raises(ValueError):
        scope["slice_geometry"](
            offsets, tile, scope["torch"].device("cuda", 0), columns
        )
    assert scope["_GEOMETRY"] == {}
    scope["torch"].tensor.assert_not_called()
    scope["torch"].cuda.device.assert_not_called()


@pytest.mark.parametrize(
    "offsets,columns",
    [
        ((0, 96, 192), None),
        ((0, 96, 192), (0, 128)),
        ((0, 96, 192, 288), (192, 0, 96)),
        ((0, 2**31 - 1), (2**31 - 1,)),
    ],
)
def test_geometry_preserves_valid_prefix_and_destination_order(offsets, columns):
    assert _geometry_host_function("_geometry_inputs")(offsets, 64, columns) == (
        offsets,
        offsets[:-1] if columns is None else columns,
    )


@pytest.mark.parametrize(
    "offsets,tile,columns",
    [
        ((False, 32, 64), 64, (0, 32)),
        ((0.0, 32, 64), 64, (0, 32)),
        ((0, 32, 64), 64, (0.0, 32)),
        ((0, 32, 64), 64.0, (0, 32)),
    ],
)
def test_geometry_warm_cache_does_not_alias_noninteger_metadata(offsets, tile, columns):
    scope = _geometry_host_namespace()
    entry, torch = scope["slice_geometry"], scope["torch"]
    device = torch.device("cuda", 0)
    valid = entry((0, 32, 64), 64, device, (0, 32))
    with pytest.raises(ValueError):
        entry(offsets, tile, device, columns)
    assert entry((0, 32, 64), 64, device) is valid
    assert len(scope["_GEOMETRY"]) == 1
    assert torch.tensor.call_count == 1


def test_geometry_cache_normalizes_columns_and_device_without_allocating_in_capture():
    scope = _geometry_host_namespace()
    entry, torch = scope["slice_geometry"], scope["torch"]
    rows = (0, 96, 192)
    first = entry(rows, 64, torch.device("cuda"))
    assert entry(list(rows), 64, torch.device("cuda", 0), [0, 96]) is first
    assert first.slice_offsets.tolist() == list(rows)
    assert first.out_offsets.tolist() == [0, 96]
    assert (first.num_column_tiles, first.uniform_width, first.full_tiles) == (
        4,
        96,
        False,
    )
    assert len(scope["_GEOMETRY"]) == 1
    torch.cuda.is_current_stream_capturing.return_value = True
    assert entry(rows, 64, torch.device("cuda", 0)) is first
    with pytest.raises(RuntimeError, match="CUDA graph capture"):
        entry((0, 96, 256), 64, torch.device("cuda", 0))
    assert len(scope["_GEOMETRY"]) == 1
    assert torch.tensor.call_count == 1
    torch.cuda.is_current_stream_capturing.return_value = False
    torch.cuda.current_device.return_value = 1
    second = entry(rows, 64, torch.device("cuda"))
    assert second is not first
    assert entry(rows, 64, torch.device("cuda", 1)) is second
    assert entry(rows, 64, torch.device("cuda", 0)) is first
    with pytest.raises(ValueError, match="CUDA"):
        entry(rows, 64, torch.device("cpu"))


@pytest.mark.parametrize("name", ["grouped_lora_b", "per_row_lora_b"])
def test_expand_rejects_geometry_bound_to_another_column_tile(name):
    launch = _geometry_host_function(name)
    args = (None, None, None, SimpleNamespace(num_rows=0))
    kwargs = dict(
        geometry=SimpleNamespace(block_size_n=64),
        config={"BLOCK_SIZE_N": 32},
        add_inplace=False,
        zero_sentinel=False,
    )
    with pytest.raises(ValueError, match="BLOCK_SIZE_N must match"):
        launch(*args, **kwargs)
    kwargs["config"] = {"BLOCK_SIZE_N": 64}
    launch(*args, **kwargs)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
