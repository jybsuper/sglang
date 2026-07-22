from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from benchmark.kernels.lora_moe import bench_local as local


def _fixture(n=1024, rank=64, slices=2):
    return SimpleNamespace(
        case=SimpleNamespace(t_local=32, adapters=SimpleNamespace(rank=rank)),
        hidden_states=SimpleNamespace(dtype=torch.bfloat16),
        lora_b=SimpleNamespace(shape=(8, 256, n, rank)),
        topk_ids=SimpleNamespace(numel=lambda: 256),
        num_slices=slices,
    )


def test_benchmark_server_args_initialized_when_absent(monkeypatch):
    sentinel = object()
    published = []

    def missing():
        raise ValueError(local._MISSING_SERVER_ARGS_ERROR)

    modules = {
        "sglang.srt.runtime_context": SimpleNamespace(
            get_server_args=missing,
            get_context=lambda: SimpleNamespace(set_server_args=published.append),
        ),
        "sglang.srt.server_args": SimpleNamespace(
            ServerArgs=lambda *, model_path: (
                sentinel if model_path == "dummy" else None
            )
        ),
    }
    monkeypatch.setattr(local, "import_module", lambda name: modules[name])

    local._ensure_benchmark_server_args()

    assert published == [sentinel]


def test_benchmark_server_args_preserves_existing_context(monkeypatch):
    sentinel = object()
    imports = []
    runtime_context = SimpleNamespace(get_server_args=lambda: sentinel)

    def import_one(name):
        imports.append(name)
        assert name == "sglang.srt.runtime_context"
        return runtime_context

    monkeypatch.setattr(local, "import_module", import_one)

    local._ensure_benchmark_server_args()

    assert imports == ["sglang.srt.runtime_context"]


def test_b_config_cli_default_and_explicit_fields():
    default = local.parse_args([])
    synthetic = local.parse_args(["--b-input-source", "synthetic"])
    fields = (
        "b_block_m",
        "b_block_n",
        "b_block_k",
        "b_group_size_m",
        "b_num_warps",
        "b_num_stages",
    )
    assert (
        default.b_config_selector,
        default.b_input_source,
        default.skip_check,
    ) == ("logical-t", "production-a", False)
    assert synthetic.b_input_source == "synthetic"
    assert tuple(getattr(default, field) for field in fields) == (64, 64, 64, 1, 4, 4)


@pytest.mark.parametrize(
    ("selector", "lookup_m", "held"),
    (("logical-t", 32, False), ("flat-tk", 256, True), ("explicit", None, True)),
)
def test_b_config_selectors(monkeypatch, selector, lookup_m, held):
    calls = []
    monkeypatch.setattr(
        local,
        "_resolve_production_b_config",
        lambda shape, dtype, m: calls.append(m) or {"BLOCK_SIZE_M": m},
    )
    selected = local._select_b_config(
        _fixture(), selector, local.ExplicitBConfig(block_m=128)
    )
    assert (selected.lookup_m, selected.held_override is not None) == (lookup_m, held)
    assert calls == ([] if lookup_m is None else [lookup_m])

    def fail(*_):
        raise ValueError("no local table")

    monkeypatch.setattr(local, "_resolve_production_b_config", fail)
    selected = local._select_b_config(
        _fixture(rank=32), "flat-tk", local.ExplicitBConfig()
    )
    assert selected.resolution_status == "local_fallback_after_value_error"
    assert selected.resolution_error == "ValueError: no local table"
    assert selected.resolved_config == selected.held_override
    assert selected.resolved_config["BLOCK_SIZE_K"] == 32


def test_missing_server_context_is_not_treated_as_config_fallback(monkeypatch):
    def fail(*_):
        raise ValueError(local._MISSING_SERVER_ARGS_ERROR)

    monkeypatch.setattr(local, "_resolve_production_b_config", fail)
    with pytest.raises(ValueError, match="Global server args is not set yet"):
        local._select_b_config(_fixture(), "logical-t", local.ExplicitBConfig())


def test_effective_config_records_direct_only_ignored_fields():
    config = {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 4,
        "num_warps": 8,
        "num_stages": 3,
    }
    midpoint = local._effective_b_config(
        _fixture(n=96, rank=48), direct=True, resolved=config
    )
    fixed = local._effective_b_config(
        _fixture(n=128, slices=1), direct=True, resolved=config
    )
    assert (midpoint["BLOCK_SIZE_N"], midpoint["BLOCK_SIZE_R"]) == (16, 64)
    assert midpoint["resolved_fields_not_consumed"] == ["BLOCK_SIZE_K", "num_stages"]
    assert fixed["resolved_fields_not_consumed"][-1] == "BLOCK_SIZE_N"
    assert local._effective_b_config(_fixture(), direct=False, resolved=config) == {
        "kernel_family": "generic_fused_moe",
        **config,
    }


def test_effective_generic_config_materializes_triton_launch_defaults():
    resolved = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }

    assert local._effective_b_config(_fixture(), direct=False, resolved=resolved) == {
        "kernel_family": "generic_fused_moe",
        **resolved,
        "num_warps": 4,
        "num_stages": 3,
    }


def test_reference_and_check_never_launch_the_opposite_b_family(monkeypatch):
    calls = []

    class Fixture:
        case = SimpleNamespace(adapters=SimpleNamespace(rank=64))
        routing_cache = {}
        output, intermediate = torch.tensor([3.0]), torch.tensor([2.0])

        def invoke(self, stage, *, direct):
            calls.append((stage, direct))

        def reset_output(self):
            self.output.fill_(3.0)

    monkeypatch.setattr(local.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(local.torch.testing, "assert_close", lambda *_, **__: None)
    fixture = Fixture()
    reference = local._production_config_reference(
        fixture, target="gate_b", variant="direct"
    )
    launch = lambda: fixture.invoke("expand", direct=True)
    op = local.PreparedOp(fixture, "gate_b", "K0", True, launch, None)
    local._check_operator(op, reference)
    assert calls == [("routing", True), ("shrink", True)] + [("expand", True)] * 3


def test_generic_gate_check_uses_direct_oracle_and_tight_tolerance(monkeypatch):
    calls = []
    tolerances = []

    class Fixture:
        case = SimpleNamespace(adapters=SimpleNamespace(rank=64))
        num_slices = 2
        routing_cache = {}
        output, intermediate = torch.tensor([3.0]), torch.tensor([2.0])

        def invoke(self, stage, *, direct):
            calls.append((stage, direct))

        def reset_output(self):
            self.output.fill_(3.0)

    monkeypatch.setattr(local.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        local.torch.testing,
        "assert_close",
        lambda *_, **kwargs: tolerances.append((kwargs["rtol"], kwargs["atol"])),
    )
    fixture = Fixture()
    reference = local._production_config_reference(
        fixture, target="gate_b", variant="generic"
    )
    op = local.PreparedOp(
        fixture,
        "gate_b",
        "K0",
        False,
        lambda: fixture.invoke("expand", direct=False),
        None,
    )
    local._check_operator(op, reference)

    assert calls == [
        ("routing", True),
        ("shrink", True),
        ("expand", True),
        ("expand", False),
        ("expand", False),
    ]
    assert tolerances == [(3e-2, 2e-4), (3e-2, 2e-4)]


@pytest.mark.parametrize("target", ["gate_b", "down_b"])
@pytest.mark.parametrize("variant", ["direct", "generic"])
def test_synthetic_b_input_skips_a_and_uses_opposite_family_oracle(
    monkeypatch, target, variant
):
    calls = []

    class Fixture:
        case = SimpleNamespace(adapters=SimpleNamespace(rank=128))
        num_slices = 2
        is_down = target == "down_b"
        routing_cache = {}
        output, intermediate = torch.tensor([3.0]), torch.tensor([2.0])

        def invoke(self, stage, *, direct):
            calls.append((stage, direct))

        def reset_output(self):
            self.output.fill_(3.0)

    monkeypatch.setattr(local.torch.cuda, "synchronize", lambda: None)
    fixture = Fixture()
    reference = local._production_config_reference(
        fixture,
        target=target,
        variant=variant,
        b_input_source="synthetic",
    )
    op = local._build_op(
        fixture,
        target=target,
        variant=variant,
        scope="K0",
        b_input_source="synthetic",
    )

    timed_direct = variant == "direct"
    assert reference is not None
    assert calls == [
        ("routing", not timed_direct),
        ("expand", not timed_direct),
        ("routing", timed_direct),
    ]
    op.launch()
    assert calls[-1] == ("expand", timed_direct)
    assert all(stage != "shrink" for stage, _ in calls)


def test_synthetic_b_input_fill_is_deterministic_and_validated():
    first = torch.empty(2, 3, 8)
    second = torch.empty_like(first)
    local._fill_synthetic_b_intermediate(first)
    local._fill_synthetic_b_intermediate(second)
    assert torch.equal(first, second)
    assert not torch.equal(first[..., :4], first[..., 4:])

    for target in ("gate_b", "down_b"):
        local._validate_b_input_source(target, "synthetic")
    with pytest.raises(ValueError, match="only valid for gate_b or down_b"):
        local._validate_b_input_source("gate_ab", "synthetic")


def test_o0_b_clears_prewarm_before_rebuilding_b(monkeypatch):
    seen = []

    class Fixture:
        case = SimpleNamespace(adapters=SimpleNamespace(rank=64))
        is_down, routing_cache = False, {}

        def invoke(self, stage, *, direct):
            if stage == "routing":
                self.routing_cache["prewarm"] = 1
            elif stage == "expand":
                seen.append(tuple(self.routing_cache))
                self.routing_cache["fresh_b"] = 1

    monkeypatch.setattr(local.torch.cuda, "synchronize", lambda: None)
    fixture = Fixture()
    op = local._build_op(fixture, target="gate_b", variant="generic", scope="O0")
    fixture.routing_cache["stale"] = 1
    op.launch()
    assert seen == [()]
    assert tuple(fixture.routing_cache) == ("fresh_b",)


def test_normal_exit_restores_unsafe_override():
    state = ["production"]

    @contextmanager
    def unsafe():
        state[0] = "experimental"
        yield
        state[0] = "production"

    with pytest.raises(RuntimeError):
        with local._exit_context_normally(unsafe()):
            raise RuntimeError
    assert state == ["production"]
