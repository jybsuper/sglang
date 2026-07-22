import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import benchmark.kernels.lora_moe.bench_indexed_shrink as indexed_shrink
import benchmark.kernels.lora_moe.bench_moe_pipeline as moe_pipeline
from benchmark.kernels.lora_moe.bench_moe_pipeline import (
    BScheduleOverrides,
    _b_schedule_override,
    _capture_production_c0_reference,
    _exit_context_normally,
    _indexed_a_override,
    _parse_b_config,
    _pipeline_two_stream_metadata,
    _resolve_c1_overlap,
    _resolve_indexed_a_configs,
    parse_args,
)


def _load_moe_runner_module():
    path = (
        Path(__file__).resolve().parents[3]
        / "python/sglang/srt/lora/sgl_lora/moe_lora_runner.py"
    )
    spec = importlib.util.spec_from_file_location("_test_moe_lora_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_indexed_a_cli_defaults_to_production_and_auto_configs():
    args = parse_args([])
    assert args.a_provider == "production"
    assert args.indexed_gate_config == "auto"
    assert args.indexed_down_config == "auto"
    assert args.c1_overlap_policy == "production_auto"
    assert args.gate_b_variant == "production"
    assert args.down_b_variant == "production"
    assert args.gate_b_config is None
    assert args.down_b_config is None

    forced = parse_args(["--c1-overlap-policy", "force"])
    assert forced.c1_overlap_policy == "force"


def test_b_schedule_cli_and_config_parser():
    args = parse_args(
        [
            "--gate-b-variant",
            "direct",
            "--down-b-variant",
            "generic",
            "--gate-b-config",
            "16,128,64,1,4,1",
            "--down-b-config",
            "16,128,32,8,4,3",
        ]
    )
    assert args.gate_b_variant == "direct"
    assert args.down_b_variant == "generic"
    assert _parse_b_config(args.gate_b_config) == {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 1,
    }
    assert _parse_b_config(args.down_b_config)["GROUP_SIZE_M"] == 8
    assert _parse_b_config(None) is None

    for malformed in ("16,32", "16,32,64,1,4,0", "16,32,x,1,4,3"):
        with pytest.raises(ValueError, match="B config"):
            _parse_b_config(malformed)


def test_b_schedule_metadata_distinguishes_routing_and_shared_outer_fallback():
    family_only = BScheduleOverrides(gate_variant="direct")
    metadata = family_only.metadata(production_variant="generic", shared_outer_b=True)
    assert not family_only.routing_config_overridden
    assert not metadata["routing_config_overridden"]
    assert metadata["gate"]["requested_variant"] == "direct"
    assert metadata["gate"]["effective_variant"] == "generic"
    assert metadata["down"]["effective_variant"] == "generic"


def test_production_two_stream_auto_policy_boundary():
    runner = _load_moe_runner_module()
    assert runner.LORA_TWO_STREAM_AUTO_MAX_TOKENS == 256
    assert runner.resolve_lora_two_stream_auto(requested=True, num_tokens=256)
    assert not runner.resolve_lora_two_stream_auto(requested=True, num_tokens=257)
    assert not runner.resolve_lora_two_stream_auto(requested=False, num_tokens=1)


def test_c1_force_policy_and_metadata_record_the_resolved_execution(monkeypatch):
    runner = _load_moe_runner_module()
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.lora.sgl_lora.moe_lora_runner",
        runner,
    )
    case = SimpleNamespace(t_local=257)
    auto_fixture = SimpleNamespace(
        case=case,
        c1_overlap_policy="production_auto",
        c1_two_stream_enabled=_resolve_c1_overlap("production_auto", 257),
    )
    forced_fixture = SimpleNamespace(
        case=case,
        c1_overlap_policy="force",
        c1_two_stream_enabled=_resolve_c1_overlap("force", 257),
    )

    auto = _pipeline_two_stream_metadata(auto_fixture, "C1")
    forced = _pipeline_two_stream_metadata(forced_fixture, "C1")

    assert not auto["effective"]
    assert auto["fallback_reason"] == "production_auto_token_threshold"
    assert not auto["benchmark_force_requested"]
    assert forced["effective"]
    assert forced["benchmark_force_requested"]
    assert forced["benchmark_force_changed_decision"]
    assert forced["production_auto_enabled"] is False
    assert forced["production_default_unchanged"] is True
    assert forced["overlap_scope"].endswith("down_lora_serial")


@pytest.mark.parametrize(
    ("device", "gate", "down"),
    (
        ("h200", "bn32-bk128-w4", "bn16-bk128-w8"),
        ("gb300", "bn32-bk128-w8", "bn8-bk128-w8"),
    ),
)
def test_indexed_a_auto_configs_use_the_cold_cache_shortlist(
    device: str, gate: str, down: str
):
    configs = _resolve_indexed_a_configs(device, "auto", "auto")
    assert configs.gate.key == gate
    assert configs.down.key == down
    assert configs.gate_source == "auto_cold_cache_shortlist"
    assert configs.down_source == "auto_cold_cache_shortlist"


def _install_fake_triton_errors(monkeypatch):
    class OutOfResources(RuntimeError):
        pass

    triton = ModuleType("triton")
    runtime = ModuleType("triton.runtime")
    errors = ModuleType("triton.runtime.errors")
    errors.OutOfResources = OutOfResources
    runtime.errors = errors
    triton.runtime = runtime
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.runtime", runtime)
    monkeypatch.setitem(sys.modules, "triton.runtime.errors", errors)
    return OutOfResources


def test_production_c0_reference_records_available(monkeypatch):
    _install_fake_triton_errors(monkeypatch)
    reference = object()
    calls = []

    def fake_run_checked(fixture, pipeline):
        calls.append((fixture, pipeline))
        return reference

    fixture = object()
    monkeypatch.setattr(moe_pipeline, "_run_checked", fake_run_checked)

    actual, status = _capture_production_c0_reference(fixture)

    assert actual is reference
    assert status == {"status": "available"}
    assert calls == [(fixture, "C0")]


def test_production_c0_reference_records_triton_resource_limit(monkeypatch):
    out_of_resources = _install_fake_triton_errors(monkeypatch)
    error = out_of_resources("shared memory limit")

    def fake_run_checked(fixture, pipeline):
        raise error

    monkeypatch.setattr(moe_pipeline, "_run_checked", fake_run_checked)

    reference, status = _capture_production_c0_reference(object())

    assert reference is None
    assert status == {
        "status": "unsupported",
        "error_type": "OutOfResources",
        "error": "shared memory limit",
    }


def test_production_c0_reference_reraises_unexpected_errors(monkeypatch):
    _install_fake_triton_errors(monkeypatch)
    error = RuntimeError("unexpected failure")

    def fake_run_checked(fixture, pipeline):
        raise error

    monkeypatch.setattr(moe_pipeline, "_run_checked", fake_run_checked)

    with pytest.raises(RuntimeError) as exc_info:
        _capture_production_c0_reference(object())
    assert exc_info.value is error


def _install_fake_virtual_experts(monkeypatch, production_ab):
    modules = {}
    names = (
        "sglang",
        "sglang.srt",
        "sglang.srt.lora",
        "sglang.srt.lora.sgl_lora",
        "sglang.srt.lora.sgl_lora.triton_ops",
    )
    for name in names:
        module = ModuleType(name)
        module.__path__ = []
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            setattr(modules[parent_name], child_name, module)

    virtual_experts = ModuleType("sglang.srt.lora.sgl_lora.triton_ops.virtual_experts")
    virtual_experts.merged_experts_fused_moe_lora_add = production_ab
    modules[names[-1]].virtual_experts = virtual_experts
    monkeypatch.setitem(sys.modules, virtual_experts.__name__, virtual_experts)
    return virtual_experts


def test_indexed_a_wrapper_replaces_only_all_stage_and_reuses_production_b(monkeypatch):
    indexed_calls = []
    production_calls = []

    def fake_indexed(*args, **kwargs):
        indexed_calls.append((args, kwargs))

    def fake_production(*args, **kwargs):
        production_calls.append((args, kwargs))
        return kwargs.get("stage", "all")

    monkeypatch.setattr(indexed_shrink, "invoke_indexed_lora_a", fake_indexed)
    virtual_experts = _install_fake_virtual_experts(monkeypatch, fake_production)
    configs = _resolve_indexed_a_configs("h200", "auto", "auto")
    down_intermediate = object()
    fixture = SimpleNamespace(indexed_down_intermediate=down_intermediate)
    gate_intermediate = object()
    common = {
        "output": object(),
        "hidden_states": object(),
        "lora_a": object(),
        "lora_b": object(),
        "topk_ids": object(),
        "topk_weights": object(),
        "token_lora_mapping": object(),
        "mul_routed_weight": False,
        "experts_shared_outer_loras_a": False,
        "experts_shared_outer_loras_b": False,
        "local_expert_offset": 0,
        "num_output_slices": 2,
        "intermediate_buffer": gate_intermediate,
    }

    with _indexed_a_override(fixture, configs):
        wrapped = virtual_experts.merged_experts_fused_moe_lora_add
        with pytest.raises(TypeError, match="requires keyword arguments"):
            wrapped(object())
        assert wrapped(**{**common, "stage": "routing"}) == "routing"
        for stage in ("shrink", "expand", "unexpected"):
            with pytest.raises(ValueError, match="supports only"):
                wrapped(**{**common, "stage": stage})
        assert wrapped(**common) == "expand"
        assert (
            wrapped(
                **{
                    **common,
                    "num_output_slices": 1,
                    "intermediate_buffer": None,
                }
            )
            == "expand"
        )

    assert virtual_experts.merged_experts_fused_moe_lora_add is fake_production
    assert [call[1].get("stage", "all") for call in production_calls] == [
        "routing",
        "expand",
        "expand",
    ]
    assert [call[0][4] for call in indexed_calls] == [
        gate_intermediate,
        down_intermediate,
    ]
    assert indexed_calls[0][1]["config"].key == "bn32-bk128-w4"
    assert indexed_calls[1][1]["config"].key == "bn16-bk128-w8"


def test_b_schedule_wrapper_selects_gate_and_down_independently(monkeypatch):
    production_calls = []
    active_configs = []

    def fake_production(*args, **kwargs):
        production_calls.append((kwargs, tuple(active_configs)))
        return kwargs["stage"]

    @contextmanager
    def fake_held_config(config):
        active_configs.append(config)
        try:
            yield
        finally:
            active_configs.pop()

    virtual_experts = _install_fake_virtual_experts(monkeypatch, fake_production)
    monkeypatch.setattr(moe_pipeline, "_held_b_config", fake_held_config)
    gate_config = _parse_b_config("16,128,64,1,4,1")
    down_config = _parse_b_config("16,128,32,8,4,3")
    schedules = BScheduleOverrides(
        gate_variant="direct",
        down_variant="generic",
        gate_config=gate_config,
        down_config=down_config,
    )

    with _b_schedule_override(schedules):
        wrapped = virtual_experts.merged_experts_fused_moe_lora_add
        assert wrapped(stage="routing", mul_routed_weight=False) == "routing"
        assert wrapped(stage="expand", mul_routed_weight=True) == "expand"

    assert virtual_experts.merged_experts_fused_moe_lora_add is fake_production
    assert production_calls == [
        (
            {
                "stage": "routing",
                "mul_routed_weight": False,
                "use_direct_expand_add": True,
            },
            (gate_config,),
        ),
        (
            {
                "stage": "expand",
                "mul_routed_weight": True,
                "use_direct_expand_add": False,
            },
            (down_config,),
        ),
    ]


def test_b_schedule_is_nested_beneath_indexed_a(monkeypatch):
    indexed_configs_seen = []
    delegated_calls = []
    active_configs = []

    def fake_indexed(*args, **kwargs):
        indexed_configs_seen.append((kwargs["config"], tuple(active_configs)))

    def fake_production(*args, **kwargs):
        delegated_calls.append((kwargs.copy(), tuple(active_configs)))
        return kwargs["stage"]

    @contextmanager
    def fake_held_config(config):
        active_configs.append(config)
        try:
            yield
        finally:
            active_configs.pop()

    monkeypatch.setattr(indexed_shrink, "invoke_indexed_lora_a", fake_indexed)
    monkeypatch.setattr(moe_pipeline, "_held_b_config", fake_held_config)
    virtual_experts = _install_fake_virtual_experts(monkeypatch, fake_production)
    indexed_configs = _resolve_indexed_a_configs("h200", "auto", "auto")
    fixture = SimpleNamespace(indexed_down_intermediate=object())
    gate_config = _parse_b_config("16,128,64,1,4,1")
    schedules = BScheduleOverrides(
        gate_variant="direct",
        gate_config=gate_config,
    )
    common = {
        "output": object(),
        "hidden_states": object(),
        "lora_a": object(),
        "lora_b": object(),
        "topk_ids": object(),
        "topk_weights": object(),
        "token_lora_mapping": object(),
        "mul_routed_weight": False,
        "experts_shared_outer_loras_a": False,
        "experts_shared_outer_loras_b": False,
        "local_expert_offset": 0,
        "num_output_slices": 2,
        "intermediate_buffer": object(),
    }

    # The entry order matches main(): B first, then indexed A as the outer wrapper.
    with _b_schedule_override(schedules):
        with _indexed_a_override(fixture, indexed_configs):
            wrapped = virtual_experts.merged_experts_fused_moe_lora_add
            assert wrapped(**common) == "expand"
            assert wrapped(**{**common, "stage": "routing"}) == "routing"

    assert virtual_experts.merged_experts_fused_moe_lora_add is fake_production
    assert indexed_configs_seen == [(indexed_configs.gate, ())]
    assert [call[0]["stage"] for call in delegated_calls] == ["expand", "routing"]
    assert all(call[0]["use_direct_expand_add"] for call in delegated_calls)
    assert all(call[1] == (gate_config,) for call in delegated_calls)


def test_exit_context_normally_restores_after_delegated_error():
    events = []

    @contextmanager
    def fragile_manager():
        events.append("enter")
        yield
        events.append("normal_exit")

    with pytest.raises(RuntimeError, match="delegated"):
        with _exit_context_normally(fragile_manager()):
            raise RuntimeError("delegated")

    assert events == ["enter", "normal_exit"]
