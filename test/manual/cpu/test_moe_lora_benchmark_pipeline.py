import sys
from types import ModuleType, SimpleNamespace

import pytest

import benchmark.kernels.lora_moe.bench_indexed_shrink as indexed_shrink
import benchmark.kernels.lora_moe.bench_moe_pipeline as moe_pipeline
from benchmark.kernels.lora_moe.bench_moe_pipeline import (
    _capture_production_c0_reference,
    _indexed_a_override,
    _resolve_indexed_a_configs,
    parse_args,
)


def test_indexed_a_cli_defaults_to_production_and_auto_configs():
    args = parse_args([])
    assert args.a_provider == "production"
    assert args.indexed_gate_config == "auto"
    assert args.indexed_down_config == "auto"


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
