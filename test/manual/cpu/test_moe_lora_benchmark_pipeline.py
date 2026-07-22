import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import benchmark.kernels.lora_moe.bench_indexed_shrink as indexed_shrink
import benchmark.kernels.lora_moe.bench_moe_pipeline as moe_pipeline
from benchmark.kernels.lora_moe.bench_moe_pipeline import (
    _NEUTRAL_BASELINE_SPECS,
    BScheduleOverrides,
    PipelineFixture,
    _b_schedule_override,
    _capture_production_c0_reference,
    _check_provider_lora_delta,
    _exit_context_normally,
    _experimental_trtllm_environment,
    _indexed_a_override,
    _matched_latency_summary,
    _make_routing,
    _parse_b_config,
    _pipeline_two_stream_metadata,
    _resolve_baseline_pipelines,
    _resolve_c1_overlap,
    _resolve_indexed_a_configs,
    _resolve_neutral_baselines,
    _run_length_encode_token_mapping,
    _smoke_case,
    _strict_delta_atol,
    _validate_neutral_case,
    _validate_all_base_sentinel,
    parse_args,
)
from benchmark.kernels.lora_moe.matrix import p0_cases


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
    assert args.neutral_baselines == "none"
    assert args.route_pattern == "lattice_control"
    assert args.route_seed == 0
    assert args.all_base_sgl_c0_sentinel is False

    forced = parse_args(["--c1-overlap-policy", "force"])
    assert forced.c1_overlap_policy == "force"


def test_neutral_baseline_cli_and_pipeline_resolution():
    assert _resolve_neutral_baselines("none") == ()
    assert _resolve_neutral_baselines("legacy_triton") == ("legacy_triton",)
    assert _resolve_neutral_baselines("all") == (
        "legacy_triton",
        "experimental_trtllm",
    )
    with pytest.raises(ValueError, match="unknown neutral baseline"):
        _resolve_neutral_baselines("missing")

    case = p0_cases("h200")[2]
    assert _resolve_baseline_pipelines("legacy_triton", "all", case) == (
        "N0",
        "C0",
    )
    assert _resolve_baseline_pipelines("experimental_trtllm", "all", case) == (
        "N0",
        "C0",
        "C1",
    )
    assert _resolve_baseline_pipelines("legacy_triton", "C1", case) == ()

    args = parse_args(["--neutral-baselines", "all"])
    assert args.neutral_baselines == "all"


def test_route_families_are_reproducible_unique_and_seed_isolated():
    import torch

    case = p0_cases("h200")[2]
    device = torch.device("cpu")
    lattice_a = _make_routing(
        case, device, pattern="lattice_control", route_seed=1
    )
    lattice_b = _make_routing(
        case, device, pattern="lattice_control", route_seed=999
    )
    assert lattice_a[3]["pattern"] == "lattice_control"
    assert lattice_a[3]["seed_effective"] is False
    assert lattice_a[3]["resolved_route_hash"] == lattice_b[3]["resolved_route_hash"]

    uniform_a = _make_routing(
        case,
        device,
        pattern="uniform_iid_without_replacement",
        route_seed=7,
    )
    uniform_b = _make_routing(
        case,
        device,
        pattern="uniform_iid_without_replacement",
        route_seed=8,
    )
    assert uniform_a[3]["sampling"] == "uniform"
    assert uniform_a[3]["resolved_route_hash"] != uniform_b[3]["resolved_route_hash"]
    # Route seeds change only expert IDs: weights and adapter assignment remain exact.
    assert torch.equal(uniform_a[1], uniform_b[1])
    assert torch.equal(uniform_a[2], uniform_b[2])

    skewed = _make_routing(
        case,
        device,
        pattern="skewed_iid_without_replacement",
        route_seed=7,
    )
    assert skewed[3]["sampling"] == "zipf_alpha_1.2"
    for topk_ids, metadata in ((uniform_a[0], uniform_a[3]), (skewed[0], skewed[3])):
        assert all(row.unique().numel() == case.model.top_k for row in topk_ids)
        assert metadata["E_hit"] == topk_ids.unique().numel()


def test_classic_adapter_segments_are_minimal_and_exact():
    import torch

    mapping = torch.tensor([2, 2, 2, -1, -1, 7, 2, 2], dtype=torch.int32)
    indptr, identities = _run_length_encode_token_mapping(mapping)
    assert indptr.tolist() == [0, 3, 5, 6, 8]
    assert identities.tolist() == [2, -1, 7, 2]
    reconstructed = torch.cat(
        [
            identities[i].expand(int(indptr[i + 1] - indptr[i]))
            for i in range(identities.numel())
        ]
    )
    assert torch.equal(reconstructed, mapping)

    single = torch.zeros(32, dtype=torch.int32)
    indptr, identities = _run_length_encode_token_mapping(single)
    assert indptr.tolist() == [0, 32]
    assert identities.tolist() == [0]


def test_all_base_sgl_c0_sentinel_is_narrow_and_opt_in():
    base = p0_cases("h200")[1]
    active = p0_cases("h200")[2]
    production = BScheduleOverrides()
    _validate_all_base_sentinel(
        base,
        enabled=True,
        execution="cuda_graph",
        a_provider="production",
        b_schedules=production,
    )
    with pytest.raises(ValueError, match="all-base P0 case"):
        _validate_all_base_sentinel(
            active,
            enabled=True,
            execution="cuda_graph",
            a_provider="production",
            b_schedules=production,
        )
    with pytest.raises(ValueError, match="requires --execution cuda_graph"):
        _validate_all_base_sentinel(
            base,
            enabled=True,
            execution="eager",
            a_provider="production",
            b_schedules=production,
        )
    assert parse_args(["--all-base-sgl-c0-sentinel"]).all_base_sgl_c0_sentinel


def test_neutral_specs_make_conversion_accounting_explicit():
    legacy = _NEUTRAL_BASELINE_SPECS["legacy_triton"]
    trtllm = _NEUTRAL_BASELINE_SPECS["experimental_trtllm"]
    assert legacy.base_provider == "stock_triton_bf16"
    assert legacy.weight_layout == "canonical_standard_bf16"
    assert "classic_lora_adapter_expert_alignment_for_active_pipeline" in (
        legacy.per_forward_preparation
    )
    assert trtllm.weight_layout == "flashinfer_trtllm_block_major_k_bf16"
    assert trtllm.supported_devices == ("gb300",)
    assert "vendored_experimental_runner_must_match_flashinfer_header_abi" in (
        trtllm.runtime_requirements
    )
    assert "standard_topk_to_trtllm_packed_topk" in (trtllm.per_forward_preparation)


def test_trtllm_control_records_architecture_and_geometry_constraints():
    smoke = _smoke_case("gb300")
    with pytest.raises(ValueError, match="divisible by 128"):
        _validate_neutral_case(smoke, ("experimental_trtllm",))
    _validate_neutral_case(_smoke_case("h200"), ("legacy_triton",))
    with pytest.raises(NotImplementedError, match="SM100"):
        _validate_neutral_case(
            p0_cases("h200")[2], ("experimental_trtllm",)
        )
    _validate_neutral_case(p0_cases("gb300")[2], ("experimental_trtllm",))


def test_experimental_environment_is_scoped(monkeypatch):
    name = "SGLANG_EXPERIMENTAL_LORA_OPTI"
    monkeypatch.delenv(name, raising=False)
    with _experimental_trtllm_environment(True):
        assert moe_pipeline.os.environ[name] == "1"
    assert name not in moe_pipeline.os.environ

    monkeypatch.setenv(name, "custom")
    with _experimental_trtllm_environment(True):
        assert moe_pipeline.os.environ[name] == "1"
    assert moe_pipeline.os.environ[name] == "custom"


def test_matched_latency_summary_never_borrows_an_unmatched_n0():
    missing = _matched_latency_summary({"C0": {"timing": {"p50_us": 120.0}}})
    assert missing == {
        "status": "unavailable_without_timed_provider_n0",
        "provider_matched": True,
    }

    summary = _matched_latency_summary(
        {
            "N0": {"timing": {"p50_us": 100.0}},
            "C0": {"timing": {"p50_us": 125.0}},
            "C1": {"timing": {"p50_us": 110.0}},
        }
    )
    assert summary["n0_p50_us"] == 100.0
    assert summary["active"]["C0"] == {
        "p50_us": 125.0,
        "active_over_n0_p50_us": 25.0,
        "n0_retention_percent": 80.0,
    }
    assert summary["active"]["C1"]["active_over_n0_p50_us"] == 10.0


def test_n0_only_neutral_check_does_not_require_active_fixture(monkeypatch):
    import torch

    calls = []
    fixture = SimpleNamespace(case=p0_cases("gb300")[2])
    monkeypatch.setattr(
        moe_pipeline,
        "_run_checked",
        lambda _fixture, pipeline: calls.append(("sgl", pipeline))
        or torch.zeros(1),
    )
    monkeypatch.setattr(
        moe_pipeline,
        "_run_neutral_checked",
        lambda _fixture, baseline, pipeline: calls.append((baseline, pipeline))
        or torch.zeros(1),
    )
    monkeypatch.setattr(torch.testing, "assert_close", lambda *args, **kwargs: None)

    moe_pipeline._check_neutral_baselines(
        fixture, ("experimental_trtllm",), "N0"
    )
    assert calls == [("sgl", "N0"), ("experimental_trtllm", "N0")]


def test_provider_delta_check_subtracts_each_providers_own_base(monkeypatch):
    import torch

    reference_base = torch.tensor([10.0, 20.0])
    candidate_base = torch.tensor([11.0, 19.0])
    delta = torch.tensor([0.25, -0.5])
    checks = {}
    compared = []
    monkeypatch.setattr(
        torch.testing,
        "assert_close",
        lambda lhs, rhs, **kwargs: compared.append(
            (lhs.tolist(), rhs.tolist(), kwargs)
        ),
    )
    _check_provider_lora_delta(
        checks,
        "matched",
        reference_base + delta,
        reference_base,
        candidate_base + delta,
        candidate_base,
        rtol=0.0,
        atol=0.0,
    )
    assert checks["matched_delta_max_abs_error"] == 0.0
    assert compared == [([0.25, -0.5], [0.25, -0.5], {"rtol": 0.0, "atol": 0.0})]


def test_strict_delta_tolerance_cannot_hide_a_dropped_adapter(monkeypatch):
    import torch

    signal = 0.0035
    assert _strict_delta_atol(signal) <= signal / 10

    base = torch.full((4,), 100.0)
    reference = base + torch.tensor([signal, 0.0, 0.0, 0.0])
    def assert_close(lhs, rhs, *, rtol, atol):
        tolerance = atol + rtol * rhs.abs()
        if not bool(((lhs - rhs).abs() <= tolerance).all()):
            raise AssertionError("not close")

    monkeypatch.setattr(torch.testing, "assert_close", assert_close)
    checks = {}
    with pytest.raises(AssertionError):
        moe_pipeline._check_lora_delta(
            checks,
            "dropped",
            reference,
            base,
            base,
        )


def test_strict_delta_check_rejects_an_all_zero_reference_signal():
    import torch

    base = torch.ones(4)
    with pytest.raises(AssertionError, match="all-zero delta"):
        moe_pipeline._check_lora_delta({}, "zero", base, base, base)


def test_legacy_control_dispatches_stock_n0_and_classic_lora_c0():
    calls = []

    class FakeRunner:
        def __init__(self, output):
            self.output = output

        def run(self, *args, **kwargs):
            calls.append((self.output, args, kwargs))
            return SimpleNamespace(hidden_states=self.output)

    fixture = SimpleNamespace(
        _dispatch_output=lambda: "standard-dispatch",
        legacy_quant_info="triton-quant",
        legacy_base_runner=FakeRunner("base-output"),
        legacy_lora_runner=FakeRunner("active-output"),
        legacy_lora_info="classic-lora-info",
        trtllm_quant_info=None,
        last_output=None,
    )
    PipelineFixture.invoke_neutral(fixture, "legacy_triton", "N0")
    assert fixture.last_output == "base-output"
    PipelineFixture.invoke_neutral(fixture, "legacy_triton", "C0")
    assert fixture.last_output == "active-output"
    assert calls == [
        (
            "base-output",
            ("standard-dispatch", "triton-quant"),
            {},
        ),
        (
            "active-output",
            ("standard-dispatch", "triton-quant"),
            {"lora_info": "classic-lora-info"},
        ),
    ]


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
