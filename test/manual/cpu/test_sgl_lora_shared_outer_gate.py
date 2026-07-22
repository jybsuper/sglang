import importlib.util
import sys
from pathlib import Path


def _load_modules():
    directory = Path(__file__).parents[3] / "python/sglang/srt/lora/sgl_lora"
    policy_path = directory / "shared_outer_gate_policy.py"
    policy_name = "sglang.srt.lora.sgl_lora.shared_outer_gate_policy"
    policy_spec = importlib.util.spec_from_file_location(policy_name, policy_path)
    assert policy_spec is not None and policy_spec.loader is not None
    policy_module = importlib.util.module_from_spec(policy_spec)
    sys.modules[policy_name] = policy_module
    policy_spec.loader.exec_module(policy_module)

    path = directory / "execution_plan.py"
    spec = importlib.util.spec_from_file_location(
        "_test_sgl_lora_shared_outer_plan", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, policy_module


PLAN, POLICY = _load_modules()


def _gate_plan(**overrides):
    args = dict(
        shared_outer=True,
        device_capability=(9, 0),
        phase="decode",
        graph_mode=True,
        num_tokens=32,
        hidden_size=2048,
        rank=128,
        top_k=8,
        has_base_rows=False,
        num_segments=8,
        max_segment_len=4,
    )
    args.update(overrides)
    return POLICY.build_shared_outer_gate_a_plan(**args)


def test_shared_outer_gate_selector_promotes_only_portable_cells():
    for capability in ((9, 0), (10, 0)):
        assert _gate_plan(device_capability=capability).uses_token_dedup
        assert _gate_plan(
            device_capability=capability,
            num_tokens=256,
            rank=64,
            has_base_rows=False,
            num_segments=4,
            max_segment_len=64,
        ).uses_token_dedup
        assert _gate_plan(
            device_capability=capability,
            hidden_size=4096,
            rank=64,
            top_k=10,
        ).uses_token_dedup


def test_shared_outer_gate_selector_keeps_wide_tiny_and_noisy_fallbacks():
    fallbacks = (
        _gate_plan(shared_outer=False),
        _gate_plan(device_capability=(8, 0)),
        _gate_plan(num_tokens=1, rank=32, num_segments=1, max_segment_len=1),
        _gate_plan(hidden_size=7168, rank=64),
        _gate_plan(rank=64),
        _gate_plan(rank=64, has_base_rows=True),
        _gate_plan(top_k=1),
        _gate_plan(rank=96),
        _gate_plan(hidden_size=1024),
        _gate_plan(phase="prefill"),
        _gate_plan(num_segments=32, max_segment_len=32),
    )
    assert all(
        plan.kernel is POLICY.SharedOuterGateAKernel.GENERIC_VIRTUAL_EXPERT
        for plan in fallbacks
    )


def test_moe_execution_plan_carries_the_static_gate_site_decision():
    plan = PLAN.build_moe_lora_execution_plan(
        phase="decode",
        graph_mode=True,
        num_tokens=32,
        rank=128,
        has_base_rows=False,
        two_stream_requested=True,
        shared_outer=True,
        device_capability=(10, 0),
        hidden_size=2048,
        top_k=8,
        num_segments=8,
        max_segment_len=4,
    )
    gate = plan.shared_outer_gate_a_plan
    assert gate is not None and gate.uses_token_dedup
    assert gate.key.site == "gate_a"
    assert gate.key.device_family == "blackwell"
    assert gate.key.graph_mode
