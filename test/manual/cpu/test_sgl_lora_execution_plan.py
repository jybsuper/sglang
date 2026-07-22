import importlib.util
import sys
from pathlib import Path


def _load_plan_module():
    directory = Path(__file__).parents[3] / "python/sglang/srt/lora/sgl_lora"
    policy_path = directory / "shared_outer_gate_policy.py"
    policy_name = "sglang.srt.lora.sgl_lora.shared_outer_gate_policy"
    policy_spec = importlib.util.spec_from_file_location(policy_name, policy_path)
    assert policy_spec is not None and policy_spec.loader is not None
    policy_module = importlib.util.module_from_spec(policy_spec)
    sys.modules[policy_name] = policy_module
    policy_spec.loader.exec_module(policy_module)

    path = directory / "execution_plan.py"
    spec = importlib.util.spec_from_file_location("_test_sgl_lora_execution_plan", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PLAN = _load_plan_module()


class _Mode:
    def __init__(self, *, decode=False, extend=False):
        self.decode = decode
        self.extend = extend

    def is_decode(self):
        return self.decode

    def is_extend(self, include_draft_extend_v2=False):
        del include_draft_extend_v2
        return self.extend


def _plan(**overrides):
    args = dict(
        phase="decode",
        graph_mode=True,
        num_tokens=32,
        rank=64,
        has_base_rows=False,
        two_stream_requested=False,
    )
    args.update(overrides)
    return PLAN.build_moe_lora_execution_plan(**args)


def test_phase_is_resolved_from_forward_mode_not_token_count():
    assert PLAN.classify_forward_phase(_Mode(decode=True)) == "decode"
    assert PLAN.classify_forward_phase(_Mode(extend=True)) == "prefill"
    assert PLAN.classify_forward_phase(_Mode()) == "other"

    decode = _plan(phase="decode", num_tokens=32)
    prefill = _plan(phase="prefill", num_tokens=32)
    assert decode.path is PLAN.MoeLoraExecutionPath.C2_FULL
    assert prefill.path is PLAN.MoeLoraExecutionPath.C2_PARTIAL


def test_static_base_rows_are_host_derived_and_graph_conservative():
    assert not PLAN.resolve_static_has_base_rows(
        weight_indices=[1, 1],
        lora_ranks=[0, 16],
        graph_mode=False,
        capture_variant=None,
    )
    assert PLAN.resolve_static_has_base_rows(
        weight_indices=[1, 0],
        lora_ranks=[0, 16],
        graph_mode=False,
        capture_variant=None,
    )
    assert PLAN.resolve_static_has_base_rows(
        weight_indices=[1, 1],
        lora_ranks=[0, 16],
        graph_mode=True,
        capture_variant="lora",
    )


def test_c3_is_only_selected_for_requested_captured_decode_through_128():
    for tokens in (1, 32, 128):
        plan = _plan(num_tokens=tokens, two_stream_requested=True)
        assert plan.path is PLAN.MoeLoraExecutionPath.C3_OVERLAP
        assert plan.uses_side_stream

    assert _plan(num_tokens=129, two_stream_requested=True).path is (
        PLAN.MoeLoraExecutionPath.C2_PARTIAL
    )
    assert _plan(num_tokens=256, two_stream_requested=True).path is (
        PLAN.MoeLoraExecutionPath.C2_PARTIAL
    )
    assert (
        _plan(graph_mode=False, num_tokens=32, two_stream_requested=True).path
        is PLAN.MoeLoraExecutionPath.C2_FULL
    )


def test_prefill_never_selects_overlap_or_complete_fused_tail():
    for graph_mode in (False, True):
        for tokens in (1, 32, 256, 2048):
            plan = _plan(
                phase="prefill",
                graph_mode=graph_mode,
                num_tokens=tokens,
                two_stream_requested=True,
            )
            assert plan.path is PLAN.MoeLoraExecutionPath.C2_PARTIAL
            assert not plan.uses_side_stream


def test_eager_and_fallback_policy_stay_in_measured_envelope():
    assert _plan(graph_mode=False, num_tokens=256).path is (
        PLAN.MoeLoraExecutionPath.C2_FULL
    )
    assert _plan(graph_mode=False, num_tokens=257).path is (
        PLAN.MoeLoraExecutionPath.C2_PARTIAL
    )
    assert _plan(rank=256).path is PLAN.MoeLoraExecutionPath.C0_SERIAL
    assert _plan(phase="other").path is PLAN.MoeLoraExecutionPath.C0_SERIAL
    assert _plan(fused_supported=False).path is PLAN.MoeLoraExecutionPath.C0_SERIAL

    quantized = _plan(
        fused_supported=False,
        provider_key="deepgemm_fp8_w8a8",
    )
    assert quantized.provider_key == "deepgemm_fp8_w8a8"
    assert "provider-neutral serial topology" in quantized.reason

    shared = _plan(base_lora_expert_domains_match=False)
    assert shared.path is PLAN.MoeLoraExecutionPath.C0_SERIAL
    assert "physical base-expert IDs include shared slots" in shared.reason


def test_rank_tiers_and_mixed_schedule_are_explicit_plan_fields():
    r16 = _plan(num_tokens=1, rank=16)
    assert r16.consumer_block_size_n == 16
    assert (r16.finalize_block_size_h, r16.finalize_num_warps) == (64, 4)

    r64 = _plan(rank=64)
    assert (r64.finalize_block_size_h, r64.finalize_num_warps) == (64, 2)

    r128 = _plan(rank=128)
    assert (r128.finalize_block_size_h, r128.finalize_num_warps) == (32, 4)

    mixed_prefill = _plan(
        phase="prefill", graph_mode=False, num_tokens=128, rank=16, has_base_rows=True
    )
    assert mixed_prefill.consumer_schedule == "pair"
