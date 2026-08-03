from __future__ import annotations

import pytest

from sglang.srt.lora.sgl_lora.execution_plan import (
    ActivationFamily,
    EarlyOverlap,
    FactorOwnership,
    LateOverlap,
    LoraAFamily,
    MiddleFamily,
    MoeLoraFactorLayout,
    RouteBuilderFamily,
)
from sglang.srt.lora.sgl_lora.selector import (
    PER_EXPERT_LAYOUT,
    SHARED_LAYOUT,
    DeviceArchitecture,
    PolicyInput,
    PolicyMode,
    ProviderKey,
    architecture_for_device,
    choices_for,
    resolve_base_gemm_provider,
    select_policy,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


def _input(**overrides) -> PolicyInput:
    values = {
        "architecture": DeviceArchitecture.H200,
        "base_gemm_provider": "cutedsl",
        "factor_layout": PER_EXPERT_LAYOUT,
        "activation": ActivationFamily.SWIGLU,
        "mode": PolicyMode.DECODE,
        "num_tokens": 4,
        "active_rank": 16,
    }
    values.update(overrides)
    return PolicyInput(**values)


def test_architecture_admission_requires_the_exact_model_and_capability() -> None:
    assert architecture_for_device("NVIDIA H200", (9, 0)) is DeviceArchitecture.H200
    assert (
        architecture_for_device("NVIDIA GB300 NVL", (10, 3)) is DeviceArchitecture.GB300
    )
    with pytest.raises(NotImplementedError, match="measured support"):
        architecture_for_device("NVIDIA H100", (9, 0))
    with pytest.raises(NotImplementedError, match="requires sm103"):
        architecture_for_device("NVIDIA GB300", (10, 0))


@pytest.mark.parametrize("architecture", list(DeviceArchitecture))
def test_auto_provider_resolves_once_to_the_measured_faster_default(
    architecture: DeviceArchitecture,
) -> None:
    assert resolve_base_gemm_provider("auto", architecture) == "cutedsl"


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"num_tokens": 17, "active_rank": 64},
        {"mode": PolicyMode.PREFILL, "num_tokens": 2048, "active_rank": 64},
        {"factor_layout": SHARED_LAYOUT, "num_tokens": 16, "active_rank": 64},
        {
            "factor_layout": SHARED_LAYOUT,
            "mode": PolicyMode.PREFILL,
            "num_tokens": 2048,
            "active_rank": 64,
        },
        {
            "architecture": DeviceArchitecture.GB300,
            "num_tokens": 16,
            "active_rank": 64,
        },
    ],
)
def test_fixed_provider_is_preserved_across_policy_regimes(overrides) -> None:
    choice = select_policy(_input(**overrides))
    assert choice.provider == "cutedsl"


def test_h200_per_expert_policy_keeps_three_winning_families() -> None:
    small = select_policy(_input(num_tokens=16))
    assert small.provider == "cutedsl"
    assert small.plan is not None
    assert small.plan.down_a is not None
    assert small.plan.down_a.family is LoraAFamily.INDEXED
    assert small.plan.early_overlap is EarlyOverlap.GATE_A_B
    assert small.plan.late_overlap is LateOverlap.DOWN_A_B

    large = select_policy(_input(num_tokens=17, active_rank=64))
    assert large.provider == "cutedsl"
    assert large.plan is not None
    assert large.plan.down_a is not None
    assert large.plan.down_a.family is LoraAFamily.GROUPED
    assert large.plan.late_overlap is LateOverlap.DOWN_A_B

    prefill = select_policy(
        _input(mode=PolicyMode.PREFILL, num_tokens=2048, active_rank=64)
    )
    assert prefill.provider == "cutedsl"
    assert prefill.plan is not None
    assert prefill.plan.middle.family is MiddleFamily.B_ACTIVATION
    assert prefill.plan.early_overlap is EarlyOverlap.GATE_A
    assert prefill.plan.late_overlap is LateOverlap.DOWN_B


@pytest.mark.parametrize("architecture", list(DeviceArchitecture))
def test_shared_policy_uses_architecture_specific_decode_and_prefill(
    architecture: DeviceArchitecture,
) -> None:
    decode = select_policy(
        _input(architecture=architecture, factor_layout=SHARED_LAYOUT, num_tokens=16)
    )
    prefill = select_policy(
        _input(
            architecture=architecture,
            factor_layout=SHARED_LAYOUT,
            mode=PolicyMode.PREFILL,
            num_tokens=2048,
            active_rank=64,
        )
    )
    assert decode.provider == "cutedsl"
    assert decode.plan is not None
    assert decode.plan.route_builder is RouteBuilderFamily.JOINT_SHARED_OUTER
    assert prefill.plan is not None
    assert prefill.plan.gate_a.family is LoraAFamily.TOKEN_DEDUP_GROUPED
    assert prefill.plan.middle.family is MiddleFamily.B_ACTIVATION
    assert prefill.provider == "cutedsl"


def test_gb300_policy_is_all_cutedsl_and_tracks_measured_regimes() -> None:
    tiny = select_policy(
        _input(architecture=DeviceArchitecture.GB300, num_tokens=4, active_rank=16)
    )
    medium = select_policy(
        _input(architecture=DeviceArchitecture.GB300, num_tokens=16, active_rank=64)
    )
    large = select_policy(
        _input(architecture=DeviceArchitecture.GB300, num_tokens=256, active_rank=64)
    )
    ordinary_prefill = select_policy(
        _input(
            architecture=DeviceArchitecture.GB300,
            mode=PolicyMode.PREFILL,
            num_tokens=2048,
            active_rank=64,
        )
    )
    large_prefill = select_policy(
        _input(
            architecture=DeviceArchitecture.GB300,
            mode=PolicyMode.PREFILL,
            num_tokens=4096,
            active_rank=64,
        )
    )
    rank128_prefill = select_policy(
        _input(
            architecture=DeviceArchitecture.GB300,
            mode=PolicyMode.PREFILL,
            num_tokens=2048,
            active_rank=128,
        )
    )
    assert all(
        choice.provider == "cutedsl"
        for choice in (
            tiny,
            medium,
            large,
            ordinary_prefill,
            large_prefill,
            rank128_prefill,
        )
    )
    assert ".tiny." in tiny.key
    assert ".medium." in medium.key
    assert ".large." in large.key
    assert "b_activation" in ordinary_prefill.key
    assert "serial_large" in large_prefill.key
    assert rank128_prefill.key == large_prefill.key


def test_relu2_is_the_q397_policy_signal() -> None:
    decode = select_policy(
        _input(
            architecture=DeviceArchitecture.GB300,
            activation=ActivationFamily.RELU2,
            num_tokens=16,
            active_rank=64,
        )
    )
    prefill = select_policy(
        _input(
            architecture=DeviceArchitecture.GB300,
            activation=ActivationFamily.RELU2,
            mode=PolicyMode.PREFILL,
            num_tokens=2048,
            active_rank=64,
        )
    )
    assert decode.key == "gb300.per_expert.decode.gab.relu2"
    assert prefill.key == "gb300.per_expert.prefill.b_activation.relu2"
    assert (
        decode.plan is not None
        and decode.plan.middle.activation is ActivationFamily.RELU2
    )
    assert (
        prefill.plan is not None
        and prefill.plan.middle.activation is ActivationFamily.RELU2
    )


@pytest.mark.parametrize("architecture", list(DeviceArchitecture))
@pytest.mark.parametrize("factor_layout", [PER_EXPERT_LAYOUT, SHARED_LAYOUT])
@pytest.mark.parametrize("activation", list(ActivationFamily))
@pytest.mark.parametrize("provider", ["cutedsl", "deepgemm"])
def test_choices_are_unique_and_prevalidatable(
    architecture: DeviceArchitecture,
    factor_layout: MoeLoraFactorLayout,
    activation: ActivationFamily,
    provider: ProviderKey,
) -> None:
    choices = choices_for(architecture, factor_layout, activation, provider)
    assert choices
    assert len({choice.key for choice in choices}) == len(choices)
    assert {choice.provider for choice in choices} == {provider}
    for choice in choices:
        assert choice.plan is not None
        assert choice.launch_config is not None
        choice.plan.validate_factor_layout(factor_layout)
        choice.launch_config.validate_for_plan(choice.plan)


def test_exact_config_values_survive_embedding() -> None:
    h200 = select_policy(_input(num_tokens=4, active_rank=16))
    assert h200.launch_config is not None
    assert h200.launch_config.down_a == {
        "BLOCK_SIZE_K": 128,
        "BLOCK_SIZE_N": 16,
        "num_stages": 2,
        "num_warps": 8,
    }
    gb300 = select_policy(
        _input(architecture=DeviceArchitecture.GB300, num_tokens=4, active_rank=16)
    )
    assert gb300.launch_config is not None
    assert gb300.launch_config.gate_a["BLOCK_SIZE_N"] == 16
    assert gb300.launch_config.down_b["num_warps"] == 8


def test_mixed_serving_layout_is_rejected() -> None:
    mixed = MoeLoraFactorLayout(
        gate_up_a=FactorOwnership.SHARED_OUTER,
        down_b=FactorOwnership.PER_EXPERT,
    )
    with pytest.raises(ValueError, match="shared-both"):
        choices_for(DeviceArchitecture.H200, mixed, ActivationFamily.SWIGLU, "cutedsl")
