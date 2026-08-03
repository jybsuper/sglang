"""Small evidence-backed policy for the production BF16 MoE-LoRA runner.

The policy intentionally keeps only the execution families that won a useful
part of the H200/GB300 sweep.  It has no CUDA imports and no runtime artifact
dependency, so serving can construct every possible runner before graph
capture via :func:`choices_for`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from functools import cache
from typing import Literal, Mapping, cast

from sglang.srt.lora.sgl_lora.execution_plan import (
    ActivationFamily,
    EarlyOverlap,
    FactorContract,
    FactorLayout,
    FactorOwnership,
    FactorSite,
    FinalizeFamily,
    FinalizeSpec,
    LateOverlap,
    LoraAFamily,
    LoraASpec,
    LoraBSpec,
    MiddleFamily,
    MiddleSpec,
    MoeLoraExecutionPlan,
    MoeLoraFactorLayout,
    RouteBuilderFamily,
)
from sglang.srt.lora.sgl_lora.launch_config import MoeLoraLaunchConfig


class DeviceArchitecture(str, Enum):
    H200 = "h200"
    GB300 = "gb300"


class PolicyMode(str, Enum):
    DECODE = "decode"
    PREFILL = "prefill"


ProviderKey = Literal["deepgemm", "cutedsl"]


@dataclass(frozen=True, slots=True, kw_only=True)
class PolicyInput:
    architecture: DeviceArchitecture
    base_gemm_provider: ProviderKey
    factor_layout: MoeLoraFactorLayout
    activation: ActivationFamily
    mode: PolicyMode
    num_tokens: int
    active_rank: int

    def __post_init__(self) -> None:
        if not isinstance(self.architecture, DeviceArchitecture):
            raise TypeError("architecture must be DeviceArchitecture")
        if self.base_gemm_provider not in ("deepgemm", "cutedsl"):
            raise ValueError("base_gemm_provider must be 'deepgemm' or 'cutedsl'")
        if not isinstance(self.factor_layout, MoeLoraFactorLayout):
            raise TypeError("factor_layout must be MoeLoraFactorLayout")
        self.factor_layout.validate()
        if not isinstance(self.activation, ActivationFamily):
            raise TypeError("activation must be ActivationFamily")
        if not isinstance(self.mode, PolicyMode):
            raise TypeError("mode must be PolicyMode")
        if self.num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if self.active_rank <= 0:
            raise ValueError("active_rank must be positive")


@dataclass(frozen=True, slots=True)
class PolicyChoice:
    key: str
    provider: ProviderKey
    plan: MoeLoraExecutionPlan
    launch_config: MoeLoraLaunchConfig

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("policy choice key must be non-empty")
        if self.provider not in ("deepgemm", "cutedsl"):
            raise ValueError("runner provider must be 'deepgemm' or 'cutedsl'")
        self.launch_config.validate_for_plan(self.plan)

    @property
    def name(self) -> str:
        return self.key


_PE = FactorOwnership.PER_EXPERT
_PAIR = FactorLayout.PAIR_MAJOR
_TOKEN = FactorLayout.TOKEN_MAJOR
PER_EXPERT_LAYOUT = MoeLoraFactorLayout.serving(False)
SHARED_LAYOUT = MoeLoraFactorLayout.serving(True)

_FUSED_32 = {
    "BLOCK_SIZE_K": 32,
    "BLOCK_SIZE_W": 64,
    "GROUP_SIZE_M": 8,
    "num_stages": 2,
    "num_warps": 4,
}
_SHARED_FINALIZE = {
    "reduce": {"BLOCK_SIZE_T": 32, "num_stages": 2, "num_warps": 4},
    "tail": {
        "BLOCK_SIZE_H": 128,
        "BLOCK_SIZE_K": 32,
        "num_stages": 3,
        "num_warps": 4,
    },
}


def _config(
    *,
    gate_a: Mapping[str, int],
    gate_b: Mapping[str, int],
    down_a: Mapping[str, int],
    down_b: Mapping[str, int],
    b_activation: Mapping[str, int] = _FUSED_32,
    gate_a_routing_block_size: int = 16,
    shared_finalize: Mapping[str, Mapping[str, int]] = _SHARED_FINALIZE,
) -> MoeLoraLaunchConfig:
    return MoeLoraLaunchConfig(
        routing_block_size=16,
        gate_a_routing_block_size=gate_a_routing_block_size,
        gate_a=dict(gate_a),
        gate_b=dict(gate_b),
        down_a=dict(down_a),
        down_b=dict(down_b),
        b_activation=dict(b_activation),
        shared_finalize={key: dict(value) for key, value in shared_finalize.items()},
    )


# H200 Step-10 compact selector. These are exact promoted launch configs.
H200_DECODE_INDEXED_CONFIG = _config(
    gate_a={
        "BLOCK_SIZE_K": 128,
        "BLOCK_SIZE_N": 64,
        "GROUP_SIZE_M": 8,
        "num_stages": 3,
        "num_warps": 4,
    },
    gate_b={
        "BLOCK_SIZE_K": 32,
        "BLOCK_SIZE_N": 128,
        "GROUP_SIZE_M": 8,
        "num_stages": 2,
        "num_warps": 4,
    },
    down_a={"BLOCK_SIZE_K": 128, "BLOCK_SIZE_N": 16, "num_stages": 2, "num_warps": 8},
    down_b={
        "BLOCK_SIZE_K": 32,
        "BLOCK_SIZE_N": 128,
        "GROUP_SIZE_M": 1,
        "num_stages": 2,
        "num_warps": 4,
    },
)
H200_DECODE_GROUPED_CONFIG = _config(
    gate_a={
        "BLOCK_SIZE_K": 64,
        "BLOCK_SIZE_N": 32,
        "GROUP_SIZE_M": 8,
        "num_stages": 3,
        "num_warps": 4,
    },
    gate_b={
        "BLOCK_SIZE_K": 32,
        "BLOCK_SIZE_N": 256,
        "GROUP_SIZE_M": 4,
        "num_stages": 2,
        "num_warps": 4,
    },
    down_a={
        "BLOCK_SIZE_K": 64,
        "BLOCK_SIZE_N": 64,
        "GROUP_SIZE_M": 8,
        "num_stages": 4,
        "num_warps": 4,
    },
    down_b={
        "BLOCK_SIZE_K": 64,
        "BLOCK_SIZE_N": 128,
        "GROUP_SIZE_M": 16,
        "num_stages": 2,
        "num_warps": 4,
    },
)
H200_PREFILL_CONFIG = _config(
    gate_a={
        "BLOCK_SIZE_K": 128,
        "BLOCK_SIZE_N": 64,
        "GROUP_SIZE_M": 8,
        "num_stages": 3,
        "num_warps": 4,
    },
    gate_b={
        "BLOCK_SIZE_K": 32,
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "GROUP_SIZE_M": 8,
        "num_stages": 2,
        "num_warps": 4,
    },
    down_a={
        "BLOCK_SIZE_K": 128,
        "BLOCK_SIZE_N": 64,
        "GROUP_SIZE_M": 8,
        "num_stages": 3,
        "num_warps": 4,
    },
    down_b={
        "BLOCK_SIZE_K": 32,
        "BLOCK_SIZE_N": 256,
        "GROUP_SIZE_M": 16,
        "num_stages": 3,
        "num_warps": 4,
    },
    b_activation={
        "BLOCK_SIZE_K": 16,
        "BLOCK_SIZE_W": 128,
        "GROUP_SIZE_M": 16,
        "num_stages": 4,
        "num_warps": 4,
    },
)
H200_SHARED_DECODE_CONFIG = _config(
    gate_a={
        "BLOCK_SIZE_K": 128,
        "BLOCK_SIZE_N": 32,
        "GROUP_SIZE_M": 8,
        "num_stages": 3,
        "num_warps": 4,
    },
    gate_b={
        "BLOCK_SIZE_K": 64,
        "BLOCK_SIZE_N": 256,
        "GROUP_SIZE_M": 4,
        "num_stages": 3,
        "num_warps": 8,
    },
    down_a={
        "BLOCK_SIZE_K": 128,
        "BLOCK_SIZE_N": 64,
        "GROUP_SIZE_M": 8,
        "num_stages": 4,
        "num_warps": 4,
    },
    down_b={
        "BLOCK_SIZE_K": 32,
        "BLOCK_SIZE_N": 128,
        "GROUP_SIZE_M": 1,
        "num_stages": 2,
        "num_warps": 4,
    },
)
H200_SHARED_PREFILL_CONFIG = _config(
    gate_a={
        "BLOCK_SIZE_K": 64,
        "BLOCK_SIZE_N": 64,
        "GROUP_SIZE_M": 8,
        "num_stages": 3,
        "num_warps": 4,
    },
    gate_b={
        "BLOCK_SIZE_K": 32,
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "GROUP_SIZE_M": 8,
        "num_stages": 2,
        "num_warps": 4,
    },
    down_a={
        "BLOCK_SIZE_K": 128,
        "BLOCK_SIZE_N": 64,
        "GROUP_SIZE_M": 8,
        "num_stages": 3,
        "num_warps": 4,
    },
    down_b={
        "BLOCK_SIZE_K": 32,
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "GROUP_SIZE_M": 8,
        "num_stages": 2,
        "num_warps": 4,
    },
    b_activation={
        "BLOCK_SIZE_K": 16,
        "BLOCK_SIZE_W": 128,
        "GROUP_SIZE_M": 1,
        "num_stages": 4,
        "num_warps": 4,
    },
    shared_finalize={
        "reduce": {"BLOCK_SIZE_T": 16, "num_stages": 2, "num_warps": 8},
        "tail": {
            "BLOCK_SIZE_H": 256,
            "BLOCK_SIZE_K": 32,
            "num_stages": 2,
            "num_warps": 4,
        },
    },
)
# GB300 exact CuTe configs. Geometry-specific decode configs avoid a large
# table while retaining the three measured token/rank regimes.
GB300_DECODE_TINY_CONFIG = _config(
    gate_a={
        "BLOCK_SIZE_K": 128,
        "BLOCK_SIZE_N": 16,
        "GROUP_SIZE_M": 8,
        "num_stages": 3,
        "num_warps": 4,
    },
    gate_b={
        "BLOCK_SIZE_K": 16,
        "BLOCK_SIZE_N": 128,
        "GROUP_SIZE_M": 4,
        "num_stages": 2,
        "num_warps": 4,
    },
    down_a={
        "BLOCK_SIZE_K": 128,
        "BLOCK_SIZE_N": 16,
        "GROUP_SIZE_M": 8,
        "num_stages": 4,
        "num_warps": 4,
    },
    down_b={
        "BLOCK_SIZE_K": 16,
        "BLOCK_SIZE_N": 256,
        "GROUP_SIZE_M": 1,
        "num_stages": 2,
        "num_warps": 8,
    },
)
GB300_DECODE_MEDIUM_CONFIG = _config(
    gate_a={
        "BLOCK_SIZE_K": 128,
        "BLOCK_SIZE_N": 64,
        "GROUP_SIZE_M": 8,
        "num_stages": 4,
        "num_warps": 4,
    },
    gate_b={
        "BLOCK_SIZE_K": 32,
        "BLOCK_SIZE_N": 512,
        "GROUP_SIZE_M": 4,
        "num_stages": 2,
        "num_warps": 4,
    },
    down_a={
        "BLOCK_SIZE_K": 128,
        "BLOCK_SIZE_N": 64,
        "GROUP_SIZE_M": 8,
        "num_stages": 3,
        "num_warps": 8,
    },
    down_b={
        "BLOCK_SIZE_K": 32,
        "BLOCK_SIZE_N": 128,
        "GROUP_SIZE_M": 1,
        "num_stages": 2,
        "num_warps": 4,
    },
)
GB300_DECODE_LARGE_CONFIG = _config(
    gate_a={
        "BLOCK_SIZE_K": 64,
        "BLOCK_SIZE_N": 64,
        "GROUP_SIZE_M": 8,
        "num_stages": 4,
        "num_warps": 4,
    },
    gate_b={
        "BLOCK_SIZE_K": 32,
        "BLOCK_SIZE_N": 256,
        "GROUP_SIZE_M": 16,
        "num_stages": 2,
        "num_warps": 4,
    },
    down_a={
        "BLOCK_SIZE_K": 128,
        "BLOCK_SIZE_N": 64,
        "GROUP_SIZE_M": 8,
        "num_stages": 4,
        "num_warps": 8,
    },
    down_b={
        "BLOCK_SIZE_K": 64,
        "BLOCK_SIZE_N": 256,
        "GROUP_SIZE_M": 4,
        "num_stages": 2,
        "num_warps": 8,
    },
)
GB300_RELU2_DECODE_CONFIG = _config(
    gate_a={
        "BLOCK_SIZE_K": 128,
        "BLOCK_SIZE_N": 32,
        "GROUP_SIZE_M": 8,
        "num_stages": 4,
        "num_warps": 4,
    },
    gate_b={
        "BLOCK_SIZE_K": 32,
        "BLOCK_SIZE_N": 256,
        "GROUP_SIZE_M": 8,
        "num_stages": 2,
        "num_warps": 4,
    },
    down_a={
        "BLOCK_SIZE_K": 128,
        "BLOCK_SIZE_N": 64,
        "GROUP_SIZE_M": 8,
        "num_stages": 4,
        "num_warps": 4,
    },
    down_b={
        "BLOCK_SIZE_K": 64,
        "BLOCK_SIZE_N": 256,
        "GROUP_SIZE_M": 8,
        "num_stages": 4,
        "num_warps": 8,
    },
)
GB300_PREFILL_CONFIG = _config(
    gate_a={
        "BLOCK_SIZE_K": 128,
        "BLOCK_SIZE_N": 64,
        "GROUP_SIZE_M": 8,
        "num_stages": 2,
        "num_warps": 4,
    },
    gate_b={
        "BLOCK_SIZE_K": 32,
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "GROUP_SIZE_M": 8,
        "num_stages": 2,
        "num_warps": 4,
    },
    down_a={
        "BLOCK_SIZE_K": 128,
        "BLOCK_SIZE_N": 64,
        "GROUP_SIZE_M": 8,
        "num_stages": 3,
        "num_warps": 4,
    },
    down_b={
        "BLOCK_SIZE_K": 32,
        "BLOCK_SIZE_N": 256,
        "GROUP_SIZE_M": 1,
        "num_stages": 2,
        "num_warps": 4,
    },
    b_activation={
        "BLOCK_SIZE_K": 16,
        "BLOCK_SIZE_W": 128,
        "GROUP_SIZE_M": 1,
        "num_stages": 2,
        "num_warps": 4,
    },
)
GB300_RELU2_PREFILL_CONFIG = _config(
    gate_a={
        "BLOCK_SIZE_K": 128,
        "BLOCK_SIZE_N": 64,
        "GROUP_SIZE_M": 8,
        "num_stages": 2,
        "num_warps": 4,
    },
    gate_b={
        "BLOCK_SIZE_K": 32,
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "GROUP_SIZE_M": 8,
        "num_stages": 2,
        "num_warps": 4,
    },
    down_a={
        "BLOCK_SIZE_K": 64,
        "BLOCK_SIZE_N": 64,
        "GROUP_SIZE_M": 8,
        "num_stages": 4,
        "num_warps": 4,
    },
    down_b={
        "BLOCK_SIZE_K": 32,
        "BLOCK_SIZE_N": 256,
        "GROUP_SIZE_M": 8,
        "num_stages": 2,
        "num_warps": 4,
    },
    b_activation={
        "BLOCK_SIZE_K": 16,
        "BLOCK_SIZE_W": 128,
        "GROUP_SIZE_M": 1,
        "num_stages": 2,
        "num_warps": 4,
    },
)
GB300_LARGE_PREFILL_CONFIG = _config(
    gate_a={
        "BLOCK_SIZE_K": 64,
        "BLOCK_SIZE_N": 64,
        "GROUP_SIZE_M": 8,
        "num_stages": 3,
        "num_warps": 4,
    },
    gate_b={
        "BLOCK_SIZE_K": 64,
        "BLOCK_SIZE_N": 128,
        "GROUP_SIZE_M": 8,
        "num_stages": 2,
        "num_warps": 4,
    },
    down_a={
        "BLOCK_SIZE_K": 128,
        "BLOCK_SIZE_N": 64,
        "GROUP_SIZE_M": 8,
        "num_stages": 2,
        "num_warps": 4,
    },
    down_b={
        "BLOCK_SIZE_K": 64,
        "BLOCK_SIZE_N": 128,
        "GROUP_SIZE_M": 16,
        "num_stages": 2,
        "num_warps": 4,
    },
    gate_a_routing_block_size=64,
)
GB300_SHARED_DECODE_CONFIG = _config(
    gate_a={
        "BLOCK_SIZE_K": 128,
        "BLOCK_SIZE_N": 64,
        "GROUP_SIZE_M": 8,
        "num_stages": 4,
        "num_warps": 4,
    },
    gate_b={
        "BLOCK_SIZE_K": 32,
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "GROUP_SIZE_M": 8,
        "num_stages": 2,
        "num_warps": 4,
    },
    down_a={
        "BLOCK_SIZE_K": 128,
        "BLOCK_SIZE_N": 64,
        "GROUP_SIZE_M": 8,
        "num_stages": 4,
        "num_warps": 4,
    },
    down_b={
        "BLOCK_SIZE_K": 32,
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "GROUP_SIZE_M": 8,
        "num_stages": 2,
        "num_warps": 4,
    },
    b_activation={
        "BLOCK_SIZE_K": 64,
        "BLOCK_SIZE_W": 64,
        "GROUP_SIZE_M": 1,
        "num_stages": 2,
        "num_warps": 4,
    },
    shared_finalize={
        "reduce": {"BLOCK_SIZE_T": 8, "num_stages": 3, "num_warps": 2},
        "tail": {
            "BLOCK_SIZE_H": 128,
            "BLOCK_SIZE_K": 64,
            "num_stages": 3,
            "num_warps": 4,
        },
    },
)
GB300_SHARED_PREFILL_CONFIG = _config(
    gate_a={
        "BLOCK_SIZE_K": 128,
        "BLOCK_SIZE_N": 64,
        "GROUP_SIZE_M": 8,
        "num_stages": 4,
        "num_warps": 4,
    },
    gate_b={
        "BLOCK_SIZE_K": 32,
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "GROUP_SIZE_M": 8,
        "num_stages": 2,
        "num_warps": 4,
    },
    down_a={
        "BLOCK_SIZE_K": 64,
        "BLOCK_SIZE_N": 64,
        "GROUP_SIZE_M": 8,
        "num_stages": 3,
        "num_warps": 4,
    },
    down_b={
        "BLOCK_SIZE_K": 32,
        "BLOCK_SIZE_N": 256,
        "GROUP_SIZE_M": 8,
        "num_stages": 2,
        "num_warps": 4,
    },
    b_activation={
        "BLOCK_SIZE_K": 16,
        "BLOCK_SIZE_W": 128,
        "GROUP_SIZE_M": 8,
        "num_stages": 4,
        "num_warps": 4,
    },
)


_MODEL_CAPABILITIES: Mapping[DeviceArchitecture, tuple[int, int]] = {
    DeviceArchitecture.H200: (9, 0),
    DeviceArchitecture.GB300: (10, 3),
}
_DEFAULT_BASE_GEMM_PROVIDERS: Mapping[DeviceArchitecture, ProviderKey] = {
    DeviceArchitecture.H200: "cutedsl",
    DeviceArchitecture.GB300: "cutedsl",
}


def architecture_for_device(
    device_name: str,
    capability: tuple[int, int],
) -> DeviceArchitecture:
    """Admit an evidence-backed GPU model, then sanity-check its ISA.

    Compute capability alone is deliberately insufficient: H100 and H200 are
    both SM90, while the policy was measured only on H200 and GB300.  NVIDIA's
    runtime name may carry form-factor suffixes (for example ``H200 NVL``), so
    the exact admitted model token is matched as a word rather than requiring
    the entire marketing string to be identical.
    """

    if not isinstance(device_name, str) or not device_name.strip():
        raise TypeError("device_name must be a non-empty string")
    if (
        not isinstance(capability, tuple)
        or len(capability) != 2
        or not all(isinstance(value, int) for value in capability)
    ):
        raise TypeError("capability must be an integer (major, minor) tuple")

    normalized_name = device_name.upper()
    matches = tuple(
        architecture
        for architecture in DeviceArchitecture
        if re.search(rf"\b{architecture.value.upper()}\b", normalized_name)
    )
    if len(matches) != 1:
        raise NotImplementedError(
            "SGL MoE-LoRA policy has measured support only for NVIDIA H200 "
            f"and GB300 GPUs; runtime reported {device_name!r}"
        )
    architecture = matches[0]
    expected_capability = _MODEL_CAPABILITIES[architecture]
    if capability != expected_capability:
        raise NotImplementedError(
            f"{architecture.value.upper()} policy requires "
            f"sm{expected_capability[0]}{expected_capability[1]}, but "
            f"{device_name!r} reported sm{capability[0]}{capability[1]}"
        )
    return architecture


def resolve_base_gemm_provider(
    requested: str,
    architecture: DeviceArchitecture,
) -> ProviderKey:
    """Resolve the immutable provider once for an SGL MoE-LoRA server."""

    if not isinstance(architecture, DeviceArchitecture):
        raise TypeError("architecture must be DeviceArchitecture")
    if requested == "auto":
        # CuTeDSL led the Qwen same-candidate provider sweep on both admitted
        # models.  The provider remains explicitly overridable for profiling.
        return _DEFAULT_BASE_GEMM_PROVIDERS[architecture]
    if requested not in ("deepgemm", "cutedsl"):
        raise ValueError("base GEMM provider must be 'auto', 'deepgemm', or 'cutedsl'")
    return cast(ProviderKey, requested)


def _build_plan(
    *,
    activation: ActivationFamily,
    factor_layout: MoeLoraFactorLayout,
    gate_a_family: LoraAFamily = LoraAFamily.GROUPED,
    down_a_family: LoraAFamily = LoraAFamily.GROUPED,
    middle_family: MiddleFamily = MiddleFamily.MATERIALIZED,
    finalize_family: FinalizeFamily = FinalizeFamily.MATERIALIZED,
    early_overlap: EarlyOverlap = EarlyOverlap.NONE,
    late_overlap: LateOverlap = LateOverlap.NONE,
    route_builder: RouteBuilderFamily = RouteBuilderFamily.STANDARD,
) -> MoeLoraExecutionPlan:
    gate_layout = _TOKEN if gate_a_family is LoraAFamily.TOKEN_DEDUP_GROUPED else _PAIR
    gate_b_contract = FactorContract(FactorSite.GATE_UP, _PE, gate_layout)
    down_b_contract = FactorContract(FactorSite.DOWN, factor_layout.down_b, _PAIR)
    consumes_gate_b = middle_family is MiddleFamily.B_ACTIVATION
    consumes_down_b = finalize_family is not FinalizeFamily.MATERIALIZED
    plan = MoeLoraExecutionPlan(
        gate_a=LoraASpec(
            FactorSite.GATE_UP, gate_a_family, factor_layout.gate_up_a, gate_layout
        ),
        gate_b=(
            None if consumes_gate_b else LoraBSpec(FactorSite.GATE_UP, _PE, gate_layout)
        ),
        middle=MiddleSpec(
            middle_family, activation, gate_b_contract if consumes_gate_b else None
        ),
        down_a=LoraASpec(FactorSite.DOWN, down_a_family, _PE, _PAIR),
        down_b=(
            None
            if consumes_down_b
            else LoraBSpec(
                FactorSite.DOWN,
                factor_layout.down_b,
                _PAIR,
            )
        ),
        finalize=FinalizeSpec(
            finalize_family, down_b_contract if consumes_down_b else None
        ),
        early_overlap=early_overlap,
        late_overlap=late_overlap,
        route_builder=route_builder,
    )
    return plan.validate_factor_layout(factor_layout)


def _choice(
    key: str,
    provider: ProviderKey,
    plan: MoeLoraExecutionPlan,
    config: MoeLoraLaunchConfig,
) -> PolicyChoice:
    return PolicyChoice(key, provider, plan, config)


@cache
def _choices_for(
    architecture: DeviceArchitecture,
    factor_layout: MoeLoraFactorLayout,
    activation: ActivationFamily,
    base_gemm_provider: ProviderKey,
) -> tuple[PolicyChoice, ...]:
    if factor_layout not in (PER_EXPERT_LAYOUT, SHARED_LAYOUT):
        raise ValueError(
            "only serving per-expert and shared-both layouts are supported"
        )
    suffix = activation.value
    if architecture is DeviceArchitecture.H200:
        if factor_layout == SHARED_LAYOUT:
            decode = _choice(
                f"h200.shared.decode.repeated_pair_joint.{suffix}",
                base_gemm_provider,
                _build_plan(
                    activation=activation,
                    factor_layout=factor_layout,
                    early_overlap=EarlyOverlap.GATE_A_B,
                    late_overlap=LateOverlap.DOWN_A_B,
                    route_builder=RouteBuilderFamily.JOINT_SHARED_OUTER,
                ),
                H200_SHARED_DECODE_CONFIG,
            )
            prefill = _choice(
                f"h200.shared.prefill.token_dedup_shared_rank.{suffix}",
                base_gemm_provider,
                _build_plan(
                    activation=activation,
                    factor_layout=factor_layout,
                    gate_a_family=LoraAFamily.TOKEN_DEDUP_GROUPED,
                    middle_family=MiddleFamily.B_ACTIVATION,
                    finalize_family=FinalizeFamily.SHARED_RANK_REDUCE,
                    early_overlap=EarlyOverlap.GATE_A,
                    late_overlap=LateOverlap.DOWN_A,
                ),
                H200_SHARED_PREFILL_CONFIG,
            )
            return (decode, prefill)
        indexed = _choice(
            f"h200.per_expert.decode.indexed_down_a.{suffix}",
            base_gemm_provider,
            _build_plan(
                activation=activation,
                factor_layout=factor_layout,
                down_a_family=LoraAFamily.INDEXED,
                early_overlap=EarlyOverlap.GATE_A_B,
                late_overlap=LateOverlap.DOWN_A_B,
            ),
            H200_DECODE_INDEXED_CONFIG,
        )
        grouped = _choice(
            f"h200.per_expert.decode.grouped.{suffix}",
            base_gemm_provider,
            _build_plan(
                activation=activation,
                factor_layout=factor_layout,
                early_overlap=EarlyOverlap.GATE_A_B,
                late_overlap=LateOverlap.DOWN_A_B,
            ),
            H200_DECODE_GROUPED_CONFIG,
        )
        prefill = _choice(
            f"h200.per_expert.prefill.b_activation.{suffix}",
            base_gemm_provider,
            _build_plan(
                activation=activation,
                factor_layout=factor_layout,
                middle_family=MiddleFamily.B_ACTIVATION,
                early_overlap=EarlyOverlap.GATE_A,
                late_overlap=LateOverlap.DOWN_B,
            ),
            H200_PREFILL_CONFIG,
        )
        return (indexed, grouped, prefill)

    if factor_layout == SHARED_LAYOUT:
        decode = _choice(
            f"gb300.shared.decode.repeated_pair_b_activation_shared_rank_joint.{suffix}",
            base_gemm_provider,
            _build_plan(
                activation=activation,
                factor_layout=factor_layout,
                middle_family=MiddleFamily.B_ACTIVATION,
                finalize_family=FinalizeFamily.SHARED_RANK_REDUCE,
                early_overlap=EarlyOverlap.GATE_A,
                late_overlap=LateOverlap.SHARED_FINALIZE,
                route_builder=RouteBuilderFamily.JOINT_SHARED_OUTER,
            ),
            GB300_SHARED_DECODE_CONFIG,
        )
        prefill = _choice(
            f"gb300.shared.prefill.token_dedup_b_activation_materialized_joint.{suffix}",
            base_gemm_provider,
            _build_plan(
                activation=activation,
                factor_layout=factor_layout,
                gate_a_family=LoraAFamily.TOKEN_DEDUP_GROUPED,
                middle_family=MiddleFamily.B_ACTIVATION,
                early_overlap=EarlyOverlap.GATE_A,
                late_overlap=LateOverlap.DOWN_A_B,
                route_builder=RouteBuilderFamily.JOINT_SHARED_OUTER,
            ),
            GB300_SHARED_PREFILL_CONFIG,
        )
        return (decode, prefill)

    gab_plan = _build_plan(
        activation=activation,
        factor_layout=factor_layout,
        early_overlap=EarlyOverlap.GATE_A_B,
        late_overlap=LateOverlap.DOWN_A_B,
    )
    if activation is ActivationFamily.RELU2:
        return (
            _choice(
                "gb300.per_expert.decode.gab.relu2",
                base_gemm_provider,
                gab_plan,
                GB300_RELU2_DECODE_CONFIG,
            ),
            _choice(
                "gb300.per_expert.prefill.b_activation.relu2",
                base_gemm_provider,
                _build_plan(
                    activation=activation,
                    factor_layout=factor_layout,
                    middle_family=MiddleFamily.B_ACTIVATION,
                    early_overlap=EarlyOverlap.GATE_A,
                    late_overlap=LateOverlap.DOWN_A_B,
                ),
                GB300_RELU2_PREFILL_CONFIG,
            ),
        )
    return (
        _choice(
            "gb300.per_expert.decode.gab.tiny.swiglu",
            base_gemm_provider,
            gab_plan,
            GB300_DECODE_TINY_CONFIG,
        ),
        _choice(
            "gb300.per_expert.decode.gab.medium.swiglu",
            base_gemm_provider,
            gab_plan,
            GB300_DECODE_MEDIUM_CONFIG,
        ),
        _choice(
            "gb300.per_expert.decode.gab.large.swiglu",
            base_gemm_provider,
            gab_plan,
            GB300_DECODE_LARGE_CONFIG,
        ),
        _choice(
            "gb300.per_expert.prefill.b_activation.swiglu",
            base_gemm_provider,
            _build_plan(
                activation=activation,
                factor_layout=factor_layout,
                middle_family=MiddleFamily.B_ACTIVATION,
                early_overlap=EarlyOverlap.GATE_A,
                late_overlap=LateOverlap.DOWN_A_B,
            ),
            GB300_PREFILL_CONFIG,
        ),
        _choice(
            "gb300.per_expert.prefill.serial_large.swiglu",
            base_gemm_provider,
            _build_plan(activation=activation, factor_layout=factor_layout),
            GB300_LARGE_PREFILL_CONFIG,
        ),
    )


def choices_for(
    device_arch: DeviceArchitecture | str,
    factor_layout: MoeLoraFactorLayout,
    activation: ActivationFamily,
    base_gemm_provider: ProviderKey,
) -> tuple[PolicyChoice, ...]:
    try:
        architecture = DeviceArchitecture(device_arch)
    except ValueError as exc:
        raise ValueError(f"unsupported device architecture {device_arch!r}") from exc
    if not isinstance(activation, ActivationFamily):
        raise TypeError("activation must be ActivationFamily")
    if base_gemm_provider not in ("deepgemm", "cutedsl"):
        raise ValueError("base_gemm_provider must be 'deepgemm' or 'cutedsl'")
    return _choices_for(
        architecture,
        factor_layout,
        activation,
        base_gemm_provider,
    )


def select_policy(policy_input: PolicyInput) -> PolicyChoice:
    if not isinstance(policy_input, PolicyInput):
        raise TypeError("policy_input must be PolicyInput")
    choices = choices_for(
        policy_input.architecture,
        policy_input.factor_layout,
        policy_input.activation,
        policy_input.base_gemm_provider,
    )

    def named(fragment: str) -> PolicyChoice:
        matches = tuple(choice for choice in choices if fragment in choice.key)
        if len(matches) != 1:
            raise RuntimeError(f"selector fragment {fragment!r} is not unique")
        return matches[0]

    if policy_input.factor_layout == SHARED_LAYOUT:
        return (
            named("shared.decode")
            if policy_input.mode is PolicyMode.DECODE
            else named("shared.prefill")
        )

    if policy_input.architecture is DeviceArchitecture.H200:
        if policy_input.mode is PolicyMode.PREFILL:
            return named("per_expert.prefill")
        return (
            named("indexed_down_a")
            if policy_input.num_tokens <= 16
            else named("decode.grouped")
        )

    if policy_input.activation is ActivationFamily.RELU2:
        return (
            named("decode")
            if policy_input.mode is PolicyMode.DECODE
            else named("prefill")
        )
    if policy_input.mode is PolicyMode.PREFILL:
        if policy_input.active_rank >= 128 or policy_input.num_tokens >= 4096:
            return named("serial_large")
        return named("b_activation")
    if policy_input.num_tokens <= 4 and policy_input.active_rank <= 16:
        return named("gab.tiny")
    if policy_input.num_tokens <= 16:
        return named("gab.medium")
    return named("gab.large")
