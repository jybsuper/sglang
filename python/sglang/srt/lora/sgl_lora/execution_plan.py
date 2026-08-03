"""Typed, policy-free execution plans for the BF16 MoE-LoRA pipeline.

This module describes *what* one forward executes.  It deliberately contains
no CUDA imports, launch configuration, device thresholds, or selector policy.
The selector and runner can therefore validate a whole-pipeline plan
before allocating a workspace or launching any kernel.

An A kernel writes a rank bridge and the matching B kernel reads it.  The
bridge contract is explicit at each site:

* ``PAIR_MAJOR`` is one row per routed ``(token, expert)`` pair.
* ``TOKEN_MAJOR`` is the shared-outer gate/up form, one row per token.

Fusion is represented by ownership, not by pretending that a consumed stage
still runs independently.  For example, ``B_ACTIVATION`` carries the
``consumed_gate_b`` factor contract and requires ``plan.gate_b is None``.
This makes illegal combinations such as gate-A+B overlap plus B+activation
fusion fail before CUDA work.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FactorSite(str, Enum):
    GATE_UP = "gate_up"
    DOWN = "down"


class FactorOwnership(str, Enum):
    """How many copies of one LoRA factor an adapter owns."""

    PER_EXPERT = "per_expert"
    SHARED_OUTER = "shared_outer"


class FactorLayout(str, Enum):
    """Logical row layout of the rank bridge between A and B."""

    PAIR_MAJOR = "pair_major"
    TOKEN_MAJOR = "token_major"


class RouteRequirement(str, Enum):
    """Route representations consumed by a whole execution plan.

    ``RAW`` materializes no derived metadata; consumers derive keys directly
    from the source tensors.  The other values are distinct products and may
    coexist.  In particular, a shared-outer forward can require both aligned
    per-expert and aligned shared-outer pair plans.
    """

    RAW = "raw"
    ALIGNED_PER_EXPERT = "aligned_per_expert"
    ALIGNED_SHARED_OUTER = "aligned_shared_outer"
    SHARED_TOKEN_PLAN = "shared_token_plan"


class RouteBuilderFamily(str, Enum):
    """Implementation used to build the required route products."""

    STANDARD = "standard"
    JOINT_SHARED_OUTER = "joint_shared_outer"


class LoraAFamily(str, Enum):
    GROUPED = "grouped"
    INDEXED = "indexed"
    TOKEN_DEDUP_GROUPED = "token_dedup_grouped"


class ActivationFamily(str, Enum):
    SWIGLU = "swiglu"
    RELU2 = "relu2"


class MiddleFamily(str, Enum):
    MATERIALIZED = "materialized"
    B_ACTIVATION = "b_activation"


class FinalizeFamily(str, Enum):
    MATERIALIZED = "materialized"
    SHARED_RANK_REDUCE = "shared_rank_reduce"


class EarlyOverlap(str, Enum):
    NONE = "none"
    GATE_A = "gate_a"
    GATE_A_B = "gate_a_b"


class LateOverlap(str, Enum):
    NONE = "none"
    DOWN_A = "down_a"
    DOWN_B = "down_b"
    DOWN_A_B = "down_a_b"
    SHARED_FINALIZE = "shared_finalize"


def _require_enum(value: object, enum_type: type[Enum], field: str) -> None:
    if not isinstance(value, enum_type):
        raise TypeError(
            f"{field} must be {enum_type.__name__}, got {type(value).__name__}"
        )


def _aligned_requirement(ownership: FactorOwnership) -> RouteRequirement:
    if ownership is FactorOwnership.PER_EXPERT:
        return RouteRequirement.ALIGNED_PER_EXPERT
    return RouteRequirement.ALIGNED_SHARED_OUTER


@dataclass(frozen=True, slots=True)
class FactorContract:
    """The factor and bridge contract of one logical A or B stage."""

    site: FactorSite
    ownership: FactorOwnership
    layout: FactorLayout

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> FactorContract:
        _require_enum(self.site, FactorSite, "site")
        _require_enum(self.ownership, FactorOwnership, "ownership")
        _require_enum(self.layout, FactorLayout, "layout")
        if self.site is FactorSite.DOWN and self.layout is FactorLayout.TOKEN_MAJOR:
            raise ValueError(
                "the down bridge is inherently pair-major: each routed expert "
                "produces a different activation"
            )
        return self


@dataclass(frozen=True, slots=True)
class MoeLoraFactorLayout:
    """Resident weight ownership independent of execution policy.

    The serving loader exposes one ``shared_outer`` flag.  Keeping the two
    physical ownership sites explicit makes the resident contract unambiguous.
    """

    gate_up_a: FactorOwnership
    down_b: FactorOwnership

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def serving(cls, shared_outer: bool) -> MoeLoraFactorLayout:
        if not isinstance(shared_outer, bool):
            raise TypeError(
                f"shared_outer must be bool, got {type(shared_outer).__name__}"
            )
        ownership = (
            FactorOwnership.SHARED_OUTER if shared_outer else FactorOwnership.PER_EXPERT
        )
        return cls(gate_up_a=ownership, down_b=ownership)

    def validate(self) -> MoeLoraFactorLayout:
        _require_enum(self.gate_up_a, FactorOwnership, "gate_up_a")
        _require_enum(self.down_b, FactorOwnership, "down_b")
        return self


@dataclass(frozen=True, slots=True)
class LoraASpec:
    """One standalone LoRA-A execution stage."""

    site: FactorSite
    family: LoraAFamily
    ownership: FactorOwnership = FactorOwnership.PER_EXPERT
    output_layout: FactorLayout = FactorLayout.PAIR_MAJOR

    def __post_init__(self) -> None:
        self.validate()

    @property
    def contract(self) -> FactorContract:
        return FactorContract(self.site, self.ownership, self.output_layout)

    def validate(self) -> LoraASpec:
        _require_enum(self.site, FactorSite, "site")
        _require_enum(self.family, LoraAFamily, "family")
        _require_enum(self.ownership, FactorOwnership, "ownership")
        _require_enum(self.output_layout, FactorLayout, "output_layout")
        self.contract.validate()

        if (
            self.site is FactorSite.DOWN
            and self.ownership is not FactorOwnership.PER_EXPERT
        ):
            raise ValueError(
                "down A is always per-expert; only gate/up A may be shared-outer"
            )
        if self.family is LoraAFamily.GROUPED:
            if self.output_layout is not FactorLayout.PAIR_MAJOR:
                raise ValueError("grouped A writes a pair-major bridge")
        elif self.family is LoraAFamily.INDEXED:
            if self.ownership is not FactorOwnership.PER_EXPERT:
                raise ValueError("indexed A is qualified only for per-expert factors")
            if self.output_layout is not FactorLayout.PAIR_MAJOR:
                raise ValueError("indexed A writes a pair-major bridge")
        else:
            if self.site is not FactorSite.GATE_UP:
                raise ValueError(f"{self.family.value} is a shared gate/up-A family")
            if self.ownership is not FactorOwnership.SHARED_OUTER:
                raise ValueError(
                    f"{self.family.value} requires shared-outer A ownership"
                )
            if self.output_layout is not FactorLayout.TOKEN_MAJOR:
                raise ValueError(f"{self.family.value} writes a token-major bridge")
        return self

    def route_requirements(self) -> frozenset[RouteRequirement]:
        if self.family is LoraAFamily.INDEXED:
            return frozenset((RouteRequirement.RAW,))
        if self.family is LoraAFamily.TOKEN_DEDUP_GROUPED:
            return frozenset((RouteRequirement.SHARED_TOKEN_PLAN,))
        return frozenset((_aligned_requirement(self.ownership),))


@dataclass(frozen=True, slots=True)
class LoraBSpec:
    """One standalone LoRA-B execution stage."""

    site: FactorSite
    ownership: FactorOwnership = FactorOwnership.PER_EXPERT
    input_layout: FactorLayout = FactorLayout.PAIR_MAJOR

    def __post_init__(self) -> None:
        self.validate()

    @property
    def contract(self) -> FactorContract:
        return FactorContract(self.site, self.ownership, self.input_layout)

    def validate(self) -> LoraBSpec:
        _require_enum(self.site, FactorSite, "site")
        _require_enum(self.ownership, FactorOwnership, "ownership")
        _require_enum(self.input_layout, FactorLayout, "input_layout")
        self.contract.validate()

        if (
            self.site is FactorSite.GATE_UP
            and self.ownership is not FactorOwnership.PER_EXPERT
        ):
            raise ValueError(
                "gate/up B is always per-expert; only down B may be shared-outer"
            )
        if (
            self.input_layout is FactorLayout.TOKEN_MAJOR
            and self.site is not FactorSite.GATE_UP
        ):
            raise ValueError("a token-major B input exists only at gate/up")
        return self

    def route_requirements(self) -> frozenset[RouteRequirement]:
        return frozenset((_aligned_requirement(self.ownership),))


@dataclass(frozen=True, slots=True)
class MiddleSpec:
    """Activation boundary and any A/B stages fused into it.

    A consumed factor names its data contract, while ``family`` names the
    fused implementation.  It is intentionally not an executable
    ``LoraASpec``/``LoraBSpec`` because the standalone family does not run.
    """

    family: MiddleFamily
    activation: ActivationFamily
    consumed_gate_b: FactorContract | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> MiddleSpec:
        _require_enum(self.family, MiddleFamily, "family")
        _require_enum(self.activation, ActivationFamily, "activation")
        if self.consumed_gate_b is not None:
            self.consumed_gate_b.validate()
            if self.consumed_gate_b.site is not FactorSite.GATE_UP:
                raise ValueError("consumed_gate_b must describe the gate/up site")
            if self.consumed_gate_b.ownership is not FactorOwnership.PER_EXPERT:
                raise ValueError("consumed gate/up B must be per-expert")
        expected_gate_b = self.family is MiddleFamily.B_ACTIVATION
        if (self.consumed_gate_b is not None) != expected_gate_b:
            raise ValueError(
                f"middle family {self.family.value} "
                f"{'requires' if expected_gate_b else 'does not consume'} gate B"
            )
        return self

    def route_requirements(self) -> frozenset[RouteRequirement]:
        if self.consumed_gate_b is None:
            return frozenset()
        return frozenset((_aligned_requirement(self.consumed_gate_b.ownership),))


@dataclass(frozen=True, slots=True)
class FinalizeSpec:
    """Final combine family and an optional down-B stage consumed by it."""

    family: FinalizeFamily
    consumed_down_b: FactorContract | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> FinalizeSpec:
        _require_enum(self.family, FinalizeFamily, "family")
        consumes_down_b = self.family is not FinalizeFamily.MATERIALIZED
        if (self.consumed_down_b is not None) != consumes_down_b:
            raise ValueError(
                f"finalize family {self.family.value} "
                f"{'requires' if consumes_down_b else 'does not consume'} down B"
            )
        if self.consumed_down_b is not None:
            self.consumed_down_b.validate()
            if self.consumed_down_b.site is not FactorSite.DOWN:
                raise ValueError("consumed_down_b must describe the down site")
        if self.family is FinalizeFamily.SHARED_RANK_REDUCE:
            consumed_down_b = self.consumed_down_b
            if consumed_down_b is None:
                raise ValueError(f"{self.family.value} requires down B")
            if consumed_down_b.ownership is not FactorOwnership.SHARED_OUTER:
                raise ValueError(
                    f"{self.family.value} requires shared-outer down-B ownership"
                )
        return self

    def route_requirements(self) -> frozenset[RouteRequirement]:
        if self.family is FinalizeFamily.MATERIALIZED:
            return frozenset()
        # The shared-rank finalizer derives its fixed-top-k keys from the raw route.
        return frozenset((RouteRequirement.RAW,))


@dataclass(frozen=True, slots=True, kw_only=True)
class MoeLoraExecutionPlan:
    """One immutable whole-pipeline MoE-LoRA execution strategy."""

    gate_a: LoraASpec
    middle: MiddleSpec
    finalize: FinalizeSpec
    gate_b: LoraBSpec | None = None
    down_a: LoraASpec | None = None
    down_b: LoraBSpec | None = None
    early_overlap: EarlyOverlap = EarlyOverlap.NONE
    late_overlap: LateOverlap = LateOverlap.NONE
    route_builder: RouteBuilderFamily = RouteBuilderFamily.STANDARD

    def __post_init__(self) -> None:
        self.validate()

    def _gate_b_contract(self) -> FactorContract:
        if self.gate_b is not None:
            return self.gate_b.contract
        consumed = self.middle.consumed_gate_b
        if consumed is None:
            raise ValueError("the execution plan has no gate-B owner")
        return consumed

    def _down_a_contract(self) -> FactorContract:
        if self.down_a is None:
            raise ValueError("the execution plan has no down-A owner")
        return self.down_a.contract

    def _down_b_contract(self) -> FactorContract:
        if self.down_b is not None:
            return self.down_b.contract
        consumed = self.finalize.consumed_down_b
        if consumed is None:
            raise ValueError("the execution plan has no down-B owner")
        return consumed

    def validate(self) -> MoeLoraExecutionPlan:
        if not isinstance(self.gate_a, LoraASpec):
            raise TypeError("gate_a must be LoraASpec")
        if not isinstance(self.middle, MiddleSpec):
            raise TypeError("middle must be MiddleSpec")
        if not isinstance(self.finalize, FinalizeSpec):
            raise TypeError("finalize must be FinalizeSpec")
        for field, value, expected in (
            ("gate_b", self.gate_b, LoraBSpec),
            ("down_a", self.down_a, LoraASpec),
            ("down_b", self.down_b, LoraBSpec),
        ):
            if value is not None and not isinstance(value, expected):
                raise TypeError(f"{field} must be {expected.__name__} or None")
        _require_enum(self.early_overlap, EarlyOverlap, "early_overlap")
        _require_enum(self.late_overlap, LateOverlap, "late_overlap")
        _require_enum(self.route_builder, RouteBuilderFamily, "route_builder")

        self.gate_a.validate()
        self.middle.validate()
        self.finalize.validate()
        if self.gate_b is not None:
            self.gate_b.validate()
        if self.down_a is not None:
            self.down_a.validate()
        if self.down_b is not None:
            self.down_b.validate()

        if self.gate_a.site is not FactorSite.GATE_UP:
            raise ValueError("gate_a must describe the gate/up site")
        if self.gate_b is not None and self.gate_b.site is not FactorSite.GATE_UP:
            raise ValueError("gate_b must describe the gate/up site")
        if self.down_a is not None and self.down_a.site is not FactorSite.DOWN:
            raise ValueError("down_a must describe the down site")
        if self.down_b is not None and self.down_b.site is not FactorSite.DOWN:
            raise ValueError("down_b must describe the down site")

        gate_b_consumed = self.middle.consumed_gate_b is not None
        if gate_b_consumed == (self.gate_b is not None):
            raise ValueError(
                "gate B must have exactly one owner: standalone gate_b or middle"
            )
        if self.down_a is None:
            raise ValueError("down A must be owned by standalone down_a")
        down_b_consumed = self.finalize.consumed_down_b is not None
        if down_b_consumed == (self.down_b is not None):
            raise ValueError(
                "down B must have exactly one owner: standalone down_b or finalize"
            )

        gate_b_contract = self._gate_b_contract()
        down_a_contract = self._down_a_contract()
        down_b_contract = self._down_b_contract()
        if self.gate_a.output_layout is not gate_b_contract.layout:
            raise ValueError("gate A output layout must match the gate B input layout")
        if down_a_contract.layout is not down_b_contract.layout:
            raise ValueError("down A output layout must match the down B input layout")

        if self.early_overlap is EarlyOverlap.GATE_A_B and self.gate_b is None:
            raise ValueError(
                "gate-A+B overlap requires standalone gate B; the middle owns it"
            )
        if (
            self.late_overlap
            in (
                LateOverlap.DOWN_A,
                LateOverlap.DOWN_A_B,
            )
            and self.down_a is None
        ):
            raise ValueError(
                f"{self.late_overlap.value} overlap requires standalone down A"
            )
        if (
            self.late_overlap
            in (
                LateOverlap.DOWN_B,
                LateOverlap.DOWN_A_B,
            )
            and self.down_b is None
        ):
            raise ValueError(
                f"{self.late_overlap.value} overlap requires standalone down B"
            )
        if self.late_overlap is LateOverlap.SHARED_FINALIZE:
            if self.finalize.family is not FinalizeFamily.SHARED_RANK_REDUCE:
                raise ValueError(
                    "shared-finalize overlap requires a shared finalize family"
                )
            if self.down_a is None:
                raise ValueError(
                    "shared-finalize overlap starts after standalone down A"
                )

        # Indexed A is retained only at down-A, where the selector uses it for
        # the small-decode frontier while all other sites keep aligned kernels.
        indexed_gate_a = self.gate_a.family is LoraAFamily.INDEXED
        if indexed_gate_a:
            raise ValueError("indexed A is evidence-qualified only at the down-A site")

        requirements = self._route_requirements_unchecked()
        if self.route_builder is RouteBuilderFamily.JOINT_SHARED_OUTER:
            needed = {
                RouteRequirement.ALIGNED_PER_EXPERT,
                RouteRequirement.ALIGNED_SHARED_OUTER,
            }
            if not needed.issubset(requirements):
                raise ValueError(
                    "the joint shared-outer route builder requires both aligned "
                    "per-expert and aligned shared-outer pair plans"
                )
        return self

    def _route_requirements_unchecked(self) -> frozenset[RouteRequirement]:
        requirements: set[RouteRequirement] = set()
        for stage in (self.gate_a, self.gate_b, self.down_a, self.down_b):
            if stage is not None:
                requirements.update(stage.route_requirements())
        requirements.update(self.middle.route_requirements())
        requirements.update(self.finalize.route_requirements())
        return frozenset(requirements)

    def route_requirements(self) -> frozenset[RouteRequirement]:
        """Return the exact union of route products consumed by this plan."""

        # The plan and every nested stage are frozen dataclasses, and
        # __post_init__ validates the complete dependency graph. Re-running
        # that proof at each call site would charge immutable plan validation
        # multiple times per layer/forward.
        return self._route_requirements_unchecked()

    def validate_factor_layout(
        self, layout: MoeLoraFactorLayout
    ) -> MoeLoraExecutionPlan:
        """Validate plan identity against the resident A/B weight layout."""

        if not isinstance(layout, MoeLoraFactorLayout):
            raise TypeError("layout must be MoeLoraFactorLayout")
        layout.validate()
        if self.gate_a.ownership is not layout.gate_up_a:
            raise ValueError(
                "plan gate-A ownership does not match resident gate/up-A weights"
            )
        if self._down_b_contract().ownership is not layout.down_b:
            raise ValueError(
                "plan down-B ownership does not match resident down-B weights"
            )
        return self
