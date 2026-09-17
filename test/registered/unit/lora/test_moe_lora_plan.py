"""CPU tests for the typed, whole-pipeline MoE-LoRA execution contract."""

from __future__ import annotations

import dataclasses
import unittest

import pydantic

from sglang.srt.lora.moe.plan import (
    ActFamily,
    ActivationFn,
    ActSpec,
    AFamily,
    ASpec,
    BFamily,
    BridgeLayout,
    BSpec,
    DownOverlap,
    FinalizeFamily,
    FinalizeSpec,
    GateUpOverlap,
    MoePlan,
    RouteBuilderFamily,
    RouteRequirement,
    Site,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _a(
    site: Site,
    family: AFamily = AFamily.GROUPED,
    is_shared_outer: bool = False,
    layout: BridgeLayout = BridgeLayout.PAIR_MAJOR,
) -> ASpec:
    return ASpec(site, family, is_shared_outer, layout)


def _b(
    site: Site,
    family: BFamily = BFamily.GROUPED,
    is_shared_outer: bool = False,
    layout: BridgeLayout = BridgeLayout.PAIR_MAJOR,
) -> BSpec:
    return BSpec(site, family, is_shared_outer, layout)


def _plan(**changes) -> MoePlan:
    values = {
        "gate_up_a": _a(Site.GATE_UP),
        "gate_up_b": _b(Site.GATE_UP),
        "act": ActSpec(ActFamily.MATERIALIZED, ActivationFn.SILU),
        "down_a": _a(Site.DOWN),
        "down_b": _b(Site.DOWN),
        "finalize": FinalizeSpec(FinalizeFamily.MATERIALIZED),
    }
    values.update(changes)
    return MoePlan(**values)


SERIAL_MATERIALIZED_REFERENCE = _plan()


class TestFactorAndKernelSpecs(CustomTestCase):
    def test_specs_reject_coercion_and_unknown_fields(self):
        for changes in (
            {"is_shared_outer": 1},
            {"site": "gate_up"},
            {"nonsense": True},
        ):
            with (
                self.subTest(changes=changes),
                self.assertRaises(pydantic.ValidationError),
            ):
                ASpec(**{"site": Site.GATE_UP, "family": AFamily.GROUPED, **changes})

    def test_a_rejects_unqualified_family_site_ownership_and_layout(self):
        invalid = (
            lambda: _a(
                Site.DOWN,
                AFamily.TOKEN_GROUPED,
                True,
                BridgeLayout.TOKEN_MAJOR,
            ),
            lambda: _a(
                Site.GATE_UP,
                AFamily.TOKEN_GROUPED,
                True,
                BridgeLayout.PAIR_MAJOR,
            ),
            lambda: _a(
                Site.GATE_UP,
                AFamily.GROUPED,
                layout=BridgeLayout.TOKEN_MAJOR,
            ),
        )
        for construct in invalid:
            with self.subTest(construct=construct), self.assertRaises(ValueError):
                construct()

        # Indexed A supports both ownership modes, but requires pair-major output.
        for site, shared in (
            (Site.GATE_UP, False),
            (Site.GATE_UP, True),
            (Site.DOWN, True),
        ):
            with self.subTest(site=site, shared=shared):
                spec = _a(site, AFamily.PER_ROW, shared)
                self.assertIs(spec.output_layout, BridgeLayout.PAIR_MAJOR)
        with self.assertRaises(ValueError):
            _a(Site.DOWN, AFamily.PER_ROW, layout=BridgeLayout.TOKEN_MAJOR)

    def test_b_rejects_down_token_layout(self):
        with self.assertRaisesRegex(ValueError, "down bridge"):
            _b(
                Site.DOWN,
                BFamily.GROUPED,
                layout=BridgeLayout.TOKEN_MAJOR,
            )

    def test_shared_finalize_requires_shared_ownership(self):
        with self.assertRaisesRegex(ValueError, "shared-outer"):
            FinalizeSpec(FinalizeFamily.SHARED_TOKEN_DELTA)
        self.assertTrue(
            FinalizeSpec(
                FinalizeFamily.SHARED_TOKEN_DELTA, is_shared_outer=True
            ).is_shared_outer
        )


class TestWholePipelineValidation(CustomTestCase):
    def test_reference_is_immutable_and_aligned(self):
        self.assertEqual(
            SERIAL_MATERIALIZED_REFERENCE.route_requirements(),
            frozenset((RouteRequirement.ALIGNED_PER_EXPERT,)),
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            SERIAL_MATERIALIZED_REFERENCE.gate_up_b = None  # type: ignore[misc]

    def test_every_required_act_and_finalize_composition_constructs(self):
        plans = (
            _plan(
                gate_up_b=None,
                act=ActSpec(ActFamily.B_ACTIVATION, ActivationFn.SILU),
            ),
            _plan(
                gate_up_b=None,
                act=ActSpec(ActFamily.B_ACTIVATION, ActivationFn.SILU),
                down_b=None,
                finalize=FinalizeSpec(
                    FinalizeFamily.SHARED_TOKEN_DELTA, is_shared_outer=True
                ),
            ),
        )
        for plan in plans:
            requirements = {RouteRequirement.ALIGNED_PER_EXPERT}
            if plan.down_b is None:
                requirements.update(
                    (
                        RouteRequirement.RAW_SHARED_OUTER,
                        RouteRequirement.SHARED_TOKEN_PLAN,
                    )
                )
            self.assertEqual(plan.route_requirements(), frozenset(requirements))

    def test_exactly_one_stage_owner_is_required(self):
        with self.assertRaisesRegex(ValueError, "exactly one owner"):
            _plan(act=ActSpec(ActFamily.B_ACTIVATION, ActivationFn.SILU))
        with self.assertRaisesRegex(ValueError, "exactly one owner"):
            _plan(gate_up_b=None)
        with self.assertRaises(pydantic.ValidationError):
            _plan(down_a=None)  # required field: down A always runs standalone
        with self.assertRaisesRegex(ValueError, "exactly one owner"):
            _plan(
                finalize=FinalizeSpec(
                    FinalizeFamily.SHARED_TOKEN_DELTA, is_shared_outer=True
                )
            )

    def test_gate_up_bridge_layouts_must_match(self):
        token_gate_up_a = _a(
            Site.GATE_UP,
            AFamily.TOKEN_GROUPED,
            True,
            BridgeLayout.TOKEN_MAJOR,
        )
        with self.assertRaisesRegex(ValueError, "gate/up A output layout"):
            _plan(gate_up_a=token_gate_up_a)

    def test_overlaps_reject_consumed_stages(self):
        with self.assertRaisesRegex(ValueError, "gate/up-A\\+B"):
            _plan(
                gate_up_b=None,
                act=ActSpec(ActFamily.B_ACTIVATION, ActivationFn.SILU),
                gate_up_overlap=GateUpOverlap.GATE_UP_A_B,
            )
        with self.assertRaisesRegex(ValueError, "standalone down B"):
            _plan(
                down_b=None,
                finalize=FinalizeSpec(
                    FinalizeFamily.SHARED_TOKEN_DELTA, is_shared_outer=True
                ),
                down_overlap=DownOverlap.DOWN_A_B,
            )


class TestRouteRequirementUnion(CustomTestCase):
    def test_routes_include_every_stage_requirement(self):
        aligned = RouteRequirement.ALIGNED_PER_EXPERT
        raw = RouteRequirement.RAW_PER_EXPERT
        shared_raw = RouteRequirement.RAW_SHARED_OUTER
        token = RouteRequirement.SHARED_TOKEN_PLAN
        cases = (
            ({"gate_up_a": _a(Site.GATE_UP, AFamily.PER_ROW)}, {raw, aligned}),
            ({"down_a": _a(Site.DOWN, AFamily.PER_ROW)}, {raw, aligned}),
            (
                {
                    "gate_up_a": _a(
                        Site.GATE_UP,
                        AFamily.TOKEN_GROUPED,
                        True,
                        BridgeLayout.TOKEN_MAJOR,
                    ),
                    "gate_up_b": _b(Site.GATE_UP, layout=BridgeLayout.TOKEN_MAJOR),
                },
                {token, aligned},
            ),
            (
                {
                    "down_b": _b(Site.DOWN, is_shared_outer=True),
                    "route_builder": RouteBuilderFamily.PARALLEL_SHARED_OUTER,
                },
                {aligned, RouteRequirement.ALIGNED_SHARED_OUTER},
            ),
            (
                {
                    "down_b": None,
                    "finalize": FinalizeSpec(FinalizeFamily.SHARED_ONE_PASS, True),
                },
                {aligned, shared_raw},
            ),
            (
                {
                    "down_b": None,
                    "finalize": FinalizeSpec(FinalizeFamily.SHARED_TOKEN_DELTA, True),
                },
                {aligned, shared_raw, token},
            ),
        )
        for changes, expected in cases:
            with self.subTest(changes=changes):
                self.assertEqual(
                    _plan(**changes).route_requirements(), frozenset(expected)
                )


class TestTokenDenseBridgeOwnership(CustomTestCase):
    def test_token_dense_a_requires_a_standalone_per_row_gate_up_b(self):
        token_dense_a = _a(
            Site.GATE_UP, AFamily.TOKEN_DENSE, True, BridgeLayout.TOKEN_MAJOR
        )
        _plan(
            gate_up_a=token_dense_a,
            gate_up_b=_b(Site.GATE_UP, BFamily.PER_ROW, True, BridgeLayout.TOKEN_MAJOR),
        )
        with self.assertRaises(ValueError):
            _plan(
                gate_up_a=token_dense_a,
                gate_up_b=_b(
                    Site.GATE_UP, BFamily.GROUPED, True, BridgeLayout.TOKEN_MAJOR
                ),
            )
        with self.assertRaises(ValueError):
            _plan(
                gate_up_a=token_dense_a,
                gate_up_b=None,
                act=ActSpec(ActFamily.B_ACTIVATION, ActivationFn.SILU),
            )


if __name__ == "__main__":
    unittest.main()
