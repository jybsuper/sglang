import argparse
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.arg_groups.overrides import _moe_runner_fusion_disable
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatcher
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.lora.backend.base_backend import BaseLoRABackend
from sglang.srt.lora.layers import FusedMoEWithLoRA
from sglang.srt.lora.sgl_lora.lora_layer import (
    _effective_sgl_lora_runner_config,
    _fp8_resident_scale_abi_violations,
    _phase1a_contract_violations,
    _use_stock_base_path,
    build_sgl_lora_quant_info,
    dispatch_sgl_lora_moe,
    validate_sgl_lora_factor_dtypes,
)
from sglang.srt.model_executor.runner_utils.capture_mode import (
    capture_lora_variant,
    get_capture_lora_variant,
    should_record_lora_graph_variants,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSglLoraExecutionSelection(unittest.TestCase):
    def test_topk_fused_routed_scale_is_not_applied_twice(self):
        config = MoeRunnerConfig(routed_scaling_factor=1.7)
        base_layer = SimpleNamespace(
            moe_runner_config=config,
            should_fuse_routed_scaling_factor_in_topk=True,
        )
        effective = _effective_sgl_lora_runner_config(base_layer)
        self.assertEqual(effective.routed_scaling_factor, 1.0)
        self.assertEqual(config.routed_scaling_factor, 1.7)

        base_layer.should_fuse_routed_scaling_factor_in_topk = False
        self.assertIs(_effective_sgl_lora_runner_config(base_layer), config)

    def test_blackwell_fp8_requires_resident_packed_scale_abi(self):
        unpacked = torch.ones((1, 1), dtype=torch.float32)
        packed = torch.ones((1, 1), dtype=torch.int32)
        packed.format_ue8m0 = True

        self.assertEqual(
            _fp8_resident_scale_abi_violations(
                unpacked, unpacked, requires_packed_ue8m0=False
            ),
            [],
        )
        self.assertEqual(
            len(
                _fp8_resident_scale_abi_violations(
                    unpacked, packed, requires_packed_ue8m0=True
                )
            ),
            1,
        )
        self.assertEqual(
            _fp8_resident_scale_abi_violations(
                packed, packed, requires_packed_ue8m0=True
            ),
            [],
        )

    def test_lora_factor_dtype_is_validated_when_pool_binds(self):
        contract = SimpleNamespace(lora_delta_dtype=torch.bfloat16)
        validate_sgl_lora_factor_dtypes(
            contract,
            gate_up_lora_a_weights=torch.empty(1, dtype=torch.bfloat16),
            gate_up_lora_b_weights=(torch.empty(1, dtype=torch.bfloat16),),
        )
        with self.assertRaisesRegex(TypeError, "gate_up_lora_a_weights"):
            validate_sgl_lora_factor_dtypes(
                contract,
                gate_up_lora_a_weights=torch.empty(1, dtype=torch.float32),
            )

    def test_up_gate_resident_bf16_layout_is_rejected(self):
        from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

        quant_method = UnquantizedFusedMoEMethod()
        quant_method.use_flashinfer_cutlass = True
        base_layer = SimpleNamespace(quant_method=quant_method)
        with self.assertRaisesRegex(NotImplementedError, r"\[Up,Gate\]"):
            build_sgl_lora_quant_info(base_layer)

    def test_nonstandard_dispatcher_is_rejected_at_attach_time(self):
        config = SimpleNamespace(
            activation="silu",
            is_gated=True,
            gemm1_alpha=None,
            gemm1_clamp_limit=None,
            swiglu_limit=None,
            apply_router_weight_on_input=False,
            no_combine=False,
            use_tp_all_gather_activation=False,
        )
        violations = _phase1a_contract_violations(
            SimpleNamespace(moe_runner_config=config, dispatcher=object())
        )
        self.assertTrue(
            any("Standard dispatch/combine ABI" in item for item in violations)
        )

    def test_no_adapter_path_uses_resident_quant_method(self):
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardCombineInput,
        )

        expected = StandardCombineInput(
            hidden_states=torch.ones((2, 4), dtype=torch.bfloat16)
        )
        dispatch_output = object()
        quant_method = SimpleNamespace(apply=Mock(return_value=expected))
        base_layer = SimpleNamespace(quant_method=quant_method)
        wrapper = SimpleNamespace(base_layer=base_layer)
        lora_info = SimpleNamespace(has_active_lora=False)

        with patch(
            "sglang.srt.model_executor.runner_utils.capture_mode.get_is_capture_mode",
            return_value=False,
        ):
            actual = dispatch_sgl_lora_moe(dispatch_output, wrapper, lora_info)

        self.assertIs(actual, expected)
        quant_method.apply.assert_called_once_with(
            layer=base_layer,
            dispatch_output=dispatch_output,
        )

    def test_no_adapter_path_honors_explicit_output_dtype(self):
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardCombineInput,
        )

        resident = StandardCombineInput(
            hidden_states=torch.ones((2, 4), dtype=torch.bfloat16)
        )
        base_layer = SimpleNamespace(
            quant_method=SimpleNamespace(apply=Mock(return_value=resident))
        )
        wrapper = SimpleNamespace(base_layer=base_layer)
        lora_info = SimpleNamespace(has_active_lora=False)

        with patch(
            "sglang.srt.model_executor.runner_utils.capture_mode.get_is_capture_mode",
            return_value=False,
        ):
            actual = dispatch_sgl_lora_moe(
                object(), wrapper, lora_info, output_dtype=torch.float32
            )

        self.assertEqual(actual.hidden_states.dtype, torch.float32)
        torch.testing.assert_close(
            actual.hidden_states, resident.hidden_states.float(), rtol=0, atol=0
        )

    def test_static_activation_fp8_is_rejected(self):
        from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod

        quant_method = object.__new__(Fp8MoEMethod)
        quant_method.with_bias = False
        quant_method.is_fp4_expert = False
        quant_method.quant_config = SimpleNamespace(activation_scheme="static")
        with self.assertRaisesRegex(NotImplementedError, "static-activation FP8"):
            build_sgl_lora_quant_info(SimpleNamespace(quant_method=quant_method))

    @staticmethod
    def _shared_expert_base(dispatcher, *, ep_size=1):
        return SimpleNamespace(
            dispatcher=dispatcher,
            moe_ep_size=ep_size,
            moe_runner_config=SimpleNamespace(
                activation="silu",
                is_gated=True,
                gemm1_alpha=None,
                gemm1_clamp_limit=None,
                swiglu_limit=None,
                apply_router_weight_on_input=False,
                no_combine=False,
                use_tp_all_gather_activation=False,
                num_fused_shared_experts=1,
            ),
        )

    def test_physical_shared_experts_are_scoped_to_supported_dispatch(self):
        unsupported = self._shared_expert_base(object())
        self.assertTrue(
            any(
                "Standard dispatch/combine ABI" in violation
                for violation in _phase1a_contract_violations(unsupported)
            )
        )

        standard = StandardDispatcher.__new__(StandardDispatcher)
        with patch(
            "sglang.srt.layers.moe.utils.uses_per_rank_fused_shared_slots",
            return_value=False,
        ):
            self.assertEqual(
                _phase1a_contract_violations(
                    self._shared_expert_base(standard, ep_size=1)
                ),
                [],
            )

        with patch(
            "sglang.srt.layers.moe.utils.uses_per_rank_fused_shared_slots",
            return_value=True,
        ):
            violations = _phase1a_contract_violations(
                self._shared_expert_base(standard, ep_size=2)
            )
        self.assertTrue(any("per-rank physical shared slots" in v for v in violations))

    def test_moe_metadata_uses_the_selected_merged_segment_bound(self):
        backend = BaseLoRABackend.__new__(BaseLoRABackend)
        backend.is_moe_lora = True
        batch_info = SimpleNamespace(
            use_cuda_graph=False,
            req_seg_indptr=None,
            req_weight_indices=None,
            num_segments=1,
            seg_indptr=torch.tensor([0, 32], dtype=torch.int32),
            weight_indices=torch.tensor([2], dtype=torch.int32),
            lora_ranks=torch.tensor([64], dtype=torch.int32),
            max_len=32,
        )
        forward_batch = SimpleNamespace(
            extend_seq_lens_cpu=[],
            forward_mode=SimpleNamespace(is_extend=lambda: False),
            batch_size=32,
        )
        adapter_enabled = torch.ones(32, dtype=torch.bool)
        mapping = torch.full((32,), 2, dtype=torch.int32)

        with patch(
            "sglang.srt.lora.backend.base_backend._compute_moe_lora_info",
            return_value=(adapter_enabled, mapping),
        ) as compute:
            result = backend._add_moe_lora_info(forward_batch, batch_info)

        self.assertEqual(compute.call_args.kwargs["max_len"], 32)
        self.assertEqual(result.moe_lora_info.max_segment_len, 32)

    def test_decode_graph_variants_are_scoped_to_sgl_lora(self):
        ordinary_decode = SimpleNamespace(is_none=lambda: True)
        speculative = SimpleNamespace(is_none=lambda: False)

        self.assertTrue(
            should_record_lora_graph_variants(
                SimpleNamespace(enable_lora=True, lora_execution_engine="sgl_lora"),
                ordinary_decode,
            )
        )
        self.assertFalse(
            should_record_lora_graph_variants(
                SimpleNamespace(enable_lora=True, lora_execution_engine="legacy"),
                ordinary_decode,
            )
        )
        self.assertFalse(
            should_record_lora_graph_variants(
                SimpleNamespace(enable_lora=True, lora_execution_engine="sgl_lora"),
                speculative,
            )
        )

    def test_capture_variant_restores_and_selects_fixed_topology(self):
        self.assertIsNone(get_capture_lora_variant())
        with capture_lora_variant("lora"):
            self.assertEqual(get_capture_lora_variant(), "lora")
            self.assertFalse(
                _use_stock_base_path(
                    has_active_lora=False,
                    capture_mode=True,
                    capture_variant=get_capture_lora_variant(),
                )
            )
            with capture_lora_variant("nolora"):
                self.assertEqual(get_capture_lora_variant(), "nolora")
                self.assertTrue(
                    _use_stock_base_path(
                        has_active_lora=False,
                        capture_mode=True,
                        capture_variant=get_capture_lora_variant(),
                    )
                )
            self.assertEqual(get_capture_lora_variant(), "lora")
        self.assertIsNone(get_capture_lora_variant())

        self.assertFalse(
            _use_stock_base_path(
                has_active_lora=True,
                capture_mode=False,
                capture_variant=None,
            )
        )

    def test_resolution_matrix(self):
        cases = (
            ("auto", "auto", "legacy", "auto", False),
            ("auto", "triton", "legacy", "triton", False),
            ("auto", "sgl_lora", "sgl_lora", "auto", True),
            ("legacy", "triton", "legacy", "triton", False),
            ("sgl_lora", "auto", "sgl_lora", "auto", True),
            ("sgl_lora", "sgl_lora", "sgl_lora", "auto", True),
            ("sgl_lora", "triton", "sgl_lora", "triton", True),
        )
        for engine, runner, expected_engine, expected_runner, expected_virtual in cases:
            with self.subTest(engine=engine, runner=runner):
                args = ServerArgs(
                    model_path="dummy",
                    lora_execution_engine=engine,
                    moe_runner_backend=runner,
                )
                self.assertEqual(args.lora_execution_engine, expected_engine)
                self.assertEqual(args.moe_runner_backend, expected_runner)
                self.assertEqual(args.lora_use_virtual_experts, expected_virtual)

    def test_engine_and_runner_alias_resolve_identically(self):
        engine_only = ServerArgs(model_path="dummy", lora_execution_engine="sgl_lora")
        runner_only = ServerArgs(model_path="dummy", moe_runner_backend="sgl_lora")
        self.assertEqual(engine_only, runner_only)

    def test_explicit_legacy_conflicts_with_runner_alias(self):
        with self.assertRaisesRegex(ValueError, "Conflicting LoRA execution"):
            ServerArgs(
                model_path="dummy",
                lora_execution_engine="legacy",
                moe_runner_backend="sgl_lora",
            )

    def test_sgl_lora_is_not_a_speculative_runner(self):
        with self.assertRaisesRegex(ValueError, "not a speculative draft"):
            ServerArgs(
                model_path="dummy",
                speculative_moe_runner_backend="sgl_lora",
            )

    def test_cli_flags(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)

        cases = (
            (["--lora-execution-engine", "sgl_lora"], False),
            (["--moe-runner-backend", "sgl_lora"], False),
            (
                ["--moe-runner-backend", "sgl_lora", "--enable-lora-two-stream"],
                True,
            ),
        )
        for extra_args, expected_two_stream in cases:
            with self.subTest(extra_args=extra_args):
                parsed = parser.parse_args(["--model-path", "dummy", *extra_args])
                args = ServerArgs.from_cli_args(parsed)
                self.assertEqual(args.lora_execution_engine, "sgl_lora")
                self.assertEqual(args.moe_runner_backend, "auto")
                self.assertTrue(args.lora_use_virtual_experts)
                self.assertEqual(args.enable_lora_two_stream, expected_two_stream)

    def test_two_stream_flag_does_not_select_engine(self):
        args = ServerArgs(model_path="dummy", enable_lora_two_stream=True)
        self.assertEqual(args.lora_execution_engine, "legacy")

    def test_legacy_path_does_not_import_sgl_lora(self):
        repo_root = Path(__file__).resolve().parents[4]
        python_path = str(repo_root / "python")
        if os.environ.get("PYTHONPATH"):
            python_path += os.pathsep + os.environ["PYTHONPATH"]
        code = """
import sys
from sglang.srt.server_args import ServerArgs
import sglang.srt.lora.layers

args = ServerArgs(model_path="dummy")
assert args.lora_execution_engine == "legacy"
prefix = "sglang.srt.lora.sgl_lora"
loaded = sorted(name for name in sys.modules if name == prefix or name.startswith(prefix + "."))
assert not loaded, loaded
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = python_path
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_alias_is_not_a_runtime_backend_and_engine_preserves_fusion(self):
        with self.assertRaises(ValueError):
            MoeRunnerBackend("sgl_lora")
        self.assertEqual(
            _moe_runner_fusion_disable(
                SimpleNamespace(
                    moe_runner_backend="auto", lora_execution_engine="sgl_lora"
                )
            ),
            {},
        )

    def test_fused_moe_dispatch_selects_engine_independent_of_base_runner(self):
        base_layer = SimpleNamespace(
            quant_method=SimpleNamespace(runner=None),
            moe_runner_config=SimpleNamespace(gemm1_alpha=None),
            dispatcher=object(),
            num_local_experts=2,
            num_experts=2,
            should_fuse_routed_scaling_factor_in_topk=False,
        )
        lora_backend = SimpleNamespace(is_moe_lora=False)

        with (
            patch(
                "sglang.srt.layers.moe.utils.get_moe_runner_backend",
                return_value=MoeRunnerBackend.TRITON,
            ),
            patch("sglang.srt.lora.sgl_lora.lora_layer.init_sgl_lora_moe") as init_sgl,
        ):
            layer = FusedMoEWithLoRA(
                base_layer,
                lora_backend,
                lora_execution_engine="sgl_lora",
                enable_lora_two_stream=True,
            )

        init_sgl.assert_called_once_with(layer, base_layer)
        self.assertEqual(layer._lora_runner_backend, MoeRunnerBackend.TRITON)
        self.assertTrue(layer._sgl_lora_two_stream)


if __name__ == "__main__":
    unittest.main()
