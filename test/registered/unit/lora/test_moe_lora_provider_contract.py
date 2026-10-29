"""Real provider methods test admission, geometry, and finalization with GPU boundaries mocked."""

from __future__ import annotations

import ast
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


ROOT = Path(__file__).resolve().parents[4]
PROVIDER = ROOT / "python/sglang/srt/lora/moe/base_gemm_provider"


def _function(source: str, name: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"function {name!r} not found")


class TestNvFp4PreparedQuantInfo(CustomTestCase):
    def test_non_bf16_activations_fail_without_mutation(self):
        from sglang.srt.lora.moe.quant_info import MoeLoraNvFp4MarlinQuantInfo

        # FP16 is the silent case: Marlin accepts half activations while the
        # runner allocates BF16 stage buffers; FP32 fails loudly in the JIT.
        layer = SimpleNamespace(
            params_dtype=torch.float16,
            w13_weight=torch.zeros((2, 4, 512), dtype=torch.int32),
            w2_weight=torch.zeros((2, 8, 128), dtype=torch.int32),
            w13_weight_scale=torch.zeros((2, 4, 256), dtype=torch.float8_e4m3fn),
            w2_weight_scale=torch.zeros((2, 8, 64), dtype=torch.float8_e4m3fn),
            w13_weight_scale_2=torch.ones(2, dtype=torch.float16),
            w2_weight_scale_2=torch.zeros(2, dtype=torch.float16),
            moe_runner_config=SimpleNamespace(is_gated=True),
            num_local_experts=2,
            intermediate_size_per_partition=128,
            hidden_size=64,
        )
        original = vars(layer).copy()
        with self.assertRaisesRegex(NotImplementedError, "BF16 activations"):
            MoeLoraNvFp4MarlinQuantInfo.from_layer(layer)
        for name, value in original.items():
            self.assertIs(getattr(layer, name), value)


class TestMxfp4PreparedQuantInfo(CustomTestCase):
    @staticmethod
    def _layer(**overrides):
        def e8m0(*shape):
            return torch.zeros(shape, dtype=torch.uint8).view(torch.float8_e8m0fnu)

        layer = SimpleNamespace(
            params_dtype=torch.bfloat16,
            w13_weight=torch.zeros((2, 16, 512), dtype=torch.int32),
            w2_weight=torch.zeros((2, 8, 512), dtype=torch.int32),
            w13_weight_scale=e8m0(2, 8, 256),
            w2_weight_scale=e8m0(2, 4, 256),
            moe_runner_config=SimpleNamespace(is_gated=True),
            num_local_experts=2,
            intermediate_size_per_partition=128,
            hidden_size=256,
        )
        vars(layer).update(overrides)
        return layer

    def test_non_bf16_activations_fail_without_mutation(self):
        from sglang.srt.lora.moe.quant_info import MoeLoraMxfp4MarlinQuantInfo

        layer = self._layer(params_dtype=torch.float16)
        original = vars(layer).copy()
        with self.assertRaisesRegex(NotImplementedError, "BF16 activations"):
            MoeLoraMxfp4MarlinQuantInfo.from_layer(layer)
        for name, value in original.items():
            self.assertIs(getattr(layer, name), value)

    def test_binds_logical_sizes_from_prepared_tiles(self):
        from sglang.srt.lora.moe.quant_info import MoeLoraMxfp4MarlinQuantInfo

        quant_info = MoeLoraMxfp4MarlinQuantInfo.from_layer(self._layer())
        self.assertEqual(
            (
                quant_info.hidden_size,
                quant_info.intermediate_size,
                quant_info.gate_up_slices,
            ),
            (256, 128, 2),
        )

    def test_loader_padded_hidden_width_is_rejected(self):
        from sglang.srt.lora.moe.quant_info import MoeLoraMxfp4MarlinQuantInfo

        # The loader rounds the hidden width to 256; stages address real columns.
        with self.assertRaisesRegex(NotImplementedError, "hidden_size % 256"):
            MoeLoraMxfp4MarlinQuantInfo.from_layer(self._layer(hidden_size=200))


class TestVendorResolution(CustomTestCase):
    def test_family_default_serves_vendors_without_a_provider(self):
        from sglang.srt.lora.moe.base_gemm_provider import VENDORS, resolve_vendor

        self.assertEqual(resolve_vendor("mxfp4", "marlin"), "marlin")
        self.assertEqual(resolve_vendor("mxfp4", "triton"), "marlin")
        self.assertEqual(resolve_vendor("mxfp8", "marlin"), "cutedsl")
        self.assertEqual(resolve_vendor("bf16", None), VENDORS["bf16"][0])

    def test_lora_runners_never_fold_routed_scaling(self):
        from sglang.srt.layers.moe.fused_moe_triton.layer import (
            _fuses_routed_scaling_factor_in_topk,
        )
        from sglang.srt.layers.moe.utils import MoeRunnerBackend
        from sglang.srt.layers.quantization.mxfp4_marlin_moe import (
            Mxfp4MarlinMoEMethod,
        )

        # The method folds for its fused runner; the LoRA finalize owns it.
        method = object.__new__(Mxfp4MarlinMoEMethod)
        self.assertTrue(
            _fuses_routed_scaling_factor_in_topk(method, MoeRunnerBackend.MARLIN)
        )
        for backend in (
            MoeRunnerBackend.LORA_MARLIN,
            MoeRunnerBackend.LORA_CUTEDSL,
            MoeRunnerBackend.LORA_TRITON,
        ):
            self.assertFalse(_fuses_routed_scaling_factor_in_topk(method, backend))


class TestProviderGeometry(CustomTestCase):
    def test_triton_fp8_k_tiles_stay_within_scale_groups(self):
        source = _function((PROVIDER / "triton_bf16.py").read_text(), "_config_for")
        namespace = {}
        exec("from __future__ import annotations\n" + source, namespace)
        config_for = namespace["_config_for"]
        module = types.ModuleType(
            "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config"
        )
        # The FP8 GEMM loads one scale per K block: a 256-wide block over
        # [128, 128] scale groups would apply the first group's scale to its
        # second half, and upstream never clamps override or table configs.
        original = {"BLOCK_SIZE_K": 256, "USE_TMA": True}
        upstream = (dict(original), dict(original))
        module.try_get_optimal_moe_config = mock.Mock(
            return_value=(upstream[0], (upstream[1], None))
        )
        provider = SimpleNamespace(
            quant_info=SimpleNamespace(
                w13_weight=SimpleNamespace(shape=(2, 256, 128)),
                w2_weight=SimpleNamespace(shape=(2, 128, 128)),
            ),
            _config_dtype_tag="fp8_w8a8",
            _config_block_shape=[128, 128],
            _configs={},
        )
        with mock.patch.dict(sys.modules, {module.__name__: module}):
            resolved = config_for(provider, 32, 2)
            self.assertIs(config_for(provider, 32, 2), resolved)
        for actual, cached in zip(resolved, upstream):
            self.assertEqual(actual, {"BLOCK_SIZE_K": 128})
            self.assertEqual(cached, original)
            self.assertIsNot(actual, cached)
        module.try_get_optimal_moe_config.assert_called_once()


class TestMaterializedFinalize(CustomTestCase):
    """Execute the host dispatch with real CPU buffers and mocked GPU launches."""

    def test_materialized_finalize_preserves_strided_weights(self):
        from sglang.srt.lora.moe.base_gemm_provider.base import MoeBaseProvider

        finalize = types.ModuleType("sglang.kernels.ops.lora.moe.finalize")
        finalize.invoke_small_finalize = mock.Mock()
        reorder = types.ModuleType("sglang.kernels.ops.moe.ep_moe_kernels")
        reorder.post_reorder_deepgemm = mock.Mock()
        # Production hands over strided top-k weight views; post_reorder_deepgemm
        # addresses them contiguously, so even a small batch must not take the
        # gather path and must receive a contiguous copy.
        tokens = 64
        weights = torch.arange(tokens * 6).view(tokens, 6)[:, ::2]
        ids = torch.zeros((tokens, 3), dtype=torch.int64)
        output = torch.empty((tokens, 5))
        delta = torch.empty((tokens, 3, 5))
        with mock.patch.dict(
            sys.modules, {finalize.__name__: finalize, reorder.__name__: reorder}
        ):
            MoeBaseProvider().finalize(
                SimpleNamespace(pair_to_row=torch.arange(tokens * 3)),
                torch.empty((tokens * 3, 5)),
                ids,
                weights,
                2.5,
                output,
                lora_delta=delta,
            )
        finalize.invoke_small_finalize.assert_not_called()
        call = reorder.post_reorder_deepgemm.call_args
        actual = call.args[4]
        self.assertTrue(actual.is_contiguous())
        torch.testing.assert_close(actual, weights)
        self.assertIs(call.kwargs["lora_delta"], delta)


if __name__ == "__main__":
    unittest.main()
