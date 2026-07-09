import argparse
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.overrides import _moe_runner_fusion_disable
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.lora.layers import FusedMoEWithLoRA
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSglLoraExecutionSelection(unittest.TestCase):
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

    def test_alias_is_not_a_runtime_backend_and_engine_disables_fusion(self):
        with self.assertRaises(ValueError):
            MoeRunnerBackend("sgl_lora")
        self.assertEqual(
            _moe_runner_fusion_disable(
                SimpleNamespace(
                    moe_runner_backend="auto", lora_execution_engine="sgl_lora"
                )
            ),
            {"disable_shared_experts_fusion": True},
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
