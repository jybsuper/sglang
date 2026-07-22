import gzip
import json
import tempfile
import unittest
from pathlib import Path

from benchmark.kernels.lora_moe.analyze_e2e_model import (
    compare,
    summarize_bench,
    summarize_correctness,
    summarize_profiles,
)


class TestMoeLoraE2EAnalysis(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _write_results(self, prefix: str, values: list[float]) -> None:
        for repetition, value in enumerate(values, start=1):
            row = {
                "run_name": f"test_{prefix}_{repetition}",
                "batch_size": 1,
                "input_len": 128,
                "output_len": 32,
                "latency": value,
                "last_ttft": value,
                "input_throughput": 100.0 / value,
                "output_throughput": 50.0 / value,
                "overall_throughput": 75.0 / value,
            }
            (self.root / f"{prefix}_r{repetition}.jsonl").write_text(
                json.dumps(row) + "\n"
            )

    def test_repeated_benchmark_summary_and_comparison(self):
        self._write_results("base", [1.0, 2.0, 3.0])
        self._write_results("lora", [2.0, 3.0, 4.0])
        summary = summarize_bench(self.root)
        self.assertEqual(summary["cells"]["base_bs1"]["latency"]["median"], 2.0)
        self.assertEqual(summary["cells"]["base_bs1"]["repetitions"], 3)
        delta = compare(summary, summary)
        self.assertEqual(
            delta["lora_bs1"]["output_throughput"][
                "sgl_minus_control_percent"
            ],
            0.0,
        )

    def test_transition_correctness_uses_token_ids(self):
        payloads = {
            "base_before": {"output_ids": [1, 2]},
            "adapter": {"output_ids": [3, 4]},
            "base_after": {"output_ids": [1, 2]},
            "mixed": [{"output_ids": [1, 2]}, {"output_ids": [3, 4]}],
        }
        for name, payload in payloads.items():
            (self.root / f"{name}.json").write_text(json.dumps(payload))
        summary = summarize_correctness(self.root)
        self.assertTrue(summary["passed"])
        self.assertTrue(summary["mixed_adapter_ids_equal"])

    def test_profile_summary_retains_crash_and_kernel_counts(self):
        profile_dir = self.root / "profiles" / "run"
        profile_dir.mkdir(parents=True)
        trace = {
            "traceEvents": [
                {"cat": "kernel", "name": "_chunked_lora_expand_kernel", "dur": 3},
                {"cat": "kernel", "name": "_chunked_lora_expand_kernel", "dur": 5},
                {"cat": "cpu_op", "name": "ignored", "dur": 99},
            ]
        }
        trace_path = profile_dir / "test-DECODE.trace.json.gz"
        with gzip.open(trace_path, "wt") as handle:
            json.dump(trace, handle)
        (self.root / "profile_lora.log").write_text(
            "Response ended prematurely; scheduler_0 crashed with exit code -11"
        )
        summary = summarize_profiles(self.root, top=10)
        decode = summary["traces"]["decode"][0]
        self.assertEqual(decode["kernel_time_us_sum"], 8.0)
        self.assertEqual(decode["kernel_launches"], 2)
        self.assertIn("profile_lora.log", summary["profile_crash_markers"])


if __name__ == "__main__":
    unittest.main()
