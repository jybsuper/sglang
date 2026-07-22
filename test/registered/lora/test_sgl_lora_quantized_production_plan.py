import sys

import pytest
import torch

from benchmark.kernels.lora_moe.bench_moe_pipeline import _single_rank_runtime
from benchmark.kernels.lora_moe.bench_quantized_moe_pipeline import (
    QuantizedPipelineFixture,
    _capture,
    _max_abs,
)
from sglang.srt.lora.sgl_lora.execution_plan import MoeLoraExecutionPath
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")


@pytest.mark.parametrize("provider", ("fp8", "marlin", "nvfp4"))
def test_quantized_provider_runs_complete_production_plan_eager_and_graph(provider):
    major, _minor = torch.cuda.get_device_capability()
    if provider == "nvfp4" and major < 10:
        pytest.skip("native NVFP4 provider requires Blackwell")
    if provider == "marlin" and major < 9:
        pytest.skip("Marlin provider requires SM90+")

    with _single_rank_runtime():
        for rank in (16, 128):
            fixture = QuantizedPipelineFixture(
                provider_name=provider,
                tokens=8,
                experts=4,
                top_k=2,
                # Use the smallest real Qwen3.5 MoE geometry. Blackwell's
                # packed UE8M0 gate input needs four packed 128-wide groups;
                # tiny square fixtures do not satisfy that provider ABI.
                hidden=2048,
                intermediate=512,
                rank=rank,
                adapters=2,
                occupancy="mixed",
                phase="decode",
                graph_mode=True,
                output_dtype=torch.float32,
                seed=20260722 + rank,
            )
            assert fixture.plan.path is MoeLoraExecutionPath.C0_SERIAL
            assert fixture.plan.provider_key == fixture.base.contract.key

            n0 = fixture.run_n0().clone()
            eager = fixture.run_active().clone()
            graph, captured = _capture(fixture.run_active)
            graph.replay()
            torch.cuda.synchronize()
            replay = captured.clone()

            assert replay.dtype is torch.float32
            assert bool(torch.isfinite(replay).all())
            graph_tolerance = 0.16 if provider == "nvfp4" else 0.12
            assert _max_abs(eager, replay) <= graph_tolerance
            mapping = fixture.lora_info.token_lora_mapping
            assert _max_abs(replay[mapping < 0], n0[mapping < 0]) <= 0.08
            assert float((replay - n0).abs().max().item()) > 0.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
