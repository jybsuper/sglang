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


def _assert_marlin_overwrites_dirty_destination(run, out):
    out.fill_(37)
    run(out)
    dirty_result = out.clone()
    out.fill_(-19)
    run(out)
    recycled_result = out.clone()
    # Split-K atomic order can move a few BF16 ulps between launches.  A stale
    # destination would instead shift every element by the 56-point sentinel
    # difference, so this tight absolute bound still isolates the contract.
    torch.testing.assert_close(dirty_result, recycled_result, rtol=0.0, atol=0.125)


def test_marlin_atomic_outputs_do_not_accumulate_recycled_buffer_contents():
    major, _minor = torch.cuda.get_device_capability()
    if major < 9:
        pytest.skip("Marlin provider requires SM90+")

    with _single_rank_runtime():
        fixture = QuantizedPipelineFixture(
            provider_name="marlin",
            tokens=8,
            experts=4,
            top_k=2,
            hidden=2048,
            intermediate=512,
            rank=16,
            adapters=2,
            occupancy="mixed",
            phase="decode",
            graph_mode=False,
            output_dtype=torch.float32,
            seed=20260723,
        )
        topk_ids = fixture.topk_output.topk_ids
        topk_weights = fixture.topk_output.topk_weights
        ws = fixture.base.prepare(
            fixture.hidden_work,
            topk_ids,
            fixture.top_k,
            topk_weights=topk_weights,
            packed_topk_ids=fixture.packed_topk_ids,
        )

        gateup_out = torch.empty(
            fixture.base.gateup_out_shape(ws),
            device=fixture.hidden_work.device,
            dtype=fixture.base.contract.gate_up_output_dtype,
        )
        _assert_marlin_overwrites_dirty_destination(
            lambda out: fixture.base.gateup(ws, out), gateup_out
        )

        activation = torch.randn(
            fixture.base.act_out_shape(ws),
            device=fixture.hidden_work.device,
            dtype=fixture.base.contract.lora_activation_dtype,
        )
        down_out = torch.empty(
            fixture.base.down_out_shape(ws),
            device=fixture.hidden_work.device,
            dtype=torch.bfloat16,
        )
        _assert_marlin_overwrites_dirty_destination(
            lambda out: fixture.base.down(ws, activation, out), down_out
        )


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
