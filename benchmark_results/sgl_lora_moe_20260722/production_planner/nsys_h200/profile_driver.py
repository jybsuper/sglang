#!/usr/bin/env python3
"""Capture final production C2F/C3 planner executors under cudaProfilerApi."""

from contextlib import ExitStack

import torch

from benchmark.kernels.lora_moe.bench_moe_pipeline import (
    _build_fixture,
    _max_abs_diff,
    _select_case,
    _single_rank_runtime,
)
from benchmark.kernels.lora_moe.profiling import cuda_profile_range, make_batch
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.lora.sgl_lora.bf16_execution import run_sgl_lora_moe_bf16_plan
from sglang.srt.lora.sgl_lora.execution_plan import build_moe_lora_execution_plan
from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode


def invoke(fixture, plan):
    result = run_sgl_lora_moe_bf16_plan(
        StandardDispatchOutput(
            hidden_states=fixture.hidden_work,
            hidden_states_scale=None,
            topk_output=fixture.topk_output,
        ),
        fixture.sgl_quant_info,
        fixture.runner_config,
        fixture.lora_info,
        fixture.sgl_base,
        plan,
    )
    fixture.last_output = result.hidden_states


def capture(fixture, plan):
    fixture.reset_hidden()
    invoke(fixture, plan)
    torch.cuda.synchronize()
    eager = fixture.last_output.clone()
    fixture.reset_hidden()
    with ExitStack() as stack:
        stack.enter_context(model_capture_mode())
        batch = make_batch(
            lambda: invoke(fixture, plan),
            execution="cuda_graph",
            inner_iterations=1,
        )
    batch.run()
    torch.cuda.synchronize()
    error = _max_abs_diff(eager, fixture.last_output)
    if error > 3e-3:
        raise AssertionError(f"eager/graph mismatch: {error}")
    return batch, error


def main():
    case = _select_case("h200", "p0-qwen3.5-35b-a3b-cap1-h200")
    with _single_rank_runtime():
        fixture = _build_fixture(case, need_lora=True, c1_overlap_policy="force")
        common = dict(
            phase="decode",
            graph_mode=True,
            num_tokens=case.t_local,
            rank=case.adapters.rank,
            has_base_rows=False,
            fused_supported=True,
        )
        c2 = build_moe_lora_execution_plan(
            two_stream_requested=False,
            **common,
        )
        c3 = build_moe_lora_execution_plan(
            two_stream_requested=True,
            **common,
        )
        c2_batch, c2_error = capture(fixture, c2)
        c3_batch, c3_error = capture(fixture, c3)

        with cuda_profile_range(
            "sgl_lora_moe::production_planner::qwen35::T32::R64::C2F_and_C3"
        ):
            for _ in range(10):
                torch.cuda.nvtx.range_push("production::C2F")
                c2_batch.run()
                torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_push("production::C3")
                c3_batch.run()
                torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        print(
            {
                "case": case.case_id,
                "c2_path": c2.path.value,
                "c3_path": c3.path.value,
                "iterations_each": 10,
                "c2_eager_graph_max_abs": c2_error,
                "c3_eager_graph_max_abs": c3_error,
                "c2_graph_resources": len(c2_batch.capture_resources),
                "c3_graph_resources": len(c3_batch.capture_resources),
            }
        )


if __name__ == "__main__":
    main()
