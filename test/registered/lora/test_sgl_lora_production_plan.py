import sys
from dataclasses import replace

import pytest
import torch

from benchmark.kernels.lora_moe.bench_moe_pipeline import (
    _build_fixture,
    _max_abs_diff,
    _single_rank_runtime,
    _smoke_case,
)
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.lora.sgl_lora.bf16_execution import run_sgl_lora_moe_bf16_plan
from sglang.srt.lora.sgl_lora.execution_plan import (
    MoeLoraExecutionPath,
    build_moe_lora_execution_plan,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=35, stage="base-b", runner_config="1-gpu-small")


def _invoke(fixture, plan, *, output_dtype=None):
    fixture.reset_hidden()
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
        output_dtype=output_dtype,
    )
    torch.cuda.synchronize()
    return result.hidden_states.clone()


@pytest.mark.parametrize(
    ("phase", "graph_mode", "two_stream", "expected_path"),
    (
        ("decode", False, False, MoeLoraExecutionPath.C2_FULL),
        ("prefill", False, False, MoeLoraExecutionPath.C2_PARTIAL),
        ("decode", True, True, MoeLoraExecutionPath.C3_OVERLAP),
    ),
)
@pytest.mark.parametrize("base_rows", (0, 1))
@pytest.mark.parametrize("rank", (8, 16))
def test_production_plan_matches_c0_delta_and_requested_output_dtype(
    phase, graph_mode, two_stream, expected_path, base_rows, rank
):
    case = _smoke_case("h200")
    case = replace(
        case,
        case_id=f"production-plan-{phase}-base-{base_rows}",
        adapters=replace(
            case.adapters,
            l_active=1,
            b_base=base_rows,
            l_capacity=2,
            rank=rank,
            physical_rank=rank,
            max_rank=rank,
        ),
    )
    has_base_rows = bool(base_rows)
    plan = build_moe_lora_execution_plan(
        phase=phase,
        graph_mode=graph_mode,
        num_tokens=case.t_local,
        rank=rank,
        has_base_rows=has_base_rows,
        two_stream_requested=two_stream,
        fused_supported=True,
    )
    assert plan.path is expected_path

    with _single_rank_runtime():
        fixture = _build_fixture(case, need_lora=True, c1_overlap_policy="force")
        c0_plan = replace(
            plan,
            path=MoeLoraExecutionPath.C0_SERIAL,
            reason="test reference",
        )
        base_only_mapping = fixture.lora_info.token_lora_mapping.clone()
        fixture.lora_info.token_lora_mapping.fill_(-1)
        n0 = _invoke(fixture, c0_plan, output_dtype=torch.float32)
        fixture.lora_info.token_lora_mapping.copy_(base_only_mapping)
        c0 = _invoke(fixture, c0_plan, output_dtype=torch.float32)
        candidate = _invoke(fixture, plan, output_dtype=torch.float32)

        assert c0.dtype is torch.float32
        assert candidate.dtype is torch.float32
        reference_delta = c0 - n0
        candidate_delta = candidate - n0
        signal = float(reference_delta.abs().max().item())
        error = _max_abs_diff(reference_delta, candidate_delta)
        assert signal > 0.0
        assert error < signal / 8.0

        mapping = fixture.lora_info.token_lora_mapping
        assert bool((mapping < 0).any()) is has_base_rows
        if has_base_rows:
            torch.testing.assert_close(
                candidate[mapping < 0], n0[mapping < 0], rtol=0.0, atol=5e-4
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
