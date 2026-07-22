import sys
from dataclasses import replace

import pytest
import torch

from benchmark.kernels.lora_moe.bench_c3_overlap import _invoke_c3
from benchmark.kernels.lora_moe.bench_moe_pipeline import (
    _build_fixture,
    _max_abs_diff,
    _single_rank_runtime,
    _smoke_case,
)
from benchmark.kernels.lora_moe.profiling import make_batch
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=35, stage="base-b", runner_config="1-gpu-small")


def _output(fixture, pipeline, schedule):
    fixture.reset_hidden()
    if pipeline == "C3":
        _invoke_c3(fixture, **schedule)
    else:
        fixture.invoke(pipeline)
    torch.cuda.synchronize()
    return fixture.last_output.clone()


@pytest.mark.parametrize(
    ("base_rows", "has_base_rows"),
    ((0, False), (1, True)),
)
def test_c3_static_base_row_contract_graph_replay_and_event_lifetime(
    base_rows, has_base_rows
):
    case = _smoke_case("h200")
    case = replace(
        case,
        case_id=f"c3-static-base-{base_rows}-smoke",
        adapters=replace(
            case.adapters,
            l_active=1,
            b_base=base_rows,
            l_capacity=2,
        ),
    )
    schedule = {
        "consumer_schedule": "aligned",
        "consumer_block_n": 32,
        "consumer_warps": 4,
        "finalize_block_h": 32,
        "finalize_warps": 4,
        "has_base_rows": has_base_rows,
    }

    with _single_rank_runtime():
        fixture = _build_fixture(
            case,
            need_lora=True,
            c1_overlap_policy="force",
        )
        fixture.runner_config.routed_scaling_factor = 1.75
        n0 = _output(fixture, "N0", schedule)
        c0 = _output(fixture, "C0", schedule)
        c3 = _output(fixture, "C3", schedule)

        reference_delta = c0.float() - n0.float()
        candidate_delta = c3.float() - n0.float()
        signal = float(reference_delta.abs().max().item())
        error = _max_abs_diff(reference_delta, candidate_delta)
        assert signal > 0.0
        assert error < signal / 8.0

        mapping = fixture.lora_info.token_lora_mapping
        assert bool((mapping >= 0).any())
        assert bool((mapping < 0).any()) is has_base_rows
        if has_base_rows:
            torch.testing.assert_close(
                c3[mapping < 0], n0[mapping < 0], rtol=0.0, atol=5e-4
            )

        eager = c3.clone()
        fixture.reset_hidden()
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            model_capture_mode,
        )

        with model_capture_mode():
            batch = make_batch(
                lambda: _invoke_c3(fixture, **schedule),
                execution="cuda_graph",
                inner_iterations=1,
            )
        assert len(batch.capture_resources) == 1
        for _ in range(3):
            fixture.reset_hidden()
            batch.run()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                fixture.last_output, eager, rtol=0.0, atol=3e-3
            )
        assert len(batch.capture_resources) == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
