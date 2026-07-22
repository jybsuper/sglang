#!/usr/bin/env python3
"""Evaluate the complete C2 down-B/base-finalize consumer at K0/O0/M0.

K0 compares a pre-routed production tail (base post-reorder plus routed B)
against one token-owned fused kernel. O0 rebuilds implementation-required
routing/allocation and measures isolated host-to-device completion. M0 compares
production C0, the partial C2P topology, and complete C2F at the full local MoE
boundary. Production dispatch is never changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from typing import Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from benchmark.kernels.lora_moe.bench_moe_pipeline import (  # noqa: E402
    PipelineFixture,
    _build_fixture,
    _case_summary,
    _detect_device,
    _environment,
    _list_cases,
    _max_abs_diff,
    _select_case,
    _single_rank_runtime,
)
from benchmark.kernels.lora_moe.profiling import (  # noqa: E402
    cuda_profile_range,
    make_batch,
    time_cuda_events,
    time_isolated_cuda_wall,
)

TAIL_VARIANTS = ("split", "indexed_split", "fused")
M0_VARIANTS = ("C0", "C2P", "C2I", "C2F")


@dataclass(slots=True)
class TailFixture:
    pipeline: PipelineFixture
    ws: object
    down_output: torch.Tensor
    down_intermediate: torch.Tensor
    output: torch.Tensor
    route_cache: dict
    input_source: str


def _with_case_overrides(case, *, rank: int | None, shared_outer: bool):
    adapters = case.adapters
    if rank is not None:
        adapters = replace(
            adapters,
            rank=rank,
            physical_rank=rank,
            max_rank=rank,
        )
    if shared_outer != adapters.shared_outer:
        adapters = replace(adapters, shared_outer=shared_outer)
    suffix = []
    if rank is not None:
        suffix.append(f"r{rank}")
    if shared_outer:
        suffix.append("shared-outer")
    return replace(
        case,
        case_id=(case.case_id + ("-" + "-".join(suffix) if suffix else "")),
        adapters=adapters,
    )


def _invoke_partial_c2(
    fixture: PipelineFixture,
    *,
    consumer_schedule: str,
    consumer_block_n: int,
    consumer_warps: int,
    down_finalize=None,
) -> None:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.lora.sgl_lora.experimental_c2 import (
        run_sgl_lora_moe_c2_experimental,
    )

    result = run_sgl_lora_moe_c2_experimental(
        StandardDispatchOutput(
            hidden_states=fixture.hidden_work,
            hidden_states_scale=None,
            topk_output=fixture.topk_output,
        ),
        fixture.sgl_quant_info,
        fixture.runner_config,
        fixture.lora_info,
        fixture.sgl_base,
        consumer_schedule=consumer_schedule,
        block_size_n=consumer_block_n,
        num_warps=consumer_warps,
        down_finalize=down_finalize,
    )
    fixture.last_output = result.hidden_states


def _invoke_full_c2(
    fixture: PipelineFixture,
    *,
    consumer_schedule: str,
    consumer_block_n: int,
    consumer_warps: int,
    finalize_block_h: int,
    finalize_warps: int,
) -> None:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.lora.sgl_lora.experimental_c2_full import (
        run_sgl_lora_moe_c2_full_experimental,
    )

    result = run_sgl_lora_moe_c2_full_experimental(
        StandardDispatchOutput(
            hidden_states=fixture.hidden_work,
            hidden_states_scale=None,
            topk_output=fixture.topk_output,
        ),
        fixture.sgl_quant_info,
        fixture.runner_config,
        fixture.lora_info,
        fixture.sgl_base,
        consumer_schedule=consumer_schedule,
        consumer_block_size_n=consumer_block_n,
        consumer_num_warps=consumer_warps,
        finalize_block_size_h=finalize_block_h,
        finalize_num_warps=finalize_warps,
    )
    fixture.last_output = result.hidden_states


def _invoke_indexed_c2(
    fixture: PipelineFixture,
    *,
    consumer_schedule: str,
    consumer_block_n: int,
    consumer_warps: int,
    finalize_block_h: int,
    finalize_warps: int,
) -> None:
    """Run C2P producer with route-free, still-unfused down finalization."""
    from sglang.srt.lora.sgl_lora.triton_ops.fused_down_finalize import (
        indexed_down_b_add,
    )

    def indexed_finalize(
        ws,
        down_out,
        down_intermediate,
        output,
        topk_ids,
        topk_weights,
        token_lora_mapping,
        callback_lora_info,
        callback_runner_config,
    ) -> None:
        scale = callback_runner_config.routed_scaling_factor or 1.0
        fixture.sgl_base.finalize(
            ws,
            down_out,
            topk_ids,
            topk_weights,
            scale,
            output,
        )
        indexed_down_b_add(
            down_intermediate,
            callback_lora_info.down_lora_b_weights,
            output,
            topk_ids,
            topk_weights,
            token_lora_mapping,
            routed_scaling_factor=scale,
            local_expert_offset=0,
            num_local_experts=fixture.sgl_quant_info.num_local_experts,
            shared_outer=callback_lora_info.experts_shared_outer_loras,
            block_size_h=finalize_block_h,
            num_warps=finalize_warps,
        )

    _invoke_partial_c2(
        fixture,
        consumer_schedule=consumer_schedule,
        consumer_block_n=consumer_block_n,
        consumer_warps=consumer_warps,
        down_finalize=indexed_finalize,
    )


def _prepare_down_route(tail: TailFixture) -> dict:
    from sglang.srt.lora.sgl_lora.triton_ops.virtual_experts import (
        merged_experts_fused_moe_lora_add,
    )

    fixture = tail.pipeline
    info = fixture.lora_info
    topk_ids = fixture.topk_output.topk_ids
    topk_weights = fixture.topk_output.topk_weights
    rank = tail.down_intermediate.shape[-1]
    route_cache: dict = {}
    merged_experts_fused_moe_lora_add(
        output=None,
        hidden_states=tail.down_intermediate.view(-1, rank),
        lora_a=info.down_lora_a_weights,
        lora_b=info.down_lora_b_weights,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        token_lora_mapping=info.token_lora_mapping,
        mul_routed_weight=True,
        experts_shared_outer_loras_a=False,
        experts_shared_outer_loras_b=info.experts_shared_outer_loras,
        routing_cache=route_cache,
        fuse_add_to_output=False,
        fuse_sum_all_reduce=True,
        use_direct_expand_add=info.max_lora_rank <= 64,
        num_output_slices=1,
        local_expert_offset=0,
        local_num_experts=fixture.sgl_quant_info.num_local_experts,
        stage="routing",
        intermediate_buffer=tail.down_intermediate,
    )
    return route_cache


def _build_tail_fixture(
    fixture: PipelineFixture,
    *,
    consumer_schedule: str,
    consumer_block_n: int,
    consumer_warps: int,
    input_source: str,
) -> TailFixture:
    captured: dict[str, object] = {}

    def capture_tail(
        ws,
        down_out,
        down_intermediate,
        output,
        _topk_ids,
        _topk_weights,
        _token_lora_mapping,
        _lora_info,
        _runner_config,
    ) -> None:
        # One-time producer snapshot. Clones are outside every K0/O0 interval.
        captured["ws"] = SimpleNamespace(src2dst=ws.src2dst.clone())
        captured["down_output"] = down_out.clone()
        captured["down_intermediate"] = down_intermediate.clone()
        output.zero_()

    resolved_source = input_source
    if input_source in ("auto", "producer"):
        fixture.reset_hidden()
        try:
            _invoke_partial_c2(
                fixture,
                consumer_schedule=consumer_schedule,
                consumer_block_n=consumer_block_n,
                consumer_warps=consumer_warps,
                down_finalize=capture_tail,
            )
            torch.cuda.synchronize()
            resolved_source = "c2_producer_snapshot"
        except Exception as exc:
            from triton.runtime.errors import OutOfResources

            if input_source != "auto" or not isinstance(exc, OutOfResources):
                raise
            # Rank-128 currently exceeds the production gate-A schedule's
            # resource envelope. Keep finalizer K0/O0 measurable, but label the
            # replacement honestly rather than treating it as a full C2 pass.
            resolved_source = "synthetic_after_producer_out_of_resources"

    if resolved_source in (
        "synthetic",
        "synthetic_after_producer_out_of_resources",
    ):
        fixture.reset_hidden()
        ws = fixture.sgl_base.prepare(
            fixture.hidden_work,
            fixture.topk_output.topk_ids,
            fixture.case.model.top_k,
        )
        generator = torch.Generator(device="cuda")
        generator.manual_seed(20260722)
        captured["ws"] = SimpleNamespace(src2dst=ws.src2dst.clone())
        captured["down_output"] = 0.1 * torch.randn(
            fixture.sgl_base.down_out_shape(ws),
            generator=generator,
            dtype=torch.bfloat16,
            device="cuda",
        )
        rank = fixture.lora_info.down_lora_b_weights.shape[-1]
        captured["down_intermediate"] = 0.05 * torch.randn(
            fixture.case.t_local,
            fixture.case.model.top_k,
            rank,
            generator=generator,
            dtype=torch.bfloat16,
            device="cuda",
        )
        torch.cuda.synchronize()
    output = torch.empty(
        fixture.case.t_local,
        fixture.case.model.h_moe,
        dtype=fixture.hidden_seed.dtype,
        device="cuda",
    )
    tail = TailFixture(
        pipeline=fixture,
        ws=captured["ws"],
        down_output=captured["down_output"],
        down_intermediate=captured["down_intermediate"],
        output=output,
        route_cache={},
        input_source=resolved_source,
    )
    tail.route_cache = _prepare_down_route(tail)
    torch.cuda.synchronize()
    return tail


def _invoke_split_tail(
    tail: TailFixture,
    *,
    output: torch.Tensor,
    route_cache: dict,
) -> None:
    from sglang.srt.lora.sgl_lora.triton_ops.virtual_experts import (
        merged_experts_fused_moe_lora_add,
    )

    fixture = tail.pipeline
    info = fixture.lora_info
    topk_ids = fixture.topk_output.topk_ids
    topk_weights = fixture.topk_output.topk_weights
    scale = fixture.runner_config.routed_scaling_factor or 1.0
    fixture.sgl_base.finalize(
        tail.ws,
        tail.down_output,
        topk_ids,
        topk_weights,
        scale,
        output,
    )
    rank = tail.down_intermediate.shape[-1]
    merged_experts_fused_moe_lora_add(
        output=output,
        hidden_states=tail.down_intermediate.view(-1, rank),
        lora_a=info.down_lora_a_weights,
        lora_b=info.down_lora_b_weights,
        topk_ids=topk_ids,
        topk_weights=topk_weights * float(scale),
        token_lora_mapping=info.token_lora_mapping,
        mul_routed_weight=True,
        experts_shared_outer_loras_a=False,
        experts_shared_outer_loras_b=info.experts_shared_outer_loras,
        routing_cache=route_cache,
        fuse_add_to_output=False,
        fuse_sum_all_reduce=True,
        use_direct_expand_add=info.max_lora_rank <= 64,
        num_output_slices=1,
        local_expert_offset=0,
        local_num_experts=fixture.sgl_quant_info.num_local_experts,
        stage="expand",
        intermediate_buffer=tail.down_intermediate,
    )


def _invoke_fused_tail(
    tail: TailFixture,
    *,
    output: torch.Tensor,
    block_h: int,
    warps: int,
) -> None:
    from sglang.srt.lora.sgl_lora.triton_ops.fused_down_finalize import (
        fused_down_b_finalize,
    )

    fixture = tail.pipeline
    fused_down_b_finalize(
        tail.down_output,
        tail.down_intermediate,
        fixture.lora_info.down_lora_b_weights,
        output,
        tail.ws.src2dst,
        fixture.topk_output.topk_ids,
        fixture.topk_output.topk_weights,
        fixture.lora_info.token_lora_mapping,
        routed_scaling_factor=fixture.runner_config.routed_scaling_factor or 1.0,
        local_expert_offset=0,
        shared_outer=fixture.lora_info.experts_shared_outer_loras,
        block_size_h=block_h,
        num_warps=warps,
    )


def _invoke_indexed_tail(
    tail: TailFixture,
    *,
    output: torch.Tensor,
    block_h: int,
    warps: int,
) -> None:
    from sglang.srt.lora.sgl_lora.triton_ops.fused_down_finalize import (
        indexed_down_b_add,
    )

    fixture = tail.pipeline
    scale = fixture.runner_config.routed_scaling_factor or 1.0
    fixture.sgl_base.finalize(
        tail.ws,
        tail.down_output,
        fixture.topk_output.topk_ids,
        fixture.topk_output.topk_weights,
        scale,
        output,
    )
    indexed_down_b_add(
        tail.down_intermediate,
        fixture.lora_info.down_lora_b_weights,
        output,
        fixture.topk_output.topk_ids,
        fixture.topk_output.topk_weights,
        fixture.lora_info.token_lora_mapping,
        routed_scaling_factor=scale,
        local_expert_offset=0,
        num_local_experts=fixture.sgl_quant_info.num_local_experts,
        shared_outer=fixture.lora_info.experts_shared_outer_loras,
        block_size_h=block_h,
        num_warps=warps,
    )


def _tail_fp32_oracle(tail: TailFixture) -> torch.Tensor:
    fixture = tail.pipeline
    info = fixture.lora_info
    ids = fixture.topk_output.topk_ids
    weights = fixture.topk_output.topk_weights
    mapping = info.token_lora_mapping
    hidden = tail.down_output.shape[-1]
    scale = fixture.runner_config.routed_scaling_factor or 1.0
    result = torch.zeros(ids.shape[0], hidden, dtype=torch.float32, device="cuda")
    down_flat = tail.down_output.view(-1, hidden)
    for token in range(ids.shape[0]):
        adapter = int(mapping[token])
        for k_idx in range(ids.shape[1]):
            expert = int(ids[token, k_idx])
            if not 0 <= expert < tail.down_output.shape[0]:
                continue
            pair = token * ids.shape[1] + k_idx
            value = down_flat[int(tail.ws.src2dst[pair])].float()
            if 0 <= adapter < info.down_lora_b_weights.shape[0]:
                b_expert = 0 if info.experts_shared_outer_loras else expert
                value = value + (
                    info.down_lora_b_weights[adapter, b_expert].float()
                    @ tail.down_intermediate[token, k_idx].float()
                )
            result[token] += weights[token, k_idx].float() * float(scale) * value
    return result


def _check_tail(tail: TailFixture, *, block_h: int, warps: int) -> dict:
    oracle = _tail_fp32_oracle(tail)
    fused_fp32 = torch.empty_like(oracle)
    _invoke_fused_tail(tail, output=fused_fp32, block_h=block_h, warps=warps)
    split = torch.empty_like(tail.output)
    _invoke_split_tail(tail, output=split, route_cache=tail.route_cache)
    indexed = torch.empty_like(tail.output)
    _invoke_indexed_tail(tail, output=indexed, block_h=block_h, warps=warps)
    torch.cuda.synchronize()
    fused_error = _max_abs_diff(fused_fp32, oracle)
    split_error = _max_abs_diff(split.float(), oracle)
    indexed_error = _max_abs_diff(indexed.float(), oracle)
    torch.testing.assert_close(fused_fp32, oracle, rtol=3e-4, atol=3e-4)
    torch.testing.assert_close(split.float(), oracle, rtol=8e-2, atol=8e-3)
    torch.testing.assert_close(indexed.float(), oracle, rtol=8e-2, atol=8e-3)
    return {
        "oracle": "independent_token_topk_fp32_base_plus_lora",
        "base_output_max_abs": float(tail.down_output.float().abs().max().item()),
        "lora_rank_max_abs": float(tail.down_intermediate.float().abs().max().item()),
        "routed_scaling_factor": tail.pipeline.runner_config.routed_scaling_factor,
        "mixed_base_rows": bool((tail.pipeline.lora_info.token_lora_mapping < 0).any()),
        "fused_fp32_max_abs_error": fused_error,
        "split_bf16_max_abs_error": split_error,
        "indexed_split_bf16_max_abs_error": indexed_error,
        "fused_rtol": 3e-4,
        "fused_atol": 3e-4,
        "split_rtol": 8e-2,
        "split_atol": 8e-3,
    }


def _tail_callable(
    tail: TailFixture,
    variant: str,
    *,
    scope: str,
    block_h: int,
    warps: int,
) -> Callable[[], None]:
    if scope == "K0":
        if variant == "split":
            return lambda: _invoke_split_tail(
                tail, output=tail.output, route_cache=tail.route_cache
            )
        if variant == "indexed_split":
            return lambda: _invoke_indexed_tail(
                tail,
                output=tail.output,
                block_h=block_h,
                warps=warps,
            )
        return lambda: _invoke_fused_tail(
            tail, output=tail.output, block_h=block_h, warps=warps
        )

    holder: list[torch.Tensor] = []

    def o0_split() -> None:
        output = torch.empty_like(tail.output)
        holder[:] = [output]
        _invoke_split_tail(tail, output=output, route_cache={})

    def o0_fused() -> None:
        output = torch.empty_like(tail.output)
        holder[:] = [output]
        _invoke_fused_tail(tail, output=output, block_h=block_h, warps=warps)

    def o0_indexed() -> None:
        output = torch.empty_like(tail.output)
        holder[:] = [output]
        _invoke_indexed_tail(tail, output=output, block_h=block_h, warps=warps)

    if variant == "split":
        return o0_split
    if variant == "indexed_split":
        return o0_indexed
    return o0_fused


def _benchmark_tail(
    tail: TailFixture,
    variant: str,
    *,
    scope: str,
    execution: str,
    block_h: int,
    warps: int,
    warmup: int,
    samples: int,
) -> dict:
    fn = _tail_callable(tail, variant, scope=scope, block_h=block_h, warps=warps)
    fn()
    torch.cuda.synchronize()
    effective_execution = "eager" if scope == "O0" else execution
    batch = make_batch(fn, execution=effective_execution, inner_iterations=1)
    timer = time_isolated_cuda_wall if scope == "O0" else time_cuda_events
    timing = timer(
        batch.run,
        launches_per_batch=1,
        warmup=warmup,
        samples=samples,
    )
    print(
        f"{tail.pipeline.case.case_id} {scope}/{variant} "
        f"BH={block_h} W={warps} {effective_execution}: "
        f"p50={timing.p50_us:.3f} us "
        f"p20/p80={timing.p20_us:.3f}/{timing.p80_us:.3f} us"
    )
    return {
        "scope": scope,
        "variant": variant,
        "execution": effective_execution,
        "schedule": (
            {"block_h": block_h, "num_warps": warps}
            if variant in ("indexed_split", "fused")
            else "production_resolved"
        ),
        "timing_domain": (
            "isolated_wall_host_to_device_completion"
            if scope == "O0"
            else "cuda_event_device"
        ),
        "timing": asdict(timing),
    }


def _invoke_m0(
    fixture: PipelineFixture,
    variant: str,
    *,
    consumer_schedule: str,
    consumer_block_n: int,
    consumer_warps: int,
    finalize_block_h: int,
    finalize_warps: int,
) -> None:
    if variant == "C0":
        fixture.invoke("C0")
    elif variant == "C2P":
        _invoke_partial_c2(
            fixture,
            consumer_schedule=consumer_schedule,
            consumer_block_n=consumer_block_n,
            consumer_warps=consumer_warps,
        )
    elif variant == "C2I":
        _invoke_indexed_c2(
            fixture,
            consumer_schedule=consumer_schedule,
            consumer_block_n=consumer_block_n,
            consumer_warps=consumer_warps,
            finalize_block_h=finalize_block_h,
            finalize_warps=finalize_warps,
        )
    else:
        _invoke_full_c2(
            fixture,
            consumer_schedule=consumer_schedule,
            consumer_block_n=consumer_block_n,
            consumer_warps=consumer_warps,
            finalize_block_h=finalize_block_h,
            finalize_warps=finalize_warps,
        )


def _checked_m0(fixture: PipelineFixture, variant: str, **kwargs) -> torch.Tensor:
    fixture.reset_hidden()
    _invoke_m0(fixture, variant, **kwargs)
    torch.cuda.synchronize()
    if fixture.last_output is None or not bool(
        torch.isfinite(fixture.last_output).all()
    ):
        raise AssertionError(f"{variant} produced no finite output")
    return fixture.last_output.clone()


def _check_m0(fixture: PipelineFixture, **kwargs) -> dict:
    c0 = _checked_m0(fixture, "C0", **kwargs)
    c2p = _checked_m0(fixture, "C2P", **kwargs)
    c2i = _checked_m0(fixture, "C2I", **kwargs)
    c2f = _checked_m0(fixture, "C2F", **kwargs)
    c2f_repeat = _checked_m0(fixture, "C2F", **kwargs)
    fixture.reset_hidden()
    fixture.invoke("N0")
    torch.cuda.synchronize()
    n0 = fixture.last_output.clone()
    # Full-output agreement is deliberately tighter than the LoRA signal.
    # The separate base-subtracted check below prevents a large base branch
    # from making this comparison vacuous.
    torch.testing.assert_close(c0, c2p, rtol=3e-2, atol=5e-4)
    torch.testing.assert_close(c2p, c2i, rtol=3e-2, atol=5e-4)
    torch.testing.assert_close(c2p, c2f, rtol=3e-2, atol=5e-4)
    reference_delta = c0.float() - n0.float()
    partial_delta = c2p.float() - n0.float()
    indexed_delta = c2i.float() - n0.float()
    fused_delta = c2f.float() - n0.float()
    signal = float(reference_delta.abs().max().item())
    if signal <= 0.0:
        raise AssertionError("M0 LoRA oracle has no nonzero delta signal")
    partial_delta_error = _max_abs_diff(reference_delta, partial_delta)
    indexed_delta_error = _max_abs_diff(reference_delta, indexed_delta)
    fused_delta_error = _max_abs_diff(reference_delta, fused_delta)
    repeat_error = _max_abs_diff(c2f, c2f_repeat)
    # BF16 base subtraction is quantized at one output ULP, so use an explicit
    # signal-relative max-error gate instead of a large absolute allclose.
    max_delta_error = signal * 0.10
    if partial_delta_error > max_delta_error:
        raise AssertionError(
            f"C2P delta error {partial_delta_error} exceeds 10% of signal {signal}"
        )
    if fused_delta_error > max_delta_error:
        raise AssertionError(
            f"C2F delta error {fused_delta_error} exceeds 10% of signal {signal}"
        )
    if indexed_delta_error > max_delta_error:
        raise AssertionError(
            f"C2I delta error {indexed_delta_error} exceeds 10% of signal {signal}"
        )
    if repeat_error > 5e-4:
        raise AssertionError(f"C2F repeat error {repeat_error} exceeds one BF16 ULP")
    return {
        "c0_c2p_max_abs": _max_abs_diff(c0, c2p),
        "c2p_c2i_max_abs": _max_abs_diff(c2p, c2i),
        "c2p_c2f_max_abs": _max_abs_diff(c2p, c2f),
        "reference_delta_max_abs": signal,
        "partial_delta_max_abs_error": partial_delta_error,
        "indexed_delta_max_abs_error": indexed_delta_error,
        "fused_delta_max_abs_error": fused_delta_error,
        "fused_delta_error_over_signal": fused_delta_error / signal,
        "max_allowed_delta_error": max_delta_error,
        "c2f_repeat_max_abs_error": repeat_error,
        "full_output_rtol": 3e-2,
        "full_output_atol": 5e-4,
        "delta_max_error_fraction": 0.10,
        "routed_scaling_factor": fixture.runner_config.routed_scaling_factor,
        "base_output_nonzero": bool(n0.float().abs().max().item() > 0),
        "has_base_only_rows": bool((fixture.lora_info.token_lora_mapping < 0).any()),
    }


def _benchmark_m0(
    fixture: PipelineFixture,
    variant: str,
    *,
    execution: str,
    warmup: int,
    samples: int,
    **kwargs,
) -> dict:
    from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode

    fixture.reset_hidden()
    _invoke_m0(fixture, variant, **kwargs)
    torch.cuda.synchronize()
    eager = fixture.last_output.clone()
    fixture.reset_hidden()
    with ExitStack() as stack:
        if execution == "cuda_graph":
            stack.enter_context(model_capture_mode())
        batch = make_batch(
            lambda: _invoke_m0(fixture, variant, **kwargs),
            execution=execution,
            inner_iterations=1,
        )
    graph_check = None
    if execution == "cuda_graph":
        fixture.reset_hidden()
        batch.run()
        torch.cuda.synchronize()
        graph_error = _max_abs_diff(eager, fixture.last_output)
        torch.testing.assert_close(eager, fixture.last_output, rtol=0.0, atol=3e-3)
        graph_check = {"eager_graph_max_abs": graph_error, "atol": 3e-3}
    timing = time_cuda_events(
        batch.run,
        launches_per_batch=1,
        warmup=warmup,
        samples=samples,
        before_sample=fixture.reset_hidden,
    )
    print(
        f"{fixture.case.case_id} M0/{variant} {execution}: "
        f"p50={timing.p50_us:.3f} us "
        f"p20/p80={timing.p20_us:.3f}/{timing.p80_us:.3f} us"
    )
    return {
        "scope": "M0",
        "variant": variant,
        "execution": execution,
        "timing": asdict(timing),
        "graph_correctness": graph_check,
    }


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(","))
    if not values or any(item <= 0 for item in values):
        raise ValueError("expected comma-separated positive integers")
    return values


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "h200", "gb300"), default="auto")
    parser.add_argument("--case-id")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--scope", choices=("K0", "O0", "M0", "all"), default="all")
    parser.add_argument("--variant", default="all")
    parser.add_argument("--rank", type=int, choices=(32, 64, 128))
    parser.add_argument("--shared-outer", action="store_true")
    parser.add_argument("--routed-scaling-factor", type=float, default=1.75)
    parser.add_argument(
        "--tail-source", choices=("auto", "producer", "synthetic"), default="auto"
    )
    parser.add_argument(
        "--consumer-schedule", choices=("pair", "aligned"), default="pair"
    )
    parser.add_argument("--consumer-block-n", type=int, default=32)
    parser.add_argument("--consumer-warps", type=int, default=4)
    parser.add_argument("--finalize-block-h", default="16,32,64")
    parser.add_argument("--finalize-warps", default="4")
    parser.add_argument(
        "--execution", choices=("eager", "cuda_graph"), default="cuda_graph"
    )
    parser.add_argument("--mode", choices=("time", "nsys", "ncu"), default="time")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--counterbalance-repeats", type=int, default=2)
    parser.add_argument("--profile-iterations", type=int, default=1)
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    device = _detect_device(args.device)
    if args.list_cases:
        _list_cases(device)
        return 0
    case = _with_case_overrides(
        _select_case(device, args.case_id),
        rank=args.rank,
        shared_outer=args.shared_outer,
    )
    if case.model.num_slices != 2 or case.model.activation != "swiglu":
        raise NotImplementedError("C2 BF16 checkpoint requires gated SwiGLU")
    block_hs = _parse_csv_ints(args.finalize_block_h)
    warp_counts = _parse_csv_ints(args.finalize_warps)
    scopes = ("K0", "O0", "M0") if args.scope == "all" else (args.scope,)
    if args.mode != "time" and (
        len(scopes) != 1 or len(block_hs) != 1 or len(warp_counts) != 1
    ):
        raise ValueError("profiling requires one scope, BLOCK_H, and warp count")

    with _single_rank_runtime():
        fixture = _build_fixture(case, need_lora=True)
        fixture.runner_config.routed_scaling_factor = args.routed_scaling_factor
        tail = _build_tail_fixture(
            fixture,
            consumer_schedule=args.consumer_schedule,
            consumer_block_n=args.consumer_block_n,
            consumer_warps=args.consumer_warps,
            input_source=args.tail_source,
        )
        checks = None
        if not args.skip_check:
            checks = {
                f"tail-bh{bh}-w{warps}": _check_tail(tail, block_h=bh, warps=warps)
                for bh in block_hs
                for warps in warp_counts
            }

        common_m0 = {
            "consumer_schedule": args.consumer_schedule,
            "consumer_block_n": args.consumer_block_n,
            "consumer_warps": args.consumer_warps,
            "finalize_block_h": block_hs[0],
            "finalize_warps": warp_counts[0],
        }
        m0_check = None
        m0_unsupported = None
        if "M0" in scopes and not args.skip_check:
            try:
                m0_check = _check_m0(fixture, **common_m0)
            except Exception as exc:
                from triton.runtime.errors import OutOfResources

                if not isinstance(exc, OutOfResources):
                    raise
                m0_unsupported = {"type": type(exc).__name__, "message": str(exc)}

        runs: list[dict] = []
        comparisons: list[dict] = []
        if args.mode == "time":
            for scope in scopes:
                variants = M0_VARIANTS if scope == "M0" else TAIL_VARIANTS
                if args.variant != "all":
                    if args.variant not in variants:
                        raise ValueError(f"{args.variant} is not valid for {scope}")
                    variants = (args.variant,)
                if scope == "M0" and m0_unsupported is not None:
                    continue
                schedules = (
                    [(block_hs[0], warp_counts[0])]
                    if scope == "M0"
                    else [(bh, w) for bh in block_hs for w in warp_counts]
                )
                for bh, warps in schedules:
                    if len(variants) == 1:
                        run = (
                            _benchmark_m0(
                                fixture,
                                variants[0],
                                execution=args.execution,
                                warmup=args.warmup,
                                samples=args.samples,
                                **common_m0,
                            )
                            if scope == "M0"
                            else _benchmark_tail(
                                tail,
                                variants[0],
                                scope=scope,
                                execution=args.execution,
                                block_h=bh,
                                warps=warps,
                                warmup=args.warmup,
                                samples=args.samples,
                            )
                        )
                        runs.append(run)
                        continue
                    for repeat in range(args.counterbalance_repeats):
                        order = (
                            variants if repeat % 2 == 0 else tuple(reversed(variants))
                        )
                        paired = {}
                        for variant in order:
                            run = (
                                _benchmark_m0(
                                    fixture,
                                    variant,
                                    execution=args.execution,
                                    warmup=args.warmup,
                                    samples=args.samples,
                                    **common_m0,
                                )
                                if scope == "M0"
                                else _benchmark_tail(
                                    tail,
                                    variant,
                                    scope=scope,
                                    execution=args.execution,
                                    block_h=bh,
                                    warps=warps,
                                    warmup=args.warmup,
                                    samples=args.samples,
                                )
                            )
                            run["repeat"] = repeat
                            run["order"] = list(order)
                            runs.append(run)
                            paired[variant] = run
                        control = "C0" if scope == "M0" else "split"
                        control_us = paired[control]["timing"]["p50_us"]
                        for candidate in variants:
                            if candidate == control:
                                continue
                            candidate_us = paired[candidate]["timing"]["p50_us"]
                            comparisons.append(
                                {
                                    "scope": scope,
                                    "block_h": bh,
                                    "num_warps": warps,
                                    "repeat": repeat,
                                    "order": list(order),
                                    "control": control,
                                    "candidate": candidate,
                                    "control_p50_us": control_us,
                                    "candidate_p50_us": candidate_us,
                                    "candidate_vs_control_pct": (
                                        candidate_us / control_us - 1.0
                                    )
                                    * 100.0,
                                }
                            )
        else:
            scope = scopes[0]
            default_variant = "C2F" if scope == "M0" else "fused"
            variant = default_variant if args.variant == "all" else args.variant
            if scope == "M0":

                def fn() -> None:
                    _invoke_m0(fixture, variant, **common_m0)

                fixture.reset_hidden()
            else:
                fn = _tail_callable(
                    tail,
                    variant,
                    scope=scope,
                    block_h=block_hs[0],
                    warps=warp_counts[0],
                )
            fn()
            torch.cuda.synchronize()
            label = (
                f"sgl_lora_moe::{scope}::down_finalize::{case.case_id}::"
                f"{variant}::BH={block_hs[0]}::W={warp_counts[0]}"
            )
            with cuda_profile_range(label):
                for _ in range(args.profile_iterations):
                    if scope == "M0":
                        fixture.reset_hidden()
                    fn()
            runs.append({"profile_label": label, "variant": variant, "scope": scope})

        summaries = []
        keys = sorted(
            {
                (row["scope"], row["block_h"], row["num_warps"], row["candidate"])
                for row in comparisons
            }
        )
        for scope, bh, warps, candidate in keys:
            values = [
                row["candidate_vs_control_pct"]
                for row in comparisons
                if (row["scope"], row["block_h"], row["num_warps"], row["candidate"])
                == (scope, bh, warps, candidate)
            ]
            summaries.append(
                {
                    "scope": scope,
                    "block_h": bh,
                    "num_warps": warps,
                    "candidate": candidate,
                    "num_pairs": len(values),
                    "median_paired_pct": median(values),
                    "min_paired_pct": min(values),
                    "max_paired_pct": max(values),
                }
            )

        result = {
            "schema_version": 1,
            "scope": "bf16_c2_down_b_finalize_experimental",
            "contract": {
                "production_dispatch_changed": False,
                "base_layout": "masked_E_mmax_H",
                "lora_input_layout": "canonical_T_K_R",
                "owner": "token_and_hidden_tile",
                "accumulation": "fp32_base_plus_lora_then_requested_output_dtype",
                "scaling": "topk_weight_times_routed_scaling_exactly_once_both_branches",
                "shared_outer_specialization": "weighted_rank_reduce_then_one_shared_B_load",
                "current_full_c2_limits": "bf16_gate_first_contiguous_gated_swiglu_offset_zero",
                "tail_input_source": tail.input_source,
            },
            "case": _case_summary(case),
            "environment": _environment(args),
            "correctness": {"tail": checks, "M0": m0_check},
            "M0_unsupported": m0_unsupported,
            "runs": runs,
            "paired_comparisons": comparisons,
            "paired_summary": summaries,
        }
        if args.json_output:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(
                json.dumps(result, indent=2, default=str) + "\n"
            )
        else:
            print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
