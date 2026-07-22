"""Matched production benchmark for shared-outer gate/up LoRA-A dispatch.

The control is the generic virtual-expert route + shrink.  The candidate is
the production static selector plus its segmented token-deduplicated kernel.
Both paths include the gate-B route preparation required by the downstream
consumer, use the same pair-major ``[T,K,2R]`` destination, and are timed in
eager and CUDA-graph modes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from sglang.srt.lora.sgl_lora.shared_outer_gate_policy import (
    build_shared_outer_gate_a_plan,
)
from sglang.srt.lora.sgl_lora.triton_ops.virtual_experts import (
    merged_experts_fused_moe_lora_add,
)


@dataclass(frozen=True, slots=True)
class GateDispatchCase:
    case_id: str
    tokens: int
    hidden_size: int
    rank: int
    top_k: int
    num_experts: int
    segment_lengths: tuple[int, ...]
    segment_adapter_ids: tuple[int, ...]
    phase: str = "decode"

    @property
    def has_base_rows(self) -> bool:
        return 0 in self.segment_adapter_ids

    @property
    def capacity(self) -> int:
        return max(self.segment_adapter_ids) + 1


CASES = (
    GateDispatchCase(
        "selected-t32-r128-active",
        32,
        2048,
        128,
        8,
        256,
        (4,) * 8,
        tuple(range(1, 9)),
    ),
    GateDispatchCase(
        "selected-t32-r128-fragmented",
        32,
        2048,
        128,
        8,
        256,
        (1,) * 32,
        tuple(1 + index % 8 for index in range(32)),
    ),
    GateDispatchCase(
        "selected-t32-r128-mixed",
        32,
        2048,
        128,
        8,
        256,
        (4,) * 8,
        (1, 2, 0, 3, 4, 0, 5, 6),
    ),
    GateDispatchCase(
        "selected-t256-r64-active",
        256,
        2048,
        64,
        8,
        256,
        (64,) * 4,
        (1, 2, 3, 4),
    ),
    GateDispatchCase(
        "selected-t32-r64-h4096",
        32,
        4096,
        64,
        10,
        512,
        (8,) * 4,
        (1, 2, 3, 4),
    ),
    GateDispatchCase(
        "fallback-t32-r64-noisy",
        32,
        2048,
        64,
        8,
        256,
        (8,) * 4,
        (1, 2, 3, 4),
    ),
    GateDispatchCase(
        "fallback-t32-r64-mixed",
        32,
        2048,
        64,
        8,
        256,
        (8,) * 4,
        (1, 0, 2, 0),
    ),
    GateDispatchCase(
        "fallback-t32-r64-wide",
        32,
        7168,
        64,
        8,
        384,
        (8,) * 4,
        (1, 2, 3, 4),
    ),
    GateDispatchCase(
        "fallback-t1-r32-tiny",
        1,
        2048,
        32,
        8,
        256,
        (1,),
        (1,),
    ),
)
CASES_BY_ID = {case.case_id: case for case in CASES}


@dataclass(slots=True)
class Fixture:
    case: GateDispatchCase
    hidden: torch.Tensor
    factor: torch.Tensor
    dummy_b: torch.Tensor
    output: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    mapping: torch.Tensor
    indptr: torch.Tensor
    adapter_ids: torch.Tensor
    plan: object


def _build_fixture(case: GateDispatchCase) -> Fixture:
    assert sum(case.segment_lengths) == case.tokens
    assert len(case.segment_lengths) == len(case.segment_adapter_ids)
    generator = torch.Generator(device="cuda").manual_seed(20260722)
    hidden = torch.empty(
        case.tokens, case.hidden_size, dtype=torch.bfloat16, device="cuda"
    ).uniform_(-0.1, 0.1, generator=generator)
    factor = torch.empty(
        case.capacity,
        1,
        2 * case.rank,
        case.hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
    ).uniform_(-0.02, 0.02, generator=generator)
    factor[0].zero_()
    dummy_b = torch.empty(
        case.capacity,
        case.num_experts,
        2,
        case.rank,
        dtype=torch.bfloat16,
        device="cuda",
    )
    token = torch.arange(case.tokens, dtype=torch.int32, device="cuda")[:, None]
    slot = torch.arange(case.top_k, dtype=torch.int32, device="cuda")[None, :]
    topk_ids = ((token * 13 + slot * 7) % case.num_experts).contiguous()
    topk_weights = torch.full(
        (case.tokens, case.top_k),
        1.0 / case.top_k,
        dtype=torch.float32,
        device="cuda",
    )
    boundaries = [0]
    mapping_values = []
    for length, adapter_id in zip(
        case.segment_lengths, case.segment_adapter_ids, strict=True
    ):
        boundaries.append(boundaries[-1] + length)
        mapping_values.extend((adapter_id,) * length)
    indptr = torch.tensor(boundaries, dtype=torch.int32, device="cuda")
    adapter_ids = torch.tensor(
        case.segment_adapter_ids, dtype=torch.int32, device="cuda"
    )
    mapping = torch.tensor(mapping_values, dtype=torch.int32, device="cuda")
    plan = _build_plan(case, graph_mode=False)
    return Fixture(
        case=case,
        hidden=hidden,
        factor=factor,
        dummy_b=dummy_b,
        output=torch.empty(
            case.tokens,
            case.top_k,
            2 * case.rank,
            dtype=torch.bfloat16,
            device="cuda",
        ),
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        mapping=mapping,
        indptr=indptr,
        adapter_ids=adapter_ids,
        plan=plan,
    )


def _build_plan(case: GateDispatchCase, *, graph_mode: bool):
    return build_shared_outer_gate_a_plan(
        shared_outer=True,
        device_capability=torch.cuda.get_device_capability(),
        phase=case.phase,
        graph_mode=graph_mode,
        num_tokens=case.tokens,
        hidden_size=case.hidden_size,
        rank=case.rank,
        top_k=case.top_k,
        has_base_rows=case.has_base_rows,
        num_segments=len(case.segment_lengths),
        max_segment_len=max(case.segment_lengths),
    )


def _invoke(fixture: Fixture, *, selected: bool, cache: dict) -> None:
    plan = fixture.plan if selected else None
    common = dict(
        output=fixture.output,
        hidden_states=fixture.hidden,
        lora_a=fixture.factor,
        lora_b=fixture.dummy_b,
        topk_ids=fixture.topk_ids,
        topk_weights=fixture.topk_weights,
        token_lora_mapping=fixture.mapping,
        mul_routed_weight=False,
        experts_shared_outer_loras_a=True,
        experts_shared_outer_loras_b=False,
        routing_cache=cache,
        fuse_add_to_output=False,
        num_output_slices=2,
        local_expert_offset=0,
        local_num_experts=fixture.case.num_experts,
        shared_outer_gate_a_plan=plan,
        segment_indptr=fixture.indptr,
        segment_lora_ids=fixture.adapter_ids,
        max_segment_len=max(fixture.case.segment_lengths),
    )
    merged_experts_fused_moe_lora_add(stage="routing", **common)
    merged_experts_fused_moe_lora_add(
        stage="shrink", intermediate_buffer=fixture.output, **common
    )


def _prepare_launch(fixture: Fixture, *, selected: bool, execution: str):
    if execution == "eager":
        return lambda: _invoke(fixture, selected=selected, cache={})
    cache = {}
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _invoke(fixture, selected=selected, cache=cache)
    return graph.replay


def _time(launch, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        launch()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) * 1000.0 / iterations


def _correctness(fixture: Fixture) -> float:
    _invoke(fixture, selected=False, cache={})
    torch.cuda.synchronize()
    generic = fixture.output.clone()
    _invoke(fixture, selected=True, cache={})
    torch.cuda.synchronize()
    return float((fixture.output.float() - generic.float()).abs().max().item())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="all", choices=("all", *CASES_BY_ID))
    parser.add_argument(
        "--execution", choices=("eager", "cuda_graph", "all"), default="all"
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--profile-replays", type=int, default=0)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    cases = CASES if args.case == "all" else (CASES_BY_ID[args.case],)
    executions = (
        ("eager", "cuda_graph") if args.execution == "all" else (args.execution,)
    )
    results = []
    for case in cases:
        fixture = _build_fixture(case)
        for execution in executions:
            fixture.plan = _build_plan(case, graph_mode=(execution == "cuda_graph"))
            selected = fixture.plan.uses_token_dedup
            max_abs = _correctness(fixture) if selected else 0.0
            generic_launch = _prepare_launch(
                fixture, selected=False, execution=execution
            )
            generic_us = _time(
                generic_launch, warmup=args.warmup, iterations=args.iterations
            )
            selected_us = None
            speedup_pct = None
            if selected:
                selected_launch = _prepare_launch(
                    fixture, selected=True, execution=execution
                )
                selected_us = _time(
                    selected_launch, warmup=args.warmup, iterations=args.iterations
                )
                speedup_pct = 100.0 * (generic_us / selected_us - 1.0)
            row = dict(
                case=asdict(case),
                execution=execution,
                selector=asdict(fixture.plan),
                generic_us=generic_us,
                selected_us=selected_us,
                selected_speedup_pct=speedup_pct,
                max_abs_vs_generic=max_abs,
            )
            results.append(row)
            print(json.dumps(row, default=str), flush=True)

            if args.profile_replays:
                launches = [("generic", generic_launch)]
                if selected:
                    launches.append(("token_dedup", selected_launch))
                for name, launch in launches:
                    with torch.cuda.nvtx.range(
                        f"shared_outer_gate::{case.case_id}::{execution}::{name}"
                    ):
                        for _ in range(args.profile_replays):
                            launch()
                torch.cuda.synchronize()
    payload = {
        "schema": "sgl_lora_shared_outer_gate_dispatch_v1",
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "results": results,
    }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
