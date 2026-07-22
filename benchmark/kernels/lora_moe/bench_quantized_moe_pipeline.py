#!/usr/bin/env python3
"""Production-plan benchmark for quantized SGL-LoRA MoE providers.

Unlike ``bench_quant_providers.py``, this driver includes the complete
virtual-expert LoRA pipeline around the provider stages:

  gate/up LoRA A+B -> provider W13 -> delta-aware activation -> provider W2
  -> provider finalize -> down LoRA A+B -> BF16/FP32 destination.

The active path is selected by ``build_moe_lora_execution_plan`` and executed
by ``run_sgl_lora_moe_plan`` exactly as the layer dispatch does.  A matched N0
control runs the same provider's base stages without LoRA.  Provider weight
construction is load-time setup and is reported separately from latency.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.kernels.lora_moe.bench_moe_pipeline import _single_rank_runtime
from benchmark.kernels.lora_moe.bench_quant_providers import _make_provider
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutputPacked
from sglang.srt.lora.lora_moe_runners import LoRAInfo
from sglang.srt.lora.sgl_lora.execution import run_sgl_lora_moe_plan
from sglang.srt.lora.sgl_lora.execution_plan import build_moe_lora_execution_plan
from sglang.srt.lora.sgl_lora.triton_ops.virtual_experts import lora_pdl_policy


def _randn(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    device: torch.device,
    scale: float,
) -> torch.Tensor:
    return torch.randn(
        shape, generator=generator, device=device, dtype=torch.bfloat16
    ).mul_(scale)


def _routes(
    tokens: int, experts: int, top_k: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    token = torch.arange(tokens, device=device, dtype=torch.int32)[:, None]
    slot = torch.arange(top_k, device=device, dtype=torch.int32)[None, :]
    topk_ids = (token * top_k + slot * 17).remainder(experts).contiguous()
    raw = (slot.to(torch.float32) + 1.0).expand(tokens, -1)
    topk_weights = (raw / raw.sum(dim=1, keepdim=True)).contiguous()
    return topk_ids, topk_weights


def _token_mapping(
    tokens: int, occupancy: str, adapters: int, device: torch.device
) -> torch.Tensor:
    if occupancy == "base":
        return torch.full((tokens,), -1, device=device, dtype=torch.int32)
    mapping = torch.arange(tokens, device=device, dtype=torch.int32).remainder(adapters)
    if occupancy == "mixed":
        mapping[::3] = -1
    return mapping


def _segments(mapping: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = mapping.tolist()
    starts = [0]
    adapters = []
    for index, value in enumerate(values):
        if index == 0 or value != values[index - 1]:
            if index:
                starts.append(index)
            adapters.append(value)
    starts.append(len(values))
    return (
        torch.tensor(starts, device=mapping.device, dtype=torch.int32),
        torch.tensor(adapters, device=mapping.device, dtype=torch.int32),
    )


class QuantizedPipelineFixture:
    def __init__(
        self,
        *,
        provider_name: str,
        tokens: int,
        experts: int,
        top_k: int,
        hidden: int,
        intermediate: int,
        rank: int,
        adapters: int,
        occupancy: str,
        phase: str,
        graph_mode: bool,
        output_dtype: torch.dtype,
        seed: int,
    ) -> None:
        device = torch.device("cuda")
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        self.provider_name = provider_name
        self.tokens = tokens
        self.experts = experts
        self.top_k = top_k
        self.hidden = hidden
        self.intermediate = intermediate
        self.rank = rank
        self.adapters = adapters
        self.occupancy = occupancy
        self.output_dtype = output_dtype

        self.hidden_seed = _randn(
            (tokens, hidden), generator=generator, device=device, scale=1.0
        )
        self.hidden_work = self.hidden_seed.clone()
        topk_ids, topk_weights = _routes(tokens, experts, top_k, device)
        self.packed_topk_ids = topk_ids.clone().contiguous()
        self.topk_output = StandardTopKOutputPacked(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=torch.empty(0, device=device, dtype=torch.float32),
            packed_topk_ids=self.packed_topk_ids,
        )

        w13 = _randn(
            (experts, 2 * intermediate, hidden),
            generator=generator,
            device=device,
            scale=1.0 / math.sqrt(hidden),
        )
        w2 = _randn(
            (experts, hidden, intermediate),
            generator=generator,
            device=device,
            scale=1.0 / math.sqrt(intermediate),
        )
        setup_started = time.perf_counter()
        self.base, _reference_w13, _reference_w2 = _make_provider(
            provider_name, w13, w2, top_k
        )
        torch.cuda.synchronize()
        self.provider_setup_ms = (time.perf_counter() - setup_started) * 1e3
        del w13, w2, _reference_w13, _reference_w2

        mapping = _token_mapping(tokens, occupancy, adapters, device)
        seg_indptr, req_to_lora = _segments(mapping)
        factor_scale = 0.02
        gate_a = _randn(
            (adapters, experts, 2 * rank, hidden),
            generator=generator,
            device=device,
            scale=factor_scale,
        )
        gate_b = _randn(
            (adapters, experts, 2 * intermediate, rank),
            generator=generator,
            device=device,
            scale=factor_scale,
        )
        down_a = _randn(
            (adapters, experts, rank, intermediate),
            generator=generator,
            device=device,
            scale=factor_scale,
        )
        down_b = _randn(
            (adapters, experts, hidden, rank),
            generator=generator,
            device=device,
            scale=factor_scale,
        )
        lora_ranks = torch.full((adapters,), rank, device=device, dtype=torch.int32)
        adapter_enabled = torch.ones_like(lora_ranks)
        self.lora_info = LoRAInfo(
            gate_up_lora_a_weights=gate_a,
            gate_up_lora_b_weights=gate_b,
            down_lora_a_weights=down_a,
            down_lora_b_weights=down_b,
            seg_indptr=seg_indptr,
            req_to_lora=req_to_lora,
            lora_ranks=lora_ranks,
            adapter_enabled=adapter_enabled,
            token_lora_mapping=mapping,
            max_lora_rank=rank,
            num_experts=experts,
            has_active_lora=occupancy != "base",
            experts_shared_outer_loras=False,
            tp_size=1,
            tp_rank=0,
            hidden_size=hidden,
            lora_use_virtual_experts=True,
            forward_phase=phase,
            use_cuda_graph=graph_mode,
            has_base_rows=occupancy != "active",
        )
        self.runner_config = SimpleNamespace(
            top_k=top_k,
            routed_scaling_factor=1.0,
        )
        self.plan = build_moe_lora_execution_plan(
            phase=phase,
            graph_mode=graph_mode,
            num_tokens=tokens,
            rank=rank,
            has_base_rows=self.lora_info.has_base_rows,
            two_stream_requested=False,
            fused_supported=False,
            provider_key=self.base.contract.key,
        )

    def reset_hidden(self) -> None:
        self.hidden_work.copy_(self.hidden_seed)

    def dispatch_output(self) -> StandardDispatchOutput:
        return StandardDispatchOutput(
            hidden_states=self.hidden_work,
            hidden_states_scale=None,
            topk_output=self.topk_output,
        )

    def run_active(self) -> torch.Tensor:
        result = run_sgl_lora_moe_plan(
            self.dispatch_output(),
            self.base.quant_info,
            self.runner_config,
            self.lora_info,
            self.base,
            self.plan,
            output_dtype=self.output_dtype,
        )
        return result.hidden_states

    def run_n0(self) -> torch.Tensor:
        """Run the provider-matched base stages without virtual-expert LoRA."""
        topk_ids = self.topk_output.topk_ids
        topk_weights = self.topk_output.topk_weights
        self.base.validate_runtime_inputs(
            self.hidden_work, output_dtype=self.output_dtype
        )
        ws = self.base.prepare(
            self.hidden_work,
            topk_ids,
            self.top_k,
            topk_weights=topk_weights,
            packed_topk_ids=self.packed_topk_ids,
        )
        if ws.packed_topk_ids is not self.packed_topk_ids:
            raise AssertionError("provider did not preserve packed top-k identity")
        gateup = torch.empty(
            self.base.gateup_out_shape(ws),
            device=self.hidden_work.device,
            dtype=self.base.contract.gate_up_output_dtype,
        )
        self.base.gateup(ws, gateup)
        activation = torch.empty(
            self.base.act_out_shape(ws),
            device=self.hidden_work.device,
            dtype=self.base.contract.lora_activation_dtype,
        )
        bridge = torch.empty(
            (self.tokens, self.top_k, self.intermediate),
            device=self.hidden_work.device,
            dtype=self.base.contract.lora_activation_dtype,
        )
        self.base.act_with_delta(ws, gateup, None, topk_ids, activation, bridge)
        down = torch.empty(
            self.base.down_out_shape(ws),
            device=self.hidden_work.device,
            dtype=torch.bfloat16,
        )
        self.base.down(ws, activation, down)
        output = torch.empty(
            (self.tokens, self.hidden),
            device=self.hidden_work.device,
            dtype=self.output_dtype,
        )
        self.base.finalize(ws, down, topk_ids, topk_weights, 1.0, output)
        return output


def _capture(fn):
    from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode

    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with model_capture_mode():
        with torch.cuda.graph(graph):
            output = fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph, output


def _time(fn, *, warmups: int, iterations: int) -> float:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) / iterations


def _max_abs(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return float((lhs.float() - rhs.float()).abs().max().item())


def _run(args: argparse.Namespace) -> dict[str, object]:
    output_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.output_dtype]
    fixture = QuantizedPipelineFixture(
        provider_name=args.provider,
        tokens=args.tokens,
        experts=args.experts,
        top_k=args.top_k,
        hidden=args.hidden,
        intermediate=args.intermediate,
        rank=args.rank,
        adapters=args.adapters,
        occupancy=args.occupancy,
        phase=args.phase,
        graph_mode=args.execution == "cuda_graph",
        output_dtype=output_dtype,
        seed=args.seed,
    )

    fixture.reset_hidden()
    n0_eager = fixture.run_n0().clone()
    fixture.reset_hidden()
    active_eager = fixture.run_active().clone()
    torch.cuda.synchronize()

    if args.execution == "cuda_graph":
        n0_graph, n0_output = _capture(fixture.run_n0)
        active_graph, active_output = _capture(fixture.run_active)
        n0_graph.replay()
        active_graph.replay()
        torch.cuda.synchronize()
        measured_n0 = n0_output.clone()
        measured_active = active_output.clone()
        n0_fn = n0_graph.replay
        active_fn = active_graph.replay
    else:
        measured_n0 = n0_eager
        measured_active = active_eager
        n0_fn = fixture.run_n0
        active_fn = fixture.run_active

    finite = bool(torch.isfinite(measured_active).all().item())
    graph_eager_max_abs = _max_abs(active_eager, measured_active)
    delta = measured_active.float() - measured_n0.float()
    delta_max_abs = float(delta.abs().max().item())
    base_mask = fixture.lora_info.token_lora_mapping < 0
    base_rows_max_abs = (
        _max_abs(measured_active[base_mask], measured_n0[base_mask])
        if bool(base_mask.any())
        else None
    )
    # Provider rounding is identical in N0 and active base rows.  The remaining
    # tolerance covers independent accumulation order in the down-LoRA no-op.
    base_tolerance = 0.08 if args.provider == "nvfp4" else 0.05
    # Rank-128 virtual-expert shrink uses the established split-K BF16 atomic
    # reduction. Independent eager/capture launches can therefore differ by a
    # few BF16 ulps before the quantized base tail amplifies them. Base rows are
    # checked separately and remain under the tighter provider-N0 tolerance.
    graph_tolerance = 0.16 if args.provider == "nvfp4" else 0.12
    passed = (
        finite
        and graph_eager_max_abs <= graph_tolerance
        and (base_rows_max_abs is None or base_rows_max_abs <= base_tolerance)
        and (args.occupancy == "base" or delta_max_abs > 0.0)
    )

    if args.profile_range:
        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(args.profile_iterations):
            active_fn()
        torch.cuda.cudart().cudaProfilerStop()
        torch.cuda.synchronize()

    n0_ms = _time(n0_fn, warmups=args.warmups, iterations=args.iterations)
    active_ms = _time(active_fn, warmups=args.warmups, iterations=args.iterations)
    props = torch.cuda.get_device_properties(0)
    return {
        "schema": "sgl_lora_quantized_production_plan_v1",
        "status": "pass" if passed else "fail",
        "provider": args.provider,
        "contract": asdict(fixture.base.contract),
        "plan": asdict(fixture.plan),
        "device": props.name,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "shape": {
            "tokens": args.tokens,
            "experts_local": args.experts,
            "top_k": args.top_k,
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "rank": args.rank,
            "adapters": args.adapters,
            "occupancy": args.occupancy,
            "phase": args.phase,
        },
        "execution": args.execution,
        "output_dtype": str(output_dtype),
        "semantics": {
            "production_plan_entrypoint": "run_sgl_lora_moe_plan",
            "virtual_experts": True,
            "packed_topk_identity_preserved": True,
            "lora_arithmetic_dtype": "torch.bfloat16",
            "provider_w2_input_dtype": str(fixture.base.contract.w2_input_dtype),
            "activation_lora_bridge_dtype": str(
                fixture.base.contract.lora_activation_dtype
            ),
        },
        "correctness": {
            "finite": finite,
            "graph_vs_eager_max_abs": graph_eager_max_abs,
            "base_rows_vs_provider_n0_max_abs": base_rows_max_abs,
            "base_tolerance": base_tolerance,
            "graph_tolerance": graph_tolerance,
            "active_delta_max_abs": delta_max_abs,
        },
        "latency_ms": {
            "provider_matched_n0": n0_ms,
            "production_plan": active_ms,
            "plan_over_n0_ratio": active_ms / n0_ms,
        },
        "provider_setup_ms_excluded": fixture.provider_setup_ms,
        "warmups": args.warmups,
        "iterations": args.iterations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("fp8", "nvfp4", "marlin"), required=True)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--rank", type=int, choices=(16, 32, 64, 128), default=64)
    parser.add_argument("--adapters", type=int, default=2)
    parser.add_argument(
        "--occupancy", choices=("active", "mixed", "base"), default="active"
    )
    parser.add_argument("--phase", choices=("decode", "prefill"), default="decode")
    parser.add_argument("--execution", choices=("eager", "cuda_graph"), default="eager")
    parser.add_argument("--output-dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--profile-range", action="store_true")
    parser.add_argument("--profile-iterations", type=int, default=3)
    parser.add_argument(
        "--pdl",
        choices=("auto", "off"),
        default="auto",
        help="LoRA virtual-expert PDL policy; provider kernels keep their native policy",
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("quantized production-plan harness requires CUDA")
    if args.top_k > args.experts:
        raise ValueError("top-k cannot exceed local experts in this local harness")

    started = time.time()
    with _single_rank_runtime():
        with lora_pdl_policy(None if args.pdl == "auto" else False):
            result = _run(args)
    result["lora_pdl_policy"] = args.pdl
    result["wall_seconds"] = time.time() - started
    encoded = json.dumps(result, indent=2, sort_keys=True, default=str)
    print(encoded)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(encoded + "\n")
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
