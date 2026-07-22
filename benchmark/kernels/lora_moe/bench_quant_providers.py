#!/usr/bin/env python3
"""Correctness and latency harness for decomposed SGL LoRA MoE providers.

The harness exercises the provider boundary directly: route/quantize, W13,
nonzero BF16 LoRA delta + SwiGLU, optional A2 quantization, W2, and weighted
FP32 finalize.  It intentionally does not import the temporary TRT-LLM LoRA
tree and does not include virtual-expert LoRA GEMM latency.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.lora.sgl_lora.base_gemm import resolve_base_gemm
from sglang.srt.lora.sgl_lora.quant_info import (
    SglLoraBf16QuantInfo,
    SglLoraFp8QuantInfo,
    SglLoraMarlinQuantInfo,
    SglLoraNvFp4QuantInfo,
)


def _routes(tokens: int, experts: int, top_k: int, device: torch.device):
    logits = torch.randn(tokens, experts, device=device, dtype=torch.float32)
    weights = logits.softmax(-1)
    weights, ids = weights.topk(top_k, dim=-1)
    weights /= weights.sum(-1, keepdim=True)
    return ids.to(torch.int32), weights.to(torch.float32)


def _block_dequant_fp8(
    quant: torch.Tensor, scale: torch.Tensor, block_n: int, block_k: int
) -> torch.Tensor:
    expanded = scale.repeat_interleave(block_n, 0).repeat_interleave(block_k, 1)
    return quant.float() * expanded[: quant.shape[0], : quant.shape[1]]


def _make_fp8_weight(weight: torch.Tensor):
    from sglang.srt.layers.quantization.fp8_utils import per_block_cast_to_fp8

    quantized, scales, references = [], [], []
    for expert_weight in weight:
        quant, scale = per_block_cast_to_fp8(expert_weight)
        quantized.append(quant)
        scales.append(scale)
        references.append(_block_dequant_fp8(quant, scale, 128, 128))
    quantized_weight = torch.stack(quantized)
    weight_scale = torch.stack(scales)
    from sglang.srt.layers import deep_gemm_wrapper

    if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
        # Match Fp8MoEMethod's Blackwell post-load representation.  DeepGEMM
        # consumes packed UE8M0 weight scales on SM100; ordinary block-FP32
        # scales are a different ABI even though the FP8 values have the same
        # logical [E,N,K] shape.
        from sglang.srt.layers.quantization.fp8_utils import (
            requant_weight_ue8m0,
        )

        quantized_weight, weight_scale = requant_weight_ue8m0(
            quantized_weight, weight_scale, [128, 128]
        )
        reference = weight
    else:
        reference = torch.stack(references).to(weight.dtype)
    return quantized_weight, weight_scale, reference


def _make_marlin_weight(weight: torch.Tensor):
    from sgl_kernel.scalar_type import scalar_types

    from sglang.test.test_marlin_utils import marlin_quantize

    references, packed, scales = [], [], []
    for expert_weight in weight:
        ref, qweight, scale, _g_idx, _sort, _ = marlin_quantize(
            expert_weight.T,
            scalar_types.uint4b8,
            128,
            False,
            None,
        )
        references.append(ref.T)
        packed.append(qweight)
        scales.append(scale)
    return (
        torch.stack(packed).contiguous(),
        torch.stack(scales),
        torch.stack(references).to(weight.dtype),
    )


def _make_nvfp4_weights(w13: torch.Tensor, w2: torch.Tensor):
    from flashinfer import scaled_fp4_grouped_quantize

    experts = w13.shape[0]
    device = w13.device
    input_scale = torch.ones(experts, device=device, dtype=torch.float32)
    a2_scale = torch.ones(experts, device=device, dtype=torch.float32)
    w13_amax = w13.abs().amax(dim=(1, 2)).float().clamp_min(1e-5)
    w2_amax = w2.abs().amax(dim=(1, 2)).float().clamp_min(1e-5)
    w13_global = 448.0 * 6.0 / w13_amax
    w2_global = 448.0 * 6.0 / w2_amax
    w13_sizes = torch.full((experts,), w13.shape[1], device=device, dtype=torch.int32)
    w2_sizes = torch.full((experts,), w2.shape[1], device=device, dtype=torch.int32)
    w13_q, w13_sf = scaled_fp4_grouped_quantize(w13, w13_sizes, w13_global)
    w2_q, w2_sf = scaled_fp4_grouped_quantize(w2, w2_sizes, w2_global)
    return SglLoraNvFp4QuantInfo(
        # grouped quant returns logical [N, K/2, E]. Keep the resident model
        # representation [E, N, K/2] without changing the packed values.
        w13_weight=w13_q.permute(2, 0, 1),
        w2_weight=w2_q.permute(2, 0, 1),
        w13_blockscale=w13_sf,
        w2_blockscale=w2_sf,
        w13_alpha=1.0 / (input_scale * w13_global),
        w2_alpha=1.0 / (a2_scale * w2_global),
        w13_input_scale=input_scale,
        w2_input_scale=a2_scale,
        num_local_experts=experts,
        intermediate_size=w2.shape[2],
        hidden_size=w13.shape[2],
    )


def _make_provider(
    name: str,
    w13: torch.Tensor,
    w2: torch.Tensor,
    top_k: int,
):
    experts, two_inter, hidden = w13.shape
    inter = two_inter // 2
    reference_w13, reference_w2 = w13, w2
    if name == "bf16":
        quant_info = SglLoraBf16QuantInfo(
            w13_weight=w13,
            w2_weight=w2,
            num_local_experts=experts,
            intermediate_size=inter,
            hidden_size=hidden,
        )
    elif name == "fp8":
        w13_q, w13_scale, reference_w13 = _make_fp8_weight(w13)
        w2_q, w2_scale, reference_w2 = _make_fp8_weight(w2)
        quant_info = SglLoraFp8QuantInfo(
            w13_weight=w13_q,
            w2_weight=w2_q,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
            block_shape=(128, 128),
            num_local_experts=experts,
            intermediate_size=inter,
            hidden_size=hidden,
        )
    elif name == "marlin":
        w13_q, w13_scale, reference_w13 = _make_marlin_weight(w13)
        w2_q, w2_scale, reference_w2 = _make_marlin_weight(w2)
        quant_info = SglLoraMarlinQuantInfo(
            w13_weight=w13_q,
            w2_weight=w2_q,
            w13_scales=w13_scale,
            w2_scales=w2_scale,
            weight_bits=4,
            num_local_experts=experts,
            intermediate_size=inter,
            hidden_size=hidden,
        )
    elif name == "nvfp4":
        quant_info = _make_nvfp4_weights(w13, w2)
    else:
        raise ValueError(f"unknown provider {name!r}")
    provider = resolve_base_gemm(quant_info, SimpleNamespace(top_k=top_k))
    return provider, reference_w13, reference_w2


def _reference(
    hidden: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens, top_k = topk_ids.shape
    inter = w2.shape[-1]
    output = torch.zeros(tokens, w2.shape[1], device=hidden.device, dtype=torch.float32)
    bridge = torch.empty(
        tokens, top_k, inter, device=hidden.device, dtype=torch.bfloat16
    )
    for token in range(tokens):
        for slot in range(top_k):
            expert = int(topk_ids[token, slot])
            gateup = torch.mv(w13[expert].float(), hidden[token].float()).to(
                torch.bfloat16
            )
            gate = gateup[:inter].float() + delta[token, slot, :inter].float()
            up = gateup[inter:].float() + delta[token, slot, inter:].float()
            activated = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
            bridge[token, slot] = activated
            down = torch.mv(w2[expert].float(), activated.float()).to(torch.bfloat16)
            output[token] += down.float() * topk_weights[token, slot]
    return output, bridge


def _invoke(
    provider,
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    packed_topk_ids: torch.Tensor,
    delta: torch.Tensor,
    *,
    collect_stats: bool = True,
):
    top_k = topk_ids.shape[1]
    provider.validate_runtime_inputs(hidden, output_dtype=torch.float32)
    ws = provider.prepare(
        hidden,
        topk_ids,
        top_k,
        topk_weights=topk_weights,
        packed_topk_ids=packed_topk_ids,
    )
    if ws.packed_topk_ids is not packed_topk_ids:
        raise AssertionError("provider replaced the caller's packed top-k object")
    gateup = torch.empty(
        provider.gateup_out_shape(ws),
        device=hidden.device,
        dtype=provider.contract.gate_up_output_dtype,
    )
    provider.gateup(ws, gateup)
    activation = torch.empty(
        provider.act_out_shape(ws),
        device=hidden.device,
        dtype=provider.contract.lora_activation_dtype,
    )
    bridge = torch.empty(
        (*topk_ids.shape, provider.quant_info.intermediate_size),
        device=hidden.device,
        dtype=provider.contract.lora_activation_dtype,
    )
    provider.act_with_delta(ws, gateup, delta, topk_ids, activation, bridge)
    down = torch.empty(
        provider.down_out_shape(ws), device=hidden.device, dtype=torch.bfloat16
    )
    provider.down(ws, activation, down)
    output = torch.empty(
        hidden.shape[0], hidden.shape[1], device=hidden.device, dtype=torch.float32
    )
    provider.finalize(ws, down, topk_ids, topk_weights, None, output)
    if not collect_stats:
        return output, bridge, {}
    stage_stats = {
        "output_max_abs": output.abs().max().item(),
    }
    mapped_rows = ws.src2dst[topk_ids.reshape(-1) >= 0].to(torch.long)
    stage_stats.update(
        {
            "mapped_gateup_max_abs": gateup.view(-1, gateup.shape[-1])[mapped_rows]
            .abs()
            .max()
            .item(),
            "mapped_activation_max_abs": activation.view(-1, activation.shape[-1])[
                mapped_rows
            ]
            .abs()
            .max()
            .item(),
            "mapped_down_max_abs": down.view(-1, down.shape[-1])[mapped_rows]
            .abs()
            .max()
            .item(),
        }
    )
    return output, bridge, stage_stats


def _timed_ms(fn, warmups: int, iterations: int) -> float:
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


def _capture_provider(
    provider,
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    packed_topk_ids: torch.Tensor,
    delta: torch.Tensor,
):
    # Seed all JIT/cache/workspace state before capture. The provider workspace
    # returned inside the graph remains alive through the captured outputs.
    _invoke(
        provider,
        hidden,
        topk_ids,
        topk_weights,
        packed_topk_ids,
        delta,
        collect_stats=False,
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _invoke(
            provider,
            hidden,
            topk_ids,
            topk_weights,
            packed_topk_ids,
            delta,
            collect_stats=False,
        )
    graph.replay()
    torch.cuda.synchronize()
    return graph, captured


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider", choices=("bf16", "fp8", "marlin", "nvfp4"), required=True
    )
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--intermediate", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--mode", choices=("eager", "graph"), default="eager")
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("quant-provider harness requires CUDA")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    scale_h = 1.0 / math.sqrt(args.hidden)
    scale_i = 1.0 / math.sqrt(args.intermediate)
    hidden = torch.randn(args.tokens, args.hidden, device=device, dtype=torch.bfloat16)
    w13 = (
        torch.randn(
            args.experts,
            2 * args.intermediate,
            args.hidden,
            device=device,
            dtype=torch.bfloat16,
        )
        * scale_h
    )
    w2 = (
        torch.randn(
            args.experts,
            args.hidden,
            args.intermediate,
            device=device,
            dtype=torch.bfloat16,
        )
        * scale_i
    )
    topk_ids, topk_weights = _routes(args.tokens, args.experts, args.top_k, device)
    # The provider treats this as opaque. A clone makes pointer replacement
    # detectable without depending on a particular bit-packing algorithm.
    packed_topk_ids = topk_ids.clone().contiguous()
    delta = (
        torch.randn(
            args.tokens,
            args.top_k,
            2 * args.intermediate,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.05
    )

    started = time.time()
    provider, ref_w13, ref_w2 = _make_provider(args.provider, w13, w2, args.top_k)
    graph = None
    if args.mode == "graph":
        graph, (actual, actual_bridge, stage_stats) = _capture_provider(
            provider, hidden, topk_ids, topk_weights, packed_topk_ids, delta
        )
    else:
        actual, actual_bridge, stage_stats = _invoke(
            provider, hidden, topk_ids, topk_weights, packed_topk_ids, delta
        )
    expected, expected_bridge = _reference(
        hidden, ref_w13, ref_w2, topk_ids, topk_weights, delta
    )
    torch.cuda.synchronize()
    output_diff = (actual - expected).abs()
    bridge_diff = (actual_bridge - expected_bridge).abs().float()
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten(), expected.flatten(), dim=0
    ).item()
    max_abs = output_diff.max().item()
    mean_abs = output_diff.mean().item()
    bridge_max_abs = bridge_diff.max().item()
    finite = bool(torch.isfinite(actual).all().item())

    limits = {
        "bf16": (0.04, 0.999),
        "fp8": (0.35, 0.97),
        "marlin": (0.35, 0.97),
        # The oracle uses the source BF16 weight because the grouped NVFP4
        # swizzle is provider-private. Cosine is the primary quantized check.
        "nvfp4": (0.8, 0.90),
    }
    abs_limit, cosine_limit = limits[args.provider]
    passed = finite and max_abs <= abs_limit and cosine >= cosine_limit
    latency_ms = _timed_ms(
        (
            graph.replay
            if graph is not None
            else lambda: _invoke(
                provider, hidden, topk_ids, topk_weights, packed_topk_ids, delta
            )
        ),
        args.warmups,
        args.iterations,
    )
    props = torch.cuda.get_device_properties(device)
    result = {
        "schema": "sgl_lora_quant_provider_v1",
        "provider": args.provider,
        "mode": args.mode,
        "contract_key": provider.contract.key,
        "status": "pass" if passed else "fail",
        "device": props.name,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "shape": {
            "tokens": args.tokens,
            "experts": args.experts,
            "top_k": args.top_k,
            "hidden": args.hidden,
            "intermediate": args.intermediate,
        },
        "semantics": {
            "gate_up_delta_dtype": str(delta.dtype),
            "activation_lora_input_dtype": str(actual_bridge.dtype),
            "output_dtype": str(actual.dtype),
            "packed_topk_identity_preserved": True,
        },
        "correctness": {
            "finite": finite,
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "cosine": cosine,
            "activation_bridge_max_abs": bridge_max_abs,
            "max_abs_limit": abs_limit,
            "cosine_limit": cosine_limit,
            "expected_output_max_abs": expected.abs().max().item(),
            **stage_stats,
        },
        "latency_ms": latency_ms,
        "warmups": args.warmups,
        "iterations": args.iterations,
        "wall_seconds": time.time() - started,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(encoded + "\n")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
