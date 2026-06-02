"""True GPU-time probe for the LoRA-B down-expand via CUDA graph replay.

do_bench re-invokes the Python callable each rep, so for a tiny kernel the
CPU launch overhead (python -> triton -> cuLaunch) floors the measurement at
~20us regardless of shape. Production decode runs under a CUDA graph, where
that overhead is captured once and replayed. This probe captures
[out.zero_(); expand] into a CUDA graph and times pure replay, isolating the
kernel's real GPU cost. Also times the shrink and a bare out.zero_() for
calibration. bs64, rank16.

Usage: python probe_expand_cudagraph.py [--model qwen35] [--t 64]
"""
import argparse

import torch

from sglang.srt.lora.triton_ops.virtual_experts import (
    _get_moe_lora_shrink_split_k,
    _invoke_moe_lora_expand_add,
    _invoke_moe_lora_shrink_splitk,
)
from bench_real_shapes import MODEL_STAGES, make_topk
from testbed import build_routing, stage_config

DEV = "cuda"
DT = torch.bfloat16


def graph_time(fn, iters=100, warmup=20):
    """Capture fn into a CUDA graph and return mean replay time in us."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3  # us


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODEL_STAGES), default="qwen35")
    p.add_argument("--t", type=int, default=64)
    args = p.parse_args()
    T = args.t
    name, E, K, rank_out, N, R, topk, mul_routed, sum_reduce = next(
        s for s in MODEL_STAGES[args.model] if s[0] == "down"
    )
    print(f"== {args.model} down-expand CUDA-graph GPU time (E={E}, N={N}, R={R}, T={T}) ==")

    g = torch.Generator(device=DEV).manual_seed(0)
    tlm = torch.zeros(T, dtype=torch.int32, device=DEV)
    topk_ids, topk_w = make_topk(T, topk, E, g)
    interm = torch.randn(T * topk, rank_out, dtype=DT, device=DEV, generator=g) * 0.02
    interm_e = interm[:, :R].contiguous()
    lora_b = torch.randn(E, N, R, dtype=DT, device=DEV, generator=g) * 0.02
    out = torch.zeros((T, N) if sum_reduce else (T * topk, N), dtype=DT, device=DEV)

    # calibration: bare memset
    print(f"  zero_(out) only            : {graph_time(lambda: out.zero_()):6.2f}us")

    # shrink for reference (the ~5us counterpart)
    hidden = torch.randn(T * topk, K, dtype=DT, device=DEV, generator=g)
    lora_a = torch.randn(E, rank_out, K, dtype=DT, device=DEV, generator=g) * 0.02
    a_cfg = stage_config(lora_a, topk, DT, T)
    s_ids_a, e_ids_a, ntpp_a, _ = build_routing(topk_ids, tlm, E, a_cfg["BLOCK_SIZE_M"], 1)
    interm_full = torch.zeros(T * topk, rank_out, dtype=DT, device=DEV)
    def _shrink():
        _invoke_moe_lora_shrink_splitk(hidden, lora_a, interm_full, topk_ids,
                                       s_ids_a, e_ids_a, ntpp_a, topk, a_cfg)
    print(f"  shrink (LoRA-A) reference  : {graph_time(_shrink):6.2f}us")

    def mk_expand(bm, bn, w):
        cfg = stage_config(lora_b, 1, DT, T)
        cfg = {**cfg, "BLOCK_SIZE_M": bm, "num_warps": w, "num_stages": 1}
        if bn:
            cfg["EXPAND_BLOCK_SIZE_N"] = bn
        s_ids, e_ids, ntpp, _ = build_routing(topk_ids, tlm, E, bm, 1)
        def _expand():
            out.zero_()
            _invoke_moe_lora_expand_add(interm_e, lora_b, out, topk_w, topk_ids,
                                        s_ids, e_ids, ntpp, cfg, mul_routed, sum_reduce)
        return _expand

    configs = [(64, None, 4), (16, None, 4), (32, 256, 4), (16, 512, 4), (16, 256, 4)]
    print("  expand [zero+kernel]:")
    for bm, bn, w in configs:
        t = graph_time(mk_expand(bm, bn, w))
        tag = "baseline" if (bm, bn, w) == (64, None, 4) else ""
        print(f"     BSM={bm:<3d} BN={str(bn):<5s} w={w}: {t:6.2f}us  {tag}")


if __name__ == "__main__":
    main()
