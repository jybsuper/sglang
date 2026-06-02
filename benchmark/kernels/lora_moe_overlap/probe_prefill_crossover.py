"""Find the BLOCK_SIZE_M crossover for the LoRA-B down-expand across batch sizes.

BSM=16 wins big at decode (bs64, ~1 token/virtual-expert). But at prefill (large
T, many tokens/expert) a larger BSM should give better throughput. This sweeps
T from decode (64) to a prefill chunk (4096) x BSM {16,32,64,128} and reports
true GPU time (CUDA-graph replay) so we can pick a decode-gated threshold that
does NOT regress prefill. rank16. BN fixed at the default 128 and the wide value
(256 for N<=2048, 512 for large N) so the BSM effect is isolated.

Usage: python probe_prefill_crossover.py [--model qwen35] [--ts 64,512,2048,4096]
"""
import argparse

import torch

from sglang.srt.lora.triton_ops.virtual_experts import _invoke_moe_lora_expand_add
from bench_real_shapes import MODEL_STAGES, make_topk
from testbed import build_routing, stage_config

DEV = "cuda"
DT = torch.bfloat16
BLOCK_MS = [16, 32, 64, 128]


def graph_time(fn, iters=50, warmup=15):
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
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record(); g.replay(); b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1e3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODEL_STAGES), default="qwen35")
    p.add_argument("--ts", default="64,256,512,2048,4096")
    args = p.parse_args()
    Ts = [int(x) for x in args.ts.split(",")]
    name, E, K, rank_out, N, R, topk, mul_routed, sum_reduce = next(
        s for s in MODEL_STAGES[args.model] if s[0] == "down"
    )
    wide_bn = 256 if N <= 2048 else 512
    print(f"== {args.model} down-expand BSM crossover (E={E}, N={N}, R={R}; wideBN={wide_bn}) ==")
    print(f"   tokens/expert ~= T/E (topk=1). BSM=16 expected to win while small.")

    for T in Ts:
        g = torch.Generator(device=DEV).manual_seed(0)
        tlm = torch.zeros(T, dtype=torch.int32, device=DEV)
        topk_ids, topk_w = make_topk(T, topk, E, g)
        interm = torch.randn(T * topk, rank_out, dtype=DT, device=DEV, generator=g) * 0.02
        interm_e = interm[:, :R].contiguous()
        lora_b = torch.randn(E, N, R, dtype=DT, device=DEV, generator=g) * 0.02
        out = torch.zeros((T, N) if sum_reduce else (T * topk, N), dtype=DT, device=DEV)

        def mk(bm, bn):
            cfg = stage_config(lora_b, 1, DT, T)
            cfg = {**cfg, "BLOCK_SIZE_M": bm, "num_warps": 4, "num_stages": 1}
            if bn:
                cfg["EXPAND_BLOCK_SIZE_N"] = bn
            s_ids, e_ids, ntpp, _ = build_routing(topk_ids, tlm, E, bm, 1)
            def _fn():
                out.zero_()
                _invoke_moe_lora_expand_add(interm_e, lora_b, out, topk_w, topk_ids,
                                            s_ids, e_ids, ntpp, cfg, mul_routed, sum_reduce)
            return _fn

        row = [f"T={T:<5d}(~{T/E:4.1f}tok/E)"]
        best = None
        for bm in BLOCK_MS:
            t128 = graph_time(mk(bm, 128))
            tw = graph_time(mk(bm, wide_bn))
            tmin = min(t128, tw)
            row.append(f"BSM{bm}:{tmin:5.2f}")
            if best is None or tmin < best[1]:
                best = (bm, tmin)
        row.append(f"-> best BSM={best[0]}")
        print("  " + "  ".join(row))


if __name__ == "__main__":
    main()
