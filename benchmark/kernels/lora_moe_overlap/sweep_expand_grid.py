"""Grid-search (BLOCK_SIZE_M, BLOCK_SIZE_N, num_warps) for the LoRA-B down-expand.

Extends the BLOCK_SIZE_M sweep: the bigger "reduce blocks" lever is the n-block
count (N / BLOCK_SIZE_N). The kernel defaults BLOCK_SIZE_N=128 when N%128==0, so
kimi's down (N=7168) launches 56 n-blocks. EXPAND_BLOCK_SIZE_N (added to the
invoke) lets us widen the N tile. Asserts output is numerically invariant to the
tiling vs the production baseline (BSM=64, BN=128).

Usage: python sweep_expand_grid.py [--model qwen35] [--ts 32,64]
"""
import argparse

import torch
import triton

from sglang.srt.lora.triton_ops.virtual_experts import _invoke_moe_lora_expand_add
from bench_real_shapes import MODEL_STAGES, make_topk
from testbed import build_routing, stage_config

DEV = "cuda"
DT = torch.bfloat16

BLOCK_MS = [16, 32]
BLOCK_NS = [128, 256, 512]
NUM_WARPS = [4, 8]


def build_expand(name, E, K, rank_out, N, R, topk, mul_routed, sum_reduce, T):
    g = torch.Generator(device=DEV).manual_seed(0)
    tlm = torch.zeros(T, dtype=torch.int32, device=DEV)
    topk_ids, topk_w = make_topk(T, topk, E, g)
    interm = torch.randn(T * topk, rank_out, dtype=DT, device=DEV, generator=g) * 0.02
    interm_e = interm[:, :R].contiguous()
    lora_b = torch.randn(E, N, R, dtype=DT, device=DEV, generator=g) * 0.02
    out_shape = (T, N) if sum_reduce else (T * topk, N)
    return dict(E=E, N=N, R=R, topk=topk, T=T, mul_routed=mul_routed, sum_reduce=sum_reduce,
                tlm=tlm, topk_ids=topk_ids, topk_w=topk_w, interm_e=interm_e, lora_b=lora_b,
                out_shape=out_shape)


def run(d, block_m, block_n, num_warps):
    cfg = stage_config(d["lora_b"], 1, DT, d["T"])
    cfg = {**cfg, "BLOCK_SIZE_M": block_m, "EXPAND_BLOCK_SIZE_N": block_n,
           "num_warps": num_warps, "num_stages": 1}
    s_ids, e_ids, ntpp, _ = build_routing(d["topk_ids"], d["tlm"], d["E"], block_m, 1)
    out = torch.zeros(d["out_shape"], dtype=DT, device=DEV)

    def _expand():
        out.zero_()
        _invoke_moe_lora_expand_add(d["interm_e"], d["lora_b"], out, d["topk_w"],
                                    d["topk_ids"], s_ids, e_ids, ntpp, cfg,
                                    d["mul_routed"], d["sum_reduce"])
    _expand()
    return out, _expand


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODEL_STAGES), default="qwen35")
    p.add_argument("--ts", default="32,64")
    args = p.parse_args()
    Ts = [int(x) for x in args.ts.split(",")]
    down = next(s for s in MODEL_STAGES[args.model] if s[0] == "down")
    N = down[4]
    print(f"== {args.model} down-expand grid (E={down[1]}, N={N}, R={down[5]}) ==")
    for T in Ts:
        d = build_expand(*down, T=T)
        # production baseline: BSM=64, BN=128 (current default path)
        base_out, base_fn = run(d, 64, 128, 4)
        base_t = triton.testing.do_bench(base_fn, warmup=50, rep=200) * 1e3
        ref = base_out.clone()
        print(f"-- T={T} -- baseline(BSM=64,BN=128,w4)={base_t:6.2f}us")
        results = []
        for bm in BLOCK_MS:
            for bn in BLOCK_NS:
                if N % bn != 0:
                    continue
                for w in NUM_WARPS:
                    out, fn = run(d, bm, bn, w)
                    diff = (out.float() - ref.float()).abs().max().item()
                    t = triton.testing.do_bench(fn, warmup=50, rep=200) * 1e3
                    results.append((t, bm, bn, w, diff))
        results.sort()
        for t, bm, bn, w, diff in results:
            spd = base_t / t
            flag = "  <-- BAD DIFF" if diff > 1e-2 else ""
            print(f"     BSM={bm:<3d} BN={bn:<4d} w={w}: {t:6.2f}us  ({spd:4.2f}x)  diff={diff:.1e}{flag}")


if __name__ == "__main__":
    main()
