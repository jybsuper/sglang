"""Sweep BLOCK_SIZE_M for the LoRA-B *expand* kernel ("reduce blocks" direction).

The expand on the down-gemm is the slow stage (~26us @ bs64 vs ~5us shrink). At
decode, T tokens scatter ~1-per-virtual-expert, so the dense-MoE-tuned
BLOCK_SIZE_M (=64) pads each M-block to mostly masked-out rows. Shrinking
BLOCK_SIZE_M cuts the moe_align padding (fewer real M-blocks) AND the kernel
grid. This script measures the expand latency for BLOCK_SIZE_M in a sweep set,
and asserts the output is numerically invariant to BLOCK_SIZE_M (it only changes
tiling/padding, not the math) by comparing each candidate against the BSM=64
baseline output.

Usage:
    python sweep_expand_block_m.py [--model qwen35] [--ts 16,32,64] [--blocks 16,32,64]
"""
import argparse

import torch
import triton

from sglang.srt.lora.triton_ops.virtual_experts import _invoke_moe_lora_expand_add
from bench_real_shapes import MODEL_STAGES, make_topk
from testbed import build_routing, stage_config

DEV = "cuda"
DT = torch.bfloat16


def build_expand(name, E, K, rank_out, N, R, topk, mul_routed, sum_reduce, T):
    """Construct the expand inputs (matches bench_real_shapes.bench_stage's expand half)."""
    g = torch.Generator(device=DEV).manual_seed(0)
    tlm = torch.zeros(T, dtype=torch.int32, device=DEV)  # single lora active
    topk_ids, topk_w = make_topk(T, topk, E, g)
    interm = torch.randn(T * topk, rank_out, dtype=DT, device=DEV, generator=g) * 0.02
    interm_e = interm[:, :R].contiguous()
    lora_b = torch.randn(E, N, R, dtype=DT, device=DEV, generator=g) * 0.02
    out_shape = (T, N) if sum_reduce else (T * topk, N)
    return dict(
        E=E, N=N, R=R, topk=topk, T=T, mul_routed=mul_routed, sum_reduce=sum_reduce,
        tlm=tlm, topk_ids=topk_ids, topk_w=topk_w, interm_e=interm_e, lora_b=lora_b,
        out_shape=out_shape,
    )


def run_expand(d, block_m):
    """Run the expand once at a given BLOCK_SIZE_M; return the output tensor."""
    cfg = stage_config(d["lora_b"], 1, DT, d["T"])
    cfg = {**cfg, "BLOCK_SIZE_M": block_m}
    s_ids, e_ids, ntpp, _ = build_routing(d["topk_ids"], d["tlm"], d["E"], block_m, 1)
    out = torch.zeros(d["out_shape"], dtype=DT, device=DEV)

    def _expand():
        out.zero_()
        _invoke_moe_lora_expand_add(
            d["interm_e"], d["lora_b"], out, d["topk_w"], d["topk_ids"],
            s_ids, e_ids, ntpp, cfg, d["mul_routed"], d["sum_reduce"],
        )

    _expand()
    return out, _expand


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODEL_STAGES), default="qwen35")
    p.add_argument("--ts", default="16,32,64")
    p.add_argument("--blocks", default="16,32,64")
    args = p.parse_args()
    Ts = [int(x) for x in args.ts.split(",")]
    blocks = [int(x) for x in args.blocks.split(",")]

    # down stage only (the slow expand). gate_up expand is comparatively cheap.
    down = next(s for s in MODEL_STAGES[args.model] if s[0] == "down")
    print(f"== {args.model} down-expand BLOCK_SIZE_M sweep (E={down[1]}, N={down[4]}, R={down[5]}) ==")
    for T in Ts:
        d = build_expand(*down, T=T)
        ref = None
        line = [f"T={T:<5d}"]
        for bm in blocks:
            out, fn = run_expand(d, bm)
            if bm == 64 or ref is None:
                ref = out.clone()
            max_diff = (out.float() - ref.float()).abs().max().item()
            t = triton.testing.do_bench(fn, warmup=50, rep=200)
            line.append(f"BSM={bm}:{t*1e3:6.2f}us(diff={max_diff:.1e})")
        print("  " + "  ".join(line))


if __name__ == "__main__":
    main()
