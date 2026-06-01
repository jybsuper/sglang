"""Kimi-K2.5-NVFP4 LoRA kernel testbed — attn + MoE shrink/expand at *runtime* shapes, + 2-stream.

Extends the MoE-LoRA testbed idea from jybsuper/sglang#3 to:
  * the **attn** LoRA sgemm kernels (q_a / q_b / kv_a / o), timed per module,
  * the **MoE** LoRA virtual-experts kernels (gate_up + down), shrink and expand timed separately,
  * **Kimi-K2.5-NVFP4 per-rank shapes** (TP8, no-EP), parametrized by --bs / --seq-len / --regime / --topk,
  * a **2-stream** overlap measurement (gate_up LoRA on a side stream vs sequential).

No server needed — just one GPU + an sglang install on the `nvfp4-lora` branch.

Shapes are derived from the live model config + adapter (see
`~/Desktop/nvfp4-lora/kimi-k25-shapes-roofline.md`):
  hidden=7168, q_lora_rank=1536, kv_lora_rank+rope=576, o_in=8192, 64 heads;
  MoE 384 routed experts (NOT EP-sharded under virtual-experts), top-8, moe_inter=2048;
  adapter r=16 / alpha=32 (scale 2.0), max_lora_rank=32 (gate_up stacks gate+up A -> rank-dim 32).
TP8 sharding: q_b -> 12288/8=1536, o input -> 8192/8=1024, moe_inter -> 2048/8=256 per rank.

The runtime token axis M:
  decode :  M = bs                 (one token per running sequence)
  prefill:  M = bs * seq_len       (chunked-prefill upper bound; the shape axis is "many tokens")
and the MoE sees M*top_k routed (token, expert) pairs.

Usage
  python bench_kimi_lora.py verify                                   # correctness gate (attn+moe vs torch)
  python bench_kimi_lora.py bench   --regime decode  --bs 16,32,64   # per-kernel latency at decode shapes
  python bench_kimi_lora.py bench   --regime prefill --bs 1 --seq-len 2048
  python bench_kimi_lora.py overlap --bs 16,32,64                    # 2-stream gate_up LoRA vs sequential
  python bench_kimi_lora.py all     --bs 16,32,64                    # verify + bench + overlap
  [--section attn|moe|both] [--rank 16] [--max-loras 1] [--dtype bf16]

`do_bench` is wall-clock incl. Python launch overhead — use for relative iteration. For absolute
in-graph kernel time, profile a live run (the real fused two-stream is behind SGLANG_LORA_TWO_STREAM=1
+ SGLANG_LORA_OVERLAP_DOWN=1, see python/sglang/srt/lora/trtllm_moe/moe_overlap.py).
"""

import argparse
import functools

import torch
import triton

# ---- production code under test (real launchers + routing primitives) ----
from sglang.srt.lora.triton_ops import (
    merged_experts_fused_moe_lora_add,
    sgemm_lora_a_fwd,
    sgemm_lora_b_fwd,
)
from sglang.srt.lora.triton_ops.virtual_experts import (
    _align_block_size_large,
    _fused_virtual_topk_ids,
    _get_moe_lora_shrink_split_k,
    _invoke_moe_lora_shrink_splitk,
    fused_sanitize_expert_ids,
)
from sglang.srt.lora.trtllm_moe.specialized_expand import _invoke_moe_lora_expand_add
from sglang.srt.lora.utils import LoRABatchInfo

DEV = "cuda"
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


# =============================================================================
# Kimi-K2.5-NVFP4 per-rank presets (TP8, no-EP)
# =============================================================================
HIDDEN = 7168

# Attn LoRA modules: (name, in_dim, out_dim) per TP rank. q_a/kv_a are Replicated; q_b/o are
# sharded /8. kv_b is *absorbed* (kv_b_lora_absorbed kernel) — listed in NOTE, not benched here.
ATTN_MODULES = [
    ("q_a", 7168, 1536),  # hidden -> q_lora_rank            (Replicated)
    ("q_b", 1536, 1536),  # q_lora_rank -> 12288/8           (ColumnParallel)
    ("kv_a", 7168, 576),  # hidden -> kv_lora_rank+rope      (Replicated)
    ("o", 1024, 7168),    # 8192/8 -> hidden                 (RowParallel input shard)
]
ATTN_NOTE = "kv_b (512->16384) is absorbed into w_kc/w_vc; benched via kv_b_lora_absorbed, not here."

# MoE LoRA stages per rank:
#   (name, E, shrink_K, shrink_rankout, expand_N, expand_R, topk, mul_routed, sum_reduce)
# gate_up: shrink reads hidden K=7168 -> rank-dim 32 (gate+up A stacked); expand N=moe_inter/8=256, R=16
# down   : shrink reads moe_inter/8=256 -> 16;        expand N=hidden=7168, R=16, topk=1 + sum-reduce
MOE_STAGES = [
    ("gate_up", 384, 7168, 32, 256, 16, 8, False, False),
    ("down", 384, 256, 16, 7168, 16, 1, True, True),
]

DEFAULTS = dict(rank=16, alpha=32, topk=8, max_loras=1)


def do_bench(fn, warmup=50, rep=200) -> float:
    """us, wall-clock (incl. launch overhead)."""
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep) * 1e3


# =============================================================================
# Attn LoRA (sgemm_lora_a / sgemm_lora_b)  — per-module shrink + expand
# =============================================================================
def make_batch_info(bs, seg_len, rank, scaling, device=DEV) -> LoRABatchInfo:
    """Triton-backend batch info: num_segments == bs, one segment per request.
    decode: seg_len=1 -> S=bs.  prefill: seg_len=seq_len -> S=bs*seq_len."""
    seg_lens = torch.full((bs,), seg_len, dtype=torch.int32, device=device)
    seg_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    seg_indptr[1:] = torch.cumsum(seg_lens, 0)
    return LoRABatchInfo(
        use_cuda_graph=False,
        bs=bs,
        num_segments=bs,
        seg_indptr=seg_indptr,
        weight_indices=torch.zeros(bs, dtype=torch.int32, device=device),  # single lora id 0
        lora_ranks=torch.tensor([rank], dtype=torch.int32, device=device),
        scalings=torch.tensor([scaling], dtype=torch.float32, device=device),
        max_len=seg_len,
        seg_lens=seg_lens,
        permutation=None,
    )


def attn_inputs(in_dim, out_dim, M, rank, dtype, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(M, in_dim, dtype=dtype, device=DEV, generator=g)
    A = torch.randn(1, rank, in_dim, dtype=dtype, device=DEV, generator=g) * 0.02  # [num_lora, r, in]
    B = torch.randn(1, out_dim, rank, dtype=dtype, device=DEV, generator=g) * 0.02  # [num_lora, out, r]
    return x, A, B


def attn_run(x, A, B, bi):
    """Production path: shrink (no scaling) then expand (applies batch_info.scalings)."""
    inter = sgemm_lora_a_fwd(x, A, bi, stack_num=1)  # [M, r]
    out = sgemm_lora_b_fwd(inter, B, bi)             # [M, out], scaled
    return out, inter


def attn_ref(x, A, B, scaling):
    return (scaling * ((x.float() @ A[0].float().T) @ B[0].float().T))


def attn_verify(M, rank, alpha, dtype):
    print(f"  attn verify (M={M}, r={rank}, scale={alpha/rank:.1f}):")
    bi = make_batch_info(bs=M, seg_len=1, rank=rank, scaling=alpha / rank)  # decode-style, S=M
    ok = True
    for name, in_dim, out_dim in ATTN_MODULES:
        x, A, B = attn_inputs(in_dim, out_dim, M, rank, dtype)
        out, _ = attn_run(x, A, B, bi)
        ref = attn_ref(x, A, B, alpha / rank).to(out.dtype)
        err = (out - ref).abs().max().item()
        rel = err / (ref.abs().max().item() + 1e-9)
        good = rel < 5e-2
        ok &= good
        print(f"    {name:5s} [{in_dim}->{out_dim}]  max|err|={err:.4e} rel={rel:.2e}  {'OK' if good else 'FAIL'}")
    print(f"    NOTE: {ATTN_NOTE}")
    return ok


def attn_bench(bs, seg_len, rank, alpha, dtype):
    # Real request segmentation drives the kernel grid: bs segments of seg_len (decode seg_len=1).
    M = bs * seg_len
    bi = make_batch_info(bs=bs, seg_len=seg_len, rank=rank, scaling=alpha / rank)
    print(f"  -- attn LoRA kernels (bs={bs} seg={seg_len} M={M}) --")
    for name, in_dim, out_dim in ATTN_MODULES:
        x, A, B = attn_inputs(in_dim, out_dim, M, rank, dtype)
        inter = sgemm_lora_a_fwd(x, A, bi, stack_num=1)
        t_s = do_bench(lambda: sgemm_lora_a_fwd(x, A, bi, stack_num=1))
        t_e = do_bench(lambda: sgemm_lora_b_fwd(inter, B, bi))
        print(f"    {name:5s} [{in_dim:>4}->{out_dim:<4}]  shrink={t_s:7.2f}us  expand={t_e:7.2f}us  (sum={t_s+t_e:7.2f}us)")


# =============================================================================
# MoE LoRA (virtual-experts)  — routing mirror + isolated shrink/expand + e2e
# (mirrors virtual_experts._get_stage_config / _get_routing; cited so drift is catchable)
# =============================================================================
def stage_config(weight, stage_top_k, hidden_dtype, n_tokens):
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
        get_config_dtype_str,
        try_get_optimal_moe_config,
    )

    config_dtype = get_config_dtype_str(dtype=hidden_dtype)
    fn = functools.partial(
        try_get_optimal_moe_config, weight.shape, weight.shape, stage_top_k, config_dtype
    )
    try:
        return fn(n_tokens)
    except ValueError:
        K_dim, N_dim = weight.shape[2], weight.shape[1]
        bk = 256 if K_dim >= 1024 else (64 if K_dim >= 64 else max(16, K_dim))
        return dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=min(64, max(16, N_dim)),
                    BLOCK_SIZE_K=min(bk, max(16, K_dim)), GROUP_SIZE_M=1, num_warps=4, num_stages=4)


def build_routing(topk_ids, tlm, num_experts, block_size, max_loras):
    vids, _mask, vne = _fused_virtual_topk_ids(topk_ids, tlm, num_experts, False, max_loras)
    if vne < 1024:
        from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
            moe_align_block_size as native_align,
        )

        s_ids, e_ids, ntpp = native_align(vids, block_size, vne)
    else:
        s_ids, e_ids, ntpp = _align_block_size_large(vids, block_size, vne)
    n_tok = topk_ids.numel()
    tight = triton.cdiv(n_tok + min(n_tok, vne) * (block_size - 1), block_size) * block_size
    s_ids = s_ids[:tight]
    e_ids = fused_sanitize_expert_ids(e_ids[: tight // block_size], vne)
    return s_ids, e_ids, ntpp


def make_topk(M, topk, E, g):
    # top-k of random router scores -> distinct experts per token, vectorized (M can be ~32k at prefill)
    scores = torch.rand(M, E, dtype=torch.float32, device=DEV, generator=g)
    w, ids = scores.topk(topk, dim=1)
    return ids.to(torch.int32), (w / w.sum(1, keepdim=True))


def moe_bench_stage(name, E, K, rank_out, N, R, topk, mul_routed, sum_reduce, M, dtype):
    """Isolated shrink + expand at the per-rank stage shape (the PR's bench_real_shapes, Kimi preset)."""
    g = torch.Generator(device=DEV).manual_seed(0)
    tlm = torch.zeros(M, dtype=torch.int32, device=DEV)  # single active lora
    topk_ids, topk_w = make_topk(M, topk, E, g)

    hidden = torch.randn(M, K, dtype=dtype, device=DEV, generator=g)
    lora_a = torch.randn(E, rank_out, K, dtype=dtype, device=DEV, generator=g) * 0.02
    a_cfg = stage_config(lora_a, topk, dtype, M)
    s_ids, e_ids, ntpp = build_routing(topk_ids, tlm, E, a_cfg["BLOCK_SIZE_M"], 1)
    interm = torch.zeros(M * topk, rank_out, dtype=dtype, device=DEV)
    split_k = _get_moe_lora_shrink_split_k(lora_a, s_ids, a_cfg)

    def _shrink():
        _invoke_moe_lora_shrink_splitk(hidden, lora_a, interm, topk_ids, s_ids, e_ids, ntpp, topk, a_cfg)

    lora_b = torch.randn(E, N, R, dtype=dtype, device=DEV, generator=g) * 0.02
    b_cfg = stage_config(lora_b, 1, dtype, M)
    s_ids_b, e_ids_b, ntpp_b = build_routing(topk_ids, tlm, E, b_cfg["BLOCK_SIZE_M"], 1)
    interm_e = interm[:, :R].contiguous()
    out = torch.zeros((M, N) if sum_reduce else (M * topk, N), dtype=dtype, device=DEV)

    def _expand():
        _invoke_moe_lora_expand_add(interm_e, lora_b, out, topk_w, topk_ids, s_ids_b, e_ids_b, ntpp_b, b_cfg, mul_routed, sum_reduce)

    _shrink(); _expand()
    t_s, t_e = do_bench(_shrink), do_bench(_expand)
    print(f"    {name:8s} [K={K:>4} rankout={rank_out} -> N={N:<4} R={R}] topk={topk}  "
          f"shrink={t_s:7.2f}us (split_k={split_k})  expand={t_e:7.2f}us  (sum={t_s+t_e:7.2f}us)")


def moe_bench(M, rank, alpha, dtype):
    print(f"  -- MoE LoRA kernels (M={M}, per-rank, E=384 not-EP-sharded) --")
    for stage in MOE_STAGES:
        moe_bench_stage(*stage, M=M, dtype=dtype)


def moe_verify(M, rank, alpha, dtype):
    """e2e merged_experts (gate_up stage) vs a torch reference (small M)."""
    print(f"  moe verify (M={M}, gate_up stage):")
    E, topk = 384, DEFAULTS["topk"]
    n_out = MOE_STAGES[0][4]  # 256
    g = torch.Generator(device=DEV).manual_seed(0)
    h = torch.randn(M, HIDDEN, dtype=dtype, device=DEV, generator=g)
    la = torch.randn(1, E, rank, HIDDEN, dtype=dtype, device=DEV, generator=g) * 0.02
    lb = torch.randn(1, E, n_out, rank, dtype=dtype, device=DEV, generator=g) * 0.02
    topk_ids, topk_w = make_topk(M, topk, E, g)
    tlm = torch.zeros(M, dtype=torch.int32, device=DEV)

    out = torch.zeros(M, topk, n_out, dtype=dtype, device=DEV)
    merged_experts_fused_moe_lora_add(
        output=out, hidden_states=h, lora_a=la, lora_b=lb, topk_ids=topk_ids,
        topk_weights=topk_w, token_lora_mapping=tlm, mul_routed_weight=False,
        experts_shared_outer_loras_a=False, experts_shared_outer_loras_b=False,
        routing_cache=None, fuse_add_to_output=False, use_direct_expand_add=rank <= 64,
    )
    ref = torch.zeros(M, topk, n_out, dtype=torch.float32, device=DEV)
    for i in range(M):
        for k in range(topk):
            e = int(topk_ids[i, k])
            ref[i, k] = h[i].float() @ la[0, e].float().T @ lb[0, e].float().T
    err = (out.float() - ref).abs().max().item()
    rel = err / (ref.abs().max().item() + 1e-9)
    good = rel < 5e-2
    print(f"    gate_up e2e vs torch:  max|err|={err:.4e} rel={rel:.2e}  {'OK' if good else 'FAIL'}")
    return good


# =============================================================================
# 2-stream overlap (toy): gate_up LoRA on a side stream vs a main-stream proxy
# =============================================================================
def time_loop(fn, iters=100, warmup=20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms


def overlap_bench(M, rank, alpha, dtype):
    """gate_up LoRA (shrink+expand via merged_experts) overlapped with a main-stream PROXY of the
    base FP4 gate_up GEMM1 (bf16 [M, hidden] x [hidden, moe_inter_per_rank]). ROUGH — single-GPU SM
    contention is shape/clock dependent; for absolute numbers profile the real path behind
    SGLANG_LORA_TWO_STREAM=1 (moe_overlap.py)."""
    E, topk = 384, DEFAULTS["topk"]
    n_out = MOE_STAGES[0][4]
    g = torch.Generator(device=DEV).manual_seed(0)
    h = torch.randn(M, HIDDEN, dtype=dtype, device=DEV, generator=g)
    la = torch.randn(1, E, rank, HIDDEN, dtype=dtype, device=DEV, generator=g) * 0.02
    lb = torch.randn(1, E, n_out, rank, dtype=dtype, device=DEV, generator=g) * 0.02
    topk_ids, topk_w = make_topk(M, topk, E, g)
    tlm = torch.zeros(M, dtype=torch.int32, device=DEV)
    gate_up_delta = torch.empty(M, topk, n_out, dtype=dtype, device=DEV)
    cache: dict = {}

    def _lora():
        merged_experts_fused_moe_lora_add(
            output=gate_up_delta, hidden_states=h, lora_a=la, lora_b=lb, topk_ids=topk_ids,
            topk_weights=topk_w, token_lora_mapping=tlm, mul_routed_weight=False,
            experts_shared_outer_loras_a=False, experts_shared_outer_loras_b=False,
            routing_cache=cache, fuse_add_to_output=False, use_direct_expand_add=rank <= 64,
        )

    # main-stream proxy: bf16 GEMM the size of the per-rank base gate_up GEMM1 (M x hidden x 2*moe_inter/8)
    w_main = torch.randn(HIDDEN, 2 * n_out, dtype=dtype, device=DEV, generator=g)

    def _main():
        return h @ w_main

    side = torch.cuda.Stream()

    def _sequential():
        _main()
        _lora()

    def _overlapped():
        side.wait_stream(torch.cuda.current_stream())
        ev = torch.cuda.Event()
        with torch.cuda.stream(side):
            _lora()
            ev.record()
        _main()
        torch.cuda.current_stream().wait_event(ev)

    t_main = time_loop(_main)
    t_lora = time_loop(_lora)
    t_seq = time_loop(_sequential)
    t_ovl = time_loop(_overlapped)
    hidden_pct = 100 * (t_seq - t_ovl) / t_seq if t_seq else 0
    print(f"    M={M:<5d} main(proxy)={t_main*1e3:6.1f}us lora={t_lora*1e3:6.1f}us  "
          f"seq={t_seq*1e3:6.1f}us overlap={t_ovl*1e3:6.1f}us  hidden={hidden_pct:4.1f}%")


# =============================================================================
# main
# =============================================================================
def run_section(section, kind, bs_list, regime, seq_len, rank, alpha, dtype):
    seg_len = 1 if regime == "decode" else seq_len
    for bs in bs_list:
        M = bs * seg_len
        tag = f"bs={bs}" + (f" seq={seq_len} -> M={M}" if regime == "prefill" else f" -> M={M}")
        print(f"[{kind}] {regime} {tag} {'='*8}")
        if kind == "bench":
            if section in ("attn", "both"):
                attn_bench(bs, seg_len, rank, alpha, dtype)
            if section in ("moe", "both"):
                moe_bench(M, rank, alpha, dtype)
        elif kind == "overlap":
            overlap_bench(M, rank, alpha, dtype)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["verify", "bench", "overlap", "all"])
    p.add_argument("--section", choices=["attn", "moe", "both"], default="both")
    p.add_argument("--regime", choices=["decode", "prefill"], default="decode")
    p.add_argument("--bs", default="16,32,64", help="comma list of batch sizes")
    p.add_argument("--seq-len", type=int, default=2048, help="prefill seq len (M = bs*seq_len)")
    p.add_argument("--rank", type=int, default=DEFAULTS["rank"])
    p.add_argument("--alpha", type=int, default=DEFAULTS["alpha"])
    p.add_argument("--dtype", choices=list(DTYPES), default="bf16")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU (run on a GB200 pod, not the dev box).")
    bs_list = [int(x) for x in args.bs.split(",")]
    dtype = DTYPES[args.dtype]
    print(f"== Kimi-K2.5-NVFP4 LoRA kernels | {torch.cuda.get_device_name()} | "
          f"r={args.rank} alpha={args.alpha} scale={args.alpha/args.rank:.1f} | {args.dtype} ==\n")

    if args.mode in ("verify", "all"):
        print("[verify] correctness vs torch reference (decode, bs=8) " + "=" * 8)
        ok = True
        if args.section in ("attn", "both"):
            ok &= attn_verify(M=8, rank=args.rank, alpha=args.alpha, dtype=dtype)
        if args.section in ("moe", "both"):
            ok &= moe_verify(M=8, rank=args.rank, alpha=args.alpha, dtype=dtype)
        print(f"  => {'ALL PASS' if ok else 'FAILURES'}\n")

    if args.mode in ("bench", "all"):
        run_section(args.section, "bench", bs_list, args.regime, args.seq_len, args.rank, args.alpha, dtype)
        print()

    if args.mode in ("overlap", "all"):
        print("[overlap] 2-stream gate_up LoRA vs main-stream proxy (ROUGH; profile real path for absolute) " + "=" * 4)
        run_section(args.section, "overlap", bs_list, "decode", args.seq_len, args.rank, args.alpha, dtype)


if __name__ == "__main__":
    main()
