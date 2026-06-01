# Kimi-K2.5-NVFP4 LoRA kernel testbed

Standalone (no server, just one GPU) testbed to **benchmark the attn and MoE LoRA kernels
separately at real Kimi-K2.5-NVFP4 per-rank shapes**, parametrized by batch size / seq len /
top-k / regime, plus a 2-stream overlap measurement. Borrows the structure of
[`jybsuper/sglang#3`](https://github.com/jybsuper/sglang/pull/3) (the MoE-LoRA testbed),
extended to the attn path and to Kimi shapes.

For a kernel colleague: edit a kernel → `verify` (correctness gate) → `bench` (per-kernel latency)
→ iterate. No 2-node MNNVL serving stack needed.

## Run

```bash
cd benchmark/kernels/kimi_lora_kernels

python bench_kimi_lora.py verify                                  # correctness vs torch (attn + moe)
python bench_kimi_lora.py bench   --regime decode  --bs 16,32,64  # decode: M = bs
python bench_kimi_lora.py bench   --regime prefill --bs 1 --seq-len 2048   # prefill: M = bs*seq_len
python bench_kimi_lora.py overlap --bs 16,32,64                   # 2-stream gate_up LoRA vs sequential
python bench_kimi_lora.py all     --bs 16,32,64                   # verify + bench + overlap

# pick one path:  --section attn | moe | both   (default both)
# overrides:      --rank 16 --alpha 32 --dtype bf16
```

Each `bench` row prints **shrink** and **expand** latency separately (the LoRA-A / LoRA-B kernels),
which is where the decode cost lives (see the roofline doc).

## What's covered (Kimi per-rank, TP8, no-EP)

**Attn LoRA** — `sgemm_lora_a_fwd` (shrink) + `sgemm_lora_b_fwd` (expand), per module:

| module | in → out | sharding |
|--------|----------|----------|
| q_a  | 7168 → 1536 | Replicated (hidden → q_lora_rank) |
| q_b  | 1536 → 1536 | ColumnParallel (12288/8) |
| kv_a | 7168 → 576  | Replicated (hidden → kv_lora_rank+rope) |
| o    | 1024 → 7168 | RowParallel input shard (8192/8 → hidden) |

> `kv_b` (512→16384) is **absorbed** into the w_kc/w_vc bmm (`kv_b_lora_absorbed` kernel) — not benched here.

**MoE LoRA** — virtual-experts `_invoke_moe_lora_shrink_splitk` + `_invoke_moe_lora_expand_add`,
isolated, plus the e2e `merged_experts_fused_moe_lora_add`:

| stage | shrink K → rank-dim | expand N ← R | top-k | note |
|-------|---------------------|--------------|-------|------|
| gate_up | 7168 → 32 | 256 ← 16 | 8 | gate+up A stacked → rank-dim 32; N = moe_inter/8 |
| down    | 256 → 16  | 7168 ← 16 | 1 | sum-reduce over top-k + all-reduce |

E = **384 routed experts per rank** — virtual-experts LoRA is **not EP-sharded**.

## Runtime token axis

| regime | M (tokens the kernels see) | MoE routed (token,expert) pairs |
|--------|----------------------------|---------------------------------|
| decode  | `bs`            | `bs * top_k` over 384 experts (≈ bs·8/384 per expert — sparse) |
| prefill | `bs * seq_len`  | `bs * seq_len * top_k` (dense) |

Attn timing uses the real request segmentation (`bs` segments of `seq_len`); the grid is
`cdiv(max_seg_len, 16) * cdiv(R, 16) × bs`.

## Relation to the real 2-stream (e2e)

`overlap` mode is a **rough** single-GPU toy: gate_up LoRA on a side stream concurrent with a bf16
GEMM proxy of the base FP4 gate_up GEMM1. SM contention is shape/clock dependent — use it to *sweep
shapes*, not for absolute numbers. The **real** fused two-stream is wired e2e in
`python/sglang/srt/lora/trtllm_moe/moe_overlap.py`:

- `SGLANG_LORA_TWO_STREAM=1` — gate_up LoRA shrink/expand on the side stream, the trtllm FP4 op waits
  on its event right before activation (permute + GEMM1 overlap the LoRA).
- `SGLANG_LORA_OVERLAP_DOWN=1` — down LoRA shrink/expand/all-reduce on the side stream, concurrent
  with the op's requant + down-GEMM + finalize.

For absolute in-graph kernel time, profile a live run (graph **on**) and isolate the `step[DECODE]`
regions (see the `kimi-regression` skill's `decode_isolate.py`).

## Caveats

- `do_bench` is wall-clock **incl. Python launch overhead** — great for relative iteration, not absolute.
- The MoE routing mirror (`build_routing` / `stage_config`) reproduces `virtual_experts._get_routing`
  / `_get_stage_config`; `verify` cross-checks it against the e2e op + a torch reference.
- Shapes are NVFP4-derived (`~/Desktop/nvfp4-lora/kimi-k25-shapes-roofline.md`); to capture *exact*
  production shapes on a live server, use the `SGLANG_DEBUG_LORA_MOE_SHAPES` shape-dump from the
  reference PR (cherry-pick if not on this branch).
