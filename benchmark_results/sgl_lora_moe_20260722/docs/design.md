# Design: Unified MoE-LoRA Runner (`sgl_lora`, née sgl_moe_lora/cutedsl_moe_lora)

> **SUPERSEDED (2026-07-22).** This document is retained only as historical
> design provenance. It predates the provider-neutral execution contract, the
> measured C2/C3 planner, separate base/LoRA graph families, the activation and
> shared-expert contracts, and the final lifecycle boundary. The authoritative
> design is `sgl_lora_lifecycle_and_orchestration_design.md`; implementation
> order and measured policy live in `sgl_lora_refactor_plan.md` and
> `sgl_lora_moe_kernel_benchmark_architecture_audit.md`. Do not implement an ABI
> or selector rule from this file when those documents differ.
>
> In particular, this historical file's no-LoRA CuTe/Triton routing, broad
> FP8/NVFP4 language, shared-expert assumptions, PDL statements, and P6 provider
> roadmap are not current support claims. At `f2f406e056`, no-LoRA calls the resident
> quant method; shared-outer and physical shared IDs have only the evidence-bounded
> scope in the authoritative docs (9 host tests and 56 tests plus 10 subtests on
> each of H200 and GB300, including TorchNative bounds, ID composition, and PDL);
> per-rank physical shared EP>1 and advanced A2A remain unpromoted;
> active FP8 rejects static activation and
> Blackwell requires resident packed UE8M0 scales; Marlin W4A16 is attachable;
> its per-invocation workspace/dirty-destination repair passed the terminal
> `f2f406e056` H200 and GB300 smoke;
> native CuTe DSL NVFP4 W4A4 remains synthetic/testbed-only and not serving-reachable.
> Route memoization is limited to one MoE layer invocation, not a whole forward.
> Final server gaps versus experimental TRTLLM are C2
> 9.43%/9.44%/10.39% and C3 5.12%/5.54%/10.48% at BS1/16/32; the
> 12.56%-14.71% range is historical pre-planner evidence only.

> **Naming v3 (2026-07-01): `sgl_lora`** — scoped to the MoE runner refactor only (attention-side two-stream O7-O11 = follow-up PR); `sgl_lora` over `fused_moe_lora` to avoid colliding with the existing hooks kernels file. Earlier: renamed from `cutedsl_moe_lora` — DeepGEMM (bf16/fp8) + flashinfer cute_dsl (nvfp4) carry the base GEMMs, so "cutedsl" was a misnomer. Self-authored CuTe DSL kernels remain the P6 optimization tier. S3 triton kernel lives in `srt/lora/sgl_lora/triton_ops/` (not jit_kernel/).

**Status:** v2 · 2026-07-01 (v1 revised after 3-lens adversarial review + csgmv/piecewise/PDL research)
**Replaces:** the experimental `trtllm_lora_temp` MoE path from PR #27329 (vendored flashinfer/trtllm cubin overlay)
**Author:** yanbin.jiang + Claude

---

## 1. Goals / non-goals

**Goals**
- ONE unified fused-MoE-LoRA runner function handling bf16 / fp8-blockscale / nvfp4 by
  delegating base-GEMM stages to per-quant CuTe DSL kernels behind a common interface.
- Kill the flashinfer-internal coupling (no vendored trtllm `.cu`, no
  `gen_trtllm_gen_fused_moe_sm100_module` monkeypatch, no cubin pin, no int-handle event ABI).
- Preserve O1 overlap semantics exactly (side-stream gate_up LoRA ∥ S1/S2; event join before
  activation; serial down-LoRA; shared-add hook).
- **csgmv-compatible by construction** (see §6): consume only the backend-agnostic
  `MoELoRABatchInfo` / `LoRAInfo` contract.
- **PDL first-class** (see §7): every kernel carries `use_pdl` + griddepcontrol hooks from
  day one; full S-chain incl. the existing triton down-LoRA gdc hooks.
- **Piecewise-cuda-graph-safe prefill** (see §8): the MoE-LoRA stage becomes a registered
  custom op (split-op first, captured later) so LoRA stops force-disabling piecewise.
- sm100 primary; sm90 for bf16 (fp8-sm90 see §5 matrix note). W4A16 stays on Marlin
  (`lora_moe_runner_marlin.py`).

**Non-goals**
- Down-LoRA/finalize overlap (bench-negative + replay corruption; seam kept, not wired).
- DeepEP / MTP / per-channel scaling / mxfp8 / mxfp4.
- Attention-side overlaps O7–O11 (stay in `trtllm_lora_temp`; note: they call
  `lora_backend._sgemm_info()` which only exists on TritonLoRABackend — pre-existing
  csgmv landmine, tracked separately).
- Lifting the piecewise LoRA auto-disable for *dense* LoRA layers (separate track, §8).

---

## 2. Decision: backend owns ALL batches; no flashinfer_trtllm delegation

v1 had a contradiction (review blocker): "standard weight layouts" is incompatible with
"delegate no-LoRA batches to flashinfer_trtllm" — same weight tensors, two layouts
(trtllm prep: fp8 `shuffle_matrix_a`, fp4 permute+block layout, bf16 BlockMajorK
[unquant.py:276-300]). Resolution:

- `cutedsl_moe_lora` **requires `--enable-lora`** (server_args validation).
- It does **not** join the `is_flashinfer_trtllm()` tuple; weights keep **standard sglang
  layouts** (triton-compatible; fp4 keeps only the hardware-mandated tcgen05 SF swizzle).
- No-active-LoRA batches run the **same CuTe pipeline with zeroed/skipped deltas**
  (captured decode already works this way today — base decode replays the LoRA-path graph
  with zeroed adapters, so the decode bar is the decomposed pipeline, not the fused cubin).
- **Prefill no-active-LoRA batches** may delegate to any *standard-layout* backend —
  `deep_gemm` (fp8, fast) or triton `fused_experts` — never flashinfer_trtllm. This
  removes the prefill-base perf risk without dual weight copies.

Consequence: the `("none", "cutedsl_moe_lora")` registered fused-func = base CuTe pipeline
(or the standard-layout delegation above), NOT `fused_experts_none_to_flashinfer_trtllm`.

---

## 3. Pipeline (unchanged shape, corrected kernel plan)

```
                     main (alt) stream                                LoRA side stream
routing: fused align (gemm-segments + lora A/B views)  ──┐
stage="routing" pre-warm (BOTH A/B cache keys, §4.2)     │ fork (side.wait_stream(main))
alloc all buffers (main stream, before fork)             ▼
[S1] per-token quant (fp8/fp4; bf16 none) ─┐          gate_up LoRA shrink (A)
[S2] gather/contiguous grouped GEMM1       │ CuTe DSL gate_up LoRA expand (B)
     raw [gate|up] output, no activation  ─┘          record lora_event
wait lora_event  ◄────────────────────────────────────────┘   (Python-side wait, §4.1)
[S3] act kernel: SwiGLU + Δgate_up (expanded idx) + requant + activation_lora_input
[S4] grouped GEMM2 (down)
[S5] finalize: scatter-add top_k × prob × routed_scaling (skip permuted_idx<0)
maybe_overlap_staged_shared_add(output)              (unchanged hook)
down-LoRA shrink+expand (triton, serial; shrink has gdc_wait already)
return StandardCombineInput(output)
```

- **S1 is a standalone kernel** (quant once per *token*, not per expanded row). Quant-in-
  gather-prologue exists in no reference (flashinfer gathers *pre-quantized* rows; LDGSTS
  cannot transform in flight) — gather+quant fusion is a P6 experiment, not the plan.
- **Prefill = same function, no fork**: gate_up LoRA runs inline on main stream between
  S2 and S3. Numerically *equivalent* (not bitwise — shrink split-K uses relaxed bf16
  atomics, pre-existing). Deletes `install_two_stream_overrides` for the MoE fn.
- Routing at prefill: P1 keeps the legacy multi-kernel align path (>512 tokens; the fused
  merged-align kernel is decode-gated today, virtual_experts.py:777-781); extending the
  fused kernel to prefill scale is P6.

### Per-stage kernel basis (evidence-checked against CUTLASS v4.5.2 + flashinfer)

**Architecture principle (v3):** the unification is the runner + `MoeLoraBaseGemm`
interface + our S1/S3/S5/routing kernels. Base GEMMs are pluggable per quant; CuTe DSL
self-authored kernels are the *optimization tier* (P6), not the foundation. Every base
tier below is an existing, pinned, battle-tested kernel with the pre-activation seam.

| Stage | bf16 | fp8 blockscale (per-128, fp32 scales) | nvfp4 |
|---|---|---|---|
| S1 | — (permute only) | per-token-group e4m3 quant (`per_token_group_quant_fp8`) + permute | `scaled_fp4_grouped_quantize` (flashinfer, already used by `flashinfer_cutedsl_moe.py`) or `fused_permute_quant.cuh` port |
| S2 | **DeepGEMM** `m_grouped_bf16_gemm_nt_contiguous` (prefill) / `_masked` (decode, graph-safe) — sm90+sm100 | **DeepGEMM** `m_grouped_fp8_gemm_nt_contiguous` / `fp8_m_grouped_gemm_nt_masked` — standard layouts, per-128 fp32 scales native, sm90+sm100, PDL via `SGLANG_DEEPGEMM_PDL`; `moe_runner/deep_gemm.py` is already this decomposed pipeline | **flashinfer `cute_dsl.blockscaled_gemm.grouped_gemm_nt_masked`** — `flashinfer_cutedsl_moe.py` already runs quant → masked GEMM → *separate* silu+quant kernel → masked GEMM; swap its act kernel for our LoRA-aware S3 and it IS the nvfp4 BaseGemm (pinned wheel, zero GEMM authoring) |
| S3 | port `fused_activation_quant.cuh` (CUDA, zero numerical risk, TE_EXACT knobs, gdc hooks; constexpr layout flags below) | same kernel, e4m3 tail | same kernel, nvfp4+SF tail (replaces `silu_and_mul_scaled_nvfp4_experts_quantize`) |
| S4 | DeepGEMM (as S2) | DeepGEMM (as S2) | `grouped_gemm_nt_masked` (as S2) |
| S5 | new scatter-add kernel (standalone first; fuse into S4 epilogue P6) | same | same |

**P6 optimization tier (CuTe DSL self-authored):** contiguous-layout nvfp4 grouped GEMM
(masked layout `[E, max_m, K]` pads per-expert — decode-friendly and graph-safe by
construction, potentially wasteful at prefill: P3 spike), gather-fusion (flashinfer
LDGSTS-warp pattern), finalize→S4-epilogue fusion, PDL in every stage, and removing the
flashinfer-API coupling if it bites. Scheduler basis for self-authored kernels: CUTLASS
`cute/blackwell/kernel/moe/*` persistent scheduler (the only graph-safe family);
**tcgen05.block_scale cannot express fp32 per-128 scales** — never crib the fp4 kernel
for fp8. Coupling note: nvfp4-via-flashinfer is version coupling again, but to a *public
Python API* (grouped_gemm_nt_masked + quant kernels), not vendored internal headers — a
far lower risk class; the self-authored kernel is the exit ramp.

**S3 layout flags (gpt-oss-class models):** S3 takes constexpr layout params
`{interleaved_gate_up: bool, gate_first: bool}` — covers standard `[gate|up]`, gpt-oss
interleaved `(g0,u0,…)`, and `[up|gate]` stack order via compile-time index mapping
(the ported kernel already has `interleavedGateUpInput`; de-interleave-on-read is free).
Base weights are NEVER relayouted at runtime; the LoRA delta stays canonical contiguous
`[gate|up]` (pool/expand unchanged) and S3 maps delta indices onto the base layout.
Adapters trained against a *fused* nonstandard module (gpt-oss fused `gate_up_proj`,
B interleaved) get their B rows permuted to canonical once at adapter load in
lora_manager. S3 also takes swiglu-variant params `(alpha, clamp_limit)` — plumbing
already exists as `MoeRunnerConfig.gemm1_alpha/gemm1_clamp_limit` (base.py:65-67) —
instead of hard-asserting plain silu.

**sm90:** bf16 adaptable from `cute/hopper/kernel/grouped_gemm` — but it computes cluster
totals on the HOST (not graph-safe for dynamic expert counts); a Hopper device-side
scheduler is novel work owned by P5. **sm90 fp8-blockscale has zero DSL reference**
(hopper grouped example is fp16/bf16-only) — moot under the DeepGEMM decision above:
fp8 uses DeepGEMM on both archs behind `MoeLoraBaseGemm`; the interface, not the DSL,
is the unification. Triton-hooks path remains the universal last-resort fallback only.

**Fork provenance:** cribbed flashinfer DSL kernels (~3.7k lines act_fusion, ~2.8k
finalize_fusion, custom `PipelineCpAsyncUmma`) are a maintained fork — ship a SOURCE.md
with the upstream flashinfer commit pinned, same as `trtllm_lora_temp/SOURCE.md`.

**Compile cache:** key on (arch, dtype, tile/cluster tactic) with **M as a dynamic layout
dim** (`from_dlpack` dynamic mode — flashinfer's pattern), NOT per-M-bucket static
compiles. Budget ~10–30 kernels, seconds each, first-run-only (4.5.2 on-disk cache is
default-on via `CUTE_DSL_DISABLE_FILE_CACHING=False`). M-buckets are used only for tile
*selection* in `tuning.py`.

---

## 4. Streams, events, cuda-graph (corrected)

### 4.1 lora_event
The wait moves to Python between S2/S3 launches — **stream-semantics-identical** to the
in-op `cudaStreamWaitEvent` (launcher.cu:3093). It does **not** change event lifetime:
torch does not manage event teardown under capture (`wait_stream`'s temp event also dies
mid-capture). Therefore **`_LORA_OVERLAP_EVENTS` keep-alive stays** (it shipped with real
capture debugging, commit `cad2ad2f48`), with a fix for unbounded growth (clear on graph
reset). Removal only behind a dedicated gate: max-loras≥2 capture across all decode
buckets + sustained mixed base/LoRA replay accuracy, or a root-cause of the original
teardown failure.

### 4.2 Allocator safety — invariant restated
**No device allocation inside the side-stream context** (not merely "caller allocates
outputs"). Concretely the runner must, on the main stream before the fork:
1. run the `stage="routing"` pre-warm so BOTH routing-cache keys are seeded — A/shrink
   (BLOCK_M=16, incl. shared-outer variant: num_experts=1) and B/expand (tuned BLOCK_M,
   ~64) (virtual_experts.py:756-760, 883-908); a side-stream cache miss allocates during
   capture → the documented `!!!!` corruption at max-loras≥2;
2. pre-allocate `gate_up_delta`, the shrink intermediate, S1–S5 buffers, `output`.
Alternative (P6 cleanup): extend `MoeLoraRouting` to carry per-stage A/B views and inject
them under the exact cache keys, or make `virtual_experts` accept explicit routing tensors.

### 4.3 Other
- `gemm2_done` dropped; re-adding is a Python event between S4/S5 if ever re-benched.
- Shared-add hook unchanged; note it only fires in capture-mode dual-stream forwards and
  falls back to the serial caller add under capture (per its own guard) — carried, not fixed.
- In-graph launch geometry must be data-independent: persistent grid a function of the
  token bucket only; device-side `total_padded`/`expert_tile_offsets`; every consumer
  skips `permuted_idx < 0` (the Kimi-IMA lesson). This is required by BOTH full decode
  graphs and piecewise prefill capture.

---

## 5. File & function layout

### 5.1 `python/sglang/jit_kernel/cutedsl_moe_lora/` (device)

| File | Contents |
|---|---|
| `__init__.py` | exports; compile cache (key: arch/dtype/tactic; M dynamic) |
| `routing.py` | `build_moe_lora_routing(...) -> MoeLoraRouting` — fused align (decode); legacy align passthrough for prefill (P1) |
| `quant_permute.py` | S1 kernels: per-token-group fp8 quant; `fused_permute_quant` port (nvfp4, CUDA); bf16 permute |
| `grouped_gemm.py` | S2/S4 CuTe DSL kernels per quant, MoE persistent scheduler, `use_pdl: cutlass.Constexpr` + gdc hooks |
| `act_quant.py` | S3 `fused_activation_quant` port (CUDA) + gdc hooks; per-quant tails |
| `finalize.py` | S5 scatter-add (+ optional fused down-delta variant) |
| `tuning.py` | tile/tactic tables per (arch, dtype, M-bucket) |
| `SOURCE.md` | provenance: flashinfer commit + CUTLASS example lineage per kernel |

### 5.2 `python/sglang/srt/lora/cutedsl_moe_lora/` (host)

| File | Contents |
|---|---|
| `moe_lora_runner.py` | **`fused_experts_cutedsl_moe_lora(dispatch_output, quant_info, runner_config, lora_info)`** — §3 pipeline; decode fork gated by `is_two_stream_active`; virtual-experts gate + `build_lora_hooks` non-virtual fallback (both per §6) |
| `base_gemm.py` | `MoeLoraBaseGemm` (gateup/act_quant/down/finalize) + `resolve_base_gemm(quant_info)`; concrete Bf16/Fp8BlockScale/Nvfp4 (+ TritonFallback for sm90-fp8 & P0) |
| `quant_info.py` | `CutedslMoe{Bf16,Fp8,Fp4}QuantInfo` msgspec.Structs. Constraints: declare 3-D `w13_weight`/`w2_weight` fields (BaseLoRABackend.init_cuda_graph_moe_buffers unpacks `E,N,_ = qinfo.w13_weight.shape`, base_backend.py:190-194); no post-hoc attr injection (msgspec forbids — the bf16 reshape hack must be a constructor arg); deliberately does NOT subclass the `MoeQuantInfo` @dataclass ABC (no isinstance sites exist) |
| `sgl_backend.py` | `@register_fused_func("none", "cutedsl_moe_lora")` → base pipeline / standard-layout delegation (§2) |
| `lora_layer.py` | `prepare_cutedsl_moe_lora(layer, base_layer)` + `dispatch_cutedsl_moe_lora(...)` (one fn, no isinstance routing) + the piecewise custom-op entry (§8) |
| `piecewise.py` | `moe_lora_forward_piecewise_cuda_graph_impl` custom op (§8) |

Kept & imported: `virtual_experts.py` (+routing cache), `shared_add_overlap.py`, side-stream
helpers, `moe_lora_merged_align`, **kimi fused-gate JIT kernel** (retained minus its
packed-topk output; `SGLANG_OPT_USE_JIT_KERNEL_KIMI_GATE` survives).

### 5.3 Injection points (full list — v1 understated ~3x)

| Site | Change |
|---|---|
| `moe/utils.py:93,115-123` | enum value + `is_cutedsl_moe_lora()`; **NOT** added to `is_flashinfer_trtllm()` tuple |
| `server_args.py:239` | `MOE_RUNNER_BACKEND_CHOICES` + help |
| `server_args.py:~5407` | validation gate: quant whitelist, `disable_shared_experts_fusion=True`, **require `--enable-lora`** |
| `moe_runner/runner.py:58-71` | `MoeRunner.__init__` branch (runner_core=None style) — else `NotImplementedError` at model build |
| `fp8.py:~1786` | `create_moe_runner` whitelist (else silent `self.runner` unset → AttributeError) |
| `modelopt_quant.py:~2274` | same for nvfp4 |
| `unquant.py:398-420` | same for bf16 (else silently selects TRITON) |
| `lora/layers.py:922,987,1056` | three gates: init / num_experts / dispatch |
| `lora/mem_pool.py:117-125` | `_moe_runner_keeps_global_expert_ids` += new backend (EP expert-id space) |
| `token_dispatcher/standard.py:98-105` | `skip_local_expert_mapping` += new backend (else double EP remap) |
| registration import site | un-gated from `_SGLANG_EXPERIMENTAL_LORA_OPTI` (see below) |
| `server_args.py:3164` | piecewise LoRA auto-disable rule narrowed (§8, last step) |

**`is_flashinfer_trtllm()` audit:** ~35 sites key on the tuple (fp8.py ×8, modelopt ×4,
compressed_tensors ×5, fused_moe_triton/layer.py padding, deepseek_v2.py correction-bias
dtype / MTP topk format / `_can_dual_stream_graph`, …). Not joining the tuple is the
decision; each kimi/qwen-relevant site gets an explicit audit line in the P1 PR
(esp. deepseek_v2.py:452 correction-bias dtype, which interacts with the kept kimi bf16 gate).

**environ:** the reused knobs (`SGLANG_TWO_STREAM_MAX_TOKENS`,
`SGLANG_OPT_LORA_SHARED_ADD_OVERLAP`, `SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC`) are hard-gated
on master `SGLANG_EXPERIMENTAL_LORA_OPTI` today (environ.py:28-47) — overrides silently
ignored when master off. For a first-class backend: re-home them to an ungated registry
(env-var-conventions skill) as part of P1; new knob `SGLANG_CUTEDSL_MOE_LORA_PDL` (§7).

---

## 6. csgmv compatibility (verified: free at the MoE layer)

- Every dense LoRA backend (triton/csgmv/ascend/torch) calls
  `BaseLoRABackend._add_moe_lora_info` → **`MoELoRABatchInfo`** (seg_indptr, req_to_lora,
  adapter_enabled, `token_lora_mapping` in original token order, −1 = none). csgmv feeds
  it from dedicated `req_seg_indptr/req_weight_indices` precisely because its own
  seg_indptr is chunk-permuted (utils.py:67-75; base_backend.py:172-188,256-311).
- `lora_use_virtual_experts` is an independent server arg, orthogonal to `--lora-backend`;
  gate on `lora_info.lora_use_virtual_experts`, never on the backend string.
- LoRA weights are always the 4D pool buffers `[num_loras, E|1, d1, d2]` — no per-backend
  store difference; `token_lora_mapping` (the routing kernel's input) is identical under
  csgmv and triton.

**Contract:** the runner consumes ONLY `LoRAInfo`/`MoELoRABatchInfo` — never
`LoRABatchInfo` internals (`permutation`, chunk `seg_indptr`, `weight_indices`), and no
`TritonLoRABackend`-only methods (`_sgemm_info` — the attention-overlap landmine). Keep
both delta paths: virtual-experts (`merged_experts_fused_moe_lora_add`) and the
non-virtual `build_lora_hooks` fallback (both already dense-backend-agnostic).
`init_cuda_graph_moe_buffers` compatibility via the `w13_weight/w2_weight` field contract (§5.2).

---

## 7. PDL (first-class, day-one hooks)

- **Gate:** `is_arch_support_pdl()` (jit_kernel/utils.py:418; sm90+, non-HIP), resolved
  once at runner init. Env kill-switch `SGLANG_CUTEDSL_MOE_LORA_PDL` (EnvBool, mirrors
  `SGLANG_DEEPGEMM_PDL`). Effective = env AND capability.
- **API (proven in-repo):** `.launch(..., use_pdl=True)` +
  `cute.arch.griddepcontrol_wait()` in loader/gather warps right before the first
  dependent global read (independent setup — smem, mbarriers, scheduler init — before the
  wait) + `cute.arch.griddepcontrol_launch_dependents()` from one elected warp after the
  last global store. Templates: `cutedsl_dsv3_fused_a_gemm.py:199-291`,
  `nvfp4_gemm_swiglu_nvfp4_quant.py`. (gdn/kda/quack have no PDL — don't copy from them.)
- **Chain:** routing → S1 → S2 → S3 → S4 → S5 → down-LoRA shrink. The triton tail already
  has gdc wired (`virtual_experts.py` shrink kernels + `get_pdl_launch_metadata`); CuTe
  producer → triton consumer is a valid PDL pair. Expand-add stays PDL-free (event-gated
  by `add_done`). CUDA-kernel ports (S1-fp4, S3) get `cudaGridDependencySynchronize`
  hooks added.
- **Composition facts (verified):** the PDL attribute only relaxes same-stream
  predecessor serialization; the interposed `lora_event` stream-wait before S3 remains a
  hard gate — composition is safe (worst case: driver degrades S2→S3 to a normal launch,
  losing only overlap). PDL works inside full-graph capture (trtllm cubins + triton decode
  attention already do this) and inside piecewise-captured segments (fp4 MoE custom op
  precedent). No capture-time disable needed.
- **Rollout:** kernels compile with the `use_pdl` constexpr from P1 (default False —
  identical behavior, stable JIT cache keys); flip env default ON after an nsys check of
  the S2→S3 boundary (the one open question: edge-type conversion under capture for the
  side-stream expand edge — verify on the first PDL PR).

---

## 8. Prefill cuda-graph: tc_piecewise AND breakable

Two prefill graph backends exist (`cuda_graph_config.py:38-58`): **tc_piecewise**
(default; torch.compile/Dynamo FX-split, token-bucket shapes) and **breakable**
(`BreakableCudaGraphBackend` — segmented real `torch.cuda.CUDAGraph`s split at
`eager_on_graph`-marked fns, NO torch.compile, request-count shapes, pool-pinning keeps
cross-segment storage stable; needs `cuda-python`).

**Breakable vs tc_piecewise for LoRA:**
- Breakable's eager-escape is one decorator (`eager_on_graph(True)(fn)`) — no custom op,
  no schema, no fake_impl, no Dynamo constraints; streams/events legal in eager fns. The
  captured route needs only buffer stability (same machinery decode full-graph already
  has via `init_cuda_graph_moe_buffers`; extend `prepare_lora_batch`'s `use_cuda_graph`
  gate to EXTEND). The dense-LoRA problem also reduces to buffer stability — no Dynamo
  wrapping track at all.
- BUT breakable is disabled for **MLA attention** (`server_args.py:3203` — q.view shape
  mismatch, "until the MLA prefill path is BCG-aware") → kimi/deepseek can't use it;
  tc_piecewise remains required for the flagship. Breakable is the faster win for
  qwen3.5-class (GQA + mamba; `bcg_unified_linear_attention_with_output` exists).
- ⚠ Today-hole: breakable has **no LoRA disable rule** (unlike tc_piecewise's
  `server_args.py:3164`), so `--cuda-graph-backend-prefill=breakable --enable-lora` is
  enabled-but-unvalidated: captured segments bake LoRA batch-info pointers that are
  reallocated per EXTEND batch → likely silent wrong-adapter replay. Reproduce early;
  consider an upstream disable rule until the stable-buffer work lands.

**Design: graph-mode-agnostic runner.** ONE entry fn + two thin adapters:
`@register_custom_op` + `@register_split_op` (tc_piecewise) and `eager_on_graph`
(breakable). The captured-route prereqs (stable max-token buffers, data-independent
launch geometry) are shared and land once (P5).

### 8.1 tc_piecewise specifics

Facts (verified): piecewise = prefill-only (decode stays FULL graph — two-stream decode
path unaffected). Dynamo FX-splits at registered split ops; non-split subgraphs are
captured per token bucket; split ops run eagerly each iteration reading live context.
LoRA is disabled by ONE blanket rule (`server_args.py:3164`, from commit `c64274c746`,
no specific justification); the rule set is skipped when the user locks
`--cuda-graph-backend-prefill=tc_piecewise` — an immediate test harness. Real blockers
today: `FusedMoEWithLoRA.forward` has no piecewise branch, zero custom ops under
`srt/lora/`, LoRA graph-buffer refresh excludes EXTEND, and the monkey-patched forwards
contain Dynamo-hostile stream/event code. Runtime Dynamo recompilation is fatal
(cuda_piecewise_backend.py:160-172) → nothing Dynamo-visible may branch on per-batch data.

**Plan:**
1. **P1 (split-op route, trivially correct):** register
   `sglang.moe_lora_forward_piecewise_cuda_graph_impl` — custom op (schema: tensors +
   `layer_id: int`; layer & lora_info via `TcPiecewiseForwardContext`; explicit
   `fake_impl` with out_shape = hidden_states; all optional tensors pre-declared;
   shape-derived ints computed inside the body) + `@register_split_op`. Add the
   `is_in_tc_piecewise_cuda_graph()` branch to the new `lora_layer.py` dispatch,
   mirroring `FusedMoE.forward` (fused_moe_triton/layer.py:1087-1110). Body runs eagerly:
   free to use side streams/events/live batch info. Padded-tail invariants already hold
   (`token_lora_mapping` = −1 in padding, scatter skips −1, zeroed logits).
2. **P6 (captured, perf phase):** drop the split-op registration so the MoE-LoRA op is
   captured in-graph. Prereqs: stable max-token side-input buffers
   (token_lora_mapping/seg/ranks/scalings) refreshed pre-replay — extend
   `prepare_lora_batch`'s `use_cuda_graph` gate to piecewise EXTEND or register prefill
   GraphSlots; capture with a representative LoRA state (today's dummy batch has
   `lora_ids=None`); data-independent launch geometry (§4.3, already required).
3. **Rule narrowing (last):** change `server_args.py:3164` from blanket-LoRA to
   "LoRA and not cutedsl_moe_lora-capable config" — NOTE this additionally requires the
   *dense* LoRA linear layers to be wrapped/Dynamo-clean (separate track; until then the
   locked-backend flag is the opt-in).

---

## 9. Phasing (rescoped after feasibility review)

| Phase | Deliverable | Gate |
|---|---|---|
| P0 | scaffolding: packages, registry, unified runner + routing + S3 port + S5, with **DeepGEMM bf16 BaseGemm** (contig+masked, sm90+sm100 — production from day one, not a placeholder; decomposed-triton BaseGemm kept as universal fallback) — proves dispatch, S2/S3 event join, overlap, decode graph wiring with near-zero new GEMM code | qwen bf16+LoRA gsm8k parity; decode tput ≥ triton path, then ≥ trtllm-lora bf16 |
| P1 | fp8 blockscale via **DeepGEMM BaseGemm** (small delta over P0) + fp8 S1; environ re-home; injection audit | qwen3.5-FP8 parity + tput ≥ current fp8 path |
| P2 | nvfp4 sm100 via **flashinfer `grouped_gemm_nt_masked` BaseGemm** + S1/S3 ports (masked-layout routing variant) | kimi-regression (acc+perf+profile) |
| P3 | prefill graphs: tc_piecewise split-op adapter + breakable `eager_on_graph` adapter; breakable+LoRA hole repro; masked-vs-contiguous nvfp4 prefill spike | LoRA prefill graphs on for qwen3.5 (breakable) + kimi (tc_piecewise) |
| P4 | PDL enable-by-default after nsys verification; two-stream + graph soak | no base-decode perturbation |
| P5 | captured-route prereqs: stable max-token LoRA buffers (EXTEND refresh), representative capture state; captured MoE-LoRA op | net-positive vs split-op/eager |
| P6 | CuTe DSL optimization tier: contiguous nvfp4 grouped GEMM, gather-fusion, finalize→S4 epilogue, fused routing at prefill, keep-alive removal gate, MoeLoraRouting cache-key injection | net-positive bench each |
| P7 | delete `trtllm_lora_temp` MoE path + vendored csrc | — |

Branch: `cutedsl-lora` dev branch; minimally-invasive injections; commit WIP.

---

## 10. Open questions

1. fp8 S2 raw-output dtype (bf16 vs e4m3+scale) — bench in P2.
2. PDL S2→S3 edge behavior with interposed event under capture (P4 nsys check).
3. Prefill launch-overhead delta of split-op vs captured MoE-LoRA op at 4k–16k buckets —
   decides if P6-captured is worth the buffer-registry work.
4. EP8 kimi validation of routing/mem_pool/token_dispatcher injection correctness (P3).
5. Whether S3 ever moves to CuTe DSL (only payoff: later S2-epilogue fusion).
