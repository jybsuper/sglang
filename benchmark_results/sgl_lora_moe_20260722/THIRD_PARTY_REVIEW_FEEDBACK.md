# Feedback: sgl_lora MoE kernel benchmark & architecture plan

Reviewer: Claude · 2026-07-22
Reviewed: `sgl_lora_refactor_plan.md` (incl. §8 matrix), `sgl_lora_moe_kernel_benchmark_architecture_audit.md`,
`sgl_lora_worklog.md`, `sgl_lora_lifecycle_and_orchestration_design.md`,
`lora_external_optimization_audit_2026-07-21.md`, branch `sgl-lora` @ `4ffaee0b41` (+ working tree),
benchmark drivers `benchmark/kernels/lora_moe/`, local `benchmark_results/`.
Method: 6 independent review lenses (methodology, evidence→conclusions, plan sequencing,
missed dimensions, cross-doc hygiene, code spot-check); 70 findings; all severity claims below
were verified against source/docs by the reviewing agents with file:line evidence.

## Verdict

This is an unusually disciplined campaign: the evidence-level ladder (U0..E0) with documented
scope reversals, discarded-not-rationalized invalid measurements, effective-config
normalization, forced-cold protocol, per-boundary unfair-comparison columns, and the
one-replay confirmation that caught batched-graph inflation are all genuinely excellent and
should be preserved verbatim (see §5). The audit's architecture description survived a hostile
line-level spot-check almost perfectly, ~40 cross-quoted numbers match exactly, and all nine
recommendations from the 2026-07-10 review were traceably adopted.

However, the review found **one critical correctness bug in the active engine**, a cluster of
**silent-wrong-result paths opened by the guard-removal cleanup**, and — most importantly for
the benchmark campaign — **three validity threats that undermine specific published
conclusions** (vacuous correctness tolerances, a mislabeled non-IID routing generator, and an
O0 boundary that double-charges a shared route plan). None of these invalidates the overall
program; all are fixable cheaply, and several past results can be re-adjudicated from retained
JSONs without new GPU time.

---

## 1. Critical

### C1. Fused shared-expert IDs silently corrupt the virtual-expert space
`sgl_lora/triton_ops/virtual_experts.py:82-84,888-892`. The virtual-ID kernel handles only
NEGATIVE sentinels (`tl.where(base < 0, base, shifted)`). With shared-experts **fusion**
(default-on for DeepSeek/GLM/Kimi-class models — the staged scope targets), every token's
`topk_ids` contains the fused shared-expert ID `== E_routed`, while LoRA factor buffers are
sized to `E_routed` only. For adapter slot `s`: `virtual = E_routed + s*E_routed =
(s+1)*E_routed` — **adapter slot s+1's expert-0 factors are applied to slot s's shared-expert
rows** (cross-adapter contamination at L>1; unsanitized OOB factor reads at L==1). The audit's
`-1` story (§9.2) covers negative IDs but misses valid-but-out-of-factor-domain IDs. Phase-1a
previously force-disabled shared-experts fusion via server_args; the guard-removal cleanup
appears to have dropped that protection. **Fix now**: either re-add the fusion-disable for
`sgl_lora` or sanitize `id >= E_factor` to `-1` in the virtual-ID kernel; add a fused-shared-ID
unit test. This must land before any DeepSeek/GLM/Kimi-class run.

---

## 2. Benchmark validity threats (fix before the next campaign)

### V1. M0/two-stream/indexed-A correctness gates are mathematically vacuous
`bench_moe_pipeline.py:867,876,896`. At the driver's synthetic scales (weights/factors ±0.02),
max|final output| ≈ 0.024 and max|total LoRA delta| ≈ 3.7e-3, while zero-LoRA parity,
production-vs-candidate, and C0-vs-C1 checks use `rtol=atol=6e-2` — the assert passes **for a
fully dropped LoRA delta or an all-zeros output**. The strict `_check_lora_delta` oracle
(atol 1.5e-3, base-subtracted) is wired only to the B-override path; every indexed-A M0
substitution (§8.1/8.3/8.4) and every C0-vs-C1 two-stream comparison relies on the vacuous
check. The graph-vs-eager check (atol 3e-3) provably cannot catch losing only the down-LoRA
contribution (≈2.2e-3). Two-stream event/join races are exactly the unguarded failure class.
**Fix (nearly free)**: run `_check_lora_delta` against the already-computed `sgl_base_only`
for C0, C1, and A-substitutions; derive atol from recorded `reference_delta_max_abs`
(≤ signal/10). Retained JSONs record observed max-abs diffs → **re-adjudicate past runs post
hoc, no new GPU time**.

Same class: `bench_local.py:731-742` — the tightened atol=2e-4 branch covers only
`gate_b/gate_ab` non-direct/synthetic; **down_b/down_ab and direct-gate_b-with-production-A
still use the vacuous 6e-2**. The §7.2 decode-large (T256/R32) B winners rest solely on this
check (no retained M0 for that cell) — a config dropping masked tail rows would win the screen
and pass "correctness". Apply the tight branch to every B target.

### V2. Routing is a deterministic lattice, not the claimed "seeded IID"
All four drivers route via `topk_ids = (t*13 + s*7) % E` (`bench_moe_pipeline.py:366`,
`bench_local.py:381`) — deterministic, never re-drawn; only weights are seeded. Measured
consequences: at T32/K8/E256 the lattice hits **211 distinct experts vs an IID mean of 163**
(~29% more fragmentation than IID) — flattering plan-free indexed-A at exactly the anchors
that selected it; at T256 it gives **exactly 8 rows to all 256 experts** — an idealized
zero-imbalance load maximally friendly to grouped/aligned schedules at the flagship
direct-vs-generic reversal anchor. Zero routing-draw variance was ever sampled; no conclusion
carries a routing error bar. §11.8's "seeded IID is not real skew" understates twice: it's
not IID and not sampled. **Fix**: correct the label in `matrix.py`/audit §4.1; rerun key
anchors over ≥3 true-IID seeds + one skewed draw (E_hit is already recorded → the
fragmentation covariate is free); re-check whether T32 indexed-vs-grouped margins survive.

### V3. O0's headline indexed-A saving double-charges a shared resource
The 55.5–71.9% route-inclusive O0 reduction (§6.3, graded High in §11.7) attributes the whole
virtualize/sort/align cost to A — but every active B family consumes the **same** aligned plan
(same cache key, §9.2), so in any real pipeline B becomes the plan's builder and the realized
saving is ~zero plan builds. The audit's own M0 numbers prove it: C0 improves ~10.45µs ≈ the
summed K0 kernel deltas (10.78µs); none of the ~80–107µs O0 route saving materialized.
**Fix**: redefine O0 for shared preparations as *marginal* cost (plan cost minus what surviving
consumers still need), or advance raw-A only in tandem with a raw/segment B; downgrade §11.7's
entry to "operator-isolated, not pipeline-realizable"; reword §1's "when route fragmentation
and planning cost dominate" (planning cost cannot dominate while any plan-consuming B runs).

### V4. Ladder blind spot: quiet-host bias + hot-L2 M0 for the bandwidth-bound site
Graph-M0 amortizes launches to zero and eager rungs run on a quiet host, so
launch-count-reducing candidates are systematically under-rewarded for the eager/E0 paths
(prefill is eager today); and the M0-decided B table was selected under hot-L2 replay in
exactly the site (down tail) where the campaign's own K0 cold data shows 30–50% swings — the
external audit's own figure says warm-L2 microbenchmarks ran 15–20% optimistic. **Fix**: add a
loaded-host eager variant (or at least record host contention as a caveat on eager rows) and
one forced-cold M0 bracket for the down-B decision before C2 freezes the fused-down design.

### V5. Published bracket deltas carry no dispersion for graph mode
Only eager drift (4.64%) and one O0 baseline split were quantified; graph-mode inter-process
drift is unmeasured, and the intra-process pipeline order is always N0→C0→C1. Cheap fix:
report p20/p80 for brackets, alternate order across processes.

---

## 3. Evidence→conclusion corrections (doc edits, mostly no new GPU time)

1. **R128 M0 evidence is conditional on benchmark-only indexed A** (production A cannot launch
   at R128). The "C1 saves 13.3–14.6µs" rank-128 rows are not production-achievable numbers and
   should be labeled as such wherever quoted. (Also: the range excludes its own largest value —
   GB300 mixed = 14.752µs; same slip in audit §8.3 and worklog Sixth-evidence.)
2. **GB300 T2048 forced-C1 eager −10.45%** is the only §8.5 delta exceeding the drift bound and
   the only one matching production reality (prefill runs eager) — yet it was never traced,
   while the sub-noise +0.67%/+1.72% figures are quoted as improvements. The eager-vs-graph sign
   flip on the same device/shape is the strongest §8 evidence that execution mode must be a
   planner key. Trace it (one nsys run) and promote "execution mode" from §6.6's list into the
   two-stream policy conclusion.
3. **T64/T128 gap at R128**: GB300's gate-B family flips between T32 (direct) and T256
   (generic) via an M0-only mechanism (join timing) that K0/O0 can't predict — interpolation is
   indefensible by the audit's own ladder lesson, and no planned cell covers R128×T64/T128 on
   any model. Two GB300 M0 bracket cells close the most common decode range.
4. **Capacity confound unpresented**: all route-cost evidence rides the ≥1024-virtual-expert
   JIT align path (cap8 = 2048). The cap1 (universe 256, native path) routing-O0 artifacts are
   retained but never shown — for mlpb=1 deployments (common), production plan cost may be far
   lower and indexed's decode margin thinner. Present the cap1-vs-cap8 pair (already on disk).
   Note the planned capacity-32 cell lands on the third (torch.compile) implementation —
   label it as such.
5. **Underreach worth claiming**: the presented N0/C0 rows imply a serial prefill LoRA tax of
   +71.6% (H200) / +82.6% (GB300) at T2048 vs ~+16% at T32 — the strongest motivation for the
   C2 fused-consumer work, and never stated as a conclusion. State it.
6. **Rows-per-hit-group ≈1 at every anchor** (K=8, E_factor=256·L): all fragmentation
   conclusions sit at one point of the real driver variable. One anchor with E_local=32-ish
   (or K↑) would test the grouped-family's actual home turf before C2 freezes.

---

## 4. Silent-wrong-result paths opened by guard removal (code, fix before broader runs)

The cleanup philosophy ("broad validation near graduation") is defensible for *missing
features*, but several removals converted **fail-fast into silently wrong numerics**:

1. **routed_scaling_factor ∉ {None, 1} is deterministically inconsistent** — the gate_up delta
   flows through base finalize (scaled: `post_reorder` folds the factor), the down delta is
   added after finalize with only topk_weights (`moe_lora_runner.py:187-215`;
   `merged_experts_fused_moe_lora_add` has no scaling parameter). Not "unvalidated" (audit
   §9.5) — provably wrong at non-unit factors. Also unfalsifiable by the matrix as designed
   (§8.4 pins cells to scaling=1). Fix: scale the down delta (or guard), add one non-unit cell.
2. **Activation semantics ignored**: `DeepGemmBf16BaseGemm` hardcodes gate-first/
   non-interleaved/plain-SiLU; `runner_config.activation/gemm1_alpha/gemm1_clamp_limit/
   swiglu_limit/apply_router_weight_on_input/no_combine` are never read (the wrapper even
   computes `_uses_interleaved_gate_up` and the sgl path ignores it). A gpt-oss/gelu/non-gated
   model silently computes wrong results **on the base path too** — worse, the eager no-adapter
   shortcut stays correct, so base-vs-LoRA outputs diverge confusingly. Fix: cheap attach-time
   asserts (activation=="silu" ∧ is_gated ∧ alpha/clamp None ∧ ¬apply_router_weight_on_input).
3. **DP attention**: `token_lora_mapping` is local-rank-sized while MoE consumes gathered
   tokens → stale in-bounds reads under graph (nondeterministic wrong adapters), OOB in eager.
   No launch guard. One assert.
4. **Workspace admission removed**: masked-DeepGEMM workspace scales as E_local×T; a T=8192
   LoRA prefill chunk ≈ **12.9 GB transient** (26 GB at 16k) that memory profiling never sees
   (profiling exercises the no-LoRA stock-Triton path) → first long LoRA prefill can OOM-kill
   a production server. Benchmark anchors stop at T=2048 (~3 GB). Until the planned
   device-aware planner exists, restore a cheap byte-estimate guard with a chunked-prefill
   hint.
5. **rank>64**: B correctly falls back to slice-correct generic (verified, `20da51d0b7`), but
   production gate_up-A crashes with an opaque Triton `OutOfResources` **mid-serving** on a
   live multi-LoRA server (adapter load-time is the right rejection point).
6. **Quantized checkpoints**: `resolve_base_gemm`'s NotImplementedError is unreachable dead
   code; FP8 serves eager-no-adapter fine, then crashes opaquely at capture/first activation.
7. Smaller: `_align_block_size_large` catches bare `Exception` → silently falls back to
   torch.compile (masks real CUDA/JIT errors); `stage='shrink'` pre-warms routing-B **on the
   side stream**, violating the engine's own capture invariant — a landmine for C3/C4 wiring;
   `_LORA_EVENTS_KEEPALIVE` still unbounded with a stale doc pointer.

Recommendation: keep the lean-guard philosophy, but adopt the rule *"a removed guard is
acceptable only where the unguarded behavior is a clean, immediate error — never where it is
silently wrong or delayed-fatal."* Items 1–4 violate it today; each fix is ≤10 lines.

---

## 5. Plan & sequencing (edits to PLAN §8 / AUDIT §17)

1. **C2 has already started while the neutral baseline is missing** — untracked `fused_c2.py`,
   `experimental_c2.py`, `test_sgl_lora_c2.py` exist in the working tree (mtimes minutes after
   the docs were saved), while PLAN §8.7/PR-item-2 sequence the matched experimental-TRTLLM /
   stock-Triton whole-M0 bracket **before** C2, and the checkpoint states "no kernel changed."
   Two problems: (a) docs/tree divergence — update the worklog; (b) every acceptance signal C2
   can earn right now is "beats internal C0/C1," and the audit concedes nothing establishes
   SGL ≥ TRTLLM at any anchor. **Run the one-day neutral TRTLLM/stock bracket at the 3-4 core
   anchors before investing weeks in C2 schedule search** — if legacy is faster at those
   anchors, the C2 target bar and possibly its design change.
2. **Within C2, build down-B+finalize first, not gate_up-fused first.** The audit's own
   evidence says so: §1 "the most important next endpoint is down-B plus finalize/collective";
   T256 trace: direct-down removed 17.456µs from the *critical tail* while gate savings were
   "much of it hidden"; T1: "the shorter down tail controls the full result." §17.3's order
   (gate-fused then down-fused) contradicts this. Down-first also de-risks C3 less (C3 needs
   the gate-fused consumer; down-fusion is topology-independent).
3. **One-shot fused A+B is missing from every list including the untested backlog** — PLAN §8's
   own 1e tier names it, the vLLM study mandates benchmarking it, C5 was pre-classified
   "diagnostic" with zero measurement. At minimum add it to AUDIT §16 and the P1 candidate set
   for the **down** site (rank-resident registers; kills both the FP32-workspace round trip and
   the atomics).
4. **The FP32-vs-BF16 split-accumulation arm is required-core in PLAN §8.4 but has no active
   implementation** — active shrink hardcodes BF16 relaxed atomics; the archived precision knob
   (evidence: BF16 median +3.51%, range FP32 +1.7% to BF16 +19.3%) was dropped. Determinism is
   also a product question (batch-invariant mode in vLLM). Reinstate the knob before A-family
   conclusions harden.
5. **Legacy counter-candidates absent**: the route-cost conclusions were measured only against
   the unfused 3-kernel align pipeline; the in-repo legacy engine ships default-ON
   `FUSED_MERGED_ALIGN` (~10.2µs/layer at decode bs16) and `PREFILL_ROUTING_REUSE` (4× align
   reuse at prefill). These are the direct competitors to indexed-A's claimed advantage and to
   the per-invocation route-cache design — add both as benchmark variants (or port them) before
   concluding on route strategy. Related: **route-plan cache lives one runner invocation**
   (§9.2) — ~48 identical rebuilds per forward at decode; cross-layer memoization (mapping
   unchanged within a forward) is a cheap, large, unmeasured win that could shrink indexed-A's
   entire O0 case.
6. **All-base tax is the cheapest, most decision-relevant unexecuted cell** (`matrix.py:80`,
   runnable today). Under graphs, *every* captured batch — including servers whose adapters are
   attention-only, the most common config — takes the SGL path with sentinel mappings through
   40–75 MoE layers. The ≤1% no-LoRA gate (PLAN §8.8) is unfalsifiable until this number
   exists, and it decides the no-LoRA/LoRA graph-family question. Run it this week.
7. **Shared-outer has no concrete anchor cells** despite "enters the matrix from the start" —
   define (T,R,L,capacity) cells now (deduped shared-A vs pair-repetition; weighted rank
   reduction vs generic shared-B), or the mandate stays aspirational.
8. **Mixed-rank cells are unimplemented, not just unexecuted**: `matrix.py:120-125` hardcodes
   `max_rank=rank` in every cell, so PLAN §8.4's required-core (R,R_max)=(32,128)/(64,128)
   cannot run; kernels execute allocated rank and ignore `lora_ranks`/`adapter_enabled` — all
   anchors silently assume homogeneous full-rank pools (unrealistic for multi-adapter serving,
   where allocated-rank execution wastes compute proportional to R_max/R).
9. **Rank 8/16 deferral hardens the ABI in the band where the current family is
   compile-illegal** — R=8 is a `tl.dot` compile cliff today and the C2 WIP kernel copies the
   same tile mins. Pull one R8/R16 compile+correctness guardrail (not perf) into P0.
10. **Per-site B winners are measured through a benchmark-side monkeypatch** of
    `merged_experts_fused_moe_lora_add` — the same pattern PLAN §8.8 bans for PDL — and
    production has no per-site seam to reproduce the composite winners. Add the seam (site→
    family/config) to production dispatch before publishing composite selections as
    recommendations.
11. **Static-graph planner tension**: §7.5's target planner keys on per-batch values
    (hit-expert distribution, physical rank) that a captured decode graph cannot switch on.
    Add a "static-policy regret" measurement (best-fixed vs oracle-per-batch) to §8.6, or the
    planner design overpromises.
12. PDL: the plan's own protocol requires PDL-off isolated K0 but the launcher physically
    cannot produce it (arch-auto). Add the off-seam before any PDL chain conclusions.
13. TMA is name-checked in §8's intro but appears in no variant list/axis — decide explicitly
    whether the fused-consumer ABI must be TMA-layout-compatible for B factors before freezing.

---

## 6. Hygiene / reproducibility (mostly cheap edits)

1. **Artifact exposure sharpened**: 100% of ACTIVE-campaign published tables (§6.1–6.5,
   §7.1–7.4, §8.1–8.5, both Nsight sets) are backed only by TTL/reprovision-sensitive GPU
   paths; only archived-prototype §15 evidence is locally retained. Copy the active artifact
   trees into `benchmark_results/` with SHA-256 + a table→filename manifest **before the next
   devbox reprovision**; §13.6 step 9 is currently unexecutable by a third party.
2. Docs/tree divergence (C2 files) — see §5.1; also update worklog changelog ordering (breaks
   around 2026-07-17) and the two inconsistent §-range pointers ("8.2–8.8" vs "8.1–8.8").
3. Dead pointers: the rejected NVFP4 factorization prototype's `/tmp` checkout + unresolvable
   commit in PLAN §6.3; README cites amended-away `65e69838d9` as branch state.
4. `design.md` (2026-07-01) is stale, contradicts three accepted decisions, and is the only
   doc in the folder without a supersession banner — add one or delete it.
5. Environment schema (`bench_local.py:352-375`) omits Triton version, driver version, clock
   state — exactly what §13.6 tells auditors to verify; the cu129-vs-cu130 split makes
   cross-device percentages operational, not architectural (say so where quoted).
6. Terminology: U0 missing from PLAN's scope ladder; "WS1" used 17× and never expanded; the
   audit's symbol table collapses distinctions PLAN §8.1 canonicalizes (L vs L_capacity,
   P variants, R vs R_max/R_phys) — align the audit's table.
7. PLAN §8.6's own protocol mandates NaN/Inf poisoning of inactive slots, sustained-replay
   correctness, and touched-bytes reporting — none is implemented in the drivers yet; either
   implement or move to explicit backlog.
8. Benchmark-fidelity couplings worth a comment: M0 overrides depend on `run_sgl_lora_moe`'s
   per-call import (a style cleanup silently disables them); the benchmark "production" A
   schedule is resolved independently and could drift from real production resolution.

---

## 7. What is strong — preserve as-is

- The K0→O0→M0 ladder with documented scope reversals (GB300 T256 gate-B O0-win/M0-loss caught,
  traced, promoted into a general rule) — exemplary and load-bearing.
- Negative-evidence hygiene: discarded wrong generic-gated-B timings with an auditor exclusion
  list; the 4.64% eager drift bound with an interleaving rule; retained known biases recorded
  in JSON (indexed-C1 prewarm penalty) instead of silently removed.
- Fair-comparison mechanics: per-family tuning, effective-config normalization (direct-B
  ignored fields), split-K zeroing charged inside the semantic launch, synthetic-B isolation at
  R128, opposite-family oracles.
- Forced-cold K0 protocol implemented exactly as documented; isolated one-replay confirmation
  exposing ~10% batched-graph pipelining inflation.
- M0 harness calls the actual production `run_sgl_lora_moe`, not a copy.
- PLAN §8.1's derived-values design (P_work/E_hit/R_phys) + guardrail-vs-tuning model split —
  structurally prevents Cartesian-matrix explosion.
- Number/pointer discipline: 14/14 P0 case tuples match `matrix.py`; 32/32 audited source
  citations resolve; commit topology claims all verified (archive branch, isolation from
  trtllm_lora_temp, amend history).
- Implementation details: single-source split-K/zeroing coupling; pool zero-then-spaced-load
  making "execute allocated rank" safe; slice-correct generic fallback after `20da51d0b7`.

---

## 8. Priority order for the plan-updating agent

**This week (before/alongside any C2 work):**
1. Fix C1 (shared-expert virtual-ID corruption) + unit test. [§1]
2. Fix V1 vacuous tolerances (wire `_check_lora_delta` everywhere; re-adjudicate retained
   JSONs) and V2 routing label + IID/skew seeds at core anchors. [§2]
3. Run the neutral TRTLLM/stock whole-M0 bracket at 3–4 core anchors. [§5.1]
4. Run the all-base-tax cell. [§5.6]
5. Land the four silent-wrong guards (routed-scaling, activation asserts, DP-attention,
   workspace byte-guard) + rank>64 load-time rejection. [§4]
6. Durable artifact bundle + manifest off the TTL filesystems. [§6.1]

**Before C2 conclusions freeze:**
7. Reorder C2 internals: down-B+finalize endpoint first. [§5.2]
8. Add one-shot A+B (down site) and FP32-accumulation arm to the candidate set. [§5.3-4]
9. Re-scope O0 as marginal-cost for shared plans; correct §1/§11.7 indexed-A claims. [§2 V3]
10. Add T64/T128 R128 GB300 cells; present cap1-vs-cap8; trace the GB300 eager −10.45%. [§3]
11. Add legacy FUSED_MERGED_ALIGN/PREFILL_ROUTING_REUSE variants + cross-layer route-plan
    memoization measurement. [§5.5]
12. Shared-outer anchor cells; mixed-rank (R,R_max) cells implemented in matrix.py; R8/R16
    compile guardrail. [§5.7-9]

**Doc edits batch:** §3 items 1–6, §6 items 2–8, production per-site seam note [§5.10],
static-policy regret [§5.11], PDL-off seam [§5.12], TMA decision [§5.13].
