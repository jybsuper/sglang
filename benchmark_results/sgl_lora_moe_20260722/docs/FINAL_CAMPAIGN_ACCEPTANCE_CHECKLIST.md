# SGL LoRA MoE final campaign acceptance checklist

This file is an execution checklist for the 2026-07-22 campaign. It is not a
new architecture document. The final architecture remains
`sgl_lora_lifecycle_and_orchestration_design.md`; the final evidence narrative
remains `sgl_lora_moe_kernel_benchmark_architecture_audit.md`.

Final rebased source placeholder: `f2f406e056` (replace after the terminal rebase).

## Correctness and review findings

- [x] Mask physical shared-expert IDs outside the LoRA factor domain.
- [x] Replace vacuous output tolerances with base-subtracted signal checks.
- [x] Separate lattice, true-IID, and skewed route policies.
- [x] Treat shared route-plan preparation as marginal pipeline cost.
- [x] Add loaded-host eager and forced-cold M0 controls.
- [x] Counterbalance timing order and report dispersion.
- [x] Fix routed scaling, activation/layout, DP-attention, workspace, rank,
  quant-provider, route-fallback, event-lifetime, and PDL failure modes.
- [x] Preserve the resident base quant method for no-active-adapter execution instead
  of forcing a synthetic Triton provider.

## BF16 execution and algorithm studies

- [x] Match N0/C0/C1 and stock/experimental-TRTLLM model controls.
- [x] Report final server gaps exactly: C2 9.43%/9.44%/10.39% and C3
  5.12%/5.54%/10.48% at BS1/16/32; label 12.56%-14.71% historical pre-planner only.
- [x] Implement down-B/finalize before promoting the planner.
- [x] Cover indexed, segmented/grouped, BMM-qualified, one-shot A+B,
  token-owned/fused, shared-factor, and precision candidates.
- [x] Cover rank 8, 16, 32, 64, and 128 guardrails/policy evidence.
- [x] Cover lattice/IID/skewed routes, capacity, occupancy, mixed/base rows,
  and multiple adapter capacities.
- [x] Measure static-policy regret and mixed allocated/physical rank.
- [x] Add separate base/adapter graphs and PDL off/on controls.
- [x] Scope route-view memoization to one MoE layer invocation; do not claim
  whole-forward or cross-layer reuse.
- [x] Validate cross-model Qwen/Kimi/GLM/Nemotron local shape and activation
  guardrails. Nemotron ReLU2 remains benchmark-only until production attachment.

## Distributed, shared, quantized, and alternate-provider work

- [x] Real TP/EP/MoE-DP D0, including two-node TP8/EP2/DP2.
- [x] Real adapter lifecycle, eviction, slot recycling, and graph replay.
- [x] Physical shared-expert policy and evidence for Standard contiguous/EP1 or
  otherwise safe non-per-rank-remapped layouts. Per-rank physical shared EP>1 and
  advanced A2A remain explicitly unpromoted.
- [x] Shared-outer evidence-bounded selector, TorchNative merged-bound repair,
  physical shared-ID composition, and PDL pairing: 9 host tests and 56 tests plus
  10 subtests on each of H200 and GB300.
- [x] FP8 W8A8 provider-neutral C0 seam and profiles; active FP8 rejects static
  activation, Blackwell requires resident packed UE8M0 scales, and broad
  checkpoint/server plus provider-native fused-LoRA graduation remain future work.
- [x] Native CuTe DSL NVFP4 W4A4 synthetic/testbed matrix and profiles. It is not
  serving-reachable and must not be presented as checkpoint/server attachment.
- [x] Marlin W4A16 attachable provider seam and profiles.
- [x] Re-run the Marlin per-invocation workspace/dirty-destination repair on H200
  and GB300 at `f2f406e056` before marking final integration complete.
- [x] Evidence-only optimized CuTe DSL/TMA comparison: K0 73.152 us versus Triton
  87.520 us (-16.42%); static M0 is invalid under route mutation, so there is no
  production integration.

## Reproducibility and handoff

- [x] `ROOT_FINALIZATION`: integrate all focused implementation/evidence commits
  in review order and replace every unresolved final-head placeholder.
- [x] Archive final H200/GB300 JSON/log/Nsight artifacts locally.
- [x] Generate and verify top-level SHA-256 and filename manifests.
- [x] Update the review-disposition ledger with final commit hashes.
- [x] Remove stale current-status claims from plan/audit/worklog/design docs.
- [x] Run final focused CPU tests, formatting, source-isolation audit, and GPU smoke.
- [ ] Push the branch and update draft PR #31882 with the commit/evidence map.

Phase 2 dense/special-layer execution, Phase 3 adapter residency/control-plane
redesign, MTP/EAGLE, advanced A2A, and broad production model graduation are
explicitly outside this Phase-1 MoE campaign; they remain named follow-on work.
