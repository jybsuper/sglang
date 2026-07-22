# Final rebased commit map

This is the authoritative review order for the tested source snapshot
`f2f406e0560bb8c95479eeb164bf93b65fb6a0c1`. Every campaign commit
range-diffed exactly across the final freshness rebase onto OSS main
`4eaa5ca6510622cb0006bcfee5947b17859ac8c7`.

| Commit | Subject |
|---|---|
| `70f25a2f4b` | [LoRA] Add isolated sgl_lora BF16 MoE execution engine |
| `32f46800e4` | [LoRA] Add MoE benchmark matrix and local A/B harness |
| `090ea8f778` | [LoRA] Tighten MoE benchmark correctness reporting |
| `fdfa3ed344` | [LoRA] Add MoE shrink schedule tuning harness |
| `2728a2cdc5` | [LoRA] Make shrink tuning robust to unsupported schedules |
| `e3a70f5828` | [LoRA] Add indexed MoE shrink benchmark candidate |
| `ae223c0934` | [LoRA] Fix indexed shrink oracle shape |
| `3c5cfbcc43` | [LoRA] Add full MoE pipeline benchmark driver |
| `8a522013bf` | [LoRA] Add cold-cache shrink benchmarking |
| `76dfc8f490` | [LoRA] Add fair route-inclusive A benchmark |
| `2d95afe741` | [LoRA] Benchmark indexed A in the full MoE pipeline |
| `18e7aa94cf` | [LoRA] Record unsupported production A in M0 |
| `c1d9d7b9d2` | [LoRA] Make MoE overlap policy benchmarkable |
| `68f379f3f8` | [LoRA] Add prefill overlap boundary cases |
| `2b2b112998` | [LoRA] Add LoRA-B config benchmark |
| `5b380e3dc3` | [LoRA] Initialize standalone MoE benchmark context |
| `9555c82e89` | [LoRA] Fix generic gated expand slices |
| `066ade402f` | [LoRA] Isolate rank-128 B benchmarking |
| `aeeadb663f` | [LoRA] Benchmark per-site B schedules in M0 |
| `e491b0c4c2` | [LoRA] Mask fused shared experts from virtual routing |
| `d163b4109d` | [LoRA] Tighten MoE B benchmark correctness |
| `0cadb71cc3` | [LoRA] Add explicit IID and skewed MoE routes |
| `50dcee9a3d` | [LoRA] Harden SGL MoE execution contracts |
| `9d1d58a995` | [LoRA] Add provider-neutral MoE pipeline controls |
| `95c1378046` | [LoRA] Add counterbalanced cold M0 controls |
| `33147be01b` | [LoRA] Prototype fused BF16 C2 consumer |
| `6c3b112430` | [LoRA] Benchmark shared-outer MoE factorizations |
| `40a3cbdc88` | [LoRA] Tile shrink ranks across output CTAs |
| `f2803ff720` | [LoRA] Prototype fused down finalize for C2 |
| `2ea74d4938` | [LoRA] Add missing rank-128 decode anchors |
| `3fcd30b7cc` | [LoRA] Admit MoE workspaces before launch |
| `ff477221cd` | [LoRA] Tie overlap events to CUDA graph lifetime |
| `d7d884b919` | [LoRA] Benchmark MoE algorithm families |
| `0ad67aedbd` | [LoRA] Add explicit PDL control matrix |
| `b5c40f795a` | [LoRA] Guard C2 across MoE activation contracts |
| `59d9f0b66d` | [LoRA] Benchmark route-plan counter-candidates |
| `aaf4c71283` | [LoRA] Capture separate base and adapter decode graphs |
| `da653f1986` | [LoRA] Benchmark C3 gate-A overlap |
| `ff1dfce9ac` | [LoRA] Compare model-level SGL and TRTLLM paths |
| `8d66c5c128` | [LoRA] Support rank-8 tensor-core tiles |
| `31cc20375c` | [LoRA] Summarize rank graduation evidence |
| `d685a885f6` | [LoRA] Benchmark mixed-rank MoE policies |
| `e43ede8bac` | [LoRA] Add quantized MoE base providers |
| `fbeffb9c78` | [LoRA] Record shared-outer kernel evidence |
| `a5f513d3f3` | [LoRA] Measure eager MoE under host contention |
| `064e4beb03` | [LoRA] Promote measured BF16 MoE execution planner |
| `1a5c89c086` | [LoRA] Validate distributed MoE execution |
| `82576884a3` | [LoRA] Validate adapter lifecycle transitions |
| `1865413e88` | [LoRA] Adapt SGL engine to kernel registry |
| `89b9cec5de` | [LoRA] Record algorithm family benchmark evidence |
| `b7ca564d80` | [LoRA] Archive rank-8 and rank-16 guardrails |
| `206567ccac` | [LoRA] Archive C2 finalizer and PDL evidence |
| `ac273dd364` | [LoRA] Archive H200 and GB300 benchmark campaign |
| `0605937808` | [LoRA] Pair virtual-expert PDL producer and consumer |
| `c851f0ac56` | [LoRA] Graduate quantized MoE providers |
| `8a420baacd` | [LoRA] Adapt quant provider to unified kernel API |
| `b2f82a8404` | [LoRA] Support physical shared-expert routing |
| `e653200ea8` | bench(lora): evaluate Blackwell CuTe MoE boundaries |
| `24c3754416` | [LoRA] Preserve resident provider contracts |
| `5beed992c6` | [LoRA] Promote shared-outer gate token dedup |
| `ca5f9079e0` | [LoRA] Isolate Marlin graph workspaces |
| `0b5469c99d` | test(lora): synchronize the P0 benchmark matrix |
| `f2f406e056` | chore(lora): normalize campaign sources |

