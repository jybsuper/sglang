# Shared-expert MoE-LoRA matrix summary

Negative percentages mean faster than conventional serial shared experts. Values are medians of each artifact's p50; use the raw distributions and timeline traces before promoting a production policy.

Correctness: `True` across `16` records; minimum cosine `0.99999052`, maximum absolute error `4.76837e-07`. The independent oracle used logical IDs only, and no shared slot received LoRA.

| Device | Case | Exec | Runs | R | S | Serial us | Overlap us (%) | Fused-global us (%) | Fused-per-rank us (%) | Winner | Map vs global |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|:---|---:|
| NVIDIA GB300 | shared-smoke-r32 | cuda_graph | 1 | 32 | 1 | 41.472 | 41.312 (-0.39%) | 37.728 (-9.03%) | 39.872 (-3.86%) | fused_global | +5.68% |
| NVIDIA GB300 | shared-smoke-r32 | eager | 1 | 32 | 1 | 887.840 | 939.264 (+5.79%) | 758.400 (-14.58%) | 702.464 (-20.88%) | fused_per_rank | -7.38% |
| NVIDIA H200 | shared-smoke-r32 | cuda_graph | 1 | 32 | 1 | 39.328 | 38.656 (-1.71%) | 34.016 (-13.51%) | 34.560 (-12.12%) | fused_global | +1.60% |
| NVIDIA H200 | shared-smoke-r32 | eager | 1 | 32 | 1 | 802.304 | 828.768 (+3.30%) | 647.936 (-19.24%) | 644.928 (-19.62%) | fused_per_rank | -0.46% |
