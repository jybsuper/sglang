# Shared-expert MoE-LoRA matrix summary

Negative percentages mean faster than conventional serial shared experts. Values are medians of each artifact's p50; use the raw distributions and timeline traces before promoting a production policy.

Correctness: `True` across `144` records; minimum cosine `0.99998337`, maximum absolute error `3.05176e-05`. The independent oracle used logical IDs only, and no shared slot received LoRA.

| Device | Case | Exec | Runs | R | S | Serial us | Overlap us (%) | Fused-global us (%) | Fused-per-rank us (%) | Winner | Map vs global |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|:---|---:|
| NVIDIA GB300 | qwen35-decode-t1-r32 | cuda_graph | 2 | 32 | 1 | 103.136 | 95.160 (-7.73%) | 93.160 (-9.67%) | 92.760 (-10.06%) | fused_per_rank | -0.43% |
| NVIDIA GB300 | qwen35-decode-t1-r32 | eager | 1 | 32 | 1 | 833.280 | 961.104 (+15.34%) | 731.264 (-12.24%) | 718.496 (-13.77%) | fused_per_rank | -1.75% |
| NVIDIA GB300 | qwen35-decode-t32-r128-two-sink-mixed | cuda_graph | 2 | 128 | 2 | 395.280 | 393.728 (-0.39%) | 388.032 (-1.83%) | 390.224 (-1.28%) | fused_global | +0.56% |
| NVIDIA GB300 | qwen35-decode-t32-r128-two-sink-mixed | eager | 1 | 128 | 2 | 996.704 | 968.480 (-2.83%) | 789.632 (-20.78%) | 792.736 (-20.46%) | fused_global | +0.39% |
| NVIDIA GB300 | qwen35-decode-t32-r64-mixed | cuda_graph | 2 | 64 | 1 | 348.232 | 345.976 (-0.65%) | 338.448 (-2.81%) | 339.480 (-2.51%) | fused_global | +0.30% |
| NVIDIA GB300 | qwen35-decode-t32-r64-mixed | eager | 1 | 64 | 1 | 844.304 | 900.576 (+6.66%) | 724.352 (-14.21%) | 702.832 (-16.76%) | fused_per_rank | -2.97% |
| NVIDIA GB300 | qwen35-prefill-t256-r128-two-sink-mixed | cuda_graph | 2 | 128 | 2 | 811.520 | 808.688 (-0.35%) | 812.424 (+0.11%) | 811.968 (+0.06%) | separate_overlap | -0.06% |
| NVIDIA GB300 | qwen35-prefill-t256-r128-two-sink-mixed | eager | 1 | 128 | 2 | 1017.856 | 1083.904 (+6.49%) | 894.336 (-12.14%) | 866.144 (-14.91%) | fused_per_rank | -3.15% |
| NVIDIA GB300 | qwen35-prefill-t256-r32-mixed | cuda_graph | 2 | 32 | 1 | 445.336 | 446.960 (+0.36%) | 440.816 (-1.01%) | 439.248 (-1.37%) | fused_per_rank | -0.36% |
| NVIDIA GB300 | qwen35-prefill-t256-r32-mixed | eager | 1 | 32 | 1 | 847.120 | 892.128 (+5.31%) | 748.128 (-11.69%) | 766.624 (-9.50%) | fused_global | +2.47% |
| NVIDIA GB300 | qwen35-prefill-t512-r64 | cuda_graph | 2 | 64 | 1 | 635.872 | 634.352 (-0.24%) | 634.168 (-0.27%) | 627.936 (-1.25%) | fused_per_rank | -0.98% |
| NVIDIA GB300 | qwen35-prefill-t512-r64 | eager | 1 | 64 | 1 | 970.816 | 1121.920 (+15.56%) | 849.280 (-12.52%) | 858.496 (-11.57%) | fused_global | +1.09% |
| NVIDIA H200 | qwen35-decode-t1-r32 | cuda_graph | 2 | 32 | 1 | 114.496 | 112.464 (-1.77%) | 104.096 (-9.08%) | 103.096 (-9.96%) | fused_per_rank | -0.96% |
| NVIDIA H200 | qwen35-decode-t1-r32 | eager | 1 | 32 | 1 | 742.496 | 804.688 (+8.38%) | 652.112 (-12.17%) | 635.648 (-14.39%) | fused_per_rank | -2.52% |
| NVIDIA H200 | qwen35-decode-t32-r128-two-sink-mixed | cuda_graph | 2 | 128 | 2 | 549.416 | 548.544 (-0.16%) | 540.688 (-1.59%) | 541.720 (-1.40%) | fused_global | +0.19% |
| NVIDIA H200 | qwen35-decode-t32-r128-two-sink-mixed | eager | 1 | 128 | 2 | 908.448 | 977.248 (+7.57%) | 788.288 (-13.23%) | 766.544 (-15.62%) | fused_per_rank | -2.76% |
| NVIDIA H200 | qwen35-decode-t32-r64-mixed | cuda_graph | 2 | 64 | 1 | 477.352 | 475.856 (-0.31%) | 469.120 (-1.72%) | 468.904 (-1.77%) | fused_per_rank | -0.05% |
| NVIDIA H200 | qwen35-decode-t32-r64-mixed | eager | 1 | 64 | 1 | 765.088 | 845.904 (+10.56%) | 667.248 (-12.79%) | 659.440 (-13.81%) | fused_per_rank | -1.17% |
| NVIDIA H200 | qwen35-prefill-t256-r128-two-sink-mixed | cuda_graph | 2 | 128 | 2 | 1113.216 | 1113.928 (+0.06%) | 1113.152 (-0.01%) | 1112.888 (-0.03%) | fused_per_rank | -0.02% |
| NVIDIA H200 | qwen35-prefill-t256-r128-two-sink-mixed | eager | 1 | 128 | 2 | 1137.712 | 1129.536 (-0.72%) | 1130.752 (-0.61%) | 1131.200 (-0.57%) | separate_overlap | +0.04% |
| NVIDIA H200 | qwen35-prefill-t256-r32-mixed | cuda_graph | 2 | 32 | 1 | 619.304 | 618.640 (-0.11%) | 609.336 (-1.61%) | 609.848 (-1.53%) | fused_global | +0.08% |
| NVIDIA H200 | qwen35-prefill-t256-r32-mixed | eager | 1 | 32 | 1 | 800.768 | 877.776 (+9.62%) | 693.248 (-13.43%) | 687.168 (-14.19%) | fused_per_rank | -0.88% |
| NVIDIA H200 | qwen35-prefill-t512-r64 | cuda_graph | 2 | 64 | 1 | 843.888 | 844.960 (+0.13%) | 836.552 (-0.87%) | 835.968 (-0.94%) | fused_per_rank | -0.07% |
| NVIDIA H200 | qwen35-prefill-t512-r64 | eager | 1 | 64 | 1 | 879.936 | 957.680 (+8.84%) | 854.240 (-2.92%) | 857.776 (-2.52%) | fused_global | +0.41% |
