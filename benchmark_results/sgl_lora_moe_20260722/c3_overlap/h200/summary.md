# C3 overlap matrix summary

Negative percentages mean C3 is faster. A winner is named only when all matched counterbalanced repeats agree on the sign and the median effect exceeds both 1% and the matched-repeat range.

| Device | T | R | Base rows | Execution | Baseline | Baseline us | C3 us | C3 vs baseline median [min, max] | Result |
|:---|---:|---:|:---:|:---|:---|---:|---:|:---|:---|
| NVIDIA H200 | 1 | 32 | False | cuda_graph | C0 | 104.384 | 90.320 | -13.48% [-13.72, -13.32] | C3 faster (all matched repeats) |
| NVIDIA H200 | 1 | 32 | False | cuda_graph | C1 | 96.768 | 90.320 | -6.64% [-6.94, -6.23] | C3 faster (all matched repeats) |
| NVIDIA H200 | 1 | 32 | False | cuda_graph | C2F | 95.792 | 90.320 | -5.70% [-6.23, -5.24] | C3 faster (all matched repeats) |
| NVIDIA H200 | 1 | 32 | False | cuda_graph | C2P | 100.256 | 90.320 | -9.81% [-10.63, -9.75] | C3 faster (all matched repeats) |
| NVIDIA H200 | 1 | 32 | False | eager | C0 | 603.664 | 529.232 | -12.12% [-13.03, -11.98] | C3 faster (all matched repeats) |
| NVIDIA H200 | 1 | 32 | False | eager | C1 | 699.456 | 529.232 | -24.19% [-24.87, -23.98] | C3 faster (all matched repeats) |
| NVIDIA H200 | 1 | 32 | False | eager | C2F | 466.144 | 529.232 | +13.67% [+12.03, +14.56] | C3 slower (all matched repeats) |
| NVIDIA H200 | 1 | 32 | False | eager | C2P | 535.856 | 529.232 | -1.92% [-2.59, -0.50] | inconclusive: effect <= dispersion floor |
| NVIDIA H200 | 32 | 32 | False | cuda_graph | C0 | 439.232 | 418.336 | -4.69% [-4.97, -3.99] | C3 faster (all matched repeats) |
| NVIDIA H200 | 32 | 32 | False | cuda_graph | C1 | 426.848 | 418.336 | -1.99% [-2.47, -1.65] | C3 faster (all matched repeats) |
| NVIDIA H200 | 32 | 32 | False | cuda_graph | C2F | 427.680 | 418.336 | -2.12% [-2.72, -2.02] | C3 faster (all matched repeats) |
| NVIDIA H200 | 32 | 32 | False | cuda_graph | C2P | 436.304 | 418.336 | -4.17% [-4.28, -3.98] | C3 faster (all matched repeats) |
| NVIDIA H200 | 32 | 32 | False | eager | C0 | 626.224 | 542.176 | -13.09% [-14.35, -12.98] | C3 faster (all matched repeats) |
| NVIDIA H200 | 32 | 32 | False | eager | C1 | 718.224 | 542.176 | -24.48% [-24.67, -23.83] | C3 faster (all matched repeats) |
| NVIDIA H200 | 32 | 32 | False | eager | C2F | 485.600 | 542.176 | +11.87% [+10.62, +12.59] | C3 slower (all matched repeats) |
| NVIDIA H200 | 32 | 32 | False | eager | C2P | 555.872 | 542.176 | -2.60% [-4.60, -1.57] | inconclusive: effect <= dispersion floor |
| NVIDIA H200 | 32 | 64 | False | cuda_graph | C0 | 470.080 | 445.056 | -5.14% [-5.64, -4.58] | C3 faster (all matched repeats) |
| NVIDIA H200 | 32 | 64 | False | cuda_graph | C1 | 453.264 | 445.056 | -1.96% [-2.33, -1.58] | C3 faster (all matched repeats) |
| NVIDIA H200 | 32 | 64 | False | cuda_graph | C2F | 457.280 | 445.056 | -2.61% [-3.32, -2.47] | C3 faster (all matched repeats) |
| NVIDIA H200 | 32 | 64 | False | cuda_graph | C2P | 466.928 | 445.056 | -4.63% [-5.34, -4.58] | C3 faster (all matched repeats) |
| NVIDIA H200 | 32 | 64 | False | eager | C0 | 604.288 | 538.912 | -11.35% [-12.29, -8.49] | C3 faster (all matched repeats) |
| NVIDIA H200 | 32 | 64 | False | eager | C1 | 700.320 | 538.912 | -23.48% [-24.68, -22.08] | C3 faster (all matched repeats) |
| NVIDIA H200 | 32 | 64 | False | eager | C2F | 480.480 | 538.912 | +11.51% [+11.13, +13.14] | C3 slower (all matched repeats) |
| NVIDIA H200 | 32 | 64 | False | eager | C2P | 550.704 | 538.912 | -1.86% [-3.98, -0.84] | inconclusive: effect <= dispersion floor |
| NVIDIA H200 | 32 | 64 | True | cuda_graph | C0 | 446.112 | 437.120 | -2.02% [-2.73, -1.11] | C3 faster (all matched repeats) |
| NVIDIA H200 | 32 | 64 | True | cuda_graph | C1 | 441.040 | 437.120 | -0.96% [-1.47, -0.48] | inconclusive: effect <= dispersion floor |
| NVIDIA H200 | 32 | 64 | True | cuda_graph | C2F | 442.448 | 437.120 | -1.14% [-2.04, -1.03] | C3 faster (all matched repeats) |
| NVIDIA H200 | 32 | 64 | True | cuda_graph | C2P | 445.680 | 437.120 | -1.92% [-2.86, -1.71] | C3 faster (all matched repeats) |
| NVIDIA H200 | 32 | 64 | True | eager | C0 | 648.256 | 604.272 | -6.72% [-7.52, -6.14] | C3 faster (all matched repeats) |
| NVIDIA H200 | 32 | 64 | True | eager | C1 | 743.216 | 604.272 | -18.76% [-19.01, -18.09] | C3 faster (all matched repeats) |
| NVIDIA H200 | 32 | 64 | True | eager | C2F | 545.040 | 604.272 | +10.94% [+9.76, +12.06] | C3 slower (all matched repeats) |
| NVIDIA H200 | 32 | 64 | True | eager | C2P | 618.624 | 604.272 | -1.85% [-3.38, -1.70] | C3 faster (all matched repeats) |
| NVIDIA H200 | 32 | 128 | False | cuda_graph | C0 | 540.432 | 500.736 | -7.33% [-7.61, -7.25] | C3 faster (all matched repeats) |
| NVIDIA H200 | 32 | 128 | False | cuda_graph | C1 | 516.496 | 500.736 | -3.00% [-3.24, -2.94] | C3 faster (all matched repeats) |
| NVIDIA H200 | 32 | 128 | False | cuda_graph | C2F | 514.496 | 500.736 | -2.60% [-2.87, -2.55] | C3 faster (all matched repeats) |
| NVIDIA H200 | 32 | 128 | False | cuda_graph | C2P | 528.016 | 500.736 | -5.13% [-5.32, -5.06] | C3 faster (all matched repeats) |
| NVIDIA H200 | 32 | 128 | False | eager | C0 | 748.560 | 558.624 | -25.43% [-25.72, -24.86] | C3 faster (all matched repeats) |
| NVIDIA H200 | 32 | 128 | False | eager | C1 | 851.472 | 558.624 | -34.36% [-34.86, -33.97] | C3 faster (all matched repeats) |
| NVIDIA H200 | 32 | 128 | False | eager | C2F | 521.520 | 558.624 | +6.92% [+6.73, +7.28] | C3 slower (all matched repeats) |
| NVIDIA H200 | 32 | 128 | False | eager | C2P | 600.640 | 558.624 | -6.98% [-7.34, -6.83] | C3 faster (all matched repeats) |
| NVIDIA H200 | 64 | 128 | False | cuda_graph | C0 | 774.384 | 707.904 | -8.58% [-8.81, -8.51] | C3 faster (all matched repeats) |
| NVIDIA H200 | 64 | 128 | False | cuda_graph | C1 | 750.928 | 707.904 | -5.76% [-5.82, -5.59] | C3 faster (all matched repeats) |
| NVIDIA H200 | 64 | 128 | False | cuda_graph | C2F | 721.968 | 707.904 | -1.99% [-2.15, -1.81] | C3 faster (all matched repeats) |
| NVIDIA H200 | 64 | 128 | False | cuda_graph | C2P | 754.112 | 707.904 | -6.11% [-6.15, -6.08] | C3 faster (all matched repeats) |
| NVIDIA H200 | 64 | 128 | False | eager | C0 | 789.264 | 717.200 | -9.22% [-9.44, -8.94] | C3 faster (all matched repeats) |
| NVIDIA H200 | 64 | 128 | False | eager | C1 | 836.096 | 717.200 | -14.24% [-14.82, -13.52] | C3 faster (all matched repeats) |
| NVIDIA H200 | 64 | 128 | False | eager | C2F | 732.736 | 717.200 | -2.20% [-2.36, -2.04] | C3 faster (all matched repeats) |
| NVIDIA H200 | 64 | 128 | False | eager | C2P | 765.936 | 717.200 | -6.53% [-6.56, -6.19] | C3 faster (all matched repeats) |
| NVIDIA H200 | 128 | 128 | False | cuda_graph | C0 | 1048.432 | 950.768 | -9.31% [-9.35, -9.25] | C3 faster (all matched repeats) |
| NVIDIA H200 | 128 | 128 | False | cuda_graph | C1 | 1027.952 | 950.768 | -7.49% [-7.59, -7.43] | C3 faster (all matched repeats) |
| NVIDIA H200 | 128 | 128 | False | cuda_graph | C2F | 961.600 | 950.768 | -1.09% [-1.28, -1.05] | C3 faster (all matched repeats) |
| NVIDIA H200 | 128 | 128 | False | cuda_graph | C2P | 1019.056 | 950.768 | -6.69% [-6.78, -6.53] | C3 faster (all matched repeats) |
| NVIDIA H200 | 128 | 128 | False | eager | C0 | 1061.696 | 954.560 | -9.98% [-10.26, -9.85] | C3 faster (all matched repeats) |
| NVIDIA H200 | 128 | 128 | False | eager | C1 | 1034.032 | 954.560 | -7.66% [-7.91, -7.50] | C3 faster (all matched repeats) |
| NVIDIA H200 | 128 | 128 | False | eager | C2F | 968.384 | 954.560 | -1.43% [-1.68, -1.19] | C3 faster (all matched repeats) |
| NVIDIA H200 | 128 | 128 | False | eager | C2P | 1027.376 | 954.560 | -7.05% [-7.29, -6.89] | C3 faster (all matched repeats) |
| NVIDIA H200 | 256 | 32 | False | cuda_graph | C0 | 529.952 | 549.440 | +3.98% [+3.40, +5.14] | C3 slower (all matched repeats) |
| NVIDIA H200 | 256 | 32 | False | cuda_graph | C1 | 521.120 | 549.440 | +5.48% [+5.17, +5.62] | C3 slower (all matched repeats) |
| NVIDIA H200 | 256 | 32 | False | cuda_graph | C2F | 554.192 | 549.440 | -0.82% [-1.21, -0.56] | inconclusive: effect <= dispersion floor |
| NVIDIA H200 | 256 | 32 | False | cuda_graph | C2P | 522.624 | 549.440 | +5.36% [+4.49, +5.43] | C3 slower (all matched repeats) |
| NVIDIA H200 | 256 | 32 | False | eager | C0 | 626.576 | 554.480 | -11.51% [-12.57, -10.95] | C3 faster (all matched repeats) |
| NVIDIA H200 | 256 | 32 | False | eager | C1 | 725.984 | 554.480 | -23.64% [-24.07, -22.40] | C3 faster (all matched repeats) |
| NVIDIA H200 | 256 | 32 | False | eager | C2F | 561.360 | 554.480 | -1.26% [-1.48, -0.87] | C3 faster (all matched repeats) |
| NVIDIA H200 | 256 | 32 | False | eager | C2P | 563.296 | 554.480 | -1.62% [-2.01, -0.80] | C3 faster (all matched repeats) |
| NVIDIA H200 | 2048 | 64 | False | cuda_graph | C0 | 900.864 | 1386.288 | +54.13% [+53.74, +55.23] | C3 slower (all matched repeats) |
| NVIDIA H200 | 2048 | 64 | False | cuda_graph | C1 | 890.288 | 1386.288 | +55.76% [+55.64, +56.18] | C3 slower (all matched repeats) |
| NVIDIA H200 | 2048 | 64 | False | cuda_graph | C2F | 1397.568 | 1386.288 | -0.58% [-0.82, -0.43] | inconclusive: effect <= dispersion floor |
| NVIDIA H200 | 2048 | 64 | False | cuda_graph | C2P | 885.296 | 1386.288 | +56.62% [+56.44, +57.21] | C3 slower (all matched repeats) |
| NVIDIA H200 | 2048 | 64 | False | eager | C0 | 916.800 | 1402.112 | +52.61% [+51.60, +62.46] | C3 slower (all matched repeats) |
| NVIDIA H200 | 2048 | 64 | False | eager | C1 | 901.056 | 1402.112 | +55.19% [+50.19, +66.87] | C3 slower (all matched repeats) |
| NVIDIA H200 | 2048 | 64 | False | eager | C2F | 1408.112 | 1402.112 | -0.74% [-0.77, +7.21] | inconclusive within dispersion |
| NVIDIA H200 | 2048 | 64 | False | eager | C2P | 900.640 | 1402.112 | +55.11% [+47.28, +67.72] | C3 slower (all matched repeats) |
