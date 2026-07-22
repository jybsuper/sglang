# C3 overlap matrix summary

Negative percentages mean C3 is faster. A winner is named only when all matched counterbalanced repeats agree on the sign and the median effect exceeds both 1% and the matched-repeat range.

| Device | T | R | Base rows | Execution | Baseline | Baseline us | C3 us | C3 vs baseline median [min, max] | Result |
|:---|---:|---:|:---:|:---|:---|---:|---:|:---|:---|
| NVIDIA GB300 | 1 | 32 | False | cuda_graph | C0 | 92.928 | 78.784 | -15.38% [-17.39, -15.19] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 1 | 32 | False | cuda_graph | C1 | 85.232 | 78.784 | -7.55% [-10.43, -5.89] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 1 | 32 | False | cuda_graph | C2F | 84.688 | 78.784 | -6.93% [-9.44, -5.16] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 1 | 32 | False | cuda_graph | C2P | 90.848 | 78.784 | -13.29% [-13.57, -11.53] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 1 | 32 | False | eager | C0 | 698.720 | 609.904 | -13.30% [-17.65, -11.38] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 1 | 32 | False | eager | C1 | 789.968 | 609.904 | -22.79% [-25.25, -21.43] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 1 | 32 | False | eager | C2F | 567.296 | 609.904 | +7.78% [+7.25, +9.99] | C3 slower (all matched repeats) |
| NVIDIA GB300 | 1 | 32 | False | eager | C2P | 646.080 | 609.904 | -5.46% [-6.00, -4.15] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 32 | 32 | False | cuda_graph | C0 | 316.160 | 293.632 | -6.88% [-7.39, -5.64] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 32 | 32 | False | cuda_graph | C1 | 301.824 | 293.632 | -2.74% [-3.94, -2.03] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 32 | 32 | False | cuda_graph | C2F | 305.408 | 293.632 | -3.53% [-5.33, -2.92] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 32 | 32 | False | cuda_graph | C2P | 314.080 | 293.632 | -6.41% [-7.20, -5.86] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 32 | 32 | False | eager | C0 | 703.744 | 632.496 | -10.15% [-16.31, -9.79] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 32 | 32 | False | eager | C1 | 781.200 | 632.496 | -19.36% [-21.90, -18.78] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 32 | 32 | False | eager | C2F | 568.288 | 632.496 | +10.80% [+9.84, +11.75] | C3 slower (all matched repeats) |
| NVIDIA GB300 | 32 | 32 | False | eager | C2P | 645.456 | 632.496 | -2.50% [-4.04, -1.06] | inconclusive: effect <= dispersion floor |
| NVIDIA GB300 | 32 | 64 | False | cuda_graph | C0 | 340.704 | 316.928 | -6.52% [-7.75, -5.88] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 32 | 64 | False | cuda_graph | C1 | 324.960 | 316.928 | -2.43% [-3.16, -1.91] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 32 | 64 | False | cuda_graph | C2F | 326.656 | 316.928 | -2.97% [-3.89, -1.89] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 32 | 64 | False | cuda_graph | C2P | 337.360 | 316.928 | -6.04% [-6.51, -5.40] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 32 | 64 | False | eager | C0 | 654.240 | 604.688 | -7.25% [-11.65, -6.88] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 32 | 64 | False | eager | C1 | 739.424 | 604.688 | -18.24% [-18.63, -16.68] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 32 | 64 | False | eager | C2F | 555.600 | 604.688 | +9.69% [+8.24, +10.44] | C3 slower (all matched repeats) |
| NVIDIA GB300 | 32 | 64 | False | eager | C2P | 621.360 | 604.688 | -2.58% [-3.46, -0.01] | inconclusive: effect <= dispersion floor |
| NVIDIA GB300 | 32 | 64 | True | cuda_graph | C0 | 323.424 | 308.304 | -4.45% [-5.23, -2.89] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 32 | 64 | True | cuda_graph | C1 | 307.328 | 308.304 | +0.12% [-0.95, +0.76] | inconclusive within dispersion |
| NVIDIA GB300 | 32 | 64 | True | cuda_graph | C2F | 318.208 | 308.304 | -3.38% [-4.04, -2.59] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 32 | 64 | True | cuda_graph | C2P | 322.304 | 308.304 | -4.17% [-5.56, -4.11] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 32 | 64 | True | eager | C0 | 737.232 | 691.392 | -6.22% [-11.60, -5.22] | inconclusive: effect <= dispersion floor |
| NVIDIA GB300 | 32 | 64 | True | eager | C1 | 828.368 | 691.392 | -16.86% [-18.30, -15.85] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 32 | 64 | True | eager | C2F | 639.616 | 691.392 | +8.15% [+7.18, +9.98] | C3 slower (all matched repeats) |
| NVIDIA GB300 | 32 | 64 | True | eager | C2P | 729.264 | 691.392 | -5.44% [-6.23, -4.59] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 32 | 128 | False | cuda_graph | C0 | 387.840 | 348.928 | -10.03% [-10.29, -9.45] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 32 | 128 | False | cuda_graph | C1 | 368.416 | 348.928 | -5.29% [-5.54, -5.00] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 32 | 128 | False | cuda_graph | C2F | 362.768 | 348.928 | -3.82% [-4.15, -3.39] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 32 | 128 | False | cuda_graph | C2P | 380.656 | 348.928 | -8.27% [-8.94, -8.06] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 32 | 128 | False | eager | C0 | 755.328 | 605.824 | -19.66% [-25.33, -19.61] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 32 | 128 | False | eager | C1 | 841.008 | 605.824 | -28.43% [-30.29, -27.66] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 32 | 128 | False | eager | C2F | 556.032 | 605.824 | +8.89% [+8.26, +9.16] | C3 slower (all matched repeats) |
| NVIDIA GB300 | 32 | 128 | False | eager | C2P | 631.072 | 605.824 | -3.94% [-5.09, -3.21] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 64 | 128 | False | cuda_graph | C0 | 563.504 | 496.384 | -11.98% [-12.00, -11.88] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 64 | 128 | False | cuda_graph | C1 | 539.344 | 496.384 | -8.06% [-8.15, -7.76] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 64 | 128 | False | cuda_graph | C2F | 509.728 | 496.384 | -2.62% [-2.99, -2.27] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 64 | 128 | False | cuda_graph | C2P | 545.520 | 496.384 | -9.01% [-9.11, -8.67] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 64 | 128 | False | eager | C0 | 816.752 | 650.384 | -20.37% [-24.73, -19.17] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 64 | 128 | False | eager | C1 | 913.008 | 650.384 | -28.75% [-30.01, -28.38] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 64 | 128 | False | eager | C2F | 603.248 | 650.384 | +9.20% [+6.29, +10.55] | C3 slower (all matched repeats) |
| NVIDIA GB300 | 64 | 128 | False | eager | C2P | 699.600 | 650.384 | -7.10% [-7.67, -5.63] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 128 | 128 | False | cuda_graph | C0 | 760.896 | 650.896 | -14.45% [-14.73, -14.31] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 128 | 128 | False | cuda_graph | C1 | 738.064 | 650.896 | -11.89% [-12.33, -11.58] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 128 | 128 | False | cuda_graph | C2F | 664.896 | 650.896 | -1.99% [-2.33, -1.73] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 128 | 128 | False | cuda_graph | C2P | 732.800 | 650.896 | -11.25% [-11.45, -11.02] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 128 | 128 | False | eager | C0 | 831.056 | 660.336 | -20.23% [-25.28, -19.83] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 128 | 128 | False | eager | C1 | 933.680 | 660.336 | -29.28% [-30.58, -27.59] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 128 | 128 | False | eager | C2F | 682.720 | 660.336 | -3.21% [-3.56, -2.46] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 128 | 128 | False | eager | C2P | 754.480 | 660.336 | -12.49% [-12.72, -11.74] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 256 | 32 | False | cuda_graph | C0 | 378.288 | 392.656 | +3.95% [+3.61, +5.50] | C3 slower (all matched repeats) |
| NVIDIA GB300 | 256 | 32 | False | cuda_graph | C1 | 367.376 | 392.656 | +6.67% [+6.36, +7.41] | C3 slower (all matched repeats) |
| NVIDIA GB300 | 256 | 32 | False | cuda_graph | C2F | 396.016 | 392.656 | -1.03% [-1.33, -0.51] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 256 | 32 | False | cuda_graph | C2P | 371.456 | 392.656 | +5.94% [+5.19, +6.08] | C3 slower (all matched repeats) |
| NVIDIA GB300 | 256 | 32 | False | eager | C0 | 691.600 | 623.712 | -9.89% [-15.34, -8.61] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 256 | 32 | False | eager | C1 | 783.408 | 623.712 | -19.47% [-24.01, -18.43] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 256 | 32 | False | eager | C2F | 580.208 | 623.712 | +8.21% [+7.22, +9.76] | C3 slower (all matched repeats) |
| NVIDIA GB300 | 256 | 32 | False | eager | C2P | 647.024 | 623.712 | -2.50% [-5.34, -0.96] | inconclusive: effect <= dispersion floor |
| NVIDIA GB300 | 2048 | 64 | False | cuda_graph | C0 | 682.720 | 1086.176 | +59.54% [+59.10, +60.56] | C3 slower (all matched repeats) |
| NVIDIA GB300 | 2048 | 64 | False | cuda_graph | C1 | 662.288 | 1086.176 | +64.22% [+63.62, +64.47] | C3 slower (all matched repeats) |
| NVIDIA GB300 | 2048 | 64 | False | cuda_graph | C2F | 1098.992 | 1086.176 | -1.16% [-1.35, -0.97] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 2048 | 64 | False | cuda_graph | C2P | 664.320 | 1086.176 | +63.44% [+63.12, +63.82] | C3 slower (all matched repeats) |
| NVIDIA GB300 | 2048 | 64 | False | eager | C0 | 854.912 | 1114.976 | +29.77% [+20.24, +30.66] | C3 slower (all matched repeats) |
| NVIDIA GB300 | 2048 | 64 | False | eager | C1 | 939.504 | 1114.976 | +18.44% [+13.58, +19.97] | C3 slower (all matched repeats) |
| NVIDIA GB300 | 2048 | 64 | False | eager | C2F | 1139.712 | 1114.976 | -2.06% [-2.58, -1.99] | C3 faster (all matched repeats) |
| NVIDIA GB300 | 2048 | 64 | False | eager | C2P | 802.192 | 1114.976 | +38.99% [+35.01, +41.06] | C3 slower (all matched repeats) |
