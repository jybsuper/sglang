# Rank-8/16 MoE-LoRA graduation guardrail

Logical rank 8 initially failed three Triton tensor-core paths because their
compile-time K tile was 8. The serving expand path, the aligned C2 consumer,
and the benchmark segmented consumer now use a masked physical-16 dot tile
while retaining logical-rank storage and zero-masked tail lanes.

The retained matrix covers H200 and GB300, T={1,32,256,2048}, logical
R={8,16}, K0/O0, eager/CUDA graph, hot/cold cache, all four sites,
forward/reverse order, all-active rows, and mixed base/adapter rows. Every
successful row has an independent numerical oracle; every graph row also
has an independent replay oracle.

Successful timed rows: 10624; graph replay oracles: 5312.

## Canonical all-active CUDA-graph/hot BF16 winners

Each cell is `family p50-us` after taking the median of forward and reverse
orders. T=1 uses the all-active fixture; the mixed fixture intentionally makes
its only row base-only and is retained as a separate semantic control.

### H200 K0 `gate_a`

| T | R8 (physical dot 16) | R16 |
|---:|---:|---:|
| 1 | aligned 10.86 | aligned 11.07 |
| 32 | aligned 11.79 | aligned 11.86 |
| 256 | aligned 12.74 | aligned 13.39 |
| 2048 | aligned 30.78 | aligned 32.42 |

### H200 K0 `gate_consumer`

| T | R8 (physical dot 16) | R16 |
|---:|---:|---:|
| 1 | indexed 12.66 | segmented 10.80 |
| 32 | indexed 15.42 | indexed 14.00 |
| 256 | aligned 25.94 | aligned 19.39 |
| 2048 | aligned 113.94 | aligned 79.86 |

### H200 K0 `down_a`

| T | R8 (physical dot 16) | R16 |
|---:|---:|---:|
| 1 | segmented 10.66 | aligned 10.66 |
| 32 | indexed 10.90 | indexed 10.78 |
| 256 | aligned 11.66 | aligned 10.91 |
| 2048 | aligned 14.42 | aligned 14.28 |

### H200 K0 `down_finalize`

| T | R8 (physical dot 16) | R16 |
|---:|---:|---:|
| 1 | indexed 10.30 | indexed 10.36 |
| 32 | indexed 12.19 | indexed 11.10 |
| 256 | aligned 21.01 | aligned 18.33 |
| 2048 | aligned 120.91 | aligned 109.73 |

### H200 O0 `gate_a`

| T | R8 (physical dot 16) | R16 |
|---:|---:|---:|
| 1 | indexed 12.63 | indexed 13.18 |
| 32 | indexed 14.18 | indexed 14.34 |
| 256 | indexed 34.65 | aligned 35.90 |
| 2048 | aligned 61.39 | aligned 63.14 |

### H200 O0 `gate_consumer`

| T | R8 (physical dot 16) | R16 |
|---:|---:|---:|
| 1 | indexed 12.63 | indexed 12.38 |
| 32 | indexed 15.50 | indexed 14.20 |
| 256 | aligned 50.36 | indexed 33.06 |
| 2048 | aligned 148.37 | aligned 114.67 |

### H200 O0 `down_a`

| T | R8 (physical dot 16) | R16 |
|---:|---:|---:|
| 1 | indexed 10.94 | indexed 11.06 |
| 32 | indexed 10.95 | indexed 11.02 |
| 256 | indexed 15.14 | indexed 15.26 |
| 2048 | aligned 44.24 | aligned 44.35 |

### H200 O0 `down_finalize`

| T | R8 (physical dot 16) | R16 |
|---:|---:|---:|
| 1 | indexed 12.90 | indexed 10.42 |
| 32 | indexed 12.24 | indexed 11.01 |
| 256 | aligned 45.45 | indexed 32.96 |
| 2048 | aligned 156.30 | aligned 144.76 |

### GB300 K0 `gate_a`

| T | R8 (physical dot 16) | R16 |
|---:|---:|---:|
| 1 | aligned 10.14 | aligned 9.22 |
| 32 | aligned 10.10 | aligned 10.95 |
| 256 | aligned 11.62 | aligned 12.25 |
| 2048 | aligned 26.56 | aligned 28.25 |

### GB300 K0 `gate_consumer`

| T | R8 (physical dot 16) | R16 |
|---:|---:|---:|
| 1 | indexed 13.02 | aligned 8.91 |
| 32 | indexed 16.32 | aligned 12.92 |
| 256 | aligned 25.18 | aligned 18.21 |
| 2048 | aligned 89.02 | aligned 55.24 |

### GB300 K0 `down_a`

| T | R8 (physical dot 16) | R16 |
|---:|---:|---:|
| 1 | segmented 9.24 | indexed 9.04 |
| 32 | indexed 9.18 | segmented 9.22 |
| 256 | segmented 9.28 | segmented 9.27 |
| 2048 | aligned 13.18 | aligned 13.30 |

### GB300 K0 `down_finalize`

| T | R8 (physical dot 16) | R16 |
|---:|---:|---:|
| 1 | indexed 9.16 | aligned 8.96 |
| 32 | indexed 10.43 | indexed 9.30 |
| 256 | aligned 19.37 | aligned 18.19 |
| 2048 | aligned 95.17 | aligned 84.94 |

### GB300 O0 `gate_a`

| T | R8 (physical dot 16) | R16 |
|---:|---:|---:|
| 1 | indexed 12.56 | indexed 13.14 |
| 32 | indexed 13.07 | indexed 13.23 |
| 256 | indexed 30.65 | indexed 32.67 |
| 2048 | aligned 59.30 | aligned 59.30 |

### GB300 O0 `gate_consumer`

| T | R8 (physical dot 16) | R16 |
|---:|---:|---:|
| 1 | indexed 13.22 | indexed 12.78 |
| 32 | indexed 16.30 | indexed 15.70 |
| 256 | aligned 49.31 | indexed 29.86 |
| 2048 | aligned 121.78 | aligned 88.00 |

### GB300 O0 `down_a`

| T | R8 (physical dot 16) | R16 |
|---:|---:|---:|
| 1 | indexed 9.26 | indexed 9.36 |
| 32 | indexed 9.34 | indexed 9.59 |
| 256 | indexed 14.46 | indexed 14.87 |
| 2048 | aligned 44.99 | aligned 46.00 |

### GB300 O0 `down_finalize`

| T | R8 (physical dot 16) | R16 |
|---:|---:|---:|
| 1 | indexed 9.49 | indexed 9.31 |
| 32 | indexed 10.90 | indexed 9.46 |
| 256 | indexed 38.83 | indexed 29.76 |
| 2048 | aligned 132.89 | aligned 122.06 |

## Interpretation

- Rank 8 is now compile-legal and numerically covered without padding the stored factors.
- Rank 8 and rank 16 share a physical tensor-core K floor, but their logical factor bytes and non-dot work remain distinct; the selector must not conflate the two ranks.
- Winner changes with site, work size, scope, and device. These rows are planner/autotune evidence, not a universal family cutoff.
- `raw_family_rows.json` retains every counterbalanced family result; `winners.csv` retains every exact comparison winner.
