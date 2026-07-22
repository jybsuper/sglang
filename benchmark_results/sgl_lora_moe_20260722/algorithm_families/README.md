# SGL LoRA MoE algorithm-family benchmark

This evidence is benchmark-only; no serving dispatch imports these candidates.
All final winners use the median p50 of forward and reverse measurement orders.
`K0` prebuilds routing metadata; `O0` charges route/segment construction,
padding/packing, clears/casts, and every consumer.

## Coverage and validity

- Final successful run records: 9216.
- Counterbalanced winner cells: 1024; order-stable: 968.
- Canonical graph/hot cells with order-dependent winners: 29/256; treat these as autotune
  boundaries, not hard dispatch thresholds.
- Devices: H200 (SM90) and GB300 (SM103).
- Core dimensions: T={1,32,256,2048}, R={16,32,64,128}, four sites,
  eager/CUDA graph, hot/cold cache, K0/O0.
- Adapter anchors: four active adapters plus separate mixed base/LoRA rows;
  regular equal-M_g, true IID, and Zipf-skewed routes.
- Every eager candidate and graph replay is checked against an independent
  chunked FP32 PyTorch oracle.

## Main decisions

- With prebuilt routing (K0), padded aligned/grouped is the default winner
  for gate-A and all prefill-scale sites. Raw indexed remains best for many
  T<=32 down-A/finalize cells; segmented is competitive for the fused gate
  consumer at decode sizes.
- When route building is charged (O0), raw indexed wins nearly every T<=32
  cell and many T=256 cells. Aligned/grouped overtakes it at T=2048.
- BMM is useful only on recorded exactly-equal M_g routes. It wins selected
  large gate-A cells when routing is prebuilt, but no canonical O0 fused
  gate-consumer or down-finalize cell after route construction is charged.
  IID/skewed mixed routes are rejected rather than padded into a misleading
  BMM result.
- The one-shot down A+B probe is decisively rejected: each H tile recomputes
  the A reduction. Its median slowdown versus the best decomposed arm is
  16.25x on H200 and 15.50x on GB300.
  Shared-A factorization requires a different cooperative schedule.
- Current aligned C2 routing omits base-only sentinel blocks. The fair mixed
  benchmark charges an explicit base-SwiGLU prepass; serving promotion must
  implement equivalent semantics.
- CuTe DSL 4.6 compiles and graph-replays the GB300 raw-indexed probe, but
  this scalar capability version is 1.95–12.40x slower than tuned Triton.
  The H200 image cannot import `cutlass`, despite installed package
  metadata.

## How candidate implementations were kept honest

- Indexed, aligned, segmented, fused-consumer, and finalizer tiles were
  swept independently before cross-family comparison. The exact candidate
  sets and per-anchor winners are in `summary.json` under `tuning`.
- Each sweep used identical weights, routes, output contracts, CUDA-graph
  replay, cache state, and independent oracle. Slower families therefore
  were not compared using a single arbitrary tile inherited from a winner.
- Forward/reverse counterbalancing exposes close boundary cells: the median
  runner-up margin in order-unstable cells is 0.91%.
  Keep those cells autotuned rather than encoding exact table transitions.
- Selected core settings are included below; they are robust defaults, not
  a claim that one tile is optimal for every site and shape.

```json
{
  "gb300": {
    "aligned_bk": 64,
    "aligned_bn": 64,
    "aligned_consumer_bn": 64,
    "aligned_warps": 4,
    "consumer_warps": 4,
    "direct_finalize_bh": 64,
    "indexed_bk": 128,
    "indexed_bn": 32,
    "indexed_warps": 4,
    "one_shot_bh": 32,
    "one_shot_bi": 64,
    "pair_consumer_bn": 64,
    "reduce_finalize_bh": 128,
    "segmented_bk": 64,
    "segmented_bn": 64,
    "segmented_consumer_bn": 64,
    "segmented_warps": 4
  },
  "h200": {
    "aligned_bk": 64,
    "aligned_bn": 64,
    "aligned_consumer_bn": 64,
    "aligned_warps": 4,
    "consumer_warps": 4,
    "direct_finalize_bh": 64,
    "indexed_bk": 128,
    "indexed_bn": 32,
    "indexed_warps": 4,
    "one_shot_bh": 32,
    "one_shot_bi": 64,
    "pair_consumer_bn": 64,
    "reduce_finalize_bh": 128,
    "segmented_bk": 64,
    "segmented_bn": 64,
    "segmented_consumer_bn": 64,
    "segmented_warps": 4
  }
}
```

## O0 BMM cost-accounting correction

- The initial O0 BMM gate-consumer and down-finalize arms reused a prebuilt
  order. The corrected closures rebuild virtual-group ids and stable-sort
  the route inside every timed invocation, matching the O0 contract.
- 512 affected records were rerun across
  both devices, both measurement orders, eager/graph, and hot/cold cache;
  every corrected run and independent graph-oracle check passed.
- 28 of 1,024 winner cells changed, including 7 canonical graph/hot cells.
  All 7 canonical changes remove previously understated BMM winners from
  O0 down-finalize. No corrected canonical O0 fused-consumer/finalize cell
  selects BMM.
- Exact before/after transitions and raw reruns are retained in
  `correction_manifest.json` and each device's `corrections/` directory.

## Canonical CUDA-graph/hot winner maps

Each cell is `family median-p50` across forward/reverse order.

#### H200 K0 — `gate_a`

| T \ R | 16 | 32 | 64 | 128 |
|---:|---:|---:|---:|---:|
| 1 | aligned 10.22 us | aligned 10.67 us | aligned 10.53 us | aligned 10.75 us |
| 32 | aligned 10.45 us | aligned 11.42 us | aligned 13.30 us | aligned 14.33 us |
| 256 | aligned 12.61 us | aligned 13.99 us | aligned 19.26 us | aligned 26.10 us |
| 2048 | aligned 29.09 us | aligned 38.11 us | aligned 67.58 us | bmm 98.82 us |

#### H200 K0 — `gate_consumer`

| T \ R | 16 | 32 | 64 | 128 |
|---:|---:|---:|---:|---:|
| 1 | segmented 10.18 us | aligned 10.54 us | aligned 10.55 us | aligned 10.58 us |
| 32 | segmented 12.25 us | segmented 11.16 us | segmented 11.85 us | aligned 13.82 us |
| 256 | aligned 16.54 us | aligned 12.75 us | aligned 14.17 us | aligned 20.72 us |
| 2048 | aligned 75.50 us | aligned 46.10 us | aligned 63.86 us | aligned 107.10 us |

#### H200 K0 — `down_a`

| T \ R | 16 | 32 | 64 | 128 |
|---:|---:|---:|---:|---:|
| 1 | segmented 10.25 us | indexed 10.54 us | indexed 10.54 us | indexed 10.54 us |
| 32 | segmented 10.28 us | segmented 10.56 us | aligned 10.56 us | segmented 12.00 us |
| 256 | aligned 10.29 us | aligned 10.54 us | aligned 11.07 us | aligned 12.76 us |
| 2048 | aligned 13.37 us | aligned 13.96 us | aligned 16.74 us | aligned 25.54 us |

#### H200 K0 — `down_finalize`

| T \ R | 16 | 32 | 64 | 128 |
|---:|---:|---:|---:|---:|
| 1 | indexed 9.73 us | indexed 10.51 us | indexed 10.52 us | aligned 11.09 us |
| 32 | indexed 10.61 us | aligned 11.72 us | aligned 12.83 us | aligned 14.14 us |
| 256 | aligned 17.36 us | aligned 18.18 us | aligned 20.92 us | aligned 30.18 us |
| 2048 | aligned 106.77 us | aligned 118.31 us | aligned 137.30 us | aligned 175.13 us |

#### H200 O0 — `gate_a`

| T \ R | 16 | 32 | 64 | 128 |
|---:|---:|---:|---:|---:|
| 1 | indexed 14.97 us | indexed 15.23 us | indexed 12.86 us | indexed 12.97 us |
| 32 | indexed 14.09 us | indexed 15.42 us | indexed 22.23 us | aligned 36.49 us |
| 256 | aligned 35.41 us | aligned 37.84 us | aligned 44.02 us | aligned 53.34 us |
| 2048 | aligned 58.18 us | aligned 68.64 us | aligned 98.22 us | aligned 162.64 us |

#### H200 O0 — `gate_consumer`

| T \ R | 16 | 32 | 64 | 128 |
|---:|---:|---:|---:|---:|
| 1 | indexed 14.39 us | indexed 14.54 us | indexed 13.27 us | indexed 13.30 us |
| 32 | indexed 14.15 us | indexed 14.49 us | indexed 15.81 us | indexed 23.82 us |
| 256 | indexed 32.93 us | aligned 35.76 us | aligned 38.47 us | aligned 44.95 us |
| 2048 | aligned 108.84 us | aligned 79.42 us | aligned 97.10 us | aligned 137.85 us |

#### H200 O0 — `down_a`

| T \ R | 16 | 32 | 64 | 128 |
|---:|---:|---:|---:|---:|
| 1 | indexed 10.93 us | indexed 10.92 us | indexed 10.69 us | indexed 10.59 us |
| 32 | indexed 10.77 us | indexed 11.06 us | indexed 10.72 us | indexed 12.33 us |
| 256 | indexed 15.16 us | indexed 15.36 us | indexed 23.45 us | aligned 36.06 us |
| 2048 | aligned 42.26 us | aligned 42.81 us | aligned 45.85 us | aligned 53.44 us |

#### H200 O0 — `down_finalize`

| T \ R | 16 | 32 | 64 | 128 |
|---:|---:|---:|---:|---:|
| 1 | indexed 10.29 us | indexed 11.02 us | indexed 10.54 us | indexed 11.47 us |
| 32 | indexed 10.97 us | indexed 12.57 us | indexed 14.90 us | indexed 24.61 us |
| 256 | indexed 32.87 us | aligned 41.71 us | aligned 45.40 us | aligned 56.63 us |
| 2048 | aligned 140.38 us | aligned 152.11 us | aligned 170.90 us | aligned 207.49 us |

#### GB300 K0 — `gate_a`

| T \ R | 16 | 32 | 64 | 128 |
|---:|---:|---:|---:|---:|
| 1 | aligned 10.44 us | aligned 10.06 us | aligned 10.15 us | aligned 10.98 us |
| 32 | aligned 10.63 us | aligned 10.27 us | aligned 11.80 us | aligned 13.76 us |
| 256 | aligned 12.98 us | aligned 14.28 us | aligned 17.85 us | aligned 22.94 us |
| 2048 | aligned 28.11 us | segmented 34.22 us | bmm 53.73 us | bmm 71.14 us |

#### GB300 K0 — `gate_consumer`

| T \ R | 16 | 32 | 64 | 128 |
|---:|---:|---:|---:|---:|
| 1 | aligned 10.35 us | segmented 10.18 us | aligned 9.93 us | aligned 9.93 us |
| 32 | aligned 12.17 us | segmented 10.08 us | aligned 10.41 us | aligned 13.70 us |
| 256 | aligned 17.70 us | aligned 13.01 us | aligned 14.76 us | aligned 19.42 us |
| 2048 | aligned 52.98 us | aligned 33.51 us | aligned 45.66 us | aligned 81.39 us |

#### GB300 K0 — `down_a`

| T \ R | 16 | 32 | 64 | 128 |
|---:|---:|---:|---:|---:|
| 1 | segmented 10.20 us | indexed 10.06 us | aligned 9.94 us | indexed 9.89 us |
| 32 | segmented 9.89 us | aligned 9.84 us | segmented 10.00 us | aligned 9.97 us |
| 256 | segmented 10.26 us | segmented 10.20 us | aligned 10.29 us | aligned 11.57 us |
| 2048 | aligned 13.70 us | aligned 14.30 us | aligned 16.82 us | aligned 22.96 us |

#### GB300 K0 — `down_finalize`

| T \ R | 16 | 32 | 64 | 128 |
|---:|---:|---:|---:|---:|
| 1 | aligned 10.24 us | indexed 9.67 us | indexed 9.98 us | indexed 10.03 us |
| 32 | indexed 9.93 us | aligned 11.51 us | aligned 12.23 us | aligned 13.68 us |
| 256 | aligned 17.83 us | aligned 17.85 us | aligned 19.93 us | aligned 24.02 us |
| 2048 | aligned 84.20 us | aligned 92.64 us | aligned 110.03 us | aligned 141.38 us |

#### GB300 O0 — `gate_a`

| T \ R | 16 | 32 | 64 | 128 |
|---:|---:|---:|---:|---:|
| 1 | indexed 13.45 us | indexed 13.76 us | indexed 13.73 us | indexed 14.23 us |
| 32 | indexed 13.77 us | indexed 17.79 us | indexed 21.90 us | indexed 33.02 us |
| 256 | indexed 33.11 us | aligned 38.32 us | aligned 40.64 us | aligned 45.54 us |
| 2048 | aligned 57.78 us | aligned 64.99 us | aligned 85.46 us | aligned 127.46 us |

#### GB300 O0 — `gate_consumer`

| T \ R | 16 | 32 | 64 | 128 |
|---:|---:|---:|---:|---:|
| 1 | indexed 13.98 us | indexed 13.73 us | indexed 13.94 us | indexed 14.30 us |
| 32 | indexed 15.90 us | indexed 16.30 us | indexed 17.80 us | indexed 22.78 us |
| 256 | indexed 30.16 us | aligned 37.31 us | aligned 36.29 us | aligned 42.45 us |
| 2048 | aligned 85.44 us | aligned 62.93 us | aligned 77.25 us | aligned 114.14 us |

#### GB300 O0 — `down_a`

| T \ R | 16 | 32 | 64 | 128 |
|---:|---:|---:|---:|---:|
| 1 | indexed 12.02 us | indexed 10.30 us | indexed 10.39 us | indexed 10.64 us |
| 32 | indexed 10.29 us | indexed 10.31 us | indexed 10.12 us | indexed 12.45 us |
| 256 | indexed 15.63 us | indexed 16.13 us | indexed 21.84 us | indexed 34.24 us |
| 2048 | aligned 44.48 us | aligned 44.08 us | aligned 46.54 us | aligned 52.67 us |

#### GB300 O0 — `down_finalize`

| T \ R | 16 | 32 | 64 | 128 |
|---:|---:|---:|---:|---:|
| 1 | indexed 11.33 us | indexed 10.09 us | indexed 10.38 us | indexed 10.19 us |
| 32 | indexed 10.26 us | indexed 11.66 us | indexed 15.72 us | indexed 26.04 us |
| 256 | indexed 30.14 us | indexed 40.38 us | aligned 42.65 us | aligned 46.54 us |
| 2048 | aligned 118.22 us | aligned 127.44 us | aligned 144.84 us | aligned 175.57 us |

## Route and accumulation evidence

```json
{
  "accumulation": {
    "gb300": {
      "down_fp32_successful_runs": 96,
      "gate_fp32_disqualifications": [],
      "gate_fp32_successful_runs": 32,
      "one_shot_accumulation": [
        {
          "fp32_vs_bf16_pct": 2.49999718056646,
          "one_shot_bf16": 12.80000014230609,
          "one_shot_fp32": 13.119999784976244,
          "rank": 16,
          "tokens": 1
        },
        {
          "fp32_vs_bf16_pct": -4.317385113564032,
          "one_shot_bf16": 13.712000101804733,
          "one_shot_fp32": 13.120000250637531,
          "rank": 32,
          "tokens": 1
        },
        {
          "fp32_vs_bf16_pct": -0.6012011658563132,
          "one_shot_bf16": 15.967999584972858,
          "one_shot_fp32": 15.87199978530407,
          "rank": 64,
          "tokens": 1
        },
        {
          "fp32_vs_bf16_pct": -0.08944611818945125,
          "one_shot_bf16": 17.88800023496151,
          "one_shot_fp32": 17.872000113129616,
          "rank": 128,
          "tokens": 1
        },
        {
          "fp32_vs_bf16_pct": -0.5464487247397454,
          "one_shot_bf16": 52.70399898290634,
          "one_shot_fp32": 52.4159986525774,
          "rank": 16,
          "tokens": 32
        },
        {
          "fp32_vs_bf16_pct": -0.07153130622993276,
          "one_shot_bf16": 67.10399687290192,
          "one_shot_fp32": 67.05599650740623,
          "rank": 32,
          "tokens": 32
        },
        {
          "fp32_vs_bf16_pct": -1.9408506882777843,
          "one_shot_bf16": 103.87200117111206,
          "one_shot_fp32": 101.85600072145462,
          "rank": 64,
          "tokens": 32
        },
        {
          "fp32_vs_bf16_pct": 1.208687841254097,
          "one_shot_bf16": 169.44000124931335,
          "one_shot_fp32": 171.48800194263458,
          "rank": 128,
          "tokens": 32
        },
        {
          "fp32_vs_bf16_pct": -2.981186572955985,
          "one_shot_bf16": 343.48800778388977,
          "one_shot_fp32": 333.24798941612244,
          "rank": 16,
          "tokens": 256
        },
        {
          "fp32_vs_bf16_pct": -1.7412603147366146,
          "one_shot_bf16": 470.46399116516113,
          "one_shot_fp32": 462.2719883918762,
          "rank": 32,
          "tokens": 256
        },
        {
          "fp32_vs_bf16_pct": -1.1645374022284605,
          "one_shot_bf16": 735.0559830665588,
          "one_shot_fp32": 726.4959812164307,
          "rank": 64,
          "tokens": 256
        },
        {
          "fp32_vs_bf16_pct": 1.0522574360929404,
          "one_shot_bf16": 1265.0879621505737,
          "one_shot_fp32": 1278.39994430542,
          "rank": 128,
          "tokens": 256
        },
        {
          "fp32_vs_bf16_pct": -2.891447139100156,
          "one_shot_bf16": 2691.5199756622314,
          "one_shot_fp32": 2613.6960983276367,
          "rank": 16,
          "tokens": 2048
        },
        {
          "fp32_vs_bf16_pct": -1.5476348385750405,
          "one_shot_bf16": 3706.3040733337402,
          "one_shot_fp32": 3648.9440202713013,
          "rank": 32,
          "tokens": 2048
        },
        {
          "fp32_vs_bf16_pct": -1.0540269705385463,
          "one_shot_bf16": 5829.071998596191,
          "one_shot_fp32": 5767.632007598877,
          "rank": 64,
          "tokens": 2048
        },
        {
          "fp32_vs_bf16_pct": 0.9561399164195628,
          "one_shot_bf16": 10070.464134216309,
          "one_shot_fp32": 10166.751861572266,
          "rank": 128,
          "tokens": 2048
        }
      ],
      "one_shot_vs_best_non_one_shot": {
        "comparison_count": 32,
        "max_slowdown_x": 60.74780291112217,
        "median_slowdown_x": 15.495538232207469,
        "min_slowdown_x": 1.1834319218377913
      }
    },
    "h200": {
      "down_fp32_successful_runs": 192,
      "gate_fp32_disqualifications": [],
      "gate_fp32_successful_runs": 64,
      "one_shot_accumulation": [
        {
          "fp32_vs_bf16_pct": 2.7247965586742806,
          "one_shot_bf16": 11.744000017642975,
          "one_shot_fp32": 12.064000125974417,
          "rank": 16,
          "tokens": 1
        },
        {
          "fp32_vs_bf16_pct": -1.624997862687394,
          "one_shot_bf16": 12.799999676644802,
          "one_shot_fp32": 12.59199995547533,
          "rank": 32,
          "tokens": 1
        },
        {
          "fp32_vs_bf16_pct": -0.7510724386849077,
          "one_shot_bf16": 14.911999925971031,
          "one_shot_fp32": 14.800000004470348,
          "rank": 64,
          "tokens": 1
        },
        {
          "fp32_vs_bf16_pct": -1.5458942957840116,
          "one_shot_bf16": 16.55999943614006,
          "one_shot_fp32": 16.303999349474907,
          "rank": 128,
          "tokens": 1
        },
        {
          "fp32_vs_bf16_pct": -3.264261785440259,
          "one_shot_bf16": 56.36800080537796,
          "one_shot_fp32": 54.52800169587135,
          "rank": 16,
          "tokens": 32
        },
        {
          "fp32_vs_bf16_pct": -0.883186132528957,
          "one_shot_bf16": 79.71200346946716,
          "one_shot_fp32": 79.00799810886383,
          "rank": 32,
          "tokens": 32
        },
        {
          "fp32_vs_bf16_pct": -0.2554594460397075,
          "one_shot_bf16": 125.2639964222908,
          "one_shot_fp32": 124.94399771094322,
          "rank": 64,
          "tokens": 32
        },
        {
          "fp32_vs_bf16_pct": -0.2711916039659701,
          "one_shot_bf16": 230.0959974527359,
          "one_shot_fp32": 229.47199642658234,
          "rank": 128,
          "tokens": 32
        },
        {
          "fp32_vs_bf16_pct": -5.753189248185975,
          "one_shot_bf16": 381.8399906158447,
          "one_shot_fp32": 359.8720133304596,
          "rank": 16,
          "tokens": 256
        },
        {
          "fp32_vs_bf16_pct": -0.9692017763076377,
          "one_shot_bf16": 566.2400126457214,
          "one_shot_fp32": 560.7520043849945,
          "rank": 32,
          "tokens": 256
        },
        {
          "fp32_vs_bf16_pct": -0.5428071638787557,
          "one_shot_bf16": 922.6079881191254,
          "one_shot_fp32": 917.600005865097,
          "rank": 64,
          "tokens": 256
        },
        {
          "fp32_vs_bf16_pct": -0.16728701799569867,
          "one_shot_bf16": 1798.1120347976685,
          "one_shot_fp32": 1795.1040267944336,
          "rank": 128,
          "tokens": 256
        },
        {
          "fp32_vs_bf16_pct": -5.756108050997755,
          "one_shot_bf16": 3011.199951171875,
          "one_shot_fp32": 2837.87202835083,
          "rank": 16,
          "tokens": 2048
        },
        {
          "fp32_vs_bf16_pct": -1.0366434428530802,
          "one_shot_bf16": 4483.695983886719,
          "one_shot_fp32": 4437.21604347229,
          "rank": 32,
          "tokens": 2048
        },
        {
          "fp32_vs_bf16_pct": -0.46844030506283385,
          "one_shot_bf16": 7394.767999649048,
          "one_shot_fp32": 7360.127925872803,
          "rank": 64,
          "tokens": 2048
        },
        {
          "fp32_vs_bf16_pct": 0.4338957303900992,
          "one_shot_bf16": 14421.855926513672,
          "one_shot_fp32": 14484.431743621826,
          "rank": 128,
          "tokens": 2048
        }
      ],
      "one_shot_vs_best_non_one_shot": {
        "comparison_count": 32,
        "max_slowdown_x": 68.03013308282503,
        "median_slowdown_x": 16.249365525509862,
        "min_slowdown_x": 1.1189024590430128
      }
    }
  },
  "route": {
    "gb300": {
      "bmm_policy": "admit only >=2 nonempty groups with exactly equal M_g",
      "disqualification_count": 48,
      "disqualification_reasons": {
        "BMM_DISQUALIFIED:nonuniform_M_g": 48
      },
      "disqualified_route_cv_m_g_max": 0.936930775642395,
      "disqualified_route_cv_m_g_min": 0.08086954057216644,
      "successful_runs": 252
    },
    "h200": {
      "bmm_policy": "admit only >=2 nonempty groups with exactly equal M_g",
      "disqualification_count": 48,
      "disqualification_reasons": {
        "BMM_DISQUALIFIED:nonuniform_M_g": 48
      },
      "disqualified_route_cv_m_g_max": 0.936930775642395,
      "disqualified_route_cv_m_g_min": 0.08086954057216644,
      "successful_runs": 672
    }
  }
}
```

## Files

- `winners.csv`: every exact winner across device/scope/T/R/site/
  eager-or-graph/hot-or-cold.
- `summary.json`: machine-readable conclusions, route admission, CuTe,
  tile tuning, stability, and FP32/BF16 evidence.
- `correction_manifest.json`: exact O0 BMM cost-accounting rerun and
  before/after winner transitions.
- `h200/` and `gb300/`: raw final, reverse-order, tuning, correctness,
  route, accumulation, correction, and CuTe artifacts.
- `SHA256SUMS`: hashes for every evidence file.
