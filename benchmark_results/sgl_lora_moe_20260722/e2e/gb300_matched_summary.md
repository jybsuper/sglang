# Matched GB300 model-level SGL-LoRA comparison

Three counterbalanced repetitions were run on the same GPU. Positive
percentages mean SGL is numerically larger; that is favorable for throughput
and unfavorable for latency/TTFT.

| Traffic | BS | SGL out tok/s | Control out tok/s | Difference | SGL input tok/s | Control input tok/s |
|---|---:|---:|---:|---:|---:|---:|
| base | 1 | 448.40 | 421.15 | +6.47% | 2788.40 | 1582.21 |
| base | 16 | 3451.40 | 3502.64 | -1.46% | 37751.00 | 31490.92 |
| base | 32 | 5914.83 | 5894.69 | +0.34% | 68507.48 | 57067.17 |
| lora | 1 | 346.90 | 406.73 | -14.71% | 1965.80 | 1633.60 |
| lora | 16 | 2966.59 | 3469.76 | -14.50% | 27853.65 | 27205.42 |
| lora | 32 | 5190.14 | 5935.44 | -12.56% | 50039.21 | 49988.20 |

## Correctness and profiling

- SGL transition checks passed: `True`.
- Control transition checks passed: `True`.
- The control produced a base trace, but its adapter profiling request
  crashed the scheduler (the exact markers and log are retained).
- SGL produced both base and adapter EXTEND/DECODE traces without a crash.

The JSON companion retains p20/p80/sample values and aggregated kernel
launch/time tables. Unprofiled `bench_one_batch_server` repetitions are the
performance authority; traces are structural evidence.
