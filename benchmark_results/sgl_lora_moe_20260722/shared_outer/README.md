# Shared-outer LoRA confirmation

This bundle evaluates the two shared-factor specializations that are required by
the MoE-LoRA plan:

- gate/up A: compute once per token/adapter and materialize the current
  pair-major `[T, K, 2R]` consumer contract;
- down B: reduce top-k weighted rank values first, then multiply the shared B
  factor once per token.

The controls invoke the current production virtual-expert kernels. The candidates
are independent Triton implementations with their own per-device configuration
sweeps. Each confirmation JSON contains 20 warmups and 200 counterbalanced CUDA
Graph samples per arm, an FP32 algebra oracle, the resolved tile configuration,
route metadata, p20/p50/p80 dispersion, and environment/source identity. The gate
comparison below charges pair materialization, because the production gate-B
consumer still requires pair-major input. The token-owned result without
materialization is retained in each JSON as an interface-change diagnostic only.

`delta` is `production / candidate - 1`; positive means that the candidate is
faster. Times are p50 microseconds.

## H200 confirmation

| case | gate production | gate token + materialize | delta | down production | down weighted-rank | delta |
|---|---:|---:|---:|---:|---:|---:|
| `kimi-t32-r64-l4` | 15.776 | 23.360 | -32.5% | 12.352 | 10.528 | +17.3% |
| `qwen35-t1-r32-l1` | 11.088 | 12.192 | -9.1% | 10.752 | 10.560 | +1.8% |
| `qwen35-t2048-r64-l3` | 70.832 | 18.592 | +281.0% | 62.688 | 14.912 | +320.4% |
| `qwen35-t256-r64-l3` | 21.600 | 12.832 | +68.3% | 15.936 | 10.704 | +48.9% |
| `qwen35-t32-r128-l8` | 16.256 | 12.640 | +28.6% | 11.072 | 11.120 | -0.4% |
| `qwen35-t32-r64-l1` | 12.768 | 12.480 | +2.3% | 11.328 | 10.656 | +6.3% |
| `qwen35-t32-r64-l4-base` | 12.896 | 12.960 | -0.5% | 11.232 | 10.560 | +6.4% |
| `qwen35-t32-r64-l4` | 12.240 | 12.048 | +1.6% | 10.400 | 9.888 | +5.2% |
| `qwen397-t32-r64-l4` | 18.432 | 17.280 | +6.7% | 12.896 | 10.592 | +21.8% |
| `smoke-t4-r16-l1` | 10.640 | 10.528 | +1.1% | 10.688 | 10.784 | -0.9% |

## GB300 confirmation

| case | gate production | gate token + materialize | delta | down production | down weighted-rank | delta |
|---|---:|---:|---:|---:|---:|---:|
| `kimi-t32-r64-l4` | 16.960 | 26.080 | -35.0% | 17.312 | 12.560 | +37.8% |
| `qwen35-t1-r32-l1` | 15.632 | 14.832 | +5.4% | 12.960 | 12.080 | +7.3% |
| `qwen35-t2048-r64-l3` | 68.176 | 21.168 | +222.1% | 60.256 | 14.240 | +323.1% |
| `qwen35-t256-r64-l3` | 24.736 | 15.680 | +57.8% | 19.552 | 12.400 | +57.7% |
| `qwen35-t32-r128-l8` | 17.280 | 14.176 | +21.9% | 12.432 | 9.760 | +27.4% |
| `qwen35-t32-r64-l1` | 14.848 | 14.016 | +5.9% | 13.376 | 11.360 | +17.7% |
| `qwen35-t32-r64-l4-base` | 15.008 | 15.120 | -0.7% | 13.488 | 12.096 | +11.5% |
| `qwen35-t32-r64-l4` | 15.344 | 17.568 | -12.7% | 14.080 | 11.968 | +17.6% |
| `qwen397-t32-r64-l4` | 19.632 | 18.160 | +8.1% | 15.216 | 10.976 | +38.6% |
| `smoke-t4-r16-l1` | 15.552 | 12.720 | +22.3% | 10.624 | 11.584 | -8.3% |

## Decision

Neither specialization is a universal production replacement.

- Gate-A token deduplication should be selected only when its saved repeated work
  amortizes pair materialization. Substantial Qwen prefill is an unambiguous win;
  wide Kimi decode and some adapter/base mixtures remain on the production path.
- Down-B weighted-rank reduction should be the preferred shared-factor path for
  substantial rows. Tiny H200 rank-16 and H200 rank-128 are fallback/tie cells;
  the device and physical rank remain policy keys.
- The production selector must key independently by site, device capability,
  tokens, hidden size, physical rank, top-k, adapter occupancy, and graph/eager
  mode. A gate decision must not force the down decision.
- Gate token ownership becomes more attractive if a future fused gate-B consumer
  removes pair materialization. These numbers do not claim that future contract.

The complete schedule sweeps, retained Nsight Systems reports, and raw confirmation
files are under the device directories. Canonical production integration remains a
separate commit so the benchmark evidence can be reviewed independently.
