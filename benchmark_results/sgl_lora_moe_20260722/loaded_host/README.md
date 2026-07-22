# Loaded-host eager M0 diagnostic

This closes reviewer item V4's quiet-host blind spot for the materialized BF16
control. It compares the same eager M0 process with zero or four independent
busy host worker processes. Each cell has 20 warmups, 100 samples, forward and
reverse pipeline order, strict active-LoRA delta checks, and p20/p50/p80 in the
raw JSON. Values below are the median p50 across the two process orders.

This is a launch-sensitivity diagnostic, not a scheduler model. The production
`one_batch_server` E0 result remains authoritative for actual server contention.

## H200

| case | pipeline | quiet us | loaded us | loaded delta |
|---|---|---:|---:|---:|
| cap1 decode | N0 | 406.224 | 406.744 | +0.13% |
| cap1 decode | C0 | 594.568 | 604.344 | +1.64% |
| cap1 decode | C1 | 693.640 | 703.568 | +1.43% |
| mixed decode | N0 | 405.544 | 405.784 | +0.06% |
| mixed decode | C0 | 618.936 | 615.384 | -0.57% |
| mixed decode | C1 | 716.136 | 714.424 | -0.24% |
| T2048 prefill | N0 | 526.952 | 527.200 | +0.05% |
| T2048 prefill | C0 | 912.472 | 912.600 | +0.01% |
| T2048 prefill | C1 | 911.936 | 911.560 | -0.04% |

## GB300

| case | pipeline | quiet us | loaded us | loaded delta |
|---|---|---:|---:|---:|
| cap1 decode | N0 | 315.712 | 340.888 | +7.97% |
| cap1 decode | C0 | 617.544 | 630.048 | +2.02% |
| cap1 decode | C1 | 700.072 | 724.608 | +3.50% |
| mixed decode | N0 | 323.904 | 347.648 | +7.33% |
| mixed decode | C0 | 660.208 | 665.656 | +0.83% |
| mixed decode | C1 | 762.784 | 767.512 | +0.62% |
| T2048 prefill | N0 | 379.624 | 377.728 | -0.50% |
| T2048 prefill | C0 | 805.496 | 791.744 | -1.71% |
| T2048 prefill | C1 | 812.056 | 809.472 | -0.32% |

## Interpretation

- Synthetic contention does not create one portable correction factor. H200
  prefill is neutral, while GB300 cap1 decode moves several percent and even
  the matched N0 base shifts more than the LoRA paths.
- The launch-heavier C1 control is more sensitive than C0 for GB300 cap1, but
  not for the mixed or prefill cells. This reinforces execution mode, device,
  and work size as planner keys; it does not justify a universal token cutoff.
- These runs use the retained materialized C0/C1 control at source snapshot
  `4ffaee0b41` plus the feedback correctness fixes. Production C2/C3 is judged
  by its separate full-server E0 and Nsight evidence.

The raw files record host-load scope, exact source/environment, resolved route,
dispersion, output correctness, and all execution parameters.
