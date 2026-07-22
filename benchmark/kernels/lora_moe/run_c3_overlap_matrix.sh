#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 {h200|gb300} ARTIFACT_DIR" >&2
  exit 2
fi

device="$1"
artifact_dir="$2"
if [[ "$device" != "h200" && "$device" != "gb300" ]]; then
  echo "device must be h200 or gb300" >&2
  exit 2
fi

mkdir -p "$artifact_dir"

run_case() {
  local label="$1"
  local case_id="$2"
  shift 2
  python3 benchmark/kernels/lora_moe/bench_c3_overlap.py \
    --device "$device" \
    --case-id "$case_id" \
    --execution both \
    --c1-overlap-policy force \
    --warmup 10 \
    --samples 30 \
    --counterbalance-repeats 5 \
    --json-output "$artifact_dir/${label}.json" \
    "$@"
}

# Token anchors plus rank anchors.  T=32 is repeated at all three ranks to
# separate the rank effect from the phase/token effect.
run_case t1-r32 "p0-qwen3.5-35b-a3b-tiny-${device}"
run_case t32-r32 "p0-qwen3.5-35b-a3b-cap1-${device}" --rank 32
run_case t32-r64 "p0-qwen3.5-35b-a3b-cap1-${device}"
run_case t32-r128 "p0-qwen3.5-35b-a3b-cap1-${device}" --rank 128
run_case t64-r128 "p0-qwen3.5-35b-a3b-decode-r128-t64-${device}"
run_case t128-r128 "p0-qwen3.5-35b-a3b-decode-r128-t128-${device}"
run_case t256-r32 "p0-qwen3.5-35b-a3b-decode-large-${device}"
run_case t2048-r64 "p0-qwen3.5-35b-a3b-prefill-${device}"

# Mixed active/base rows are a semantic guardrail and a performance anchor.
run_case t32-r64-mixed "p0-qwen3.5-35b-a3b-mixed-${device}"
