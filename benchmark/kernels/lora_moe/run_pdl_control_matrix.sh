#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 {h200|gb300} OUTPUT_DIR [forward|reverse]" >&2
  exit 2
fi

device=$1
output_dir=$2
order=${3:-forward}
case "$device" in
  h200|gb300) ;;
  *)
    echo "unsupported device: $device" >&2
    exit 2
    ;;
esac

case "$order" in
  forward) policies=(auto off) ;;
  reverse) policies=(off auto) ;;
  *)
    echo "unsupported policy order: $order" >&2
    exit 2
    ;;
esac

mkdir -p "$output_dir"

for cell in cap1 prefill; do
  case_id="p0-qwen3.5-35b-a3b-${cell}-${device}"
  for site in gate down; do
    for execution in eager cuda_graph; do
      for pdl in "${policies[@]}"; do
        output="${output_dir}/${cell}_${site}_${execution}_${pdl}.json"
        python benchmark/kernels/lora_moe/bench_shrink_schedules.py \
          --device "$device" \
          --case-id "$case_id" \
          --site "$site" \
          --config production \
          --scope K0 \
          --execution "$execution" \
          --pdl "$pdl" \
          --cache-state hot \
          --inner-iterations 10 \
          --warmup 20 \
          --samples 100 \
          --json-output "$output"
      done
    done
  done
done
