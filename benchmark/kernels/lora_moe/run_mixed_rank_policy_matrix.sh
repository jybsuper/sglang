#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 5 ]]; then
  echo "usage: $0 DEVICE OUTPUT_DIR [WARMUP=20] [SAMPLES=100] [SUSTAINED_REPLAYS=50]" >&2
  exit 2
fi

device=$1
output_dir=$2
warmup=${3:-20}
samples=${4:-100}
sustained_replays=${5:-50}
driver=benchmark/kernels/lora_moe/bench_mixed_rank_policy.py

mkdir -p "${output_dir}/raw"
mapfile -t cases < <(
  PYTHONPATH="python${PYTHONPATH:+:${PYTHONPATH}}" \
    python3 "${driver}" --device "${device}" --list-cases
)

for case_id in "${cases[@]}"; do
  for execution in eager cuda_graph; do
    for order in forward reverse; do
      output="${output_dir}/raw/${case_id}__${execution}__${order}.json"
      PYTHONPATH="python${PYTHONPATH:+:${PYTHONPATH}}" \
        python3 "${driver}" \
          --device "${device}" \
          --case-id "${case_id}" \
          --execution "${execution}" \
          --order "${order}" \
          --warmup "${warmup}" \
          --samples "${samples}" \
          --sustained-replays "${sustained_replays}" \
          --json-output "${output}"
    done
  done
done

PYTHONPATH="python${PYTHONPATH:+:${PYTHONPATH}}" \
  python3 benchmark/kernels/lora_moe/summarize_mixed_rank_policy.py \
    "${output_dir}"
