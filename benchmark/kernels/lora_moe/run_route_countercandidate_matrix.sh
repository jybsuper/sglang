#!/usr/bin/env bash
# Run one deterministic shard.  Select a GPU with CUDA_VISIBLE_DEVICES and run
# shards concurrently when the node has spare devices.
set -euo pipefail

if [[ $# -lt 4 || $# -gt 7 ]]; then
  echo "usage: $0 OUTPUT_JSON DEVICE_LABEL SHARD_INDEX NUM_SHARDS [REPS] [WARMUP] [MACRO_LAYERS]" >&2
  exit 2
fi

output=$1
device_label=$2
shard_index=$3
num_shards=$4
reps=${5:-15}
warmup=${6:-3}
macro_layers=${7:-8}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export PYTHONPATH="${repo_root}/python${PYTHONPATH:+:${PYTHONPATH}}"

python3 "${repo_root}/benchmark/kernels/lora_moe/bench_route_countercandidates.py" \
  --output "${output}" \
  --device-label "${device_label}" \
  --shard-index "${shard_index}" \
  --num-shards "${num_shards}" \
  --reps "${reps}" \
  --warmup "${warmup}" \
  --macro-layers "${macro_layers}"
