#!/usr/bin/env bash
# Run each cross-model C2 schedule in a fresh CUDA process.  Besides isolating
# graph pools, this preserves a status/log artifact when an experimental kernel
# poisons its CUDA context (for example an architecture-specific illegal access).

set -u

if [[ $# -ne 3 ]]; then
  echo "usage: $0 OUT_DIR SHARD_INDEX NUM_SHARDS" >&2
  exit 2
fi

out_dir=$1
shard_index=$2
num_shards=$3
mkdir -p "$out_dir"

models=(
  qwen3.5-397b-a17b
  kimi-k2.5
  glm-5.2
  nemotron-3-super
  nemotron-3-nano
  odd-provider-padding
)
tokens=(1 32 256)
ranks=(16 32 64)
schedules=(pair aligned)
block_ns=(16 32 64)
case_index=0
selected=0
failed=0

run_one() {
  local model=$1
  local token_count=$2
  local rank=$3
  local schedule=$4
  local block_n=$5
  local stem="${model}_T${token_count}_R${rank}_${schedule}_BN${block_n}"
  local json_path="$out_dir/${stem}.json"
  local log_path="$out_dir/${stem}.log"
  local status_path="$out_dir/${stem}.status"

  python3 benchmark/kernels/lora_moe/bench_c2_cross_model_contracts.py \
    --models "$model" \
    --tokens "$token_count" \
    --ranks "$rank" \
    --schedules "$schedule" \
    --block-n "$block_n" \
    --warmup 5 \
    --samples 20 \
    --json-output "$json_path" >"$log_path" 2>&1
  local status=$?
  printf '%s\n' "$status" >"$status_path"
  if [[ $status -ne 0 ]]; then
    failed=$((failed + 1))
  fi
  selected=$((selected + 1))
}

for model in "${models[@]}"; do
  model_tokens=("${tokens[@]}")
  model_ranks=("${ranks[@]}")
  if [[ $model == odd-provider-padding ]]; then
    model_tokens=(32)
    model_ranks=(32)
  fi
  for token_count in "${model_tokens[@]}"; do
    for rank in "${model_ranks[@]}"; do
      if (( case_index % num_shards == shard_index )); then
        for schedule in "${schedules[@]}"; do
          for block_n in "${block_ns[@]}"; do
            run_one "$model" "$token_count" "$rank" "$schedule" "$block_n"
          done
        done
      fi
      case_index=$((case_index + 1))
    done
  done
done

printf 'selected_configs=%s failed=%s total_shape_cells=%s\n' \
  "$selected" "$failed" "$case_index" >"$out_dir/shard_summary.txt"
printf 'completed shard %s/%s: selected_configs=%s failed=%s shape_cells=%s\n' \
  "$shard_index" "$num_shards" "$selected" "$failed" "$case_index"
