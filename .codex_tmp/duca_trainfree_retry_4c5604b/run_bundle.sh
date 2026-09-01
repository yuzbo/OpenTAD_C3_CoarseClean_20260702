#!/usr/bin/env bash
set -euo pipefail

run_root="${DUCA_RUN_ROOT:?DUCA_RUN_ROOT is required}"
arm_runner="${DUCA_ARM_RUNNER:?DUCA_ARM_RUNNER is required}"
IFS=',' read -r -a variants <<< "${DUCA_BUNDLE_VARIANTS:?DUCA_BUNDLE_VARIANTS is required}"
mkdir -p "${run_root}/logs" "${run_root}/arms"

pids=()
for variant in "${variants[@]}"; do
  log_path="${run_root}/logs/${variant}.srun.log"
  srun --exact --exclusive --nodes=1 --ntasks=1 \
    --gpus=1 --gpus-per-task=1 --cpus-per-task=8 \
    bash "${arm_runner}" "${variant}" >"${log_path}" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
exit "${status}"
