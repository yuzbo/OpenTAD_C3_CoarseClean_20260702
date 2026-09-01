#!/usr/bin/env bash
set -euo pipefail

STAGE=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_recovery_submit_ca40c9c
OLD_ROOT=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_t1_trainfree_f81aef4_formal_20260723_1048
ROOT_BASE=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe
RUN_ROOT="${ROOT_BASE}/duca_t1_919aa55_recovery_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/jobs"

variants=(t1_true_time_residual_g0 t1_reversed_time_residual_g0)
: > "${RUN_ROOT}/jobs/jobs.tsv"
for variant in "${variants[@]}"; do
  p0="${OLD_ROOT}/arms/${variant}/p0/work/gpu1_id0/checkpoint/epoch_19.pth"
  job="$(sbatch --parsable \
    --job-name="t1fix-${variant}" \
    --output="${RUN_ROOT}/logs/${variant}.%j.out" \
    --error="${RUN_ROOT}/logs/${variant}.%j.err" \
    --export="ALL,RUN_ROOT=${RUN_ROOT},VARIANT=${variant},P0_CHECKPOINT=${p0}" \
    "${STAGE}/recover_t1_919aa55.sbatch")"
  printf '%s\t%s\n' "${variant}" "${job}" | tee -a "${RUN_ROOT}/jobs/jobs.tsv"
done

cat > "${RUN_ROOT}/deployment_manifest.txt" <<EOF
task=offline_tad_t1_runtime_binding_recovery
code_commit=919aa555d1aa36191ee318477409dfbfdfb0e807
code_snapshot=/data/run01/sczc063/yuzibo/projects/opentad_duca_t1_919aa55_20260723
source_p0_root=${OLD_ROOT}
source_p0_sha256=acb6e30673e811f34ce84d710442581bec8a74ca68e9187eb71e005e01536c9b
run_root=${RUN_ROOT}
EOF
sha256sum "${RUN_ROOT}/deployment_manifest.txt" "${RUN_ROOT}/jobs/jobs.tsv" \
  > "${RUN_ROOT}/deployment_hashes.sha256"
printf 'RUN_ROOT=%s\n' "${RUN_ROOT}"
cat "${RUN_ROOT}/deployment_hashes.sha256"
