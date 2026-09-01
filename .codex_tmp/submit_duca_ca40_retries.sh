#!/usr/bin/env bash
set -euo pipefail

STAGE_DIR=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_recovery_submit_ca40c9c
SBATCH_FILE="${STAGE_DIR}/recover_duca_r2_r3_ca40c9c.sbatch"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_ca40c9c_retry_${STAMP}"
mkdir -p "${RUN_ROOT}/jobs" "${RUN_ROOT}/logs"

submit_arm() {
    local name="$1"
    local variant="$2"
    local p0="$3"
    local p0_sha="$4"
    local arm_root="${RUN_ROOT}/arms/${name}"
    local job_id
    job_id="$(sbatch --parsable \
        --job-name="duca-${name}" \
        --output="${RUN_ROOT}/logs/${name}.%j.out" \
        --error="${RUN_ROOT}/logs/${name}.%j.err" \
        --export="ALL,RECOVERY_VARIANT=${variant},RECOVERY_P0=${p0},RECOVERY_P0_SHA=${p0_sha},RECOVERY_ARM_ROOT=${arm_root}" \
        "${SBATCH_FILE}")"
    printf '%s\t%s\t%s\t%s\t%s\n' "${job_id}" "${name}" "${variant}" "${p0_sha}" "${arm_root}" >> "${RUN_ROOT}/jobs/jobs.tsv"
    printf '%s\n' "JOB ${job_id} ${name}"
}

printf 'job_id\tarm\texperiment_variant\tp0_sha256\tarm_root\n' > "${RUN_ROOT}/jobs/jobs.tsv"

submit_arm \
    soft_detached \
    boundary_burst_r2q3_soft_detached_g0 \
    /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_9f97f2c_formal_20260722_2343/bundles/r2_r3_core/boundary_burst_r2q3_soft_detached_g0/arm/p0/work/gpu1_id0/checkpoint/epoch_19.pth \
    d331107fdd344918669c89e283981eec8f52cb2b1279e94da2b4429044f4b02b

submit_arm \
    hard_detached \
    boundary_burst_r2q3_hard_detached_g0 \
    /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_9f97f2c_formal_20260722_2343/bundles/r2_r3_core/boundary_burst_r2q3_hard_detached_g0/arm/p0/work/gpu1_id0/checkpoint/epoch_19.pth \
    b45be54ca04f3fbf081df84ec30adf88bd97c2b19efb530858f3d881befa7df4

submit_arm \
    soft_adapted \
    boundary_burst_r2q3_soft_adapted_g0 \
    /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_9f97f2c_formal_20260722_2343/bundles/r2_r3_adapted/boundary_burst_r2q3_soft_adapted_g0/arm/p0/work/gpu1_id0/checkpoint/epoch_19.pth \
    75b2e526ba492fff08ba38a5012dd2da5860698da2fdd0ffb724f1fd3cf6aa21

cat > "${RUN_ROOT}/deployment_manifest.txt" <<EOF
task=offline_tad_duca_r2_r3_recovery
code_commit=ca40c9c5a097e8ab083ba3ffd2ff7f5709841010
code_snapshot=/data/run01/sczc063/yuzibo/projects/opentad_duca_boundary_ca40c9c_20260723
source_p0_commit=9f97f2c7f081b10fbf1f63d0602a621c6b43a780
run_root=${RUN_ROOT}
jobs_tsv=${RUN_ROOT}/jobs/jobs.tsv
reason=resume sealed completed P0 after state-buffer and runtime-policy contract fixes
EOF

sha256sum "${RUN_ROOT}/deployment_manifest.txt" "${RUN_ROOT}/jobs/jobs.tsv" > "${RUN_ROOT}/deployment_hashes.sha256"
printf 'RUN_ROOT %s\n' "${RUN_ROOT}"
cat "${RUN_ROOT}/jobs/jobs.tsv"
cat "${RUN_ROOT}/deployment_hashes.sha256"
