#!/usr/bin/env bash
set -euo pipefail

snapshot="/data/run01/sczc063/yuzibo/projects/opentad_duca_t1_trainfree_4c5604b_20260723"
run_root="/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_trainfree_4c5604b_retry_20260723_1107"
expected_commit="4c5604b4a0abde9e59f625d519934e855bfe1519"
pretrain_path="/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
pretrain_sha256="4b96b7f403f8ae0396437855b785af6a0064f11a9d76e2268e5a76a04e0de251"
arm_runner="${run_root}/sbatch/run_one_arm.sh"
bundle_runner="${run_root}/sbatch/run_bundle.sh"
jobs_tsv="${run_root}/jobs.tsv"

test "$(git -C "${snapshot}" rev-parse HEAD)" = "${expected_commit}"
test -z "$(git -C "${snapshot}" status --porcelain --untracked-files=normal)"
test "$(sha256sum /data/run01/sczc063/yuzibo/.cache/torch/hub/checkpoints/SLOWFAST_8x8_R50.pyth | awk '{print $1}')" = "454f39e1c1f985df2bee2aa27887ed53ff56e74ed8b8cca11203a1a1264d7cc2"
bash -n "${arm_runner}"
bash -n "${bundle_runner}"
printf 'bundle\tjob_id\tvariants\n' >"${jobs_tsv}"

submit_bundle() {
  local name="$1"
  local variants="$2"
  local tasks="$3"
  local job_id
  job_id="$(
    DUCA_SNAPSHOT="${snapshot}" \
    DUCA_RUN_ROOT="${run_root}" \
    DUCA_EXPECTED_COMMIT="${expected_commit}" \
    DUCA_ADATAD_PRETRAIN_PATH="${pretrain_path}" \
    DUCA_ADATAD_PRETRAIN_SHA256="${pretrain_sha256}" \
    DUCA_ARM_RUNNER="${arm_runner}" \
    DUCA_BUNDLE_VARIANTS="${variants}" \
    sbatch --parsable --partition=gpu --nodes=1 --ntasks="${tasks}" \
      --cpus-per-task=8 --gres="gpu:${tasks}" --time=48:00:00 \
      --job-name="${name}" \
      --output="${run_root}/slurm/%x.%j.out" \
      --error="${run_root}/slurm/%x.%j.err" \
      --export=ALL "${bundle_runner}"
  )"
  printf '%s\t%s\t%s\n' "${name}" "${job_id}" "${variants}" >>"${jobs_tsv}"
  printf '%s=%s\n' "${name}" "${job_id}"
}

submit_bundle \
  dtf_mb4c \
  trainfree_mobilenet_feature_change,trainfree_mobilenet_semantic,trainfree_mobilenet_fusion_r2q3 \
  3
submit_bundle dtf_fast4c trainfree_slowfast_fast_fusion_r2q3 1

sha256sum "${arm_runner}" "${bundle_runner}" "${jobs_tsv}" >"${run_root}/deployment_hashes.sha256"
