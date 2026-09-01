#!/usr/bin/env bash
set -euo pipefail

snapshot="/data/run01/sczc063/yuzibo/projects/opentad_duca_t1_trainfree_4c5604b_20260723"
run_root="/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_trainfree_mobilenet_4c5604b_retry2_20260723_1115"
expected_commit="4c5604b4a0abde9e59f625d519934e855bfe1519"
pretrain_path="/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
pretrain_sha256="4b96b7f403f8ae0396437855b785af6a0064f11a9d76e2268e5a76a04e0de251"
mobilenet_path="/data/run01/sczc063/yuzibo/.cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth"
mobilenet_sha256="047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f"
arm_runner="${run_root}/sbatch/run_one_arm.sh"
bundle_runner="${run_root}/sbatch/run_bundle.sh"
jobs_tsv="${run_root}/jobs.tsv"

test "$(git -C "${snapshot}" rev-parse HEAD)" = "${expected_commit}"
test -z "$(git -C "${snapshot}" status --porcelain --untracked-files=normal)"
test "$(sha256sum "${mobilenet_path}" | awk '{print $1}')" = "${mobilenet_sha256}"
bash -n "${arm_runner}"
bash -n "${bundle_runner}"

variants="trainfree_mobilenet_feature_change,trainfree_mobilenet_semantic,trainfree_mobilenet_fusion_r2q3"
job_id="$(
  DUCA_SNAPSHOT="${snapshot}" \
  DUCA_RUN_ROOT="${run_root}" \
  DUCA_EXPECTED_COMMIT="${expected_commit}" \
  DUCA_ADATAD_PRETRAIN_PATH="${pretrain_path}" \
  DUCA_ADATAD_PRETRAIN_SHA256="${pretrain_sha256}" \
  DUCA_ARM_RUNNER="${arm_runner}" \
  DUCA_BUNDLE_VARIANTS="${variants}" \
  sbatch --parsable --partition=gpu --nodes=1 --ntasks=3 \
    --cpus-per-task=8 --gres=gpu:3 --time=48:00:00 \
    --job-name=dtf_mb4c_r2 \
    --output="${run_root}/slurm/%x.%j.out" \
    --error="${run_root}/slurm/%x.%j.err" \
    --export=ALL "${bundle_runner}"
)"
printf 'bundle\tjob_id\tvariants\n' >"${jobs_tsv}"
printf 'dtf_mb4c_r2\t%s\t%s\n' "${job_id}" "${variants}" >>"${jobs_tsv}"
sha256sum "${arm_runner}" "${bundle_runner}" "${jobs_tsv}" "${mobilenet_path}" \
  >"${run_root}/deployment_hashes.sha256"
printf 'dtf_mb4c_r2=%s\n' "${job_id}"
