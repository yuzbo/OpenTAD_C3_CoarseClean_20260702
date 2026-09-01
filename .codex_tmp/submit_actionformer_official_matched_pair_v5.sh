#!/usr/bin/env bash
set -eo pipefail
source /etc/profile >/dev/null 2>&1
set -u

mode="${1:-}"
case "$mode" in
  test-only|formal) ;;
  *) echo "usage: $0 {test-only|formal}" >&2; exit 2 ;;
esac

run_root="/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_official_matched_pair_k384_seed1234567891_20260730_v5"
launcher="/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_diagnostic_closure_20260730_v28/scripts/run_actionformer_official_matched_pair_n16r4.sbatch"
test -d "$run_root"
test -f "$run_root/DEPLOYMENT_IDENTITY.json"
test -f "$launcher"

export_arg="ALL"
export_arg+=",CANDIDATE_ROOT=/data/run01/sczc063/yuzibo/projects/actionformer_sparsehead_official_matched_20260730_v7"
export_arg+=",AUDIT_ROOT=/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_diagnostic_closure_20260730_v28"
export_arg+=",OFFICIAL_SOURCE_ROOT=/data/run01/sczc063/yuzibo/projects/actionformer_official_61ea7eb_20260729_v1"
export_arg+=",OFFICIAL_DATA_ROOT=/data/run01/sczc063/yuzibo/projects/actionformer_official_61ea7eb_20260729_v1/data/thumos"
export_arg+=",OFFICIAL_RECORD_ROOT=/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_official_anchor_reseal_20260730_v3/official_record"
export_arg+=",RUN_ROOT=$run_root"
export_arg+=",PAIR_TAG=official_k384_seed1234567891_v5"
export_arg+=",SOURCE_DIFF_ATTESTATION=/data/run01/sczc063/yuzibo/projects/preflight/actionformer_native_grid_global_pack_remote_source_diff_20260730_v1/SOURCE_DIFF_ATTESTATION.json"
export_arg+=",EXPECTED_SOURCE_DIFF_SHA256=a07d038d87632d1f8cc984ba24af44ca7ce9a9902e30e501f5de80a32265d46b"
export_arg+=",EXPECTED_CANDIDATE_COMMIT=d86a4acda21e35a1609f19f1a46bc470ee18b7e1"
export_arg+=",EXPECTED_CANDIDATE_TREE=327c032a1ab3c14d0e34d6339df36f8a33ec6907"
export_arg+=",EXPECTED_AUDIT_COMMIT=98f5b875315b4a2b5c6829f5d74ccce68f478e47"
export_arg+=",EXPECTED_AUDIT_TREE=2e6b4bba6868c323d70c97140f7cbed044eb1a7b"
export_arg+=",EXPECTED_OFFICIAL_COMMIT=61ea7eb9308a568b0cf45e3804830836e30061de"
export_arg+=",EXPECTED_OFFICIAL_TREE=7b06c5261ba244788c942a0d73e304581bc35154"
export_arg+=",EXPECTED_ANNOTATION_SHA256=3b025685a07fb98fc58d2399fb5fa9493c2168632d8ae1a8c3f4689897d2fbb2"
export_arg+=",EXPECTED_DATA_MANIFEST_SHA256=7ae43a960702cbc6cb2d56f69913257b9553fdf4a43b1fd3a91699da1411e9fe"
export_arg+=",EXPECTED_FEATURE_MANIFEST_SHA256=cda269dace393b9af1f6fcb87a9a531beed69e3c71279ba3ca2cee76e198d59a"
export_arg+=",EXPECTED_EVALUATOR_MANIFEST_SHA256=0e406ea34e96a1b9cf603d407417c76075de62ff4367a7ce156b1ad6b48bc7a9"
export_arg+=",ACTIONFORMER_PYTHON_ENV=/data/run01/sczc063/yuzibo/projects/python_envs/actionformer_official_runtime_20260730_v2"
export_arg+=",ACTIONFORMER_ENVIRONMENT_RECEIPT=/data/run01/sczc063/yuzibo/projects/preflight/actionformer_official_runtime_environment_20260730_v2/ENVIRONMENT_RECEIPT.json"
export_arg+=",EXPECTED_ENVIRONMENT_RECEIPT_SHA256=13d57c1161905f059204f7101f26029503a03da7f5eb44b81c418a0b97999f24"
export_arg+=",ACTIONFORMER_NMS_EXTENSION=/data/run01/sczc063/yuzibo/projects/python_envs/actionformer_official_runtime_20260730_v2/lib/python3.10/site-packages/nms_1d_cpu.cpython-310-x86_64-linux-gnu.so"
export_arg+=",EXPECTED_ACTIONFORMER_NMS_EXTENSION_SHA256=b67e0e41f9f55cd69e8b90cfc75a1947214365857d851a510047838ad49ed98d"
export_arg+=",SCREENING_SEED=1234567891"
export_arg+=",EXPECTED_TERMINAL_EPOCH=35"

common_args=(
  --job-name=af-k384-pair-r5
  "--output=$run_root/slurm-%j.out"
  "--error=$run_root/slurm-%j.err"
  "--export=$export_arg"
)

if [[ "$mode" == "test-only" ]]; then
  test ! -e "$run_root/sbatch_test_only.txt"
  sbatch --test-only "${common_args[@]}" "$launcher" \
    2>&1 | tee "$run_root/sbatch_test_only.txt"
  exit "${PIPESTATUS[0]}"
fi

test -s "$run_root/sbatch_test_only.txt"
test ! -e "$run_root/SUBMITTED_JOB_ID.txt"
test ! -e "$run_root/SUBMISSION_RECEIPT.json"
formal_output=$(sbatch --parsable "${common_args[@]}" "$launcher")
job_id="${formal_output%%;*}"
case "$job_id" in
  *[!0-9]*|"") echo "unexpected sbatch output: $formal_output" >&2; exit 2 ;;
esac
printf "%s\n" "$job_id" > "$run_root/SUBMITTED_JOB_ID.txt"
printf "%s\n" "$job_id"
