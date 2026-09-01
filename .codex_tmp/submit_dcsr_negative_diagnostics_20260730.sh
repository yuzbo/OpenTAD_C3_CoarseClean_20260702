#!/bin/bash
set -eo pipefail
source /etc/profile >/dev/null 2>&1
set -u
module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

diagnostic_runtime=/data/run01/sczc063/yuzibo/projects/actionformer_dcsr_negative_diagnostics_20260730_v3
source_run_root=/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_dcsr_g0_g1_internal_20260730_v5
diagnostic_run_root=/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_dcsr_negative_diagnostics_20260730_v3
official_data_root=/data/run01/sczc063/yuzibo/projects/actionformer_official_61ea7eb_20260729_v1/data/thumos
manifest="$source_run_root/DCSR_INTERNAL_HOLDOUT.json"
aggregate="$source_run_root/DCSR_G1_AGGREGATE.json"
environment_path=/data/run01/sczc063/yuzibo/projects/python_envs/actionformer_official_runtime_20260730_v2
environment_receipt=/data/run01/sczc063/yuzibo/projects/preflight/actionformer_official_runtime_environment_20260730_v2/ENVIRONMENT_RECEIPT.json
nms_extension="$environment_path/lib/python3.10/site-packages/nms_1d_cpu.cpython-310-x86_64-linux-gnu.so"

diagnostic_commit=8d6f6e5e7fcf8c27b6aa46870bc4c0b242f6314b
diagnostic_tree=1ac5a68c6b8d0b1c9028ea3154765ae20e87622a
source_training_commit=bf0df83d7400c89fc61f38d169d68085420a2263
source_training_tree=2f9346fcfd2bfb7fc5a76a86ef65545030a67469
annotation_sha256=3b025685a07fb98fc58d2399fb5fa9493c2168632d8ae1a8c3f4689897d2fbb2
manifest_sha256=ba683bc5ddbb1fe219fab0545e9d808808d9b25fc9b32e7c5c0b6339b68b9bbb
aggregate_sha256=b98d59468ef39aa6fe6de387adfd6f872c848ab8f63b26c3bf1bf6161f5f7939
config_sha256=59aa4ad9eba74471017aef3ffe943e4c20c35a10947d555feb83c2e21e3e97f0
environment_receipt_sha256=13d57c1161905f059204f7101f26029503a03da7f5eb44b81c418a0b97999f24
nms_extension_sha256=b67e0e41f9f55cd69e8b90cfc75a1947214365857d851a510047838ad49ed98d
full_content_sha256=4f9b9fef1eff3f829628f717c6ab57db554163a9f5c866086076a178d73dbbef
linux_tests_sha256=f423c4ce51ed8381c29acc9e0c5bd133721849b56a0620978cfe3e8854a19f37

counterfactual_launcher="$diagnostic_runtime/scripts/run_dcsr_counterfactual_diagnostics_n16r4.sbatch"
analysis_launcher="$diagnostic_runtime/scripts/run_dcsr_negative_analysis_n16r4.sbatch"
deployment_identity="$diagnostic_run_root/DEPLOYMENT_IDENTITY.json"
submission_receipt="$diagnostic_run_root/SUBMISSION_RECEIPT.json"
test_only_log="$diagnostic_run_root/SBATCH_TEST_ONLY.log"
test ! -e "$deployment_identity"
test ! -e "$submission_receipt"
test ! -e "$test_only_log"
test "$(git -C "$diagnostic_runtime" rev-parse HEAD)" = "$diagnostic_commit"
test "$(git -C "$diagnostic_runtime" rev-parse 'HEAD^{tree}')" = "$diagnostic_tree"
test -z "$(git -C "$diagnostic_runtime" status --porcelain=v1 --untracked-files=all)"
test "$(sha256sum "$manifest" | awk '{print $1}')" = "$manifest_sha256"
test "$(sha256sum "$aggregate" | awk '{print $1}')" = "$aggregate_sha256"
test "$(sha256sum "$official_data_root/annotations/thumos14.json" | awk '{print $1}')" = \
  "$annotation_sha256"
test "$(sha256sum "$diagnostic_runtime/configs/thumos_i3d_dcsr_dev_g1_uniform.yaml" | awk '{print $1}')" = \
  "$config_sha256"
test "$(sha256sum "$environment_receipt" | awk '{print $1}')" = \
  "$environment_receipt_sha256"
test "$(sha256sum "$nms_extension" | awk '{print $1}')" = \
  "$nms_extension_sha256"
test "$(sha256sum "$diagnostic_run_root/FULL_CONTENT_SHA256.txt" | awk '{print $1}')" = \
  "$full_content_sha256"
test "$(sha256sum "$diagnostic_run_root/linux_exact_tests.log" | awk '{print $1}')" = \
  "$linux_tests_sha256"

DIAGNOSTIC_RUNTIME="$diagnostic_runtime" \
DIAGNOSTIC_RUN_ROOT="$diagnostic_run_root" \
SOURCE_RUN_ROOT="$source_run_root" \
OFFICIAL_DATA_ROOT="$official_data_root" \
DIAGNOSTIC_COMMIT="$diagnostic_commit" \
DIAGNOSTIC_TREE="$diagnostic_tree" \
SOURCE_TRAINING_COMMIT="$source_training_commit" \
SOURCE_TRAINING_TREE="$source_training_tree" \
FULL_CONTENT_SHA256="$full_content_sha256" \
LINUX_TESTS_SHA256="$linux_tests_sha256" \
/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["DIAGNOSTIC_RUN_ROOT"])
payload = {
    "schema_version": "actionformer_dcsr_negative_diagnostic_deployment_v1",
    "validation_pass": True,
    "diagnostic_only": True,
    "training_performed": False,
    "paper_performance_row_allowed": False,
    "diagnostic_runtime": os.environ["DIAGNOSTIC_RUNTIME"],
    "source_run_root": os.environ["SOURCE_RUN_ROOT"],
    "official_data_root": os.environ["OFFICIAL_DATA_ROOT"],
    "diagnostic_commit": os.environ["DIAGNOSTIC_COMMIT"],
    "diagnostic_tree": os.environ["DIAGNOSTIC_TREE"],
    "source_training_commit": os.environ["SOURCE_TRAINING_COMMIT"],
    "source_training_tree": os.environ["SOURCE_TRAINING_TREE"],
    "linux_exact_focused_suite": {
        "passed": 38,
        "log_sha256": os.environ["LINUX_TESTS_SHA256"],
    },
    "full_content_manifest_sha256": os.environ["FULL_CONTENT_SHA256"],
    "preserved_pre_slurm_failure_signatures": [
        "diagnostic_deployment_inline_ssh_quoting_v1",
        "diagnostic_deployment_nonlogin_module_function_v1",
        "diagnostic_deployment_profile_under_nounset_v1",
    ],
}
temporary = root / "DEPLOYMENT_IDENTITY.json.tmp"
temporary.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, root / "DEPLOYMENT_IDENTITY.json")
PY

export_spec="ALL,DIAGNOSTIC_RUNTIME=$diagnostic_runtime,SOURCE_RUN_ROOT=$source_run_root,DIAGNOSTIC_RUN_ROOT=$diagnostic_run_root,OFFICIAL_DATA_ROOT=$official_data_root,EXPECTED_DIAGNOSTIC_COMMIT=$diagnostic_commit,EXPECTED_DIAGNOSTIC_TREE=$diagnostic_tree,SOURCE_TRAINING_COMMIT=$source_training_commit,SOURCE_TRAINING_TREE=$source_training_tree,EXPECTED_ANNOTATION_SHA256=$annotation_sha256,DCSR_INTERNAL_HOLDOUT_MANIFEST=$manifest,DCSR_INTERNAL_HOLDOUT_MANIFEST_SHA256=$manifest_sha256,EXPECTED_DCSR_CONFIG_SHA256=$config_sha256,DCSR_G1_AGGREGATE=$aggregate,EXPECTED_DCSR_G1_AGGREGATE_SHA256=$aggregate_sha256,ACTIONFORMER_PYTHON_ENV=$environment_path,ACTIONFORMER_ENVIRONMENT_RECEIPT=$environment_receipt,EXPECTED_ENVIRONMENT_RECEIPT_SHA256=$environment_receipt_sha256,ACTIONFORMER_NMS_EXTENSION=$nms_extension,EXPECTED_ACTIONFORMER_NMS_EXTENSION_SHA256=$nms_extension_sha256"

cd "$diagnostic_run_root"
sbatch --test-only --export="$export_spec" "$counterfactual_launcher" \
  2>&1 | tee "$test_only_log"
array_job_id=$(sbatch --parsable --export="$export_spec" "$counterfactual_launcher")
test -n "$array_job_id"
analysis_job_id=$(sbatch --parsable \
  --dependency="afterok:$array_job_id" \
  --export="$export_spec" \
  "$analysis_launcher")
test -n "$analysis_job_id"

DIAGNOSTIC_RUN_ROOT="$diagnostic_run_root" \
ARRAY_JOB_ID="$array_job_id" \
ANALYSIS_JOB_ID="$analysis_job_id" \
DEPLOYMENT_IDENTITY="$deployment_identity" \
TEST_ONLY_LOG="$test_only_log" \
COUNTERFACTUAL_LAUNCHER="$counterfactual_launcher" \
ANALYSIS_LAUNCHER="$analysis_launcher" \
/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["DIAGNOSTIC_RUN_ROOT"])

def record(path):
    path = Path(path)
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }

payload = {
    "schema_version": "actionformer_dcsr_negative_diagnostic_submission_v1",
    "validation_pass": True,
    "counterfactual_array_job_id": os.environ["ARRAY_JOB_ID"],
    "analysis_job_id": os.environ["ANALYSIS_JOB_ID"],
    "analysis_dependency": "afterok:" + os.environ["ARRAY_JOB_ID"],
    "duplicate_submission_allowed": False,
    "diagnostic_only": True,
    "training_performed": False,
    "artifacts": {
        "deployment_identity": record(os.environ["DEPLOYMENT_IDENTITY"]),
        "sbatch_test_only": record(os.environ["TEST_ONLY_LOG"]),
        "counterfactual_launcher": record(
            os.environ["COUNTERFACTUAL_LAUNCHER"]
        ),
        "analysis_launcher": record(os.environ["ANALYSIS_LAUNCHER"]),
    },
}
temporary = root / "SUBMISSION_RECEIPT.json.tmp"
temporary.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, root / "SUBMISSION_RECEIPT.json")
PY

squeue -j "$array_job_id,$analysis_job_id" \
  -o "%.18i %.16j %.8T %.10M %.20R"
