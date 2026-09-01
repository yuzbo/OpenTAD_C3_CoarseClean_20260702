#!/bin/bash
set -o pipefail
source /etc/profile >/dev/null 2>&1
set -eu
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

project_root=/data/run01/sczc063/yuzibo/projects
stage_root="$project_root/preflight/actionformer_odfcr_deployment_20260731_v3"
runtime="$project_root/actionformer_odfcr_internal_20260731_v3"
run_root="$project_root/phystime_tad/runs/actionformer_odfcr_internal_20260731_v3"
main_job=1209259
expected_commit=01cdb78d2b7668098b6b13a1e49433d48fbc1a8d
expected_tree=e70d2956a197b1204e721239178e76152efe282b
python_env="$project_root/python_envs/actionformer_official_runtime_20260730_v2"
environment_receipt="$project_root/preflight/actionformer_official_runtime_environment_20260730_v2/ENVIRONMENT_RECEIPT.json"
expected_environment_receipt_sha256=13d57c1161905f059204f7101f26029503a03da7f5eb44b81c418a0b97999f24

test "$(git -C "$runtime" rev-parse HEAD)" = "$expected_commit"
test "$(git -C "$runtime" rev-parse 'HEAD^{tree}')" = "$expected_tree"
test -z "$(git -C "$runtime" status --porcelain=v1 --untracked-files=all)"
test -d "$run_root"
test ! -e "$run_root/jobs.tsv"
test ! -e "$run_root/DEPLOYMENT_RECEIPT.json"
test -x "$python_env/bin/python"
test "$(sha256sum "$environment_receipt" | awk '{print $1}')" = \
  "$expected_environment_receipt_sha256"
scontrol --clusters=n16r4 show job -o "${main_job}_0" \
  | grep -F "Command=$runtime/scripts/run_odfcr_internal_factorial_n16r4.sbatch"
if squeue --clusters=n16r4 -h -u sczc063 -n af-odfcr-g2 | grep -q .; then
  echo "an ODF-CR G2 job already exists; refusing duplicate submission" >&2
  exit 2
fi

export PATH="$python_env/bin:$PATH"
export PYTHONNOUSERSITE=1
hash -r
test "$(readlink -f "$(command -v python)")" = \
  "$(readlink -f "$python_env/bin/python")"

g2_export="ALL,CANDIDATE_ROOT=$runtime,RUN_ROOT=$run_root,EXPECTED_CANDIDATE_COMMIT=$expected_commit,EXPECTED_CANDIDATE_TREE=$expected_tree,ACTIONFORMER_PYTHON_ENV=$python_env,ACTIONFORMER_ENVIRONMENT_RECEIPT=$environment_receipt,EXPECTED_ENVIRONMENT_RECEIPT_SHA256=$expected_environment_receipt_sha256"
sbatch --clusters=n16r4 --test-only --gpus=1 \
  --dependency="afterok:$main_job" \
  --export="$g2_export" \
  "$runtime/scripts/aggregate_odfcr_internal_factorial_n16r4.sbatch" \
  | tee "$stage_root/sbatch_g2_recovery_test_only.log"

g2_submit=$(sbatch --clusters=n16r4 --parsable --gpus=1 \
  --dependency="afterok:$main_job" \
  --output="$run_root/slurm-g2-%j.out" \
  --error="$run_root/slurm-g2-%j.err" \
  --export="$g2_export" \
  "$runtime/scripts/aggregate_odfcr_internal_factorial_n16r4.sbatch")
g2_job=${g2_submit%%;*}
case "$g2_job" in
  ''|*[!0-9]*) echo "invalid G2 aggregate Job ID: $g2_submit" >&2; exit 2 ;;
esac

printf 'role\tjob_id\tdependency\nfactorial_array\t%s\t\naggregate_g2\t%s\tafterok:%s\n' \
  "$main_job" "$g2_job" "$main_job" > "$run_root/jobs.tsv"

STAGE_ROOT="$stage_root" \
RUNTIME="$runtime" \
RUN_ROOT="$run_root" \
EXPECTED_COMMIT="$expected_commit" \
EXPECTED_TREE="$expected_tree" \
MAIN_JOB="$main_job" \
G2_JOB="$g2_job" \
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

def receipt(path):
    path = Path(path).resolve()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }

stage = Path(os.environ["STAGE_ROOT"])
run_root = Path(os.environ["RUN_ROOT"])
manifest = run_root / "ODFCR_INTERNAL_HOLDOUT_V2.json"
payload = {
    "schema_version": "actionformer_odfcr_deployment_receipt_v1",
    "validation_pass": True,
    "runtime": str(Path(os.environ["RUNTIME"]).resolve()),
    "run_root": str(run_root.resolve()),
    "git_commit": os.environ["EXPECTED_COMMIT"],
    "git_tree": os.environ["EXPECTED_TREE"],
    "runtime_clean": True,
    "manifest": receipt(manifest),
    "linux_focused_suite": receipt(stage / "linux_focused_suite.log"),
    "holdout_v2_build": receipt(stage / "holdout_v2_build.log"),
    "factorial_test_only": receipt(stage / "sbatch_factorial_test_only.log"),
    "g2_recovery_test_only": receipt(
        stage / "sbatch_g2_recovery_test_only.log"
    ),
    "jobs": {
        "factorial_array": int(os.environ["MAIN_JOB"]),
        "aggregate_g2": int(os.environ["G2_JOB"]),
    },
    "pre_submission_engineering_events": [
        {
            "signature": "remote_profile_nonzero_under_errexit_v1",
            "stage_root": (
                "/data/run01/sczc063/yuzibo/projects/preflight/"
                "actionformer_odfcr_deployment_20260731_v1"
            ),
            "slurm_job_created": False,
        },
        {
            "signature": "yaml_1_1_off_coercion_residual_support_v1",
            "stage_root": (
                "/data/run01/sczc063/yuzibo/projects/preflight/"
                "actionformer_odfcr_deployment_20260731_v2"
            ),
            "slurm_job_created": False,
        },
        {
            "signature": "aggregate_submit_gpu_count_missing_v1",
            "stage_root": str(stage.resolve()),
            "factorial_array_already_submitted": int(os.environ["MAIN_JOB"]),
            "duplicate_factorial_submission": False,
            "recovery": "submit only G2 with explicit --gpus=1",
        },
    ],
    "source_split": "validation",
    "test_gt_used": False,
    "test_predictions_used": False,
    "paper_performance_row_allowed": False,
    "official_test_authorized": False,
    "efficiency_claim_allowed": False,
}
output = run_root / "DEPLOYMENT_RECEIPT.json"
temporary = output.with_suffix(output.suffix + ".tmp")
temporary.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
temporary.replace(output)
print(json.dumps({
    "factorial_array_job": payload["jobs"]["factorial_array"],
    "aggregate_g2_job": payload["jobs"]["aggregate_g2"],
    "deployment_receipt": receipt(output),
}, sort_keys=True))
PY

squeue --clusters=n16r4 -j "$main_job,$g2_job" \
  -o '%.18i %.12j %.2t %.10M %.30R'
