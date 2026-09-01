#!/bin/bash
set -eo pipefail
source /etc/profile >/dev/null 2>&1
set -u

runtime=/data/run01/sczc063/yuzibo/projects/actionformer_dcsr_negative_diagnostics_20260730_v3
source_runtime=/data/run01/sczc063/yuzibo/projects/actionformer_dcsr_g0_g1_20260730_v5
bundle=/data/run01/sczc063/yuzibo/projects/preflight/dcsr_negative_diagnostics_8d6f6e5.bundle
run_root=/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_dcsr_negative_diagnostics_20260730_v3
expected_commit=8d6f6e5e7fcf8c27b6aa46870bc4c0b242f6314b
expected_tree=1ac5a68c6b8d0b1c9028ea3154765ae20e87622a

legacy_failure_root=/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_dcsr_negative_diagnostics_20260730_v1
LEGACY_FAILURE_ROOT="$legacy_failure_root" \
  /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["LEGACY_FAILURE_ROOT"])
output = root / "DEPLOYMENT_FAILURE.json"
if root.is_dir() and not output.exists():
    temporary = root / "DEPLOYMENT_FAILURE.json.tmp"
    temporary.write_text(
        json.dumps(
            {
                "schema_version": "actionformer_dcsr_diagnostic_deployment_failure_v1",
                "failure_class": "engineering_or_environment_failure",
                "failure_signature": "diagnostic_deployment_nonlogin_module_function_v1",
                "stage": "linux_exact_test_environment_bootstrap",
                "exit_code": 127,
                "slurm_job_created": False,
                "model_result_claim_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
PY

legacy_failure_root_v2=/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_dcsr_negative_diagnostics_20260730_v2
LEGACY_FAILURE_ROOT="$legacy_failure_root_v2" \
  /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["LEGACY_FAILURE_ROOT"])
output = root / "DEPLOYMENT_FAILURE.json"
if root.is_dir() and not output.exists():
    temporary = root / "DEPLOYMENT_FAILURE.json.tmp"
    temporary.write_text(
        json.dumps(
            {
                "schema_version": "actionformer_dcsr_diagnostic_deployment_failure_v1",
                "failure_class": "engineering_or_environment_failure",
                "failure_signature": "diagnostic_deployment_profile_under_nounset_v1",
                "stage": "linux_exact_test_environment_bootstrap",
                "exit_code": 1,
                "slurm_job_created": False,
                "model_result_claim_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
PY

test ! -e "$runtime"
test ! -e "$run_root"
test "$(sha256sum "$bundle" | awk '{print $1}')" = \
  8215a2235b0d8932c92e307d6f1d211fea1bdbcc7cdcd3830b7fef33b92b2255

git clone --no-local "$source_runtime" "$runtime"
git -C "$runtime" fetch "$bundle" \
  refs/heads/codex/actionformer-dcsr-g0-g1-20260730:refs/remotes/bundle/dcsr
git -C "$runtime" checkout -B \
  codex/actionformer-dcsr-g0-g1-20260730 refs/remotes/bundle/dcsr
test "$(git -C "$runtime" rev-parse HEAD)" = "$expected_commit"
test "$(git -C "$runtime" rev-parse 'HEAD^{tree}')" = "$expected_tree"
test -z "$(git -C "$runtime" status --porcelain=v1 --untracked-files=all)"

mkdir "$run_root"
module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate
export PATH=/data/run01/sczc063/yuzibo/projects/python_envs/actionformer_official_runtime_20260730_v2/bin:"$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH=/data/run01/sczc063/yuzibo/projects/python_envs/actionformer_official_runtime_20260730_v2/lib/python3.10/site-packages
hash -r

cd "$runtime"
python -m py_compile \
  tools/run_dcsr_counterfactual_eval.py \
  tools/analyze_dcsr_internal_predictions.py \
  tools/analyze_dcsr_checkpoint_dynamics.py
bash -n \
  scripts/run_dcsr_counterfactual_diagnostics_n16r4.sbatch \
  scripts/run_dcsr_negative_analysis_n16r4.sbatch
python -m pytest \
  tests/test_dcsr_negative_diagnostics.py \
  tests/test_dcsr_heads.py \
  tests/test_dcsr_internal_holdout.py \
  tests/test_dcsr_launcher_contract.py \
  tests/test_native_grid_sparse_heads.py \
  tests/test_sparsehead_official_config.py \
  -q 2>&1 | tee "$run_root/linux_exact_tests.log"
git ls-files -z | sort -z | xargs -0 sha256sum \
  > "$run_root/FULL_CONTENT_SHA256.txt"
sha256sum \
  "$run_root/FULL_CONTENT_SHA256.txt" \
  "$run_root/linux_exact_tests.log"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
