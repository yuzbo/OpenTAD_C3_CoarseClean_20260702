#!/bin/bash
set -o pipefail

source /etc/profile >/dev/null 2>&1
set -eu
module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

project_root=/data/run01/sczc063/yuzibo/projects
stage_root="$project_root/preflight/actionformer_odfcr_deployment_20260731_v3"
bundle="$stage_root/actionformer_odfcr_01cdb78.bundle"
runtime="$project_root/actionformer_odfcr_internal_20260731_v3"
run_root="$project_root/phystime_tad/runs/actionformer_odfcr_internal_20260731_v3"
branch=codex/actionformer-densefloor-factorial-20260731
expected_commit=01cdb78d2b7668098b6b13a1e49433d48fbc1a8d
expected_tree=e70d2956a197b1204e721239178e76152efe282b
expected_bundle_sha256=0367478cc3436f3c51bd2df7ca41ca32bae8479b1c31c56f92dfd2b1a2064167

official_data="$project_root/actionformer_official_61ea7eb_20260729_v1/data/thumos"
annotation="$official_data/annotations/thumos14.json"
features="$official_data/i3d_features"
expected_annotation_sha256=3b025685a07fb98fc58d2399fb5fa9493c2168632d8ae1a8c3f4689897d2fbb2
previous_manifest="$project_root/phystime_tad/runs/actionformer_dcsr_g0_g1_internal_20260730_v5/DCSR_INTERNAL_HOLDOUT.json"
expected_previous_manifest_sha256=ba683bc5ddbb1fe219fab0545e9d808808d9b25fc9b32e7c5c0b6339b68b9bbb

python_env="$project_root/python_envs/actionformer_official_runtime_20260730_v2"
environment_receipt="$project_root/preflight/actionformer_official_runtime_environment_20260730_v2/ENVIRONMENT_RECEIPT.json"
expected_environment_receipt_sha256=13d57c1161905f059204f7101f26029503a03da7f5eb44b81c418a0b97999f24
nms_extension="$python_env/lib/python3.10/site-packages/nms_1d_cpu.cpython-310-x86_64-linux-gnu.so"
expected_nms_extension_sha256=b67e0e41f9f55cd69e8b90cfc75a1947214365857d851a510047838ad49ed98d

test -f "$bundle"
test "$(sha256sum "$bundle" | awk '{print $1}')" = "$expected_bundle_sha256"
test ! -e "$runtime"
test ! -e "$run_root"
test -f "$annotation"
test -d "$features"
test "$(sha256sum "$annotation" | awk '{print $1}')" = "$expected_annotation_sha256"
test -f "$previous_manifest"
test "$(sha256sum "$previous_manifest" | awk '{print $1}')" = "$expected_previous_manifest_sha256"
test -x "$python_env/bin/python"
test -f "$environment_receipt"
test "$(sha256sum "$environment_receipt" | awk '{print $1}')" = "$expected_environment_receipt_sha256"
test -f "$nms_extension"
test "$(sha256sum "$nms_extension" | awk '{print $1}')" = "$expected_nms_extension_sha256"

git -C "$project_root/actionformer_dcsr_g0_g1_20260730_v5" \
  bundle verify "$bundle"
git clone --branch "$branch" "$bundle" "$runtime"
test "$(git -C "$runtime" rev-parse HEAD)" = "$expected_commit"
test "$(git -C "$runtime" rev-parse 'HEAD^{tree}')" = "$expected_tree"
test "$(git -C "$runtime" branch --show-current)" = "$branch"
test -z "$(git -C "$runtime" status --porcelain=v1 --untracked-files=all)"

mkdir "$run_root"
exec > >(tee -a "$stage_root/deployment.log") 2>&1

export PATH="$python_env/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$runtime:$(dirname "$nms_extension")"
hash -r
test "$(readlink -f "$(command -v python)")" = "$(readlink -f "$python_env/bin/python")"
python - <<'PY'
import numpy
import torch
import nms_1d_cpu

assert torch.__version__ == "2.0.1"
assert torch.version.cuda == "11.8"
assert numpy.__version__ == "1.23.5"
assert nms_1d_cpu is not None
print({
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "numpy": numpy.__version__,
    "cuda_available_on_login": torch.cuda.is_available(),
})
PY

(
  cd "$runtime"
  python -m py_compile \
    libs/datasets/thumos14.py \
    libs/modeling/meta_archs.py \
    libs/modeling/sparse_heads.py \
    tools/*odfcr*.py \
    tests/test_odfcr_*.py
  bash -n \
    scripts/run_odfcr_internal_factorial_n16r4.sbatch \
    scripts/aggregate_odfcr_internal_factorial_n16r4.sbatch \
    scripts/run_odfcr_k384_replay_n16r4.sbatch \
    scripts/aggregate_odfcr_k384_n16r4.sbatch
  python -m pytest \
    tests/test_native_grid_sparse_heads.py \
    tests/test_sparsehead_official_config.py \
    tests/test_dcsr_heads.py \
    tests/test_dcsr_internal_holdout.py \
    tests/test_odfcr_heads.py \
    tests/test_odfcr_internal_holdout.py \
    tests/test_odfcr_matrix_contract.py \
    tests/test_odfcr_k384_contract.py \
    tests/test_odfcr_launcher_contract.py \
    -q
) | tee "$stage_root/linux_focused_suite.log"

manifest="$run_root/ODFCR_INTERNAL_HOLDOUT_V2.json"
(
  cd "$runtime"
  python -m tools.build_odfcr_internal_holdout_v2 \
    --previous-manifest "$previous_manifest" \
    --annotation "$annotation" \
    --feature-folder "$features" \
    --seed 2026073100 \
    --output "$manifest"
) | tee "$stage_root/holdout_v2_build.log"
manifest_sha256=$(sha256sum "$manifest" | awk '{print $1}')

MANIFEST="$manifest" \
PREVIOUS_MANIFEST="$previous_manifest" \
ANNOTATION="$annotation" \
EXPECTED_PREVIOUS_MANIFEST_SHA256="$expected_previous_manifest_sha256" \
EXPECTED_ANNOTATION_SHA256="$expected_annotation_sha256" \
python - <<'PY'
import json
import os
from tools.build_odfcr_internal_holdout_v2 import (
    _read_json,
    _sha256_file,
    validate_manifest_contract,
)

manifest = _read_json(os.environ["MANIFEST"])
previous = _read_json(os.environ["PREVIOUS_MANIFEST"])
annotation = _read_json(os.environ["ANNOTATION"])
validate_manifest_contract(
    manifest,
    previous,
    os.environ["EXPECTED_PREVIOUS_MANIFEST_SHA256"],
    annotation,
    os.environ["EXPECTED_ANNOTATION_SHA256"],
)
assert _sha256_file(os.environ["PREVIOUS_MANIFEST"]) == os.environ[
    "EXPECTED_PREVIOUS_MANIFEST_SHA256"
]
assert _sha256_file(os.environ["ANNOTATION"]) == os.environ[
    "EXPECTED_ANNOTATION_SHA256"
]
print({
    "schema_version": manifest["schema_version"],
    "train_video_count": manifest["train_video_count"],
    "holdout_video_count": manifest["holdout_video_count"],
    "new_holdout_disjoint_previous_holdout": manifest[
        "new_holdout_disjoint_previous_holdout"
    ],
})
PY
chmod 0444 "$manifest"

export_string="ALL,CANDIDATE_ROOT=$runtime,RUN_ROOT=$run_root,OFFICIAL_DATA_ROOT=$official_data,EXPECTED_CANDIDATE_COMMIT=$expected_commit,EXPECTED_CANDIDATE_TREE=$expected_tree,EXPECTED_ANNOTATION_SHA256=$expected_annotation_sha256,ODFCR_INTERNAL_HOLDOUT_MANIFEST=$manifest,ODFCR_INTERNAL_HOLDOUT_MANIFEST_SHA256=$manifest_sha256,ODFCR_PREVIOUS_HOLDOUT_MANIFEST=$previous_manifest,ODFCR_PREVIOUS_HOLDOUT_MANIFEST_SHA256=$expected_previous_manifest_sha256,ACTIONFORMER_PYTHON_ENV=$python_env,ACTIONFORMER_ENVIRONMENT_RECEIPT=$environment_receipt,EXPECTED_ENVIRONMENT_RECEIPT_SHA256=$expected_environment_receipt_sha256,ACTIONFORMER_NMS_EXTENSION=$nms_extension,EXPECTED_ACTIONFORMER_NMS_EXTENSION_SHA256=$expected_nms_extension_sha256,ODFCR_RUN_MODE=factorial"

sbatch --clusters=n16r4 --test-only \
  --export="$export_string" \
  "$runtime/scripts/run_odfcr_internal_factorial_n16r4.sbatch" \
  | tee "$stage_root/sbatch_factorial_test_only.log"

main_submit=$(sbatch --clusters=n16r4 --parsable \
  --output="$run_root/slurm-%A_%a.out" \
  --error="$run_root/slurm-%A_%a.err" \
  --export="$export_string" \
  "$runtime/scripts/run_odfcr_internal_factorial_n16r4.sbatch")
main_job=${main_submit%%;*}
case "$main_job" in
  ''|*[!0-9]*) echo "invalid formal factorial Job ID: $main_submit" >&2; exit 2 ;;
esac

g2_export="ALL,CANDIDATE_ROOT=$runtime,RUN_ROOT=$run_root,EXPECTED_CANDIDATE_COMMIT=$expected_commit,EXPECTED_CANDIDATE_TREE=$expected_tree,ACTIONFORMER_PYTHON_ENV=$python_env,ACTIONFORMER_ENVIRONMENT_RECEIPT=$environment_receipt,EXPECTED_ENVIRONMENT_RECEIPT_SHA256=$expected_environment_receipt_sha256"
g2_submit=$(sbatch --clusters=n16r4 --parsable \
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

RUNTIME="$runtime" \
RUN_ROOT="$run_root" \
BRANCH="$branch" \
EXPECTED_COMMIT="$expected_commit" \
EXPECTED_TREE="$expected_tree" \
BUNDLE="$bundle" \
BUNDLE_SHA256="$expected_bundle_sha256" \
MANIFEST="$manifest" \
MANIFEST_SHA256="$manifest_sha256" \
PREVIOUS_MANIFEST="$previous_manifest" \
PREVIOUS_MANIFEST_SHA256="$expected_previous_manifest_sha256" \
ANNOTATION="$annotation" \
ANNOTATION_SHA256="$expected_annotation_sha256" \
PYTHON_ENV="$python_env" \
ENVIRONMENT_RECEIPT="$environment_receipt" \
ENVIRONMENT_RECEIPT_SHA256="$expected_environment_receipt_sha256" \
NMS_EXTENSION="$nms_extension" \
NMS_EXTENSION_SHA256="$expected_nms_extension_sha256" \
MAIN_JOB="$main_job" \
G2_JOB="$g2_job" \
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

def receipt(path, expected_sha256=None):
    path = Path(path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 is not None:
        assert digest == expected_sha256
    return {
        "path": str(path),
        "sha256": digest,
        "size_bytes": path.stat().st_size,
    }

payload = {
    "schema_version": "actionformer_odfcr_deployment_receipt_v1",
    "validation_pass": True,
    "runtime": str(Path(os.environ["RUNTIME"]).resolve()),
    "run_root": str(Path(os.environ["RUN_ROOT"]).resolve()),
    "branch": os.environ["BRANCH"],
    "git_commit": os.environ["EXPECTED_COMMIT"],
    "git_tree": os.environ["EXPECTED_TREE"],
    "runtime_clean": True,
    "bundle": receipt(os.environ["BUNDLE"], os.environ["BUNDLE_SHA256"]),
    "manifest": receipt(
        os.environ["MANIFEST"], os.environ["MANIFEST_SHA256"]
    ),
    "previous_manifest": receipt(
        os.environ["PREVIOUS_MANIFEST"],
        os.environ["PREVIOUS_MANIFEST_SHA256"],
    ),
    "annotation": receipt(
        os.environ["ANNOTATION"], os.environ["ANNOTATION_SHA256"]
    ),
    "python_environment": str(Path(os.environ["PYTHON_ENV"]).resolve()),
    "environment_receipt": receipt(
        os.environ["ENVIRONMENT_RECEIPT"],
        os.environ["ENVIRONMENT_RECEIPT_SHA256"],
    ),
    "nms_extension": receipt(
        os.environ["NMS_EXTENSION"], os.environ["NMS_EXTENSION_SHA256"]
    ),
    "jobs": {
        "factorial_array": int(os.environ["MAIN_JOB"]),
        "aggregate_g2": int(os.environ["G2_JOB"]),
    },
    "source_split": "validation",
    "test_gt_used": False,
    "test_predictions_used": False,
    "paper_performance_row_allowed": False,
    "official_test_authorized": False,
    "efficiency_claim_allowed": False,
}
output = Path(os.environ["RUN_ROOT"]) / "DEPLOYMENT_RECEIPT.json"
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
