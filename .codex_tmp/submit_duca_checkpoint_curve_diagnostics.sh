#!/usr/bin/env bash
set -euo pipefail

BASE=/data/run01/sczc063/yuzibo
STAGE_DIR="${BASE}/projects/c3_lowres_action_probe/duca_checkpoint_curve_stage_20260723"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${BASE}/projects/c3_lowres_action_probe/duca_checkpoint_curve_${STAMP}"
SBATCH_FILE="${STAGE_DIR}/run_checkpoint_curve_bundle.sbatch"
mkdir -p "${STAGE_DIR}" "${RUN_ROOT}/logs" "${RUN_ROOT}/jobs"

cat > "${SBATCH_FILE}" <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --clusters=n16r4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00

source /etc/profile
module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate
set -euo pipefail

: "${BUNDLE:?set BUNDLE}"
: "${RUN_ROOT:?set RUN_ROOT}"
PRETRAIN=/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth

declare -A ARM_ROOTS=(
  [exact_uniform]=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_t1_trainfree_f81aef4_formal_20260723_1048/arms/two_stage_exact_uniform/official60
  [R2Q3]=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_t1_26ce86d_recovery_20260723_050516/arms/boundary_burst_r2q3_g0/official60
  [R4Q5]=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_487a178_recovery_20260723_034220/arms/boundary_burst_r4q5_g0/official60
  [soft_detached]=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_ca40c9c_retry_20260723_035756/arms/soft_detached/official60
  [hard_detached]=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_ca40c9c_retry_20260723_035756/arms/hard_detached/official60
  [soft_adapted]=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_ca40c9c_retry_20260723_035756/arms/soft_adapted/official60
)

if [[ "${BUNDLE}" == A ]]; then
  ARMS=(exact_uniform R2Q3 R4Q5)
elif [[ "${BUNDLE}" == B ]]; then
  ARMS=(soft_detached hard_detached soft_adapted)
else
  echo "unknown bundle ${BUNDLE}" >&2
  exit 2
fi

for arm in "${ARMS[@]}"; do
  official_root="${ARM_ROOTS[$arm]}"
  audit="${official_root}/work/gpu1_id0/duca_selected_axis_training_audit.json"
  test -f "${audit}"
  readarray -t fields < <(python - "${audit}" <<'PY'
import json
import sys
from pathlib import Path
d = json.loads(Path(sys.argv[1]).read_text())
init = d.get("selector_initialization_contract") or {}
print(Path(d["source_config_path"]).resolve())
print(d["git_commit"])
print(init.get("checkpoint_path", ""))
print(init.get("checkpoint_sha256", ""))
print(init.get("checkpoint_epoch", ""))
PY
  )
  source_config="${fields[0]}"
  expected_commit="${fields[1]}"
  repo="${source_config%%/configs/*}"
  export DUCA_FRONTEND_CHECKPOINT="${fields[2]}"
  export DUCA_FRONTEND_CHECKPOINT_SHA256="${fields[3]}"
  export DUCA_FRONTEND_CHECKPOINT_EPOCH="${fields[4]}"
  export DUCA_EXPECTED_COMMIT="${expected_commit}"

  cd "${repo}"
  test "$(git rev-parse HEAD)" = "${expected_commit}"
  test -z "$(git status --porcelain)"

  arm_out="${RUN_ROOT}/${arm}"
  mkdir -p "${arm_out}/configs"
  diagnostic_config="${arm_out}/configs/checkpoint_curve.py"
  cat > "${diagnostic_config}" <<EOF
_base_ = [r"${source_config}"]

workflow = dict(
    formal_protocol="duca_checkpoint_curve_diagnostic_v1",
    checkpoint_curve_diagnostic=True,
)
EOF

  python - "${diagnostic_config}" <<'PY'
import sys
from mmengine.config import Config
cfg = Config.fromfile(sys.argv[1])
assert cfg.workflow.formal_protocol == "duca_checkpoint_curve_diagnostic_v1"
assert int(cfg.model.frame_selector.budget) == 384
assert int(cfg.model.frame_selector.max_unselected_hole) == 2
PY

  for epoch in 9 19 29 39 49; do
    checkpoint="${official_root}/work/gpu1_id0/checkpoint/epoch_${epoch}.pth"
    test -f "${checkpoint}" || continue
    epoch_out="${arm_out}/epoch_${epoch}"
    metrics="${epoch_out}/evaluation.json"
    test -f "${metrics}" && continue
    mkdir -p "${epoch_out}"
    python -m torch.distributed.run --nproc_per_node=1 \
      --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
      --rdzv_id="duca-curve-${SLURM_JOB_ID}-${arm}-${epoch}" \
      tools/test.py "${diagnostic_config}" --checkpoint "${checkpoint}" \
      --checkpoint-state-key state_dict_ema --expected-checkpoint-epoch "${epoch}" \
      --metrics-json "${metrics}" --id 0 --seed 3407 --cfg-options \
        "work_dir=${epoch_out}/work" \
        "model.backbone.custom.pretrain=${PRETRAIN}" \
        "post_processing.save_dict=True" \
        "inference.load_from_raw_predictions=False" \
      2>&1 | tee "${epoch_out}/eval.out"
    sha256sum "${checkpoint}" "${metrics}" > "${epoch_out}/hashes.sha256"
  done

  python - "${arm}" "${arm_out}" "${source_config}" "${expected_commit}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

arm, root, source_config, commit = sys.argv[1:]
root = Path(root)
evaluations = {}
for path in sorted(root.glob("epoch_*/evaluation.json")):
    epoch = int(path.parent.name.split("_")[-1])
    payload = json.loads(path.read_text())
    evaluations[str(epoch)] = {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "metrics": payload.get("metrics"),
        "evaluator": payload.get("evaluator"),
    }
summary = {
    "schema": "duca_checkpoint_curve_diagnostic_v1",
    "status": "diagnostic_only_not_for_checkpoint_selection",
    "arm": arm,
    "git_commit": commit,
    "source_config": source_config,
    "checkpoint_state_key": "state_dict_ema",
    "seed": 3407,
    "evaluations": evaluations,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY
done
SBATCH

printf 'job_id\tbundle\trun_root\n' > "${RUN_ROOT}/jobs/jobs.tsv"
for bundle in A B; do
  job_id="$(sbatch --parsable \
    --job-name="duca-curve-${bundle}" \
    --output="${RUN_ROOT}/logs/bundle_${bundle}.%j.out" \
    --error="${RUN_ROOT}/logs/bundle_${bundle}.%j.err" \
    --export="ALL,BUNDLE=${bundle},RUN_ROOT=${RUN_ROOT}" \
    "${SBATCH_FILE}")"
  printf '%s\t%s\t%s\n' "${job_id}" "${bundle}" "${RUN_ROOT}" >> "${RUN_ROOT}/jobs/jobs.tsv"
done

cat > "${RUN_ROOT}/deployment_manifest.txt" <<EOF
task=offline_tad_duca_checkpoint_curve_diagnostic
protocol=full_validation_every_10_one_based_epochs_state_dict_ema
epochs=10,20,30,40,50; terminal_60_reuses_formal_evaluation
arms=exact_uniform,R2Q3,R4Q5,soft_detached,hard_detached,soft_adapted
purpose=diagnose_learning_trajectory_only; never select paper checkpoint
run_root=${RUN_ROOT}
EOF
sha256sum "${RUN_ROOT}/deployment_manifest.txt" "${RUN_ROOT}/jobs/jobs.tsv" > "${RUN_ROOT}/deployment_hashes.sha256"
echo "RUN_ROOT ${RUN_ROOT}"
cat "${RUN_ROOT}/jobs/jobs.tsv"
cat "${RUN_ROOT}/deployment_hashes.sha256"
