#!/usr/bin/env bash
source /etc/profile
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
ETTRC_PRETRAIN="${ETTRC_PRETRAIN:-${BASE}/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}"
[[ -f "$ETTRC_PRETRAIN" ]] || { echo "missing ET-TRC pretrain: $ETTRC_PRETRAIN" >&2; exit 1; }
SEED=4407
STRIDE_K=4
MAX_JOBS_IN_QUEUE="${ETTRC_MAX_JOBS_IN_QUEUE:-14}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEED="$2"; shift 2;;
    --stride-k) STRIDE_K="$2"; shift 2;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
cd "$PROJECT_DIR"
[[ -z "$(git status --porcelain)" ]] || { echo "ET-TRC checkout is not clean" >&2; exit 2; }
EXPECTED_COMMIT="${ETTRC_EXPECTED_COMMIT:?ETTRC_EXPECTED_COMMIT must be the full 40-character target SHA}"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-fA-F]{40}$ ]] || { echo "ETTRC_EXPECTED_COMMIT must be a full SHA" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || { echo "ET-TRC HEAD mismatch" >&2; exit 2; }
[[ "$MAX_JOBS_IN_QUEUE" =~ ^[1-9][0-9]*$ ]] || { echo "ETTRC_MAX_JOBS_IN_QUEUE must be positive" >&2; exit 2; }
(( MAX_JOBS_IN_QUEUE >= 3 )) || { echo "ETTRC_MAX_JOBS_IN_QUEUE must allow three submissions" >&2; exit 2; }
SHORT_COMMIT="$(git rev-parse --short HEAD)"
RUN_ROOT="${ETTRC_RUN_ROOT:-${BASE}/projects/zoomtoken_et_trc_fix_${SHORT_COMMIT}}"
mkdir -p "${RUN_ROOT}"
REGISTRY="${RUN_ROOT}/submission_registry.tsv"
if [[ -s "$REGISTRY" ]]; then
  echo "ET-TRC submission already recorded: $REGISTRY" >&2
  exit 2
fi
export ETTRC_PRETRAIN

while true; do
  current_jobs="$(squeue -u "$USER" -h | wc -l)"
  if (( current_jobs <= MAX_JOBS_IN_QUEUE - 3 )); then
    break
  fi
  echo "ET-TRC queue has ${current_jobs}/${MAX_JOBS_IN_QUEUE} jobs; retrying in 60 seconds"
  sleep 60
done

common_wrap="source /etc/profile; set -euo pipefail; module load cuda/11.8; module load miniforge3/24.11; source \"${BASE}/conda_envs/opentad/bin/activate\"; export ETTRC_PRETRAIN=\"${ETTRC_PRETRAIN}\" ETTRC_EXPECTED_COMMIT=\"${EXPECTED_COMMIT}\"; cd \"${PROJECT_DIR}\"; test \"\$(git rev-parse HEAD)\" = \"\${ETTRC_EXPECTED_COMMIT}\"; test -z \"\$(git status --porcelain)\""
admission_cfg="configs/adatad/thumos/et_trc_videomae_s_768x1_160_adapter_seed${SEED}.py"
admission_job="$(sbatch --parsable --account=sczc063 --qos=normal --partition=gpu --gres=gpu:2 --cpus-per-task=8 --time=02:00:00 \
  --job-name="et-trc-admission-s${SEED}" \
  --output="${BASE}/slurm_logs/%x_%j.out" --error="${BASE}/slurm_logs/%x_%j.err" \
  --wrap="bash -lc '${common_wrap}; torchrun --standalone --nproc_per_node=2 tools/train.py \"${admission_cfg}\" --seed \"${SEED}\" --cfg-options model.backbone.backbone.stride_k=${STRIDE_K} model.backbone.custom.pretrain=\"${ETTRC_PRETRAIN}\" workflow.end_epoch=1 workflow.max_train_iters=1 workflow.val_start_epoch=2 work_dir=\"${RUN_ROOT}/admission\"'")"
printf 'stage\tjob_id\tdependency\tconfig\tcommit\n' > "$REGISTRY"
printf 'admission\t%s\t\t%s\t%s\n' "$admission_job" "$admission_cfg" "$EXPECTED_COMMIT" >> "$REGISTRY"

for mode in off on; do
  if [[ "$mode" == off ]]; then
    cfg="configs/adatad/thumos/et_trc_videomae_s_768x1_160_adapter_off_seed${SEED}.py"
  else
    cfg="configs/adatad/thumos/et_trc_videomae_s_768x1_160_adapter_seed${SEED}.py"
  fi
  [[ -f "$cfg" ]] || { echo "missing config: $cfg" >&2; exit 1; }
    job_id="$(sbatch --parsable --export=ALL,ETTRC_PRETRAIN="$ETTRC_PRETRAIN",ETTRC_EXPECTED_COMMIT="$EXPECTED_COMMIT" --account=sczc063 --qos=normal --partition=gpu --gres=gpu:2 --cpus-per-task=8 --time=72:00:00 \
    --dependency="afterok:${admission_job%%;*}" \
    --job-name="et-trc-${mode}-s${SEED}" \
    --output="${BASE}/slurm_logs/%x_%j.out" --error="${BASE}/slurm_logs/%x_%j.err" \
      --wrap="bash -lc '${common_wrap}; torchrun --standalone --nproc_per_node=2 tools/train.py \"${cfg}\" --seed \"${SEED}\" --cfg-options model.backbone.backbone.stride_k=${STRIDE_K} model.backbone.custom.pretrain=\"${ETTRC_PRETRAIN}\" work_dir=\"${RUN_ROOT}/${mode}\"'")"
    printf '%s\t%s\tafterok:%s\t%s\t%s\n' "$mode" "$job_id" "${admission_job%%;*}" "$cfg" "$EXPECTED_COMMIT" >> "$REGISTRY"
done
cat "$REGISTRY"
