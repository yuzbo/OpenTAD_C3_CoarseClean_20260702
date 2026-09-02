#!/usr/bin/env bash
# Diagnostic-only replay of the previously failing Evidence F/A6 arms.
# The command intentionally stops on the first forward non-finite tensor.
# It never resumes a corrected formal run or writes into the old run root.
#SBATCH --job-name=duca_rec_diag
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source /etc/profile
set -uo pipefail

module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

REPO_ROOT="${DUCA_REPO_ROOT:?DUCA_REPO_ROOT is required}"
OLD_RUN_ROOT="${DUCA_OLD_RUN_ROOT:?DUCA_OLD_RUN_ROOT is required}"
TARGET_TASKS="${DUCA_TARGET_TASKS:-1,7}"
DIAG_ITERS="${DUCA_DIAG_ITERS:-120}"
YUZIBO_ROOT="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
PRETRAIN="${DUCA_VIDEOMAE_PRETRAIN:-${YUZIBO_ROOT}/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}"

if [[ ! -d "${REPO_ROOT}/.git" ]]; then
  echo "[ERROR] not a git checkout: ${REPO_ROOT}" >&2
  exit 2
fi
if [[ ! -d "${OLD_RUN_ROOT}" ]]; then
  echo "[ERROR] old run root not found: ${OLD_RUN_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${PRETRAIN}" ]]; then
  echo "[ERROR] VideoMAE checkpoint not found: ${PRETRAIN}" >&2
  exit 2
fi

cd "${REPO_ROOT}"
export YUZIBO_ROOT DUCA_VIDEOMAE_PRETRAIN="${PRETRAIN}"
export DUCA_H65_LEDGER_ROOT="${DUCA_H65_LEDGER_ROOT:-${YUZIBO_ROOT}/projects/c3_lowres_action_probe/ledger_exports/c3_official_asformer_delta_ledgers_20260702_052357_+0800}"
export DUCA_H65_TRAIN_LEDGER_PATH="${DUCA_H65_TRAIN_LEDGER_PATH:-${DUCA_H65_LEDGER_ROOT}/train/value_transport_ledger_delta_p_action_384.jsonl}"
export DUCA_H65_VAL_LEDGER_PATH="${DUCA_H65_VAL_LEDGER_PATH:-${DUCA_H65_LEDGER_ROOT}/val/value_transport_ledger_delta_p_action_384.jsonl}"
export DUCA_H65_TEST_LEDGER_PATH="${DUCA_H65_TEST_LEDGER_PATH:-${DUCA_H65_LEDGER_ROOT}/test/value_transport_ledger_delta_p_action_384.jsonl}"
export HOME="${YUZIBO_ROOT}/tmp/home"
export XDG_CACHE_HOME="${YUZIBO_ROOT}/tmp/xdg_cache"
export XDG_CONFIG_HOME="${YUZIBO_ROOT}/tmp/xdg_config"
export HF_HOME="${YUZIBO_ROOT}/hf_cache"
export PYTHONNOUSERSITE=1

IFS=',' read -r -a TASKS <<< "${TARGET_TASKS}"
status=0
for task in "${TASKS[@]}"; do
  case "${task}" in
    1) arm="F"; cfg="configs/adatad/thumos/duca_evidence_recovery_full.py" ;;
    7) arm="A6"; cfg="configs/adatad/thumos/duca_evidence_recovery_h65_selection.py" ;;
    *) echo "[ERROR] unsupported diagnostic task: ${task}" >&2; status=2; continue ;;
  esac

  work_dir="${OLD_RUN_ROOT}/diagnostic_${arm}_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}"
  mkdir -p "${work_dir}"
  resume_args=()
  if [[ "${arm}" == "F" ]]; then
    resume="$(find "${OLD_RUN_ROOT}/${arm}/seed_8261" -type f -name '*.pth' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)"
    if [[ -n "${resume}" ]]; then
      resume_args=(--resume "${resume}")
    fi
  fi

  echo "[INFO] diagnostic arm=${arm} config=${cfg} resume=${resume_args[*]:-none} work_dir=${work_dir}"
  # The historical F/A6 run predates deploy-ledger provenance fields. D0
  # replays its fixed positions only; formal runs keep strict checks.
  set +e
  torchrun --nproc_per_node=1 tools/train.py "${cfg}" \
    --seed 8261 \
    "${resume_args[@]}" \
    --cfg-options \
      "workflow.max_train_iters=${DIAG_ITERS}" \
      workflow.max_nonfinite_loss_retries=0 \
      workflow.max_amp_retries_per_batch=0 \
      workflow.require_finite_train_loss=True \
      dataset.train.pipeline.3.require_deployable=False \
      dataset.train.pipeline.3.source='' \
      dataset.train.pipeline.3.config_hash='' \
      dataset.val.pipeline.3.require_deployable=False \
      dataset.val.pipeline.3.source='' \
      dataset.val.pipeline.3.config_hash='' \
      dataset.test.pipeline.3.require_deployable=False \
      dataset.test.pipeline.3.source='' \
      dataset.test.pipeline.3.config_hash='' \
      "workflow.training_update_audit_json=${work_dir}/update_audit.json" \
      "work_dir=${work_dir}"
  rc=$?
  set -e
  echo "[INFO] diagnostic arm=${arm} exit_code=${rc}"
  if (( rc == 0 )); then
    echo "[ERROR] diagnostic arm=${arm} completed without a terminal non-finite event" >&2
    status=1
  elif [[ ! -f "${work_dir}/update_audit.json" ]]; then
    echo "[ERROR] diagnostic arm=${arm} failed without update_audit.json" >&2
    status=1
  fi
done

exit "${status}"
