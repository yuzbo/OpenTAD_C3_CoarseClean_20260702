#!/usr/bin/env bash
#SBATCH --job-name=duca-ev-time-precheck
#SBATCH --partition=gpu
#SBATCH --account=sczc063
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

source /etc/profile
set -euo pipefail

PROJECT_DIR="${DUCA_REPO_ROOT:?DUCA_REPO_ROOT is required}"
EXPECTED_COMMIT="${DUCA_EVIDENCE_EXPECTED_COMMIT:?DUCA_EVIDENCE_EXPECTED_COMMIT is required}"
RUN_ROOT="${DUCA_TIME_PRECHECK_ROOT:?DUCA_TIME_PRECHECK_ROOT is required}"
cd "$PROJECT_DIR"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]

module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

export YUZIBO_ROOT=/data/run01/sczc063/yuzibo
export XDG_CACHE_HOME="$YUZIBO_ROOT/tmp/xdg_cache"
export XDG_CONFIG_HOME="$YUZIBO_ROOT/tmp/xdg_config"
export HF_HOME="$YUZIBO_ROOT/hf_cache"
export THUMOS14_ANNOTATION_PATH="$YUZIBO_ROOT/thumos14/annotations/thumos_14_anno.json"
export THUMOS14_CLASS_MAP="$YUZIBO_ROOT/thumos14/annotations/category_idx.txt"
export THUMOS14_TRAIN_DATA_PATH="$YUZIBO_ROOT/raw/Validation Data/validation"
export THUMOS14_TEST_DATA_PATH="$YUZIBO_ROOT/raw/Test Data/TH14_test_set_mp4"
export DUCA_VIDEOMAE_PRETRAIN="$YUZIBO_ROOT/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
LEGACY_LEDGER_ROOT="$YUZIBO_ROOT/projects/c3_lowres_action_probe/ledger_exports/c3_official_asformer_delta_ledgers_20260702_052357_+0800"
FULLGRID_LEDGER_ROOT="$YUZIBO_ROOT/projects/c3_lowres_action_probe/ledger_exports/c3_official_asformer_delta_ledgers_fullgrid_ec0bb6a_20260702_065702_+0800"
export DUCA_H65_TRAIN_LEDGER_PATH="$LEGACY_LEDGER_ROOT/train/value_transport_ledger_delta_p_action_384.jsonl"
export DUCA_H65_VAL_LEDGER_PATH="$LEGACY_LEDGER_ROOT/val/value_transport_ledger_delta_p_action_384.jsonl"
export DUCA_H65_TEST_LEDGER_PATH="$FULLGRID_LEDGER_ROOT/test/value_transport_ledger_delta_p_action_384.jsonl"
export DUCA_H65_LEDGER_SOURCE=c3_lowres_probe_delta_p_action
export DUCA_H65_LEDGER_CONFIG_HASH=""

case "${SLURM_ARRAY_TASK_ID:?array task must be 2 or 7}" in
  2) CONFIG=duca_evidence_recovery_a1_optimizer_precheck.py; ARM=A1 ;;
  7) CONFIG=duca_evidence_recovery_a6_optimizer_precheck.py; ARM=A6 ;;
  *) echo "Unsupported precheck task" >&2; exit 1 ;;
esac

python -m pytest tests/test_evidence_time_optimizer.py -q
mkdir -p "$RUN_ROOT/$ARM"
torchrun --nproc_per_node=1 --master_port="$((31000 + SLURM_JOB_ID % 10000))" \
  tools/train.py "configs/adatad/thumos/$CONFIG" \
  --seed 8261 --cfg-options work_dir="$RUN_ROOT/$ARM"
