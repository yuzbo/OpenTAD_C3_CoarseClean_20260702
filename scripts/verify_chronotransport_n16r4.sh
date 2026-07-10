#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"

case "${BASE}" in
  /data/run01/sczc063/yuzibo|/data/run01/sczc063/yuzibo/*) ;;
  *) echo "YUZIBO_ROOT must stay under /data/run01/sczc063/yuzibo" >&2; exit 2 ;;
esac

module load cuda/11.8
module load miniforge3/24.11
# shellcheck disable=SC1091
source "${BASE}/conda_envs/opentad/bin/activate"

export HOME="${BASE}/tmp/home"
export XDG_CACHE_HOME="${BASE}/tmp/xdg_cache"
export XDG_CONFIG_HOME="${BASE}/tmp/xdg_config"
export HF_HOME="${BASE}/hf_cache"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""

cd "${ROOT}"

TESTS=(
  tests/test_chronotransport_core.py
  tests/test_chronotransport_repository_contract.py
  tests/test_chronotransport_vit_adapter_integration.py
  tests/test_chronotransport_pipeline.py
  tests/test_chronotransport_stage_a_smoke.py
  tests/test_chronotransport_opentad_replay.py
  tests/test_c3_coarse_classifier_model_matrix.py
  tests/test_c3_asformer_delta_ledger_full_train.py
)
python -m pytest -p no:cacheprovider "${TESTS[@]}" -q

PY_FILES=(
  tools/train.py
  tools/test.py
  tools/bata/train_lowres_action_probe.py
  tools/bata/validate_chronotransport_adatad.py
  tools/bata/run_chronotransport_paired_replay.py
  tools/bata/train_chronotransport_stage_b.py
  tools/bata/profile_chronotransport_schedules.py
  tools/bata/check_chronotransport_checkpoint.py
  tools/bata/chronotransport_opentad_factory.py
)
python -m py_compile "${PY_FILES[@]}"

bash -n scripts/run_chronotransport_adatad_gpu1.sh
bash -n scripts/run_chronotransport_paired_replay_gpu1.sh
echo "CHRONOTRANSPORT_REMOTE_VERIFY_PASS"
