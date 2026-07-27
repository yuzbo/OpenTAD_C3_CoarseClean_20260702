#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_RIME_CODE_GATE][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="${DUCA_RIME_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EXPECTED_COMMIT="${DUCA_RIME_EXPECTED_COMMIT:-}"
OUTPUT_ROOT="${DUCA_RIME_CODE_GATE_ROOT:-}"
PYTHON="${PYTHON:-python}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "the code gate must run inside Slurm"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "an exact expected commit is required"
[[ -n "${OUTPUT_ROOT}" && ! -e "${OUTPUT_ROOT}" ]] || fail "a fresh external output root is required"
[[ -d "${REPO_ROOT}" ]] || fail "repository snapshot is missing"
cd "${REPO_ROOT}"

if [[ -d .git ]]; then
  [[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "Git commit drift"
  [[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "Git tree is dirty"
else
  SOURCE_MANIFEST="${REPO_ROOT}/.codex_source_manifest"
  [[ -f "${SOURCE_MANIFEST}" ]] || fail "archive source manifest is missing"
  grep -qx "commit=${EXPECTED_COMMIT}" "${SOURCE_MANIFEST}" \
    || fail "archive commit drift"
  OVERLAY_MANIFEST="${DUCA_RIME_OVERLAY_SHA256_MANIFEST:-}"
  [[ -f "${OVERLAY_MANIFEST}" ]] || fail "overlay SHA-256 manifest is required"
  sha256sum -c "${OVERLAY_MANIFEST}"
fi

mkdir -p "${OUTPUT_ROOT}/fixtures" "${OUTPUT_ROOT}/logs"
printf 'fixture-block-entry\n' > "${OUTPUT_ROOT}/fixtures/train_block.txt"
printf 'fixture-block-entry\n' > "${OUTPUT_ROOT}/fixtures/development_block.txt"
printf '{}\n' > "${OUTPUT_ROOT}/fixtures/targets.jsonl"
printf '{}\n' > "${OUTPUT_ROOT}/fixtures/replay.jsonl"
printf '{}\n' > "${OUTPUT_ROOT}/fixtures/protocol.json"

export DUCA_RIME_TRAIN_BLOCK_LIST="${OUTPUT_ROOT}/fixtures/train_block.txt"
export DUCA_RIME_DEVELOPMENT_BLOCK_LIST="${OUTPUT_ROOT}/fixtures/development_block.txt"
export DUCA_RIME_PHASE2_EVAL_BLOCK_LIST="${OUTPUT_ROOT}/fixtures/development_block.txt"
export DUCA_RIME_TARGETS_JSONL="${OUTPUT_ROOT}/fixtures/targets.jsonl"
export DUCA_RIME_TARGETS_SHA256="$(sha256sum "${DUCA_RIME_TARGETS_JSONL}" | awk '{print $1}')"
export DUCA_RIME_BUDGET_PROTOCOL_JSON="${OUTPUT_ROOT}/fixtures/protocol.json"
export DUCA_RIME_BUDGET_PROTOCOL_SHA256="$(
  sha256sum "${DUCA_RIME_BUDGET_PROTOCOL_JSON}" | awk '{print $1}'
)"
unset DUCA_RIME_REPLAY_JSONL DUCA_RIME_REPLAY_SHA256

"${PYTHON}" -m py_compile \
  tools/train.py \
  tools/test.py \
  tools/bata/train_lowres_action_probe.py \
  tools/bata/create_duca_rime_splits.py \
  tools/bata/create_duca_rime_training_exposure.py \
  tools/bata/compact_duca_rime_checkpoint.py \
  tools/bata/evaluate_duca_rime_predictions.py \
  tools/bata/finalize_duca_rime_cost.py \
  tools/bata/finalize_duca_rime_phase3_arm.py \
  tools/bata/bootstrap_duca_rime_phase4.py \
  tools/bata/finalize_duca_rime_phase4_cell.py \
  tools/bata/build_duca_rime_gate_records.py \
  tools/bata/profile_duca_full_stack_cost.py \
  tools/bata/duca_p0_evaluation.py \
  tools/bata/build_duca_rime_training_targets.py \
  tools/bata/duca_rime_phase2.py \
  tools/bata/duca_rime_stage_contract.py \
  tools/bata/duca_rime_training.py \
  tools/bata/finalize_duca_rime_inference_ledger.py \
  tools/bata/build_duca_rime_budget_replay.py \
  opentad/models/duca/rime.py \
  opentad/models/selectors/duca_rime_frame_selector.py \
  opentad/models/backbones/backbone_wrapper.py \
  opentad/models/detectors/actionformer.py \
  opentad/models/detectors/tridet.py \
  opentad/models/dense_heads/tridet_head.py

bash -n \
  scripts/run_duca_rime_code_gate.sh \
  scripts/run_duca_rime_phase1_gate.sh \
  scripts/run_duca_rime_phase2_gates.sh \
  scripts/run_duca_rime_phase3_train_arm.sh \
  scripts/run_duca_rime_phase4_train_cell.sh \
  scripts/run_duca_rime_evaluate_arm.sh \
  scripts/run_duca_rime_cost_cell.sh \
  scripts/run_duca_rime_phase3_seal.sh \
  scripts/run_duca_rime_phase4_seal_cell.sh \
  scripts/run_duca_rime_phase4_seal_matrix.sh \
  scripts/run_duca_rime_phase0_measurements.sh \
  scripts/run_duca_rime_phase2_record_build.sh \
  scripts/run_duca_rime_phase2_baseline_eval.sh

"${PYTHON}" -m pytest \
  tests/test_duca_rime.py \
  tests/test_duca_rime_phase2.py \
  tests/test_duca_rime_targets.py \
  tests/test_duca_rime_stage_contract.py \
  tests/test_duca_rime_launchers.py \
  tests/test_duca_rime_inference_ledger.py \
  tests/test_duca_rime_training_contract.py \
  tests/test_duca_rime_exposure.py \
  tests/test_duca_rime_checkpoint_retention.py \
  tests/test_duca_rime_prediction_metrics.py \
  tests/test_duca_rime_gate_records.py \
  tests/test_profile_duca_full_stack_cost_cli.py \
  tests/test_duca_rime_tridet.py \
  tests/test_c3_coarse_classifier_model_matrix.py \
  tests/test_c3_asformer_delta_ledger_full_train.py \
  -q 2>&1 | tee "${OUTPUT_ROOT}/logs/pytest.out"

export DUCA_RIME_REPLAY_JSONL="${OUTPUT_ROOT}/fixtures/replay.jsonl"
export DUCA_RIME_REPLAY_SHA256="$(sha256sum "${DUCA_RIME_REPLAY_JSONL}" | awk '{print $1}')"

"${PYTHON}" - <<'PY' 2>&1 | tee "${OUTPUT_ROOT}/logs/configs.out"
from mmengine.config import Config

configs = (
    "configs/adatad/thumos/duca_rime_uniform_fixed384_total60.py",
    "configs/adatad/thumos/duca_rime_uniform_phase2_baseline.py",
    "configs/adatad/thumos/duca_rime_fixed_bound_total60.py",
    "configs/adatad/thumos/duca_rime_dynamic_no_risk_total60.py",
    "configs/adatad/thumos/duca_rime_dynamic_shuffle_total60.py",
    "configs/adatad/thumos/duca_adaptok_tad_direct_total60.py",
    "configs/adatad/thumos/duca_rime_full_total60.py",
    "configs/adatad/thumos/duca_rime_full_tridet_total60.py",
    "configs/adatad/thumos/duca_rime_uniform_fixed_tridet_total60.py",
    "configs/adatad/thumos/duca_rime_uniform_same_k_eval.py",
    "configs/adatad/thumos/duca_rime_uniform_same_k_tridet_eval.py",
    "configs/adatad/thumos/duca_rime_full_formal_validation.py",
    "configs/adatad/thumos/duca_rime_full_tridet_formal_validation.py",
    "configs/adatad/thumos/duca_rime_uniform_fixed_formal_validation.py",
    "configs/adatad/thumos/duca_rime_uniform_fixed_tridet_formal_validation.py",
    "configs/adatad/thumos/duca_rime_uniform_same_k_formal_validation.py",
    "configs/adatad/thumos/duca_rime_uniform_same_k_tridet_formal_validation.py",
)
for path in configs:
    cfg = Config.fromfile(path)
    assert cfg.solver.train.batch_size == 1
    assert cfg.post_processing.save_dict is True
    if "duca_rime_contract" in cfg:
        assert cfg.duca_rime_contract.pad_to_kmax is False
        assert cfg.duca_rime_contract.execution_quantum == 16
    print(f"CONFIG_OK {path}")
PY

printf '%s\n' \
  "schema=duca_rime_code_gate_v1" \
  "status=passed" \
  "commit=${EXPECTED_COMMIT}" \
  "slurm_job_id=${SLURM_JOB_ID}" \
  > "${OUTPUT_ROOT}/gate.receipt"

echo "[DUCA_RIME_CODE_GATE] PASS ${OUTPUT_ROOT}/gate.receipt"
