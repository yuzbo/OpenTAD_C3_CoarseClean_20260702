#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_TWO_STAGE_TRAIN][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="${DUCA_REPO_ROOT:-$(pwd -P)}"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
export DUCA_CELLCF_TRAINING_PROFILE=official60
source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"

VARIANT="${DUCA_SELECTED_OPT_VARIANT:-}"
case "${VARIANT}" in
  two_stage_exact_uniform)
    CONFIG="configs/adatad/thumos/duca_two_stage_exact_uniform_fixed384_official60.py"
    ;;
  two_stage_scratch)
    CONFIG="configs/adatad/thumos/duca_two_stage_scratch_fixed384_official60.py"
    ;;
  two_stage_pretrained_joint)
    CONFIG="configs/adatad/thumos/duca_two_stage_pretrained_joint_fixed384_official60.py"
    ;;
  two_stage_pretrained_frozen)
    CONFIG="configs/adatad/thumos/duca_two_stage_pretrained_frozen_fixed384_official60.py"
    ;;
  global_curriculum_g0)
    CONFIG="configs/adatad/thumos/duca_global_curriculum_g0_no_feedback_fixed384_official60.py"
    ;;
  global_curriculum_g1)
    CONFIG="configs/adatad/thumos/duca_global_curriculum_g1_protected_fixed384_official60.py"
    ;;
  global_curriculum_g2)
    CONFIG="configs/adatad/thumos/duca_global_curriculum_g2_uni_companion_fixed384_official60.py"
    ;;
  gaussian_matched_g0)
    CONFIG="configs/adatad/thumos/duca_global_curriculum_g0_no_feedback_fixed384_official60.py"
    ;;
  boundary_burst_r2q3_g0)
    CONFIG="configs/adatad/thumos/duca_boundary_burst_g0_no_feedback_fixed384_official60.py"
    ;;
  boundary_burst_r4q5_g0)
    CONFIG="configs/adatad/thumos/duca_boundary_burst_r4q5_g0_no_feedback_fixed384_official60.py"
    ;;
  boundary_burst_r2q3_g1)
    CONFIG="configs/adatad/thumos/duca_boundary_burst_g1_protected_fixed384_official60.py"
    ;;
  boundary_burst_r2q3_g2)
    CONFIG="configs/adatad/thumos/duca_boundary_burst_g2_uni_companion_fixed384_official60.py"
    ;;
  boundary_burst_r4q5_g1)
    CONFIG="configs/adatad/thumos/duca_boundary_burst_r4q5_g1_protected_fixed384_official60.py"
    ;;
  boundary_burst_r4q5_g2)
    CONFIG="configs/adatad/thumos/duca_boundary_burst_r4q5_g2_uni_companion_fixed384_official60.py"
    ;;
  *)
    fail "unknown two-stage variant: ${VARIANT}"
    ;;
esac

EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
DECISION="${DUCA_FRONTEND_DECISION_JSON:-}"
DECISION_SHA256="${DUCA_FRONTEND_DECISION_SHA256:-}"
GATE_SUITE="${DUCA_SELECTED_OPT_GATE_SUITE:-}"
GATE_SUITE_SHA256="${DUCA_SELECTED_OPT_GATE_SUITE_SHA256:-}"
RUN_DIR="${RUN_DIR:-}"
WORK_DIR="${WORK_DIR:-}"
DIAGNOSTIC_ONLY="${DUCA_BOUNDARY_BURST_DIAGNOSTIC_ONLY:-0}"
ALIGNMENT_JSON="${DUCA_BOUNDARY_BURST_ALIGNMENT_JSON:-}"
ALIGNMENT_SHA256="${DUCA_BOUNDARY_BURST_ALIGNMENT_SHA256:-}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "Slurm allocation is required"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "Slurm did not expose a GPU"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "exact commit is required"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "clean tree required"
[[ -f "${DECISION}" ]] || fail "frontend decision is missing"
[[ "$(sha256sum "${DECISION}" | awk '{print $1}')" == "${DECISION_SHA256}" ]] \
  || fail "frontend decision hash drift"
[[ -f "${GATE_SUITE}" ]] || fail "two-stage gate suite is missing"
[[ "$(sha256sum "${GATE_SUITE}" | awk '{print $1}')" == "${GATE_SUITE_SHA256}" ]] \
  || fail "two-stage gate suite hash drift"
readarray -t execution_policy < <("${PYTHON}" - "${DECISION}" \
  "${DECISION_SHA256}" "${GATE_SUITE}" "${GATE_SUITE_SHA256}" \
  "${EXPECTED_COMMIT}" "${VARIANT}" "${DIAGNOSTIC_ONLY}" \
  "${CONFIG}" "${ALIGNMENT_JSON}" "${ALIGNMENT_SHA256}" <<'PY'
import hashlib
import sys
from pathlib import Path
from tools.bata.duca_boundary_burst_hard_swap_alignment import (
    FAMILY_FEEDBACK_ROUTES,
    validate_alignment_artifact,
)
from tools.bata.select_duca_boundary_burst_candidates import (
    GAUSSIAN_OFFICIAL_VARIANT,
    GAUSSIAN_P0_VARIANT,
    R0_PROJECTED_FAMILY_ROUTES,
    UNIFORM_OFFICIAL_VARIANT,
    validate_frontend_decision,
    validate_full_model_gate,
)

decision = validate_frontend_decision(
    decision_path=sys.argv[1],
    decision_sha256=sys.argv[2],
    expected_commit=sys.argv[5],
)
validate_full_model_gate(
    gate_path=sys.argv[3],
    gate_sha256=sys.argv[4],
    decision_path=sys.argv[1],
    decision_sha256=sys.argv[2],
    expected_commit=sys.argv[5],
)
routing = decision["family_routing"]
variant = sys.argv[6]
diagnostic_only = sys.argv[7] == "1"
config = sys.argv[8]
alignment_path = sys.argv[9]
alignment_sha256 = sys.argv[10]
feedback_variants = {
    value[key]
    for value in FAMILY_FEEDBACK_ROUTES.values()
    for key in ("g1_variant", "g2_variant")
}
if variant in feedback_variants:
    binding = validate_alignment_artifact(
        path=alignment_path,
        digest=alignment_sha256,
        expected_commit=sys.argv[5],
        expected_variant=variant,
        source_config_path=config,
        source_config_sha256=hashlib.sha256(Path(config).read_bytes()).hexdigest(),
    )
    if (
        binding["selected_weakest_projected_family"]
        != routing["selected_weakest_projected_family"]
    ):
        raise SystemExit("hard-swap artifact selected-family drift")
    execution_role = "required_main"
elif variant in routing["required_official60_variants"]:
    execution_role = "required_main"
elif diagnostic_only and variant in routing["diagnostic_official60_variants"]:
    execution_role = "optional_diagnostic"
else:
    raise SystemExit(f"R0 family routing did not authorize {variant}")
variant_to_p0 = {GAUSSIAN_OFFICIAL_VARIANT: GAUSSIAN_P0_VARIANT}
variant_to_p0.update(
    {
        route["official60_variant"]: route["p0_variant"]
        for route in R0_PROJECTED_FAMILY_ROUTES.values()
    }
)
if variant in feedback_variants:
    frontend_variant = routing["selected_p0_variant"]
else:
    frontend_variant = (
        "not_applicable"
        if variant == UNIFORM_OFFICIAL_VARIANT
        else variant_to_p0[variant]
    )
print(execution_role)
print(frontend_variant)
PY
)
EXECUTION_ROLE="${execution_policy[0]}"
FRONTEND_VARIANT="${execution_policy[1]}"
[[ -n "${RUN_DIR}" && ! -e "${RUN_DIR}" ]] || fail "fresh RUN_DIR is required"
[[ -n "${WORK_DIR}" && ! -e "${WORK_DIR}" ]] || fail "fresh WORK_DIR is required"
[[ "$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')" == "1" ]] \
  || fail "exactly one Slurm-visible GPU is required"

if [[ "${VARIANT}" == "two_stage_exact_uniform" ]]; then
  "${PYTHON}" - "${DECISION}" "${EXPECTED_COMMIT}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("ok") is not True or payload.get("test_subset_consumed") is not False:
    raise SystemExit("frontend decision did not authorize the matched suite")
if payload.get("git_commit") not in {None, sys.argv[2]}:
    raise SystemExit("frontend decision commit mismatch")
PY
  unset DUCA_FRONTEND_CHECKPOINT DUCA_FRONTEND_CHECKPOINT_SHA256 DUCA_FRONTEND_CHECKPOINT_EPOCH
  FRONTEND_BINDING="not_applicable_exact_uniform"
  FRONTEND_CHECKPOINT_SHA256_VALUE=""
  FRONTEND_CHECKPOINT_EPOCH_VALUE=""
else
  readarray -t winner < <("${PYTHON}" - "${DECISION}" "${DECISION_SHA256}" \
    "${EXPECTED_COMMIT}" "${FRONTEND_VARIANT}" <<'PY'
import sys
from tools.bata.select_duca_boundary_burst_candidates import validate_frontend_decision

payload = validate_frontend_decision(
    decision_path=sys.argv[1],
    decision_sha256=sys.argv[2],
    expected_commit=sys.argv[3],
)
key = sys.argv[4]
if key not in payload["winners"]:
    raise SystemExit(f"no frontend winner is registered for diagnostic/main variant {key}")
winner = payload["winners"][key]
print(winner["checkpoint_path"])
print(winner["checkpoint_sha256"])
print(int(winner["epoch_one_based"]) - 1)
PY
  )
  export DUCA_FRONTEND_CHECKPOINT="${winner[0]}"
  export DUCA_FRONTEND_CHECKPOINT_SHA256="${winner[1]}"
  export DUCA_FRONTEND_CHECKPOINT_EPOCH="${winner[2]}"
  [[ -f "${DUCA_FRONTEND_CHECKPOINT}" ]] || fail "selected frontend checkpoint is missing"
  [[ "$(sha256sum "${DUCA_FRONTEND_CHECKPOINT}" | awk '{print $1}')" == "${DUCA_FRONTEND_CHECKPOINT_SHA256}" ]] \
    || fail "selected frontend checkpoint hash drift"
  FRONTEND_BINDING="variant_matched_p0_winner"
  FRONTEND_CHECKPOINT_SHA256_VALUE="${DUCA_FRONTEND_CHECKPOINT_SHA256}"
  FRONTEND_CHECKPOINT_EPOCH_VALUE="${DUCA_FRONTEND_CHECKPOINT_EPOCH}"
fi

mkdir -p "${RUN_DIR}" "${WORK_DIR}"
CONFIG_SHA256="$(sha256sum "${CONFIG}" | awk '{print $1}')"
"${PYTHON}" - "${RUN_DIR}/launch_manifest.json" "${VARIANT}" \
  "${EXPECTED_COMMIT}" "${CONFIG}" "${CONFIG_SHA256}" \
  "${DECISION}" "${DECISION_SHA256}" "${GATE_SUITE}" \
  "${GATE_SUITE_SHA256}" "${FRONTEND_BINDING}" \
  "${FRONTEND_CHECKPOINT_SHA256_VALUE}" "${FRONTEND_CHECKPOINT_EPOCH_VALUE}" \
  "${EXECUTION_ROLE}" "${SLURM_JOB_ID}" "${ALIGNMENT_JSON}" \
  "${ALIGNMENT_SHA256}" <<'PY'
import sys
from pathlib import Path

from tools.bata.duca_boundary_burst_hard_swap_alignment import (
    FAMILY_FEEDBACK_ROUTES,
    validate_alignment_artifact,
)
from tools.bata.select_duca_boundary_burst_candidates import (
    _atomic_write_json,
    validate_frontend_decision,
    validate_full_model_gate,
)

(
    output,
    variant,
    commit,
    config,
    config_sha256,
    decision_path,
    decision_sha256,
    gate_path,
    gate_sha256,
    frontend_binding,
    frontend_checkpoint_sha256,
    frontend_checkpoint_epoch,
    execution_role,
    slurm_job_id,
    alignment_path,
    alignment_sha256,
) = sys.argv[1:]
decision = validate_frontend_decision(
    decision_path=decision_path,
    decision_sha256=decision_sha256,
    expected_commit=commit,
)
validate_full_model_gate(
    gate_path=gate_path,
    gate_sha256=gate_sha256,
    decision_path=decision_path,
    decision_sha256=decision_sha256,
    expected_commit=commit,
)
payload = {
    "schema": "duca_two_stage_curriculum_launch_v1",
    "fail_closed": True,
    "task": "offline_temporal_action_detection",
    "git_commit": commit,
    "variant": variant,
    "execution_role": execution_role,
    "seed": 3407,
    "config": config,
    "config_sha256": config_sha256,
    "frontend_decision_path": str(Path(decision_path).resolve()),
    "frontend_decision_sha256": decision_sha256,
    "family_manifest": decision["family_manifest"],
    "r0_headroom_gate": decision["r0_headroom_gate"],
    "family_routing": decision["family_routing"],
    "p0_training_asformer_consumer": decision[
        "p0_training_asformer_consumer"
    ],
    "frontend_checkpoint_binding": frontend_binding,
    "frontend_checkpoint_sha256": frontend_checkpoint_sha256 or None,
    "frontend_checkpoint_epoch_zero_based": (
        int(frontend_checkpoint_epoch) if frontend_checkpoint_epoch else None
    ),
    "gate_path": str(Path(gate_path).resolve()),
    "gate_suite_sha256": gate_sha256,
    "uniform_detector_warmup_successful_updates": 1000,
    "official_training_successful_updates": 6000,
    "detector_extra_updates": 0,
    "terminal_checkpoint": "epoch_59.pth/state_dict_ema",
    "checkpoint_interval": 5,
    "slurm_job_id": slurm_job_id,
}
feedback_variants = {
    value[key]
    for value in FAMILY_FEEDBACK_ROUTES.values()
    for key in ("g1_variant", "g2_variant")
}
if variant in feedback_variants:
    payload["hard_swap_alignment"] = validate_alignment_artifact(
        path=alignment_path,
        digest=alignment_sha256,
        expected_commit=commit,
        expected_variant=variant,
        source_config_path=config,
        source_config_sha256=config_sha256,
    )
_atomic_write_json(Path(output).resolve(), payload, require_absent=True)
PY

"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  --rdzv_id="duca-two-stage-${SLURM_JOB_ID}-${VARIANT}-train" \
  tools/train.py "${CONFIG}" \
  --id 0 \
  --seed 3407 \
  --cfg-options \
    "work_dir=${WORK_DIR}" \
    "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" \
  2>&1 | tee "${RUN_DIR}/train.out"

ACTUAL_WORK_DIR="${WORK_DIR}/gpu1_id0"
CHECKPOINT="${ACTUAL_WORK_DIR}/checkpoint/epoch_59.pth"
EVAL_ROOT="${RUN_DIR}/terminal_eval"
EVAL_JSON="${RUN_DIR}/terminal_evaluation.json"
[[ -f "${CHECKPOINT}" ]] || fail "terminal epoch_59 checkpoint is missing"

"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  --rdzv_id="duca-two-stage-${SLURM_JOB_ID}-${VARIANT}-eval" \
  tools/test.py "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --checkpoint-state-key state_dict_ema \
  --expected-checkpoint-epoch 59 \
  --metrics-json "${EVAL_JSON}" \
  --id 0 \
  --seed 3407 \
  --cfg-options \
    "work_dir=${EVAL_ROOT}" \
    "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" \
    "post_processing.save_dict=True" \
    "inference.load_from_raw_predictions=False" \
  2>&1 | tee "${RUN_DIR}/terminal_eval.out"

"${PYTHON}" - "${RUN_DIR}" "${VARIANT}" "${EXPECTED_COMMIT}" \
  "${CHECKPOINT}" "${EVAL_JSON}" "${DECISION_SHA256}" "${GATE_SUITE_SHA256}" \
  "${CONFIG}" "${CONFIG_SHA256}" "${DUCA_ADATAD_PRETRAIN_SHA256}" \
  "${RUN_DIR}/launch_manifest.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from tools.bata.select_duca_boundary_burst_candidates import (
    _atomic_write_json,
    validate_frontend_decision,
    validate_full_model_gate,
)

(
    run_dir,
    variant,
    commit,
    checkpoint,
    evaluation,
    decision_sha,
    gate_sha,
    config,
    config_sha,
    pretrain_sha,
    launch_manifest,
) = sys.argv[1:]
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(payload):
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
metrics = json.loads(Path(evaluation).read_text(encoding="utf-8"))
if metrics.get("schema_version") != "duca_selected_axis_terminal_evaluation_v1":
    raise SystemExit("terminal evaluation schema mismatch")
unsigned_metrics = dict(metrics)
expected_self_hash = unsigned_metrics.pop("evaluation_sha256", None)
if expected_self_hash != canonical(unsigned_metrics):
    raise SystemExit("terminal evaluation self-hash mismatch")
if (
    metrics.get("git_commit") != commit
    or metrics.get("variant") != variant
    or metrics.get("seed") != 3407
    or Path(metrics.get("config_path", "")).resolve() != Path(config).resolve()
    or metrics.get("config_sha256") != config_sha
    or Path(metrics.get("checkpoint_path", "")).resolve() != Path(checkpoint).resolve()
    or metrics.get("checkpoint_sha256") != digest(checkpoint)
    or metrics.get("checkpoint_epoch") != 59
    or metrics.get("checkpoint_state_key") != "state_dict_ema"
):
    raise SystemExit("terminal evaluation identity mismatch")
identity = metrics.get("training_identity")
if (
    not isinstance(identity, dict)
    or identity.get("variant") != variant
    or identity.get("seed") != 3407
    or identity.get("successful_optimizer_updates") != 6000
    or identity.get("gate_suite_sha256") != gate_sha
    or identity.get("pretrain_sha256") != pretrain_sha
):
    raise SystemExit("terminal training identity mismatch")
for path_key, sha_key in (
    ("checkpoint_sidecar_path", "checkpoint_sidecar_sha256"),
    ("training_audit_path", "training_audit_sha256"),
):
    artifact = Path(identity.get(path_key, "")).resolve()
    if not artifact.is_file() or digest(artifact) != identity.get(sha_key):
        raise SystemExit(f"terminal training artifact drift: {path_key}")
prediction = Path(metrics.get("prediction_path", "")).resolve()
if not prediction.is_file() or digest(prediction) != metrics.get("prediction_sha256"):
    raise SystemExit("terminal prediction artifact drift")
launch_path = Path(launch_manifest).resolve()
launch = json.loads(launch_path.read_text(encoding="utf-8"))
if (
    launch.get("schema") != "duca_two_stage_curriculum_launch_v1"
    or launch.get("fail_closed") is not True
    or launch.get("git_commit") != commit
    or launch.get("variant") != variant
    or launch.get("config_sha256") != config_sha
    or launch.get("frontend_decision_sha256") != decision_sha
    or launch.get("gate_suite_sha256") != gate_sha
    or launch.get("execution_role") not in {"required_main", "optional_diagnostic"}
):
    raise SystemExit("launch manifest evidence drift")
alignment = launch.get("hard_swap_alignment")
if alignment is not None and identity.get("hard_swap_alignment") != alignment:
    raise SystemExit("hard-swap alignment training identity drift")
decision = validate_frontend_decision(
    decision_path=launch["frontend_decision_path"],
    decision_sha256=decision_sha,
    expected_commit=commit,
)
validate_full_model_gate(
    gate_path=launch["gate_path"],
    gate_sha256=gate_sha,
    decision_path=launch["frontend_decision_path"],
    decision_sha256=decision_sha,
    expected_commit=commit,
)
if (
    launch.get("family_manifest") != decision["family_manifest"]
    or launch.get("r0_headroom_gate") != decision["r0_headroom_gate"]
    or launch.get("family_routing") != decision["family_routing"]
    or launch.get("p0_training_asformer_consumer")
    != decision["p0_training_asformer_consumer"]
):
    raise SystemExit("completion propagation evidence drift")
payload = {
    "schema": "duca_two_stage_curriculum_completion_v1",
    "ok": True,
    "fail_closed": True,
    "git_commit": commit,
    "variant": variant,
    "execution_role": launch["execution_role"],
    "checkpoint_path": str(Path(checkpoint).resolve()),
    "checkpoint_sha256": digest(checkpoint),
    "evaluation_path": str(Path(evaluation).resolve()),
    "evaluation_sha256": digest(evaluation),
    "evaluation_self_sha256": expected_self_hash,
    "prediction_path": str(prediction),
    "prediction_sha256": metrics["prediction_sha256"],
    "metrics": metrics["metrics"],
    "training_identity": identity,
    "launch_manifest_path": str(launch_path),
    "launch_manifest_sha256": digest(launch_path),
    "frontend_decision_path": launch["frontend_decision_path"],
    "frontend_decision_sha256": decision_sha,
    "family_manifest": decision["family_manifest"],
    "r0_headroom_gate": decision["r0_headroom_gate"],
    "family_routing": decision["family_routing"],
    "p0_training_asformer_consumer": decision[
        "p0_training_asformer_consumer"
    ],
    "gate_path": launch["gate_path"],
    "gate_suite_sha256": gate_sha,
    "hard_swap_alignment": alignment,
}
_atomic_write_json(
    Path(run_dir, "completion.json").resolve(), payload, require_absent=True
)
PY

echo "[DUCA_TWO_STAGE_TRAIN] completed ${RUN_DIR}/completion.json"
