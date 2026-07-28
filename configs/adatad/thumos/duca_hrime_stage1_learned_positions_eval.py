_base_ = ["./duca_rime_physical_total60_base.py"]

import os


if not os.environ.get("DUCA_RIME_REPLAY_JSONL", "").strip():
    raise RuntimeError("H-RIME Stage-1 learned positions require a frozen replay")
if os.environ.get("DUCA_RIME_ALLOW_ORACLE_REPLAY", "").strip().lower() not in {
    "1",
    "true",
}:
    raise RuntimeError("H-RIME Stage-1 requires explicit oracle replay permission")
decision_role = os.environ.get("HRIME_STAGE1_DECISION_ROLE", "").strip()
learned_role_uses_gt = {
    "hrime_stage1_independent_exact_total": False,
    "hrime_stage1_joint_oracle": True,
    "hrime_stage1_shuffled_null": True,
}
if decision_role not in learned_role_uses_gt:
    raise RuntimeError(
        "H-RIME Stage-1 learned positions require a registered decision role"
    )

model = dict(
    frame_selector=dict(
        rime_arm="hrime_stage1_learned_positions",
        require_frozen_protocol=True,
        allow_oracle_replay=True,
    ),
)

workflow = dict(
    formal_protocol="duca_rime_physical_dynamic_k_v1",
    evaluation_protocol="hrime_stage1_oracle_execution_v1",
)

hrime_stage1_execution_contract = dict(
    protocol="hrime_stage1_oracle_execution_v1",
    decision_role=decision_role,
    position_policy="frozen_rime_selector",
    uses_gt_at_decision=learned_role_uses_gt[decision_role],
    runtime_gt_input_to_selector=False,
    oracle_only=True,
    evaluation_only=True,
    deployment_candidate=False,
    uses_official_final=False,
)

duca_rime_variant = dict(
    arm="H-RIME-Stage1-Learned-Positions",
    trainable=False,
    evaluation_only=True,
    oracle_only=True,
    deployment_candidate=False,
    positions="frozen_rime_selector",
    per_window_k="hash_bound_same_total_video_replay",
)

duca_rime_contract = dict(
    official_final_subset_consumed=False,
    empirically_supported=False,
    paper_ready=False,
    oracle_only=True,
    deployment_candidate=False,
)

work_dir = "exps/thumos/adatad/duca_hrime_stage1_learned_positions_eval"

del decision_role, learned_role_uses_gt
