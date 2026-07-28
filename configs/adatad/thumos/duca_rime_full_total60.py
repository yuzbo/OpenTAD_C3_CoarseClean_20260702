_base_ = ["./duca_rime_physical_total60_base.py"]

import os

_formal_target = float(os.environ.get("DUCA_RIME_TARGET_MEAN_COST", "384"))
_dynamic_panel = _formal_target > 192.0

model = dict(
    frame_selector=dict(
        rime_arm="rime_full",
        require_frozen_protocol=True,
    ),
)

duca_rime_variant = dict(
    arm="RIME-full",
    dynamic_budget=_dynamic_panel,
    pair_risk=True,
    pair_risk_used_for_allocation=_dynamic_panel,
    allocation=(
        "frozen_per_video_dual"
        if _dynamic_panel
        else "fixed_floor_budget_position_only"
    ),
    budget_panel_semantics=(
        "content_conditioned_dynamic_budget_panel"
        if _dynamic_panel
        else "exact_k192_learned_position_stress_panel"
    ),
    inference_batch_invariant=True,
)

work_dir = "exps/thumos/adatad/duca_rime_full_total60"

del _formal_target, _dynamic_panel
