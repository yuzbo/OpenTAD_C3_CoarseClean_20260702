_base_ = ["./duca_rime_physical_total60_base.py"]

model = dict(
    frame_selector=dict(
        rime_arm="dynamic_no_risk",
        require_frozen_protocol=True,
    ),
)

duca_rime_variant = dict(
    arm="D-no-risk",
    dynamic_budget=True,
    pair_risk=False,
    allocation="frozen_per_video_dual",
)

work_dir = "exps/thumos/adatad/duca_rime_dynamic_no_risk_total60"
