_base_ = ["./duca_rime_physical_total60_base.py"]

model = dict(
    frame_selector=dict(
        rime_arm="rime_full",
        require_frozen_protocol=True,
    ),
)

duca_rime_variant = dict(
    arm="RIME-full",
    dynamic_budget=True,
    pair_risk=True,
    allocation="frozen_per_video_dual",
    inference_batch_invariant=True,
)

work_dir = "exps/thumos/adatad/duca_rime_full_total60"
