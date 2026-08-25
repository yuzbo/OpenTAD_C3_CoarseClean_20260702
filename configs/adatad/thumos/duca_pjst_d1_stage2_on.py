_base_ = ["./duca_sampling_rate_curriculum_stage2_joint384.py"]

# Matched PJST-D1 intervention arm.  This arm reuses the identical Stage-1
# learned H65 nonuniform selector acquisition (policy_alpha pinned to 1.0,
# never the exact-uniform fallback) and differs from the OFF arm ONLY in
# ``model.backbone.custom.pjst_derivative_only`` (True) and ``work_dir``.  All
# detector-to-selector and auxiliary-selector adaptation routes are frozen to
# zero so the two arms observe byte-identical positions/RGB/mask exposures.
model = dict(
    frame_selector=dict(
        loss_weight_schedule=dict(
            policy_alpha=dict(start=1.0, end=1.0),
            detector_gradient=dict(start=0.0, end=0.0),
            detector_contribution=dict(start=0.0, end=0.0),
            asformer_adapt=dict(start=0.0, end=0.0),
        ),
    ),
    backbone=dict(custom=dict(pjst_derivative_only=True)),
)

checkpoint_interval_epochs = 5
checkpoint_policy = dict(resumable=True, keep_latest=3, milestones=True, final=True, final_ema=True)
workflow = dict(require_resumable_training_state=True, checkpoint_interval=5)
work_dir = "exps/thumos/adatad/duca_pjst_d1_matched_on"
