_base_ = ["./duca_local_residual_fixed384_official60_base.py"]


duca_transition_only_contract = dict(
    route="DUCA_LOCAL_RESIDUAL_R1_FEEDBACK_FIXED384_OFFICIAL60",
    optimization_variant="bounded_residual_with_local_cell_detector_feedback",
    detector_gradient_updates="residual_scorer_only",
    detector_gradient_final_weight=0.25,
)


model = dict(
    frame_selector=dict(
        local_cell_residual_scale=0.25,
        detector_gradient_mode="local_cell_straight_through",
        inference_policy_alpha=1.0,
        loss_weight_schedule=dict(
            detector_gradient=dict(
                start=0.0,
                end=0.25,
                warmup_steps=2500,
                transition_steps=1500,
            ),
        ),
    ),
)


work_dir = "exps/thumos/adatad/duca_local_residual_r1_feedback_fixed384_official60"
