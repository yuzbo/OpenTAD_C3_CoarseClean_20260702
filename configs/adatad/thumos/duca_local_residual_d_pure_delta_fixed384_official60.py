_base_ = ["./duca_local_residual_fixed384_official60_base.py"]


duca_transition_only_contract = dict(
    route="DUCA_LOCAL_RESIDUAL_D_PURE_DELTA_FIXED384_OFFICIAL60",
    optimization_variant="detached_abs_delta_only",
    residual_policy="disabled",
    detector_gradient_updates="none",
)


model = dict(
    frame_selector=dict(
        local_cell_residual_scale=0.0,
        detector_gradient_mode="none",
        inference_policy_alpha=1.0,
        loss_weight_schedule=dict(
            transition=dict(start=0.0, end=0.0),
            transition_boundary=dict(start=0.0, end=0.0),
            detector_gradient=dict(start=0.0, end=0.0),
        ),
    ),
)


work_dir = "exps/thumos/adatad/duca_local_residual_d_pure_delta_fixed384_official60"
