_base_ = ["./duca_local_residual_fixed384_official60_base.py"]


duca_transition_only_contract = dict(
    route="DUCA_LOCAL_RESIDUAL_U_EXACT_UNIFORM_FIXED384_OFFICIAL60",
    optimization_variant="exact_uniform_control",
    base_policy="none",
    residual_policy="disabled",
    detector_gradient_updates="none",
)


model = dict(
    frame_selector=dict(
        local_cell_force_exact_uniform=True,
        local_cell_residual_scale=0.0,
        detector_gradient_mode="none",
        inference_policy_alpha=0.0,
        loss_weight_schedule=dict(
            transition=dict(start=0.0, end=0.0),
            transition_boundary=dict(start=0.0, end=0.0),
            policy_alpha=dict(
                start=0.0,
                end=0.0,
                warmup_steps=0,
                transition_steps=0,
            ),
            detector_gradient=dict(
                start=0.0,
                end=0.0,
                warmup_steps=0,
                transition_steps=0,
            ),
        ),
    ),
)


work_dir = "exps/thumos/adatad/duca_local_residual_u_exact_uniform_fixed384_official60"
