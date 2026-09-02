_base_ = ["./base_h65_pro_strict60.py"]

h65_pro_experiment_id = "F07"
h65_pro_factor_policy = dict(
    phase=False,
    ct=True,
    mod=True,
    taylor=False,
    curriculum=True,
    generator="Resolution-V" if "F07".startswith("F") else "canonical",
    fixed_relation="E=ABCD",
)

model = dict(
    frame_selector=dict(
        acquisition_policy="budget_calibrated_sampling_rate",
        detector_contribution_mode="abs_grad_times_input",
        loss_weight_schedule=dict(
        _delete_=True,
        type="progressive_joint",
        shape="cosine",
        warmup_steps=0,
        transition_steps=1,
        actionness=dict(start=1.0, end=1.0),
        transition=dict(start=0.5, end=0.5),
        transition_boundary=dict(start=0.0, end=0.25, warmup_steps=1500, transition_steps=2000),
        detector_gradient=dict(start=0.0, end=0.25, warmup_steps=1500, transition_steps=2000),
        policy_alpha=dict(start=0.0, end=1.0, warmup_steps=1500, transition_steps=2000),
        detector_contribution=dict(start=0.0, end=1.0, warmup_steps=1500, transition_steps=2000),
        asformer_adapt=dict(start=0.0, end=1.0, warmup_steps=1500, transition_steps=2000),
    )
    ),
    backbone=dict(
        backbone=dict(
            amod_config=dict(
        _delete_=True,
        enabled=True,
        name="Mixture-of-Depths",
        amod_layers=[1, 3, 5, 7, 9, 11],
        capacity_schedule=dict(
            warmup_steps=1500,
            transition_steps=2000,
            start_capacity=1.0,
            end_capacity=0.5,
        ),
        min_capacity=0.5,
    ),
        ),
    ),
    rpn_head=dict(
        conv_cfg=dict(
    _delete_=True,
    type="ContinuousTimeScaleAdaptiveConv1d",
    local_ref_delta_t=2.0,
    context_ref_delta_t=2.0,
    context_level_base=2.0,
    max_abs_offset=64.0,
),
    ),
)

work_dir = "exps/thumos/adatad/h65_pro_fullmatrix_20260902/f07"
