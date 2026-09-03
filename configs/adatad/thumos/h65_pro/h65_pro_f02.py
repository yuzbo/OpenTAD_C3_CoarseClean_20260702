_base_ = ["./base_h65_pro_strict60.py"]

h65_pro_experiment_id = "F02"
h65_pro_factor_policy = dict(
    phase=False,
    ct=False,
    mod=False,
    taylor=True,
    curriculum=False,
    generator="Resolution-V" if "F02".startswith("F") else "canonical",
    fixed_relation="E=ABCD",
)

model = dict(
    frame_selector=dict(
        acquisition_policy="budget_calibrated_sampling_rate",
        detector_contribution_mode="signed_removal_utility",
        loss_weight_schedule=dict(
        _delete_=True,
        type="progressive_joint",
        shape="linear",
        warmup_steps=0,
        transition_steps=3000,
        actionness=dict(start=1.0, end=1.0),
        transition=dict(start=0.5, end=0.5),
        transition_boundary=dict(start=0.0, end=0.25, warmup_steps=0, transition_steps=3000),
        detector_gradient=dict(start=0.0, end=0.25, warmup_steps=2100, transition_steps=1500),
        policy_alpha=dict(start=0.0, end=1.0, warmup_steps=0, transition_steps=3000),
        detector_contribution=dict(start=0.0, end=1.0, warmup_steps=1500, transition_steps=900),
        asformer_adapt=dict(start=0.0, end=1.0, warmup_steps=1500, transition_steps=900),
    )
    ),
    backbone=dict(
        backbone=dict(
            amod_config=dict(_delete_=True, enabled=False),
        ),
    ),
    rpn_head=dict(
        conv_cfg=None,
    ),
)

work_dir = "exps/thumos/adatad/h65_pro_fullmatrix_20260902/f02"
