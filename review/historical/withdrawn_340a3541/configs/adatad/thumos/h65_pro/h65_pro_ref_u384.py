_base_ = ["./base_h65_pro_strict60.py"]

h65_pro_experiment_id = "REF-U384"
h65_pro_factor_policy = dict(
    phase=False,
    ct=False,
    mod=False,
    taylor=False,
    curriculum=False,
    frames=384,
    reference="uniform_k384",
)

model = dict(
    frame_selector=dict(
        inference_policy_alpha=0.0,
        training_uniform_companion_fraction=0.0,
        training_uniform_companion_normalize_learned_gradient=False,
        sampling_rate_utility_components="none",
        detector_contribution_distillation_weight=0.0,
        detector_contribution_components="none",
        detector_contribution_mode="abs_grad_times_input",
        loss_weight_schedule=dict(
            _delete_=True,
            type="progressive_joint",
            shape="linear",
            warmup_steps=0,
            transition_steps=1,
            actionness=dict(start=0.0, end=0.0),
            transition=dict(start=0.0, end=0.0),
            transition_boundary=dict(start=0.0, end=0.0),
            detector_gradient=dict(start=0.0, end=0.0),
            policy_alpha=dict(start=0.0, end=0.0),
            detector_contribution=dict(start=0.0, end=0.0),
            asformer_adapt=dict(start=0.0, end=0.0),
        ),
    ),
    backbone=dict(backbone=dict(amod_config=dict(_delete_=True, enabled=False))),
    rpn_head=dict(conv_cfg=None),
)
work_dir = "exps/thumos/adatad/h65_pro_fullmatrix_20260902/ref_u384"
