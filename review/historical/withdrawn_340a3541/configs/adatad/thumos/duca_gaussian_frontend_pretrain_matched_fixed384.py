_base_ = ["./duca_frontend_pretrain_fixed384_base.py"]


duca_transition_only_contract = dict(
    route="DUCA_GAUSSIAN_FRONTEND_PRETRAIN_MATCHED_FIXED384",
    transition_objective="legacy_gaussian_mass_control",
    detector_executed=False,
    paper_claim_allowed=False,
)


model = dict(
    frame_selector=dict(
        transition_objective="gaussian_mass",
        coarse_trunk_lr=5.0e-5,
        action_head_lr=1.0e-4,
        transition_scorer_lr=1.0e-4,
        loss_weights=dict(
            transition=0.10,
            transition_boundary=2.0,
        ),
        loss_weight_schedule=dict(
            transition=dict(end=0.10),
            transition_boundary=dict(end=2.0),
        ),
    ),
)


work_dir = "exps/thumos/adatad/duca_gaussian_frontend_pretrain_matched_fixed384"
