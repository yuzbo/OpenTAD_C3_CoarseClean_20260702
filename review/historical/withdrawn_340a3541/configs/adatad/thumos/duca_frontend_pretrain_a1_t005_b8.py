_base_ = ["./duca_frontend_pretrain_fixed384_base.py"]


duca_transition_only_contract = dict(
    weight_variant="action_first",
    actionness_loss_weight=1.0,
    transition_distribution_loss_weight=0.05,
    transition_boundary_coverage_loss_weight=8.0,
)

model = dict(
    frame_selector=dict(
        loss_weights=dict(transition=0.05, transition_boundary=8.0),
        loss_weight_schedule=dict(
            transition=dict(end=0.05),
            transition_boundary=dict(end=8.0),
        ),
    ),
)

work_dir = "exps/thumos/adatad/duca_frontend_pretrain_a1_t005_b8"
