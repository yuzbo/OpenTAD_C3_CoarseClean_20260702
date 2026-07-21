_base_ = ["./duca_frontend_pretrain_fixed384_base.py"]


duca_transition_only_contract = dict(
    weight_variant="transition_boundary_strong",
    actionness_loss_weight=1.0,
    transition_distribution_loss_weight=0.20,
    transition_boundary_coverage_loss_weight=32.0,
)

model = dict(
    frame_selector=dict(
        loss_weights=dict(transition=0.20, transition_boundary=32.0),
        loss_weight_schedule=dict(
            transition=dict(end=0.20),
            transition_boundary=dict(end=32.0),
        ),
    ),
)

work_dir = "exps/thumos/adatad/duca_frontend_pretrain_a1_t020_b32"
