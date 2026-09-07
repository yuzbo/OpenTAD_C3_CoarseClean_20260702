_base_ = ["./duca_frontend_pretrain_fixed384_base.py"]


duca_transition_only_contract = dict(
    weight_variant="fixed_a1_t010_b16",
    learning_rate_variant="coarse_first_c50_a100_s25",
    actionness_loss_weight=1.0,
    transition_distribution_loss_weight=0.10,
    transition_boundary_coverage_loss_weight=16.0,
)

model = dict(
    frame_selector=dict(
        coarse_trunk_lr=5.0e-5,
        action_head_lr=1.0e-4,
        transition_scorer_lr=2.5e-5,
    ),
)

work_dir = "exps/thumos/adatad/duca_frontend_pretrain_lr_coarse50_action100_scorer25"
