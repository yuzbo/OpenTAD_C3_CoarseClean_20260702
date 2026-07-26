_base_ = ["./duca_frontend_pretrain_fixed384_base.py"]


duca_transition_only_contract = dict(
    weight_variant="balanced_by_observed_loss_scale",
    actionness_loss_weight=1.0,
    transition_distribution_loss_weight=0.10,
    transition_boundary_coverage_loss_weight=16.0,
)

work_dir = "exps/thumos/adatad/duca_frontend_pretrain_a1_t010_b16"
