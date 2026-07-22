_base_ = ["./duca_boundary_burst_frontend_pretrain_fixed384.py"]


duca_transition_only_contract = dict(
    route="DUCA_BOUNDARY_BURST_HARD_DETACHED_FRONTEND_PRETRAIN_FIXED384",
    transition_supervision_updates_coarse_representation=False,
    auxiliary_hidden_gradient_scale=0.0,
    transition_distribution_updates="detached",
    boundary_coverage_updates="detached",
    policy_hidden_gradient_scale=0.0,
    paper_claim_allowed=False,
)


model = dict(
    frame_selector=dict(
        boundary_burst_require_bilateral_offsets=True,
        boundary_burst_require_global_mandatory_groups=True,
        auxiliary_hidden_gradient_scale=0.0,
        policy_hidden_gradient_scale=0.0,
        actionness_source_cfg=dict(policy_hidden_gradient_scope="none"),
    ),
)


work_dir = (
    "exps/thumos/adatad/"
    "duca_boundary_burst_hard_detached_frontend_pretrain_fixed384"
)
