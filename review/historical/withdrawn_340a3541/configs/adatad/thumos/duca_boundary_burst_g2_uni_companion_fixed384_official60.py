_base_ = ["./duca_boundary_burst_g1_protected_fixed384_official60.py"]


duca_transition_only_contract = dict(
    route="DUCA_BOUNDARY_BURST_R2Q3_G2_UNI_COMPANION_FIXED384_OFFICIAL60",
    optimization_variant="r0_selected_boundary_burst_with_uniform_companion",
    detector_gradient_updates="transition_scorer_only_on_learned_rows",
    training_uniform_companion_fraction=0.50,
    learned_row_detector_gradient_normalization="batch_size_over_learned_rows",
    detector_forward_passes_per_batch=1,
    inference_uses_learned_policy_only=True,
    inference_extra_companion_cost=False,
    hard_swap_alignment_required=True,
    paper_claim_allowed=False,
)


model = dict(
    frame_selector=dict(
        training_uniform_companion_fraction=0.50,
        training_uniform_companion_normalize_learned_gradient=True,
    ),
)


work_dir = (
    "exps/thumos/adatad/"
    "duca_boundary_burst_r2q3_g2_uni_companion_fixed384_official60"
)
