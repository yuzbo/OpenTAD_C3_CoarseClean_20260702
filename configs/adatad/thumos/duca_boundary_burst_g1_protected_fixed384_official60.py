_base_ = ["./duca_boundary_burst_g0_no_feedback_fixed384_official60.py"]


duca_transition_only_contract = dict(
    route="DUCA_BOUNDARY_BURST_R2Q3_G1_PROTECTED_FIXED384_OFFICIAL60",
    optimization_variant="r0_selected_boundary_burst_with_protected_detector_feedback",
    detector_gradient_bridge="protected_structured_transport",
    detector_gradient_final_weight=0.25,
    detector_gradient_updates="transition_scorer_only",
    hard_swap_alignment_required=True,
    paper_claim_allowed=False,
)


model = dict(
    frame_selector=dict(
        detector_gradient_mode="protected_structured_transport",
        training_uniform_companion_fraction=0.0,
        training_uniform_companion_normalize_learned_gradient=False,
        loss_weight_schedule=dict(
            detector_gradient=dict(
                start=0.0,
                end=0.25,
            ),
        ),
    ),
)


work_dir = (
    "exps/thumos/adatad/"
    "duca_boundary_burst_r2q3_g1_protected_fixed384_official60"
)
