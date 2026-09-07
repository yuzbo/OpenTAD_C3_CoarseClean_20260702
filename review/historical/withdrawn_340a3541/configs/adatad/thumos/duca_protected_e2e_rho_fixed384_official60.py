_base_ = ["./duca_protected_e2e_fixed384_official60.py"]


duca_policy_hidden_gradient_scale = 0.05

duca_transition_only_contract = dict(
    route="DUCA_PROTECTED_E2E_RHO_FIXED384_OFFICIAL60",
    detector_gradient_updates=(
        "transition_scorer_and_official_asformer_last_encoder_layer_only"
    ),
    policy_hidden_gradient_scale=duca_policy_hidden_gradient_scale,
    policy_hidden_gradient_scope="asformer_last_encoder_layer",
    asformer_trunk_detector_gradient=True,
    action_head_detector_gradient=False,
    earlier_asformer_detector_gradient=False,
)

model = dict(
    frame_selector=dict(
        policy_hidden_gradient_scale=duca_policy_hidden_gradient_scale,
        actionness_source_cfg=dict(
            policy_hidden_gradient_scope="asformer_last_encoder_layer",
        ),
    ),
)

work_dir = "exps/thumos/adatad/duca_protected_e2e_rho_fixed384_official60"
