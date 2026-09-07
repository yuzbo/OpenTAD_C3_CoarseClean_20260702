_base_ = ["./duca_trainfree_fixed384_official60_base.py"]


duca_trainfree_contract = dict(
    evidence="fixed_075_feature_change_020_semantic_005_uncertainty_peak",
    allocation="parameter_free_r2q3_boundary_burst",
)


model = dict(
    frame_selector=dict(
        transition_objective="boundary_burst",
        transition_boundary_radius=2,
        boundary_burst_quota=3.0,
        boundary_burst_budget_fraction=0.25,
        boundary_burst_context_weight=0.05,
        boundary_burst_center_temperature=0.7,
        boundary_burst_offset_temperature=1.0,
        boundary_burst_require_bilateral_offsets=True,
        boundary_burst_require_global_mandatory_groups=True,
        actionness_source_cfg=dict(
            source_name="frozen_imagenet_mobilenetv3_transition_fusion",
            train_free_evidence_mode="frozen_transition_fusion",
        ),
    ),
)


work_dir = "exps/thumos/adatad/duca_trainfree_mobilenet_fusion_r2q3_fixed384_official60"
