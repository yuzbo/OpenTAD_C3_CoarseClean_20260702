_base_ = ["./duca_trainfree_mobilenet_fusion_r2q3_fixed384_official60.py"]


duca_trainfree_contract = dict(
    encoder="kinetics_pretrained_slowfast_r50_fast_pathway_only",
    efficiency_role="high_cost_frozen_video_prior_diagnostic",
    slow_path_executed=False,
    lateral_fusion_executed=False,
)


model = dict(
    frame_selector=dict(
        actionness_source_cfg=dict(
            source_name="frozen_kinetics_slowfast_r50_fast_transition_fusion",
            probe_model="slowfast-fast",
            slowfast_fast_pretrained=True,
            temporal_probe_stride=4,
            training_dataset="Kinetics-400",
        ),
    ),
)


work_dir = "exps/thumos/adatad/duca_trainfree_slowfast_fast_fusion_r2q3_fixed384_official60"
