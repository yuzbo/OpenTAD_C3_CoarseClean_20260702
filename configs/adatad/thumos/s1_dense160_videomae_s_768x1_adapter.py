_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]

spatial_zoom_s1_contract = dict(
    schema_version="spatial_zoom_s1_config_v1",
    gate="S1_spatial_resolution_headroom",
    runtime_resolution=160,
    train_short_side=182,
    temporal_window=768,
    detector_time_grid=768,
    tubelet_points=384,
    fit_gate_manifest_required=True,
    official_test_sealed_until_protocol_freeze=True,
    checkpoint_selection_rule="max_gate_high_tiou_headroom_earliest_epoch_tie",
    training_seeds=[3407, 3408, 3409],
    roi_policy_enabled=False,
    teacher_oracle_enabled=False,
    new_detector_enabled=False,
    paper_claim_allowed=False,
)

work_dir = "exps/thumos/adatad/spatial_zoom_s1_dense160"
