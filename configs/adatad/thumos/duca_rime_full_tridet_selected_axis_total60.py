_base_ = ["./duca_rime_full_tridet_total60.py"]


model = dict(
    frame_selector=dict(
        detector_coordinate_mode="selected_axis_plugin",
    ),
    rpn_head=dict(
        physical_grid_actionformer=None,
    ),
)

workflow = dict(
    formal_protocol="duca_rime_tridet_selected_axis_plugin_v2",
)

duca_rime_variant = dict(
    arm="RIME-full-TriDet-selected-axis",
    detector_coordinate_mode="selected_axis_plugin",
    dense_physical_training_axis=False,
    q_to_t_before_nms=True,
    paper_mainline_replication_candidate=True,
    phase4_submission_enabled=False,
    official_final_sealed=True,
    empirically_supported=False,
    paper_ready=False,
)

duca_rime_contract = dict(
    pre_backbone_plugin=True,
    detector_coordinate_mode="selected_axis_plugin",
    detector_output_coordinate_space="selected_axis_index",
    inverse_map_before_official_nms=True,
    gt_remapped_to_selected_axis=True,
    physical_head_enabled=False,
    detector_head_modified=False,
    paper_mainline_allowed=True,
    admission_schema="duca_acquisition_admission_v2",
    official_final_subset_consumed=False,
    empirically_supported=False,
    paper_ready=False,
)

work_dir = "exps/thumos/adatad/duca_rime_full_tridet_selected_axis_total60"
