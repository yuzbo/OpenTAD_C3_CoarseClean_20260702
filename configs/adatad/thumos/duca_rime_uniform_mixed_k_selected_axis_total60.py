_base_ = ["./duca_rime_uniform_mixed_k_total60.py"]


model = dict(
    frame_selector=dict(
        detector_coordinate_mode="selected_axis_plugin",
    ),
    rpn_head=dict(
        physical_grid_actionformer=None,
    ),
)

workflow = dict(
    formal_protocol="duca_rime_phase2_mixed_k_selected_axis_v2",
)

duca_rime_variant = dict(
    arm="U-mixed-K-selected-axis",
    paper_mainline_control=True,
    detector_coordinate_mode="selected_axis_plugin",
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

work_dir = (
    "exps/thumos/adatad/duca_rime_uniform_mixed_k_selected_axis_total60"
)
