_base_ = ["./duca_rime_physical_total60_base.py"]


# Paper-mainline coordinate boundary: DUCA-RIME changes only acquisition before
# the heavy backbone.  The detector receives a compact selected-axis sequence,
# uses its standard head/loss/decode, and maps proposals back before official NMS.
model = dict(
    frame_selector=dict(
        detector_coordinate_mode="selected_axis_plugin",
    ),
    rpn_head=dict(
        physical_grid_actionformer=None,
    ),
)

workflow = dict(
    formal_protocol="duca_rime_selected_axis_plugin_v2",
)

duca_rime_contract = dict(
    pre_backbone_plugin=True,
    detector_coordinate_mode="selected_axis_plugin",
    detector_output_coordinate_space="selected_axis_index",
    inverse_map_before_official_nms=True,
    gt_remapped_to_selected_axis=True,
    physical_head_enabled=False,
    detector_head_modified=False,
    integration_scope="pure_pre_backbone_acquisition",
    paper_mainline_allowed=True,
    admission_schema="duca_acquisition_admission_v2",
    official_final_subset_consumed=False,
    empirically_supported=False,
    paper_ready=False,
)

work_dir = "exps/thumos/adatad/duca_rime_selected_axis_base_do_not_run"
