_base_ = ["./duca_allocation_ceiling_validation_windows.py"]

import os


allocation_artifact_path = os.environ.get("DUCA_ALLOCATION_ARTIFACT_PATH", "")
allocation_artifact_sha256 = os.environ.get("DUCA_ALLOCATION_ARTIFACT_SHA256", "")
allocation_family_key = os.environ.get("DUCA_ALLOCATION_FAMILY_KEY", "D_deploy_score")
allocation_allow_privileged = os.environ.get(
    "DUCA_ALLOCATION_ALLOW_PRIVILEGED",
    "0",
) == "1"


allocation_ceiling_replay_contract = dict(
    task="offline_temporal_action_detection",
    execution="frozen_checkpoint_physical_grid_mAP_diagnostic",
    trains_model=False,
    artifact_path=allocation_artifact_path,
    artifact_sha256=allocation_artifact_sha256,
    family_key=allocation_family_key,
    privileged_family_allowed=allocation_allow_privileged,
    runtime_gt_input=False,
    selected_axis_gt_remap=False,
    physical_grid_actionformer=True,
    detector_output_coordinate_space="true_time_dense_index",
    paper_deployable=False,
)


model = dict(
    frame_selector=dict(
        type="DucaAllocationArtifactReplaySelector",
        artifact_path=allocation_artifact_path,
        artifact_sha256=allocation_artifact_sha256,
        family_key=allocation_family_key,
        allow_privileged_family=allocation_allow_privileged,
        remap_gt_to_selected_axis=False,
        selected_axis_remap_required=False,
        detector_output_coordinate_space="true_time_dense_index",
    ),
)


workflow = dict(formal_protocol="duca_allocation_ceiling_replay_v1")
post_processing = dict(save_dict=True)
inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
work_dir = "exps/thumos/adatad/duca_allocation_ceiling_physical_grid_replay"
