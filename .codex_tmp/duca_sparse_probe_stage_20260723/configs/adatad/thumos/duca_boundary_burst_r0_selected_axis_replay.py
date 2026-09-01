_base_ = [
    "./duca_cellcf_transition_beta0_fixed384_official_adatad_backend_full_train.py"
]

import os


artifact_path = os.environ.get("DUCA_ALLOCATION_ARTIFACT_PATH", "")
artifact_sha256 = os.environ.get("DUCA_ALLOCATION_ARTIFACT_SHA256", "")
family_key = os.environ.get("DUCA_ALLOCATION_FAMILY_KEY", "A_exact_uniform")
allow_privileged = os.environ.get("DUCA_ALLOCATION_ALLOW_PRIVILEGED", "0") == "1"
holdout_block_list = os.environ.get("DUCA_FRONTEND_HOLDOUT_BLOCK_LIST", "")
if not holdout_block_list:
    raise ValueError("DUCA_FRONTEND_HOLDOUT_BLOCK_LIST is required for R0 replay")
r0_eval_blocked_videos = os.environ.get("DUCA_R0_EVAL_BLOCKED_VIDEOS", "")
if not r0_eval_blocked_videos:
    raise ValueError("DUCA_R0_EVAL_BLOCKED_VIDEOS is required for R0 replay")

yuzibo_root = os.environ.get("YUZIBO_ROOT", os.path.expanduser("~/run/yuzibo"))
thumos14_root = os.path.join(yuzibo_root, "thumos14")
annotation_path = os.environ.get(
    "THUMOS14_ANNOTATION_PATH",
    os.path.join(thumos14_root, "annotations", "thumos_14_anno.json"),
)
class_map = os.environ.get(
    "THUMOS14_CLASS_MAP",
    os.path.join(thumos14_root, "annotations", "category_idx.txt"),
)
train_data_path = os.environ.get(
    "THUMOS14_TRAIN_DATA_PATH",
    os.path.join(thumos14_root, "train"),
)


duca_r0_replay_contract = dict(
    task="offline_temporal_action_detection",
    execution="frozen_checkpoint_selected_axis_train_holdout_map",
    model_training=False,
    runtime_gt_input=False,
    selector_uses_artifact_only=True,
    test_subset_consumed=False,
    detector_output_coordinate_space="selected_axis_index",
    prediction_inverse_map_required=True,
    paper_claim_allowed=False,
)


model = dict(
    frame_selector=dict(
        type="DucaAllocationArtifactReplaySelector",
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha256,
        family_key=family_key,
        allow_privileged_family=allow_privileged,
        remap_gt_to_selected_axis=True,
        selected_axis_remap_required=True,
        detector_output_coordinate_space="selected_axis_index",
    ),
)


dataset = dict(
    test=dict(
        _delete_=True,
        type="ThumosSlidingDataset",
        ann_file=annotation_path,
        subset_name="training",
        block_list=holdout_block_list,
        class_map=class_map,
        data_path=train_data_path,
        filter_gt=False,
        test_mode=True,
        feature_stride=4,
        sample_stride=1,
        window_size=768,
        window_overlap_ratio=0.5,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=1),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(
                type="Collect",
                inputs="imgs",
                keys=["masks"],
                meta_keys=[
                    "video_name",
                    "data_path",
                    "fps",
                    "avg_fps",
                    "duration",
                    "total_frames",
                    "snippet_stride",
                    "window_start_frame",
                    "frame_inds",
                    "window_size",
                    "offset_frames",
                ],
            ),
        ],
    ),
)


solver = dict(test=dict(batch_size=1, num_workers=2))
post_processing = dict(save_dict=True)
inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
evaluation = dict(
    subset="training",
    blocked_videos=r0_eval_blocked_videos,
)
workflow = dict(formal_protocol="duca_r0_selected_axis_holdout_replay_v1")
work_dir = "exps/thumos/adatad/duca_boundary_burst_r0_selected_axis_replay"
