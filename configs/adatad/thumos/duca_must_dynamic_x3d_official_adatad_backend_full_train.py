_base_ = ["./duca_must_dynamic_official_adatad_backend_full_train.py"]

import os


def _env_int(name, default):
    value = os.environ.get(name, default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


yuzibo_root = os.environ.get("YUZIBO_ROOT", os.path.expanduser("~/run/yuzibo"))
dense_window_size = _env_int("DUCA_MUST_DENSE_WINDOW_SIZE", 768)
scale_factor = 1
duca_x3d_actionness_jsonl = os.environ.get(
    "DUCA_X3D_ACTIONNESS_JSONL",
    os.path.join(
        yuzibo_root,
        "projects",
        "c3_lowres_action_probe",
        "trainfree_frozen_actionness",
        "best_x3d_actionness.jsonl",
    ),
)

duca_x3d_external_meta_keys = [
    "video_name",
    "data_path",
    "fps",
    "duration",
    "snippet_stride",
    "window_start_frame",
    "resize_length",
    "window_size",
    "offset_frames",
    "irregular_selected_positions",
    "irregular_selected_valid_len",
    "irregular_native_axis",
    "selected_dense_indices",
    "selected_valid_len",
    "irregular_dense_valid_len",
    "remap_gt_to_selected_axis",
    "gt_remapped_to_selected_axis",
    "pc_ot_mras_prebackbone_remap_gt_to_selected_axis",
    "duca_external_p_action",
    "duca_external_actionness_logits",
    "duca_external_actionness_valid",
    "duca_external_actionness_provenance",
    "duca_external_actionness_source",
    "duca_external_actionness_observation_times",
    "duca_external_actionness_jsonl",
]

duca_must_dynamic_contract = dict(
    stage="duca_must_dynamic_x3d_official_adatad_backend_full_train",
    detector_stack="official_OpenTAD_AdaTAD_VideoMAE-S_ActionFormerHead_plus_DUCA_MUST_prebackbone_plugin_plus_X3D_JSONL_actionness",
    external_actionness_source="train_free_x3d_jsonl",
    requires_external_actionness=True,
    train_free_actionness=True,
    x3d_downstream_detector_full_train=True,
    x3d_actionness_jsonl=duca_x3d_actionness_jsonl,
    uses_offline_deploy_selection_ledger=False,
    deploy_claim_allowed=False,
    metric_claim_allowed=False,
    paper_claim_allowed=False,
)

dataset = dict(
    train=dict(
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="random_trunc",
                trunc_len=dense_window_size,
                trunc_thresh=0.75,
                crop_ratio=[0.9, 1.0],
                scale_factor=scale_factor,
            ),
            dict(type="DucaExternalActionnessFromJsonl", actionness_jsonl=duca_x3d_actionness_jsonl),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 182)),
            dict(type="mmaction.RandomResizedCrop"),
            dict(type="mmaction.Resize", scale=(160, 160), keep_ratio=False),
            dict(type="mmaction.Flip", flip_ratio=0.5),
            dict(type="mmaction.ImgAug", transforms="default"),
            dict(type="mmaction.ColorJitter"),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"], meta_keys=duca_x3d_external_meta_keys),
        ],
    ),
    val=dict(
        window_size=dense_window_size,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=scale_factor),
            dict(type="DucaExternalActionnessFromJsonl", actionness_jsonl=duca_x3d_actionness_jsonl),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"], meta_keys=duca_x3d_external_meta_keys),
        ],
    ),
    test=dict(
        window_size=dense_window_size,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=scale_factor),
            dict(type="DucaExternalActionnessFromJsonl", actionness_jsonl=duca_x3d_actionness_jsonl),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(type="Collect", inputs="imgs", keys=["masks"], meta_keys=duca_x3d_external_meta_keys),
        ],
    ),
)

model = dict(
    frame_selector=dict(
        external_actionness_meta_key="duca_external_p_action",
        external_actionness_logits_meta_key="duca_external_actionness_logits",
        external_actionness_provenance_meta_key="duca_external_actionness_provenance",
        external_actionness_source_meta_key="duca_external_actionness_source",
        require_external_actionness=True,
        actionness_source_cfg=dict(
            type="ZeroShotActionnessSource",
            source_name="train_free_x3d_jsonl_actionness",
            mode="motion",
            thumos_trained=False,
            uses_labels=False,
            uses_teacher=False,
            uses_gt=False,
            uses_prediction_cache=False,
            no_train_gt=True,
            no_teacher=True,
            no_oracle=True,
            no_raw_prediction_cache=True,
            no_gt_generation=True,
            calibration_split="none",
            checkpoint_hash="external_jsonl_validated_at_pipeline",
        ),
    ),
)

work_dir = "exps/thumos/adatad/duca_must_dynamic_x3d_official_adatad_backend_full_train"
