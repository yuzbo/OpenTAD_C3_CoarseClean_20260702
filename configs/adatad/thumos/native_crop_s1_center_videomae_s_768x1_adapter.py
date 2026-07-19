_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]

window_size = 768
scale_factor = 1

native_crop_train_pipeline = [
    dict(type="PrepareVideoInfo", format="mp4"),
    dict(type="mmaction.DecordInit", num_threads=4),
    dict(
        type="LoadFrames",
        num_clips=1,
        method="random_trunc",
        trunc_len=window_size,
        trunc_thresh=0.75,
        crop_ratio=[0.9, 1.0],
        scale_factor=scale_factor,
    ),
    dict(type="mmaction.DecordDecode"),
    dict(
        type="NativeCropSourceViews",
        global_size=96,
        local_size=128,
        output_key="native_crop_inputs",
        allow_local_padding=False,
    ),
    dict(type="ConvertToTensor", keys=["gt_segments", "gt_labels"]),
    dict(
        type="Collect",
        inputs="native_crop_inputs",
        keys=["masks", "gt_segments", "gt_labels"],
    ),
]

native_crop_val_pipeline = [
    dict(type="PrepareVideoInfo", format="mp4"),
    dict(type="mmaction.DecordInit", num_threads=4),
    dict(
        type="LoadFrames",
        num_clips=1,
        method="sliding_window",
        scale_factor=scale_factor,
    ),
    dict(type="mmaction.DecordDecode"),
    dict(
        type="NativeCropSourceViews",
        global_size=96,
        local_size=128,
        output_key="native_crop_inputs",
        allow_local_padding=False,
    ),
    dict(type="ConvertToTensor", keys=["gt_segments", "gt_labels"]),
    dict(
        type="Collect",
        inputs="native_crop_inputs",
        keys=["masks", "gt_segments", "gt_labels"],
    ),
]

dataset = dict(
    train=dict(
        subset_name="training",
        pipeline=native_crop_train_pipeline,
    ),
    val=dict(
        subset_name="training",
        window_size=window_size,
        # The 0.5 overlap is required to cover the final short action in
        # video_validation_0000054; the inherited 0.25 path drops that video.
        window_overlap_ratio=0.5,
        pipeline=native_crop_val_pipeline,
    ),
    # The vertical slice is development-only. A future formal config must be
    # separately reviewed before any official-test dataset is materialized.
    test=None,
)

model = dict(
    backbone=dict(
        custom=dict(
            wrapper_type="native_crop_shared_videomae",
            native_crop_global_key="global",
            native_crop_local_key="local",
            native_crop_global_size=96,
            native_crop_local_size=128,
            native_crop_chunk_num=48,
            native_crop_intermediate_length=384,
            native_crop_output_length=window_size,
            native_crop_fusion_mode="fixed_mean",
        )
    )
)

native_crop_s1_gate = dict(
    route="spatial-zoom-native-crop-s1",
    stage="development-only-no-training-vertical-slice",
    precheck_only=True,
    allow_detector_training=False,
    allow_tools_train=False,
    allow_tools_test=False,
    allow_detector_map=False,
    allowed_entrypoints=[
        "tools/bata/run_native_crop_s1_precheck.py",
        "tools/bata/native_crop_s1_geometry_census.py",
    ],
    allowed_checks=[
        "development geometry census",
        "source-pixel crop tests",
        "synthetic full-model forward/backward",
        "cost-schema smoke",
    ],
    official_test_open_allowed=False,
    teacher_allowed=False,
    oracle_allowed=False,
    learned_crop_policy_allowed=False,
    paper_claim_allowed=False,
)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)

work_dir = "exps/thumos/adatad/native_crop_s1_center_videomae_s_768x1_adapter"
