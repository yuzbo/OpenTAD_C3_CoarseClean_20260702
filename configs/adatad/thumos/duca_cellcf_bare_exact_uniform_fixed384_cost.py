_base_ = ["./duca_cellcf_fixed384_official_adatad_backend_full_train.py"]


duca_cellcf_cost_contract = dict(
    schema="duca_cellcf_bare_uniform_cost_v1",
    task="offline_temporal_action_detection",
    purpose="inference_cost_lower_bound_only",
    main_method=False,
    exact_uniform_definition="canonical_round_endpoint_half_even",
    dense_window_size=768,
    detector_input_size=384,
    builds_probe=False,
    builds_selector=False,
    builds_counterfactual_teacher=False,
    paper_accuracy_claim_allowed=False,
)


_train_pipeline = [
    dict(type="PrepareVideoInfo", format="mp4"),
    dict(type="mmaction.DecordInit", num_threads=4),
    dict(
        type="LoadFrames",
        num_clips=1,
        method="exact_uniform_fixed_subsample",
        method_base="random_trunc",
        source_len=768,
        target_len=384,
        trunc_thresh=0.75,
        crop_ratio=[0.9, 1.0],
        scale_factor=1,
        remap_gt_to_selected_axis=True,
    ),
    dict(type="mmaction.DecordDecode"),
    dict(type="mmaction.Resize", scale=(-1, 182)),
    dict(type="mmaction.RandomResizedCrop"),
    dict(type="mmaction.Resize", scale=(160, 160), keep_ratio=False),
    dict(type="mmaction.Flip", flip_ratio=0.5),
    dict(type="mmaction.ImgAug", transforms="default"),
    dict(type="mmaction.ColorJitter"),
    dict(type="mmaction.FormatShape", input_format="NCTHW"),
    dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
    dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
]

_test_pipeline = [
    dict(type="PrepareVideoInfo", format="mp4"),
    dict(type="mmaction.DecordInit", num_threads=4),
    dict(
        type="LoadFrames",
        num_clips=1,
        method="exact_uniform_fixed_subsample",
        method_base="sliding_window",
        target_len=384,
        scale_factor=1,
        remap_gt_to_selected_axis=True,
    ),
    dict(type="mmaction.DecordDecode"),
    dict(type="mmaction.Resize", scale=(-1, 160)),
    dict(type="mmaction.CenterCrop", crop_size=160),
    dict(type="mmaction.FormatShape", input_format="NCTHW"),
    dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
    dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
]

_inference_pipeline = [
    dict(type="PrepareVideoInfo", format="mp4"),
    dict(type="mmaction.DecordInit", num_threads=4),
    dict(
        type="LoadFrames",
        num_clips=1,
        method="exact_uniform_fixed_subsample",
        method_base="sliding_window",
        target_len=384,
        scale_factor=1,
        remap_gt_to_selected_axis=True,
    ),
    dict(type="mmaction.DecordDecode"),
    dict(type="mmaction.Resize", scale=(-1, 160)),
    dict(type="mmaction.CenterCrop", crop_size=160),
    dict(type="mmaction.FormatShape", input_format="NCTHW"),
    dict(type="ConvertToTensor", keys=["imgs"]),
    dict(type="Collect", inputs="imgs", keys=["masks"]),
]


dataset = dict(
    train=dict(pipeline=_train_pipeline),
    val=dict(window_size=768, pipeline=_test_pipeline),
    test=dict(window_size=768, pipeline=_inference_pipeline),
)


model = dict(frame_selector=None)


work_dir = "exps/thumos/adatad/duca_cellcf_bare_exact_uniform_fixed384_cost"
