_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]

import os


VARIANT_SPECS = {
    "detector_aware_fixed_384": dict(
        target_len=384,
        require_selected_count=384,
        strategy="detector_aware_fixed_384",
        ledger_name="detector_aware_fixed_384",
        source="learned_detector_aware_policy_checkpoint",
        config_hash="stage2_detector_aware_fixed384_no_uniform_v1",
    ),
    "detector_aware_fixed_768": dict(
        target_len=768,
        require_selected_count=768,
        strategy="detector_aware_fixed_768",
        ledger_name="detector_aware_fixed_768",
        source="learned_detector_aware_policy_checkpoint",
        config_hash="stage2_detector_aware_fixed768_no_uniform_v1",
    ),
    "detector_aware_dynamic": dict(
        target_len=768,
        require_selected_count=None,
        strategy="detector_aware_dynamic",
        ledger_name="detector_aware_dynamic",
        source="learned_detector_aware_policy_checkpoint",
        config_hash="stage2_detector_aware_dynamic_no_uniform_v1",
    ),
}


detector_aware_ledger_variant = os.environ.get("C3_DETECTOR_AWARE_LEDGER_VARIANT", "detector_aware_fixed_384")
if detector_aware_ledger_variant not in VARIANT_SPECS:
    raise ValueError(f"unknown C3_DETECTOR_AWARE_LEDGER_VARIANT={detector_aware_ledger_variant}")
_variant = VARIANT_SPECS[detector_aware_ledger_variant]

yuzibo_root = os.environ.get("YUZIBO_ROOT", os.path.expanduser("~/run/yuzibo"))
thumos14_root = os.path.join(yuzibo_root, "thumos14")
detector_aware_ledger_root = os.environ.get(
    "C3_DETECTOR_AWARE_LEDGER_ROOT",
    os.path.join(yuzibo_root, "projects/c3_lowres_action_probe/detector_aware_ledgers"),
)

annotation_path = os.environ.get("THUMOS14_ANNOTATION_PATH", os.path.join(thumos14_root, "annotations", "thumos_14_anno.json"))
class_map = os.environ.get("THUMOS14_CLASS_MAP", os.path.join(thumos14_root, "annotations", "category_idx.txt"))
train_data_path = os.environ.get("THUMOS14_TRAIN_DATA_PATH", os.path.join(thumos14_root, "train"))
test_data_path = os.environ.get("THUMOS14_TEST_DATA_PATH", os.path.join(thumos14_root, "test"))
adatad_pretrain_filename = "vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
c3_detector_aware_adatad_pretrain_path = os.environ.get(
    "C3_DETECTOR_AWARE_ADATAD_PRETRAIN_PATH",
    os.path.join(yuzibo_root, "pretrained", adatad_pretrain_filename),
)

_ledger_name = _variant["ledger_name"]
train_ledger_path = os.environ.get(
    "C3_DETECTOR_AWARE_TRAIN_LEDGER_PATH",
    os.path.join(detector_aware_ledger_root, "train", f"value_transport_ledger_{_ledger_name}.jsonl"),
)
val_ledger_path = os.environ.get(
    "C3_DETECTOR_AWARE_VAL_LEDGER_PATH",
    os.path.join(detector_aware_ledger_root, "val", f"value_transport_ledger_{_ledger_name}.jsonl"),
)
test_ledger_path = os.environ.get(
    "C3_DETECTOR_AWARE_TEST_LEDGER_PATH",
    os.path.join(detector_aware_ledger_root, "test", f"value_transport_ledger_{_ledger_name}.jsonl"),
)
c3_value_transport_source = os.environ.get("C3_DETECTOR_AWARE_LEDGER_SOURCE", _variant["source"])
c3_value_transport_config_hash = os.environ.get("C3_DETECTOR_AWARE_LEDGER_CONFIG_HASH", _variant["config_hash"])

window_size = int(_variant["target_len"])
dense_window_size = 768
scale_factor = 1
chunk_num = window_size * scale_factor // 16
detector_aware_ledger_strategy = _variant["strategy"]
detector_aware_require_selected_count = _variant["require_selected_count"]

baseline_comparison = dict(
    question="Can dense AdaTAD teacher utility train an acquisition policy better than p_action-only?",
    matched_budget_baselines=["p_action_only", "GAS-VT"],
    variants=["fixed_384", "fixed_768", "dynamic"],
    decision_metrics=[
        "detector_utility_coverage",
        "detector_utility_ndcg",
        "boundary_bracket_support",
        "action_interior_bin_coverage",
        "max_unselected_hole",
        "p95_unselected_hole",
        "mean_uniform_similarity",
        "AdaTAD_mAP_after_full_train",
    ],
    full_detector_map_required_for_claim=True,
)

experiment_scope = dict(
    route="C3_STAGE2_DIVERGENT_ROUTE",
    route_variant="DIVERGENT_INNOVATION_DETECTOR_AWARE_UTILITY_DO_NOT_MERGE_WITH_C3",
    stage="Stage-2 detector-aware offline selector",
    selector_source="learned_detector_aware_policy_checkpoint",
    selection_strategy=detector_aware_ledger_strategy,
    detector_aware_ledger_variant=detector_aware_ledger_variant,
    detector_stack="original_adatad_actionformer_adapter",
    changes_input_sampling=True,
    changes_detector_head=False,
    changes_neck=False,
    changes_loss_assignment=False,
    changes_post_processing=False,
    uses_offline_deploy_selection_ledger=True,
    uses_train_only_teacher_utility_for_policy_pretraining=True,
    passes_teacher_or_value_to_forward_test=False,
    end_to_end=False,
    uses_uniform_scaffold=False,
    uses_uniform_fill=False,
    deploy_claim_allowed=False,
    runtime_flops_claim_allowed=False,
    paper_claim_allowed=False,
    map_claim_allowed=False,
)

c3_detector_aware_full_train_gate = dict(
    route="C3_STAGE2_DIVERGENT_ROUTE",
    route_variant="DIVERGENT_INNOVATION_DETECTOR_AWARE_UTILITY_DO_NOT_MERGE_WITH_C3",
    stage="Stage-2 detector-aware offline selector",
    default_off=True,
    explicit_config_opt_in=True,
    full_train_candidate=True,
    requires_launch_gate=True,
    launch_gate_passed=False,
    allow_tools_train=True,
    allow_tools_test=False,
    allow_detector_map=True,
    allow_remote_sync=True,
    allow_precheck_only=True,
    allow_slurm=True,
    allow_gpu=True,
    required_gpu="GPU1",
    allowed_entrypoints=("tools/train.py",),
    forbidden_entrypoints=("tools/test.py",),
    metric_claim_allowed=False,
    paper_claim_allowed=False,
    runtime_flops_claim_allowed=False,
    deploy_claim_allowed=False,
)

c3_detector_aware_meta_keys = [
    "video_name",
    "data_path",
    "fps",
    "duration",
    "snippet_stride",
    "window_start_frame",
    "resize_length",
    "window_size",
    "offset_frames",
    "selected_valid_len",
    "irregular_selected_positions",
    "irregular_selected_valid_len",
    "irregular_dense_valid_len",
    "irregular_native_axis",
    "bata_score_source",
    "bata_diagnostic_only",
    "bata_selected_dense_indices",
    "bata_value_transport_selection_row",
    "bata_value_transport_config_hash",
]


def c3_detector_aware_loadframes(ledger_path):
    return dict(
        type="LoadFrames",
        num_clips=1,
        method="bata_value_transport_ledger_subsample",
        method_base="sliding_window",
        keep_ratio=float(window_size) / float(dense_window_size),
        target_len=window_size,
        scale_factor=scale_factor,
        remap_gt_to_selected_axis=True,
        bata_value_transport_ledger_path=ledger_path,
        bata_value_transport_allow_missing_fallback=False,
        bata_value_transport_require_deployable=True,
        bata_value_transport_require_selected_count=detector_aware_require_selected_count,
        bata_value_transport_allow_short_valid_ratio_count=True,
        bata_value_transport_source=c3_value_transport_source,
        bata_value_transport_config_hash=c3_value_transport_config_hash,
    )


dataset = dict(
    train=dict(
        type="ThumosSlidingDataset",
        ann_file=annotation_path,
        subset_name="training",
        class_map=class_map,
        data_path=train_data_path,
        filter_gt=False,
        feature_stride=4,
        sample_stride=1,
        window_size=dense_window_size,
        window_overlap_ratio=0.25,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            c3_detector_aware_loadframes(train_ledger_path),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 182)),
            dict(type="mmaction.RandomResizedCrop"),
            dict(type="mmaction.Resize", scale=(160, 160), keep_ratio=False),
            dict(type="mmaction.Flip", flip_ratio=0.5),
            dict(type="mmaction.ImgAug", transforms="default"),
            dict(type="mmaction.ColorJitter"),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"], meta_keys=c3_detector_aware_meta_keys),
        ],
    ),
    val=dict(
        type="ThumosSlidingDataset",
        ann_file=annotation_path,
        subset_name="validation",
        class_map=class_map,
        data_path=test_data_path,
        filter_gt=False,
        feature_stride=4,
        sample_stride=1,
        window_size=dense_window_size,
        window_overlap_ratio=0.25,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            c3_detector_aware_loadframes(val_ledger_path),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"], meta_keys=c3_detector_aware_meta_keys),
        ],
    ),
    test=dict(
        type="ThumosSlidingDataset",
        ann_file=annotation_path,
        subset_name="validation",
        class_map=class_map,
        data_path=test_data_path,
        filter_gt=False,
        test_mode=True,
        feature_stride=4,
        sample_stride=1,
        window_size=dense_window_size,
        window_overlap_ratio=0.5,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            c3_detector_aware_loadframes(test_ledger_path),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(type="Collect", inputs="imgs", keys=["masks"], meta_keys=c3_detector_aware_meta_keys),
        ],
    ),
)

model = dict(
    backbone=dict(
        backbone=dict(total_frames=window_size * scale_factor),
        custom=dict(
            pretrain=c3_detector_aware_adatad_pretrain_path,
            pre_processing_pipeline=[
                dict(type="Rearrange", keys=["frames"], ops="b n c (t1 t) h w -> (b t1) n c t h w", t1=chunk_num),
            ],
            post_processing_pipeline=[
                dict(type="Reduce", keys=["feats"], ops="b n c t h w -> b c t", reduction="mean"),
                dict(type="Rearrange", keys=["feats"], ops="(b t1) c t -> b c (t1 t)", t1=chunk_num),
                dict(type="Interpolate", keys=["feats"], size=window_size),
            ],
        ),
    ),
    projection=dict(max_seq_len=window_size),
)

solver = dict(
    train=dict(batch_size=2, num_workers=2),
    val=dict(batch_size=2, num_workers=2),
    test=dict(batch_size=2, num_workers=2),
    clip_grad_norm=1,
    amp=True,
    fp16_compress=True,
    static_graph=True,
    ema=True,
)

optimizer = dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=0.05,
    paramwise=True,
    backbone=dict(
        lr=0,
        weight_decay=0,
        custom=[dict(name="adapter", lr=2e-4, weight_decay=0.05)],
        exclude=["backbone"],
    ),
)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=100)

evaluation = dict(
    type="mAP",
    subset="validation",
    tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
    ground_truth_filename=annotation_path,
)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)

workflow = dict(
    logging_interval=50,
    checkpoint_interval=2,
    val_loss_interval=-1,
    val_eval_interval=10,
    val_eval_interval_anchor_epoch=10,
    val_start_epoch=9,
    end_epoch=60,
    max_train_iters=None,
)

work_dir = f"exps/thumos/adatad/c3_detector_aware_stage2_adatad_full_train/{detector_aware_ledger_variant}"
