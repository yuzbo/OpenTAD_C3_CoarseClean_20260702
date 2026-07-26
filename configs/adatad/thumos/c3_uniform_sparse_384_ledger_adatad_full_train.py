_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]

import os


yuzibo_root = os.environ.get("YUZIBO_ROOT", os.path.expanduser("~/run/yuzibo"))
thumos14_root = os.path.join(yuzibo_root, "thumos14")
uniform_ledger_root = os.environ.get(
    "C3_UNIFORM_SPARSE_LEDGER_ROOT",
    os.path.join(yuzibo_root, "projects/c3_lowres_action_probe/uniform_sparse_384_adatad/ledgers"),
)

annotation_path = os.environ.get(
    "THUMOS14_ANNOTATION_PATH",
    os.path.join(thumos14_root, "annotations", "thumos_14_anno.json"),
)
class_map = os.environ.get(
    "THUMOS14_CLASS_MAP",
    os.path.join(thumos14_root, "annotations", "category_idx.txt"),
)
train_data_path = os.environ.get("THUMOS14_TRAIN_DATA_PATH", os.path.join(thumos14_root, "train"))
test_data_path = os.environ.get("THUMOS14_TEST_DATA_PATH", os.path.join(thumos14_root, "test"))
adatad_pretrain_filename = "vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
c3_uniform_sparse_adatad_pretrain_path = os.environ.get(
    "C3_UNIFORM_SPARSE_ADATAD_PRETRAIN_PATH",
    os.path.join(yuzibo_root, "pretrained", adatad_pretrain_filename),
)

uniform_sparse_ledger_variant = "uniform_sparse_384"
train_ledger_path = os.environ.get(
    "C3_UNIFORM_SPARSE_TRAIN_LEDGER_PATH",
    os.path.join(uniform_ledger_root, "train", "value_transport_ledger_uniform_sparse_384.jsonl"),
)
val_ledger_path = os.environ.get(
    "C3_UNIFORM_SPARSE_VAL_LEDGER_PATH",
    os.path.join(uniform_ledger_root, "val", "value_transport_ledger_uniform_sparse_384.jsonl"),
)
test_ledger_path = os.environ.get(
    "C3_UNIFORM_SPARSE_TEST_LEDGER_PATH",
    os.path.join(uniform_ledger_root, "test", "value_transport_ledger_uniform_sparse_384.jsonl"),
)
c3_value_transport_source = os.environ.get("C3_UNIFORM_SPARSE_LEDGER_SOURCE", "uniform_exact_sparse_384")
c3_value_transport_config_hash = os.environ.get(
    "C3_UNIFORM_SPARSE_LEDGER_CONFIG_HASH",
    "uniform_exact_sparse_384_generator_v1_target384_no_gt_no_teacher_no_oracle",
)

window_size = 384
dense_window_size = 768
scale_factor = 1
chunk_num = window_size * scale_factor // 16

experiment_scope = dict(
    route="C3_MAINLINE_OPTIMIZATION",
    route_variant="C3_UNIFORM_SPARSE_EXACT_384_BASELINE",
    stage="uniform_sparse_384_ledger_original_adatad_full_train",
    selector_source="exact_uniform_sparse_ledger_generator",
    selector_decoder="exact_uniform_local_dense_index_generator",
    selection_strategy="uniform_exact_384",
    selection_geometry_constraint="strict_uniform_exact_no_value_model_no_teacher_no_oracle",
    detector_stack="original_adatad_actionformer_adapter",
    changes_input_sampling=True,
    changes_detector_head=False,
    changes_neck=False,
    changes_loss_assignment=False,
    changes_post_processing=False,
    uses_offline_deploy_selection_ledger=True,
    uses_uniform_scaffold=True,
    uses_uniform_fill=False,
    uses_gt=False,
    uses_teacher=False,
    uses_oracle=False,
    uses_checkpoint=False,
    oracle_shell_semantics="strict_uniform_sparse_384_baseline_not_oracle_shell",
    deploy_claim_allowed=False,
    runtime_flops_claim_allowed=False,
    paper_claim_allowed=False,
)

c3_uniform_sparse_384_full_train_gate = dict(
    route="C3_MAINLINE_OPTIMIZATION",
    route_variant="C3_UNIFORM_SPARSE_EXACT_384_BASELINE",
    stage="uniform_sparse_384_ledger_original_adatad_full_train",
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
    required_gpu="GPU0",
    allowed_entrypoints=("tools/train.py",),
    forbidden_entrypoints=("tools/test.py",),
    forbidden_routes=(
        "teacher_at_test",
        "gt_at_test_selection",
        "oracle_selection",
        "raw_prediction_cache",
        "learned_paction_checkpoint",
    ),
    metric_claim_allowed=False,
    paper_claim_allowed=False,
    runtime_flops_claim_allowed=False,
    deploy_claim_allowed=False,
)

c3_uniform_sparse_meta_keys = [
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


def c3_uniform_sparse_loadframes(ledger_path):
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
        bata_value_transport_require_selected_count=window_size,
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
            c3_uniform_sparse_loadframes(train_ledger_path),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 182)),
            dict(type="mmaction.RandomResizedCrop"),
            dict(type="mmaction.Resize", scale=(160, 160), keep_ratio=False),
            dict(type="mmaction.Flip", flip_ratio=0.5),
            dict(type="mmaction.ImgAug", transforms="default"),
            dict(type="mmaction.ColorJitter"),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"], meta_keys=c3_uniform_sparse_meta_keys),
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
            c3_uniform_sparse_loadframes(val_ledger_path),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"], meta_keys=c3_uniform_sparse_meta_keys),
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
            c3_uniform_sparse_loadframes(test_ledger_path),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(type="Collect", inputs="imgs", keys=["masks"], meta_keys=c3_uniform_sparse_meta_keys),
        ],
    ),
)

model = dict(
    backbone=dict(
        backbone=dict(total_frames=window_size * scale_factor),
        custom=dict(
            pretrain=c3_uniform_sparse_adatad_pretrain_path,
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

work_dir = "exps/thumos/adatad/c3_uniform_sparse_384_ledger_original_adatad_full_train/uniform_sparse_384"
