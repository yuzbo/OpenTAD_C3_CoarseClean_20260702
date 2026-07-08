_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]

import os


VARIANT_SPECS = {
    "learned_fixed_384": dict(
        target_len=384,
        require_selected_count=384,
        strategy="learned_paction_gap_loss_value",
        ledger_name="learned_fixed_384",
        source="learned_paction_gap_loss_policy_checkpoint",
        config_hash="c3_paction_learned_gap_loss_fixed384_no_uniform_v1",
    ),
    "learned_fixed_768": dict(
        target_len=768,
        require_selected_count=768,
        strategy="learned_paction_gap_loss_value",
        ledger_name="learned_fixed_768",
        source="learned_paction_gap_loss_policy_checkpoint",
        config_hash="c3_paction_learned_gap_loss_fixed768_no_uniform_v1",
    ),
    "learned_dynamic": dict(
        target_len=768,
        require_selected_count=None,
        strategy="learned_paction_gap_loss_dynamic_budget",
        ledger_name="learned_dynamic",
        source="learned_paction_gap_loss_policy_checkpoint",
        config_hash="c3_paction_learned_gap_loss_dynamic_budget_no_uniform_v1",
    ),
    "paction_lattice_radius_score_only_move25": dict(
        target_len=384,
        require_selected_count=384,
        strategy="paction_lattice_radius_score_only_move25",
        ledger_name="paction_lattice_radius_score_only_move25",
        source="learned_paction_gap_loss_policy_checkpoint",
        config_hash="c3_paction_lattice_radius_score_only_move25_v1",
        selector_decoder="score_only_lattice_replacement_with_adaptive_radius_v1",
        geometry_constraint="score_only_local_lattice_replacement_move25_adaptive_radius_0_16",
        route_variant="C3_PACTION_SCORE_ONLY_LATTICE_REPLACEMENT_ADAPTIVE_RADIUS",
        use_expanded_positions=True,
    ),
    "paction_lattice_replace_score_only_move25": dict(
        target_len=384,
        require_selected_count=384,
        strategy="paction_lattice_replace_score_only_move25",
        ledger_name="paction_lattice_replace_score_only_move25",
        source="learned_paction_gap_loss_policy_checkpoint",
        config_hash="c3_paction_lattice_replace_score_only_move25_v1",
        selector_decoder="score_only_lattice_replacement_v1",
        geometry_constraint="score_only_local_lattice_replacement_move25",
        route_variant="C3_PACTION_SCORE_ONLY_LATTICE_REPLACEMENT",
    ),
    "paction_lattice_replace_score_only_move50": dict(
        target_len=384,
        require_selected_count=384,
        strategy="paction_lattice_replace_score_only_move50",
        ledger_name="paction_lattice_replace_score_only_move50",
        source="learned_paction_gap_loss_policy_checkpoint",
        config_hash="c3_paction_lattice_replace_score_only_move50_v1",
        selector_decoder="score_only_lattice_replacement_v1",
        geometry_constraint="score_only_local_lattice_replacement_move50",
        route_variant="C3_PACTION_SCORE_ONLY_LATTICE_REPLACEMENT",
    ),
    "paction_lattice_replace_score_only_move75": dict(
        target_len=384,
        require_selected_count=384,
        strategy="paction_lattice_replace_score_only_move75",
        ledger_name="paction_lattice_replace_score_only_move75",
        source="learned_paction_gap_loss_policy_checkpoint",
        config_hash="c3_paction_lattice_replace_score_only_move75_v1",
        selector_decoder="score_only_lattice_replacement_v1",
        geometry_constraint="score_only_local_lattice_replacement_move75",
        route_variant="C3_PACTION_SCORE_ONLY_LATTICE_REPLACEMENT",
    ),
    "paction_lattice_replace_score_only_no_protect": dict(
        target_len=384,
        require_selected_count=384,
        strategy="paction_lattice_replace_score_only_no_protect",
        ledger_name="paction_lattice_replace_score_only_no_protect",
        source="learned_paction_gap_loss_policy_checkpoint",
        config_hash="c3_paction_lattice_replace_score_only_no_protect_v1",
        selector_decoder="score_only_lattice_replacement_v1",
        geometry_constraint="score_only_local_lattice_replacement_no_protect",
        route_variant="C3_PACTION_SCORE_ONLY_LATTICE_REPLACEMENT",
    ),
}


paction_ledger_variant = os.environ.get("C3_PACTION_LEDGER_VARIANT", "learned_fixed_384")
if paction_ledger_variant not in VARIANT_SPECS:
    raise ValueError(f"unknown C3_PACTION_LEDGER_VARIANT={paction_ledger_variant}")
_variant = VARIANT_SPECS[paction_ledger_variant]

yuzibo_root = os.environ.get("YUZIBO_ROOT", os.path.expanduser("~/run/yuzibo"))
thumos14_root = os.path.join(yuzibo_root, "thumos14")
learned_ledger_root = os.environ.get(
    "C3_PACTION_LEARNED_LEDGER_ROOT",
    os.path.join(yuzibo_root, "projects/c3_lowres_action_probe/paction_learned_ledgers"),
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
c3_paction_adatad_pretrain_path = os.environ.get(
    "C3_PACTION_ADATAD_PRETRAIN_PATH",
    os.path.join(yuzibo_root, "pretrained", adatad_pretrain_filename),
)
paction_adatad_checkpoint_interval = int(os.environ.get("C3_PACTION_ADATAD_CHECKPOINT_INTERVAL", "10"))
paction_adatad_disable_checkpoint = os.environ.get("C3_PACTION_ADATAD_DISABLE_CHECKPOINT", "0") == "1"
paction_adatad_val_eval_interval = int(os.environ.get("C3_PACTION_ADATAD_VAL_EVAL_INTERVAL", "10"))
paction_adatad_val_eval_anchor_epoch = int(
    os.environ.get("C3_PACTION_ADATAD_VAL_EVAL_INTERVAL_ANCHOR_EPOCH", str(paction_adatad_val_eval_interval))
)
paction_adatad_val_start_epoch = int(
    os.environ.get("C3_PACTION_ADATAD_VAL_START_EPOCH", str(max(0, paction_adatad_val_eval_interval - 1)))
)

_ledger_name = _variant["ledger_name"]
train_ledger_path = os.environ.get(
    "C3_PACTION_TRAIN_LEDGER_PATH",
    os.path.join(learned_ledger_root, "train", f"value_transport_ledger_{_ledger_name}.jsonl"),
)
val_ledger_path = os.environ.get(
    "C3_PACTION_VAL_LEDGER_PATH",
    os.path.join(learned_ledger_root, "val", f"value_transport_ledger_{_ledger_name}.jsonl"),
)
test_ledger_path = os.environ.get(
    "C3_PACTION_TEST_LEDGER_PATH",
    os.path.join(learned_ledger_root, "test", f"value_transport_ledger_{_ledger_name}.jsonl"),
)
c3_value_transport_source = os.environ.get("C3_PACTION_LEDGER_SOURCE", _variant["source"])
c3_value_transport_config_hash = os.environ.get("C3_PACTION_LEDGER_CONFIG_HASH", _variant["config_hash"])

window_size = int(_variant["target_len"])
dense_window_size = 768
scale_factor = 1
chunk_num = window_size * scale_factor // 16
paction_ledger_strategy = _variant["strategy"]
paction_require_selected_count = _variant["require_selected_count"]

experiment_scope = dict(
    route="C3_MAINLINE_OPTIMIZATION",
    route_variant=_variant.get("route_variant", "C3_PACTION_LEARNED_STRICT_LEDGER"),
    stage="paction_learned_ledger_original_adatad_full_train",
    selector_source="learned_paction_gap_loss_policy_checkpoint",
    selector_decoder=_variant.get("selector_decoder", "learned_paction_gap_loss_value_decoder"),
    selection_strategy=paction_ledger_strategy,
    selection_geometry_constraint=_variant.get("geometry_constraint", "learned_score_topk_or_dynamic_budget_no_manual_slots"),
    paction_ledger_variant=paction_ledger_variant,
    detector_stack="original_adatad_actionformer_adapter",
    changes_input_sampling=True,
    changes_detector_head=False,
    changes_neck=False,
    changes_loss_assignment=False,
    changes_post_processing=False,
    uses_offline_deploy_selection_ledger=True,
    uses_uniform_scaffold=False,
    uses_uniform_fill=False,
    gap_control="learned_gap_hole_loss_no_uniform_fill",
    oracle_shell_semantics="strict_learned_paction_value_transport_ledger_not_random_online_oracle_shell",
    deploy_claim_allowed=False,
    runtime_flops_claim_allowed=False,
    paper_claim_allowed=False,
)

c3_paction_learned_ledger_full_train_gate = dict(
    route="C3_MAINLINE_OPTIMIZATION",
    route_variant=_variant.get("route_variant", "C3_PACTION_LEARNED_STRICT_LEDGER"),
    stage="paction_learned_ledger_original_adatad_full_train",
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
    forbidden_routes=(
        "DIVERGENT_INNOVATION_BH_SDC",
        "DIVERGENT_INNOVATION_EVENT_SURPRISE",
        "DIVERGENT_INNOVATION_BOUNDARY_MICROSCOPE",
        "DIVERGENT_INNOVATION_FRAME_TOKEN_HYBRID",
        "P2",
        "PQR",
        "raw_prediction_cache",
        "teacher_at_test",
        "gt_at_test_selection",
        "uniform_scaffold",
        "uniform_fill",
    ),
    metric_claim_allowed=False,
    paper_claim_allowed=False,
    runtime_flops_claim_allowed=False,
    deploy_claim_allowed=False,
)

c3_paction_meta_keys = [
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


def c3_paction_loadframes(ledger_path):
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
        bata_value_transport_require_selected_count=paction_require_selected_count,
        bata_value_transport_allow_short_valid_ratio_count=True,
        bata_value_transport_source=c3_value_transport_source,
        bata_value_transport_config_hash=c3_value_transport_config_hash,
        bata_value_transport_use_expanded_positions=bool(_variant.get("use_expanded_positions", False)),
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
            c3_paction_loadframes(train_ledger_path),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 182)),
            dict(type="mmaction.RandomResizedCrop"),
            dict(type="mmaction.Resize", scale=(160, 160), keep_ratio=False),
            dict(type="mmaction.Flip", flip_ratio=0.5),
            dict(type="mmaction.ImgAug", transforms="default"),
            dict(type="mmaction.ColorJitter"),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"], meta_keys=c3_paction_meta_keys),
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
            c3_paction_loadframes(val_ledger_path),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"], meta_keys=c3_paction_meta_keys),
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
            c3_paction_loadframes(test_ledger_path),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(type="Collect", inputs="imgs", keys=["masks"], meta_keys=c3_paction_meta_keys),
        ],
    ),
)

model = dict(
    backbone=dict(
        backbone=dict(total_frames=window_size * scale_factor),
        custom=dict(
            pretrain=c3_paction_adatad_pretrain_path,
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
    checkpoint_interval=paction_adatad_checkpoint_interval,
    val_loss_interval=-1,
    val_eval_interval=paction_adatad_val_eval_interval,
    val_eval_interval_anchor_epoch=paction_adatad_val_eval_anchor_epoch,
    val_start_epoch=paction_adatad_val_start_epoch,
    end_epoch=60,
    max_train_iters=None,
    disable_checkpoint=paction_adatad_disable_checkpoint,
)

work_dir = f"exps/thumos/adatad/c3_paction_learned_ledger_original_adatad_full_train/{paction_ledger_variant}"
