_base_ = ["./duca_protected_physical_fixed384_official60_base.py"]

import os

from tools.bata.duca_cellcf_protocol import protocol_for_name


def _required_env(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required by the fail-closed DUCA-RIME config")
    return value


def _int_tuple(name, default):
    raw = os.environ.get(name, default)
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if len(values) < 2 or tuple(sorted(set(values))) != values:
        raise RuntimeError(f"{name} must contain unique increasing integer budgets")
    return values


candidate_budgets = _int_tuple(
    "DUCA_RIME_CANDIDATE_BUDGETS",
    "128,192,256,384,512",
)
candidate_costs = tuple(float(value) for value in candidate_budgets)
max_budget = candidate_budgets[-1]
dense_window_size = 768
target_mean_cost = float(os.environ.get("DUCA_RIME_TARGET_MEAN_COST", "384"))
if not candidate_costs[0] < target_mean_cost < candidate_costs[-1]:
    raise RuntimeError(
        "RIME target mean cost must have candidate budgets on both sides"
    )

train_block_list = _required_env("DUCA_RIME_TRAIN_BLOCK_LIST")
development_block_list = _required_env("DUCA_RIME_DEVELOPMENT_BLOCK_LIST")
targets_jsonl = _required_env("DUCA_RIME_TARGETS_JSONL")
targets_sha256 = _required_env("DUCA_RIME_TARGETS_SHA256")
budget_protocol_path = os.environ.get("DUCA_RIME_BUDGET_PROTOCOL_JSON", "").strip()
budget_protocol_sha256 = os.environ.get(
    "DUCA_RIME_BUDGET_PROTOCOL_SHA256",
    "",
).strip()
replay_jsonl = os.environ.get("DUCA_RIME_REPLAY_JSONL", "").strip()
replay_sha256 = os.environ.get("DUCA_RIME_REPLAY_SHA256", "").strip()

duca_training_protocol = protocol_for_name("official60")
duca_meta_keys = [
    "video_name",
    "data_path",
    "fps",
    "avg_fps",
    "duration",
    "total_frames",
    "snippet_stride",
    "window_start_frame",
    "resize_length",
    "window_size",
    "offset_frames",
    "frame_inds",
    "rime_requested_k_replay",
    "rime_requested_k_replay_provenance",
    "rime_budget_replay_jsonl",
    "rime_budget_replay_sha256",
]
target_transform = [
    dict(
        type="DucaRimeTargetsFromJsonl",
        targets_jsonl=targets_jsonl,
        targets_sha256=targets_sha256,
        candidate_budgets=candidate_budgets,
    )
]
replay_transform = (
    [
        dict(
            type="DucaRimeBudgetReplayFromJsonl",
            replay_jsonl=replay_jsonl,
            replay_sha256=replay_sha256,
            candidate_budgets=candidate_budgets,
        )
    ]
    if replay_jsonl
    else []
)

dataset = dict(
    train=dict(
        type="DucaStatelessThumosPaddingDataset",
        stateless_seed=3407,
        block_list=train_block_list,
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
                scale_factor=1,
                emit_boundary_validity=True,
            ),
            *target_transform,
            *replay_transform,
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 182)),
            dict(type="mmaction.RandomResizedCrop"),
            dict(type="mmaction.Resize", scale=(160, 160), keep_ratio=False),
            dict(type="mmaction.Flip", flip_ratio=0.5),
            dict(type="mmaction.ImgAug", transforms="default"),
            dict(type="mmaction.ColorJitter"),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(
                type="ConvertToTensor",
                keys=[
                    "imgs",
                    "gt_segments",
                    "gt_labels",
                    "gt_boundary_validity",
                    "rime_utility_target",
                    "rime_risk_target",
                    "rime_target_mask",
                    "rime_hard_frame_utility",
                ],
            ),
            dict(
                type="Collect",
                inputs="imgs",
                keys=[
                    "masks",
                    "gt_segments",
                    "gt_labels",
                    "gt_boundary_validity",
                    "rime_utility_target",
                    "rime_risk_target",
                    "rime_target_mask",
                    "rime_hard_frame_utility",
                ],
                meta_keys=duca_meta_keys,
            ),
        ],
    ),
    val=None,
    test=dict(
        subset_name="training",
        block_list=development_block_list,
        test_mode=True,
        window_size=dense_window_size,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="sliding_window",
                scale_factor=1,
            ),
            *replay_transform,
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(
                type="Collect",
                inputs="imgs",
                keys=["masks"],
                meta_keys=duca_meta_keys,
            ),
        ],
    ),
)

evaluation = dict(
    subset="training",
)

model = dict(
    frame_selector=dict(
        _delete_=True,
        type="DucaRimeFrameSelector",
        in_channels=3,
        rime_arm="rime_full",
        candidate_budgets=candidate_budgets,
        candidate_costs=candidate_costs,
        fixed_budget=384,
        dense_window_size=dense_window_size,
        frozen_price=0.0,
        target_mean_cost=target_mean_cost,
        risk_weight=1.0,
        risk_threshold=0.35,
        uncertainty_z=1.645,
        decoder_family="independent",
        weak_overlap_fraction=0.50,
        controller_hidden_dim=128,
        execution_quantum=16,
        budget_protocol_path=budget_protocol_path or None,
        budget_protocol_sha256=budget_protocol_sha256 or None,
        require_frozen_protocol=True,
        coarse_hidden_dim=96,
        selector_hidden_dim=64,
        coverage_floor_weight=0.10,
        score_temperature=0.70,
        path_temperature=1.0,
        transition_target_sigma=2.0,
        transition_target_radius=4,
        transition_boundary_radius=4,
        transition_distribution_temperature=0.70,
        action_loss_weight=1.0,
        transition_loss_weight=0.50,
        transition_boundary_loss_weight=0.25,
        budget_utility_loss_weight=1.0,
        budget_risk_loss_weight=1.0,
        budget_uncertainty_loss_weight=0.25,
        rank_alignment_loss_weight=0.25,
        coarse_trunk_lr=2.5e-5,
        action_head_lr=5.0e-5,
        selector_lr=1.0e-4,
        detector_bridge_gradient_scale=1.0,
        strict_physical_metadata=True,
        forbid_raw_prediction_cache=True,
        actionness_source_cfg=dict(
            type="C3CoarseProbeActionnessSource",
            source_name="rime_official_asformer_binary_actionness",
            probe_model="official-action-seg",
            official_action_seg_backend="official_asformer",
            spatial_size=64,
            tcn_hidden_dim=96,
            official_num_layers=2,
            dropout=0.0,
            return_hidden_features=True,
            require_hidden_features=True,
            hidden_output_kind="official_asformer_encoder_hidden",
            policy_hidden_gradient_scope="none",
            checkpoint_path="",
            require_checkpoint=False,
            frozen=False,
            trainable=True,
            train_split_supervised=True,
            calibration_split="none",
            calibration_temperature=1.0,
            calibration_bias=0.0,
            thumos_trained=True,
            uses_labels=True,
            uses_teacher=False,
            uses_gt=True,
            uses_prediction_cache=False,
            trained_with_thumos_labels=True,
            trained_with_gt_segments=True,
            training_dataset="THUMOS14",
            training_supervision_scope="train_only",
        ),
    ),
    backbone=dict(
        backbone=dict(
            total_frames=max_budget,
            with_cp=False,
        ),
        custom=dict(
            dynamic_temporal_bucket=True,
            dynamic_temporal_clip_len=16,
        ),
    ),
    projection=dict(max_seq_len=max_budget),
    rpn_head=dict(
        physical_grid_actionformer=dict(
            enabled=True,
            required=True,
            strict=True,
            contract="duca_rime_physical_dynamic_k_v1",
        ),
    ),
)

solver = dict(
    train=dict(batch_size=1, num_workers=2),
    val=dict(batch_size=1, num_workers=2),
    test=dict(batch_size=1, num_workers=2),
    static_graph=False,
    find_unused_parameters=True,
)

scheduler = dict(
    type="LinearWarmupCosineAnnealingLR",
    warmup_epoch=5,
    max_epoch=duca_training_protocol.end_epoch,
)

workflow = dict(
    formal_protocol="duca_rime_physical_dynamic_k_v1",
    training_profile=duca_training_protocol.name,
    logging_interval=50,
    checkpoint_interval=duca_training_protocol.checkpoint_interval,
    checkpoint_retention=1,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_eval_interval_anchor_epoch=9999,
    val_start_epoch=9999,
    end_epoch=duca_training_protocol.end_epoch,
    formal_successful_update_contract=True,
    expected_train_batches_per_epoch=duca_training_protocol.steps_per_epoch,
    expected_successful_optimizer_updates=(
        duca_training_protocol.expected_successful_optimizer_updates
    ),
    max_amp_retries_per_batch=8,
    fail_on_amp_replay_exhaustion=True,
    require_finite_train_loss=True,
    primary_checkpoint_epoch=duca_training_protocol.terminal_epoch,
    primary_checkpoint_state_key=duca_training_protocol.terminal_state_key,
    checkpoint_criterion=duca_training_protocol.checkpoint_criterion,
    seal_eval_dataloaders_during_training=True,
    derive_train_loader_contract=True,
)

duca_rime_contract = dict(
    task="offline_temporal_action_detection",
    online_tad=False,
    streaming=False,
    pre_backbone_plugin=True,
    candidate_budgets=candidate_budgets,
    target_mean_cost=target_mean_cost,
    dynamic_heavy_compute=True,
    pad_to_kmax=False,
    execution_quantum=16,
    batch_size=1,
    exact_successful_updates=(
        duca_training_protocol.expected_successful_optimizer_updates
    ),
    development_split="train_only_certification_development",
    official_final_subset_consumed=False,
    empirically_supported=False,
    paper_ready=False,
)

post_processing = dict(save_dict=True)

work_dir = "exps/thumos/adatad/duca_rime_base_do_not_run"

del duca_training_protocol
