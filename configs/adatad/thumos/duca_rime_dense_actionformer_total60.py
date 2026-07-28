_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]

import os


train_block_list = os.environ.get("DUCA_RIME_TRAIN_BLOCK_LIST", "").strip()
development_block_list = os.environ.get(
    "DUCA_RIME_DEVELOPMENT_BLOCK_LIST",
    "",
).strip()
if not train_block_list or not development_block_list:
    raise RuntimeError(
        "dense ActionFormer training requires the frozen train/development roles"
    )

dataset = dict(
    train=dict(block_list=train_block_list),
    val=None,
    test=dict(
        subset_name="training",
        block_list=development_block_list,
        test_mode=True,
        window_size=768,
    ),
)

evaluation = dict(
    subset="training",
    blocked_videos=development_block_list,
)

model = dict(
    backbone=dict(
        backbone=dict(with_cp=False),
    ),
)

solver = dict(
    train=dict(batch_size=1, num_workers=2),
    val=dict(batch_size=1, num_workers=2),
    test=dict(batch_size=1, num_workers=2),
    static_graph=False,
)

scheduler = dict(
    type="LinearWarmupCosineAnnealingLR",
    warmup_epoch=5,
    max_epoch=60,
)

workflow = dict(
    formal_protocol="duca_rime_dense_actionformer_cost_baseline_v1",
    logging_interval=50,
    checkpoint_interval=10,
    checkpoint_retention=1,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=9999,
    end_epoch=60,
    primary_checkpoint_epoch=59,
    primary_checkpoint_state_key="state_dict_ema",
    seal_eval_dataloaders_during_training=True,
)

duca_rime_dense_contract = dict(
    role="dense_adatad_baseline",
    task="offline_temporal_action_detection",
    detector_backend="ActionFormer",
    backbone="VideoMAE-S-AdaTAD",
    dense_window_size=768,
    detector_projection_in_channels=384,
    selector=None,
    dynamic_budget=False,
    train_role="detector_selector_train",
    evaluation_role="certification_development",
    official_final_subset_consumed=False,
    claim_scope="trained_dense_actionformer_cost_reference_not_candidate_method",
    empirically_supported=False,
    paper_ready=False,
)

post_processing = dict(save_dict=True)

work_dir = "exps/thumos/adatad/duca_rime_dense_actionformer_total60"

del train_block_list
del development_block_list
