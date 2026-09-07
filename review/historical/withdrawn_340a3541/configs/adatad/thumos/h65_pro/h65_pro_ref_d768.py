_base_ = ["../e2e_thumos_videomae_s_768x1_160_adapter.py"]

from tools.bata.duca_cellcf_protocol import protocol_for_name

duca_training_protocol = protocol_for_name("official60")
seed = 3407
total_epochs = 60
max_updates = 6000
h65_pro_experiment_id = "REF-D768"
h65_pro_factor_policy = dict(
    phase=False,
    ct=False,
    mod=False,
    taylor=False,
    curriculum=False,
    frames=768,
    reference="dense_768_no_acquisition",
)

model = dict(
    frame_selector=None,
    backbone=dict(
        backbone=dict(
            total_frames=768,
            num_frames=16,
            tubelet_size=2,
            amod_config=dict(_delete_=True, enabled=False),
        ),
    ),
    projection=dict(max_seq_len=768),
    rpn_head=dict(conv_cfg=None),
)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=60)
workflow = dict(
    formal_protocol="h65_pro_dense_reference_official60_v1",
    training_profile=duca_training_protocol.name,
    logging_interval=50,
    checkpoint_interval=5,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_eval_interval_anchor_epoch=9999,
    val_start_epoch=9999,
    end_epoch=60,
    formal_successful_update_contract=True,
    expected_train_batches_per_epoch=duca_training_protocol.steps_per_epoch,
    expected_successful_optimizer_updates=duca_training_protocol.expected_successful_optimizer_updates,
    max_amp_retries_per_batch=8,
    fail_on_amp_replay_exhaustion=True,
    require_finite_train_loss=True,
    selector_schedule_required=False,
    primary_checkpoint_epoch=59,
    primary_checkpoint_state_key="state_dict_ema",
    checkpoint_criterion=duca_training_protocol.checkpoint_criterion,
)
work_dir = "exps/thumos/adatad/h65_pro_fullmatrix_20260902/ref_d768"
