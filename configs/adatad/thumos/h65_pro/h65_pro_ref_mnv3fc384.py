_base_ = ["../duca_trainfree_fixed384_official60_base.py"]

from tools.bata.duca_cellcf_protocol import protocol_for_name

duca_training_protocol = protocol_for_name("official60")
seed = 3407
total_epochs = 60
max_updates = 6000
h65_pro_experiment_id = "REF-MNV3FC384"
h65_pro_factor_policy = dict(
    phase=False,
    ct=False,
    mod=False,
    taylor=False,
    curriculum=False,
    frames=384,
    reference="frozen_mobilenetv3_feature_change",
)

model = dict(
    backbone=dict(backbone=dict(amod_config=dict(_delete_=True, enabled=False))),
    rpn_head=dict(conv_cfg=None),
)
workflow = dict(
    formal_protocol="duca_selected_axis_optimization_v1",
    training_profile=duca_training_protocol.name,
    checkpoint_interval=5,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=9999,
    end_epoch=60,
    formal_successful_update_contract=True,
    expected_train_batches_per_epoch=duca_training_protocol.steps_per_epoch,
    expected_successful_optimizer_updates=duca_training_protocol.expected_successful_optimizer_updates,
    max_amp_retries_per_batch=8,
    fail_on_amp_replay_exhaustion=True,
    require_finite_train_loss=True,
    primary_checkpoint_epoch=59,
    primary_checkpoint_state_key="state_dict_ema",
    checkpoint_criterion=duca_training_protocol.checkpoint_criterion,
)
work_dir = "exps/thumos/adatad/h65_pro_fullmatrix_20260902/ref_mnv3fc384"
