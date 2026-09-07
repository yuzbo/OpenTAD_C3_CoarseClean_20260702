_base_ = ["./duca_two_stage_joint_fixed384_official60_base.py"]

import os


def _required(name):
    value = os.environ.get(name, "")
    if not value:
        raise ValueError(f"{name} is required for two-stage frontend initialization")
    return value


duca_frontend_checkpoint = _required("DUCA_FRONTEND_CHECKPOINT")
duca_frontend_checkpoint_sha256 = _required("DUCA_FRONTEND_CHECKPOINT_SHA256")
duca_frontend_checkpoint_epoch = int(_required("DUCA_FRONTEND_CHECKPOINT_EPOCH"))


duca_transition_only_contract = dict(
    route="DUCA_TWO_STAGE_PRETRAINED_JOINT_FIXED384_OFFICIAL60",
    frontend_initialization="train_only_pretrained_selector_state",
    frontend_checkpoint_state_key="state_dict_ema",
    frontend_schedule_state_reset=True,
    coarse_probe_training="low_lr_joint_finetune",
)


workflow = dict(
    selector_initialization=dict(
        enabled=True,
        checkpoint_path=duca_frontend_checkpoint,
        checkpoint_sha256=duca_frontend_checkpoint_sha256,
        state_key="state_dict_ema",
        expected_checkpoint_epoch=duca_frontend_checkpoint_epoch,
        reset_state_keys=["_loss_weight_schedule_step"],
    ),
)


work_dir = "exps/thumos/adatad/duca_two_stage_pretrained_joint_fixed384_official60"
