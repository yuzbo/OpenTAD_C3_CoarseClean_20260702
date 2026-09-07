"""Two real-loader epochs for the no-temporal-merge arm."""
_base_ = ["./duca_evidence_recovery_no_merge.py"]

workflow = dict(
    formal_protocol="", formal_successful_update_contract=False,
    logging_interval=10, checkpoint_interval=-1, disable_checkpoint=True,
    val_loss_interval=-1, val_eval_interval=-1, val_start_epoch=9999,
    end_epoch=2, max_train_iters=100, seal_eval_dataloaders_during_training=True,
)
dataset = dict(val=None, test=None)
work_dir = "exps/thumos/adatad/duca_evidence_a4_optimizer_precheck"
