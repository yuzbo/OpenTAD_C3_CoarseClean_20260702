"""Three-update real-loader numerical precheck for the A6 H65 replay arm."""

_base_ = ["./duca_evidence_recovery_h65_selection.py"]

workflow = dict(
    formal_protocol="",
    logging_interval=1,
    checkpoint_interval=-1,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=9999,
    end_epoch=1,
    max_train_iters=3,
    disable_checkpoint=True,
    seal_eval_dataloaders_during_training=True,
)

dataset = dict(val=None, test=None)
work_dir = "exps/thumos/adatad/duca_evidence_recovery_a6_precheck"
