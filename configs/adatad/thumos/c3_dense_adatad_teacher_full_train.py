_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]

teacher_route = dict(
    route="c3_dense_adatad_teacher",
    purpose="train full dense AdaTAD teacher for train-only detector utility export",
    base_config="./e2e_thumos_videomae_s_768x1_160_adapter.py",
    claim_lock="teacher checkpoint only; no sparse acquisition claim without downstream mAP",
)

workflow = dict(
    logging_interval=50,
    checkpoint_interval=10,
    val_loss_interval=-1,
    val_eval_interval=10,
    val_eval_interval_anchor_epoch=10,
    val_start_epoch=9,
    end_epoch=60,
)

work_dir = "exps/thumos/adatad/c3_dense_adatad_teacher_full_train"
