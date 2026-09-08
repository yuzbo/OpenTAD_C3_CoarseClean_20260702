_base_ = ["./et_trc_videomae_s_768x1_160_adapter_seed4407.py"]

test_guided_exploratory = True
workflow = dict(
    checkpoint_interval=5,
    val_eval_interval=5,
    val_start_epoch=4,
    val_eval_interval_anchor_epoch=5,
    val_eval_epochs=list(range(5, 61, 5)),
)
inference = dict(save_raw_prediction=False)
post_processing = dict(save_dict=True)
work_dir = "exps/thumos/adatad/ettrc_test_guided_on_seed4407"
