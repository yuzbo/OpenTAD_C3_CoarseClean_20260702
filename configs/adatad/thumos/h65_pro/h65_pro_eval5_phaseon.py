_base_ = ["./h65_pro_f09.py"]

h65_pro_experiment_id = "TEST-PHASEON"
test_guided_exploratory = True
h65_pro_factor_policy = dict(generator="matched_phase_pair", fixed_relation=None)
workflow = dict(
    val_loss_interval=-1,
    val_eval_interval=5,
    val_start_epoch=4,
    val_eval_interval_anchor_epoch=5,
    val_eval_epochs=list(range(5, 61, 5)),
    intermediate_validation_role="full_curve_and_best_validation_checkpoint",
    intermediate_validation_selects_checkpoint=True,
)
inference = dict(save_raw_prediction=False, load_from_raw_predictions=False)
post_processing = dict(save_dict=True)
work_dir = "exps/thumos/h65_tia_test_guided/phaseon"
