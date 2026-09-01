_base_ = ["./c3_chronotransport_adatad_videomae_s_768x1_160_stage_a.py"]

chronotransport_training = dict(
    stage="B",
    paired_dense_reference=True,
    dense_reference_no_grad=True,
    same_batch_same_augmentation_same_rng=True,
    fit_calibration_evaluation_split_isolation=True,
    one_sided_detector_regret=True,
    scheduler_argmin_backprop=False,
    trainable=("chronotransport.transport", "chronotransport.risk_predictor"),
)

chronotransport_claims = dict(
    deploy=False,
    metric=False,
    latency=False,
    paper=False,
)
