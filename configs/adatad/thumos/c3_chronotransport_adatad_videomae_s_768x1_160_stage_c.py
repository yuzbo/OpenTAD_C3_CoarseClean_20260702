_base_ = ["./c3_chronotransport_adatad_videomae_s_768x1_160_stage_b.py"]

chronotransport_training = dict(
    stage="C",
    paired_dense_reference=True,
    dense_reference_no_grad=True,
    frozen_videomae_heavy=True,
    trainable=("adatad_adapter", "chronotransport.transport", "chronotransport.risk_predictor"),
    seeds=(3407, 3408, 3409),
    duplicate_detector_loss_forbidden=True,
)

chronotransport_claims = dict(
    deploy=False,
    metric=False,
    latency=False,
    paper=False,
)
