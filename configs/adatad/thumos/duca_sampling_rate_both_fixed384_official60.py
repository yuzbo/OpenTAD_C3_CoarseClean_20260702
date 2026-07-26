_base_ = ["./duca_sampling_rate_fixed384_official60.py"]

duca_sampling_rate_contract = dict(
    variant="rate_plus_cls_reg_contribution_distillation",
    contribution_components="both",
    asformer_last_layer_adapted=False,
)

model = dict(
    frame_selector=dict(
        sampling_rate_utility_components="both",
        detector_contribution_distillation_weight=1.0,
        detector_contribution_components="both",
    )
)

work_dir = "exps/thumos/adatad/duca_sampling_rate_both_fixed384_official60"
