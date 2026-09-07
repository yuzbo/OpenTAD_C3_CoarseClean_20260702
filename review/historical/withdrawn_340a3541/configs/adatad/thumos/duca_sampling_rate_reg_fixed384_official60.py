_base_ = ["./duca_sampling_rate_fixed384_official60.py"]

duca_sampling_rate_contract = dict(
    variant="rate_plus_reg_contribution_distillation",
    contribution_components="reg",
    asformer_last_layer_adapted=False,
)

model = dict(
    frame_selector=dict(
        sampling_rate_utility_components="reg",
        detector_contribution_distillation_weight=1.0,
        detector_contribution_components="reg",
    )
)

work_dir = "exps/thumos/adatad/duca_sampling_rate_reg_fixed384_official60"
