_base_ = ["./duca_sampling_rate_fixed384_official60.py"]

duca_sampling_rate_contract = dict(
    variant="rate_plus_cls_contribution_distillation",
    contribution_components="cls",
    asformer_last_layer_adapted=False,
)

model = dict(
    frame_selector=dict(
        sampling_rate_utility_components="cls",
        detector_contribution_distillation_weight=1.0,
        detector_contribution_components="cls",
    )
)

work_dir = "exps/thumos/adatad/duca_sampling_rate_cls_fixed384_official60"
