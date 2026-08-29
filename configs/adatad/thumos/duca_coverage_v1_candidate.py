_base_ = ["./duca_coverage_v1_matched_h65_control.py"]


duca_coverage_v1_contract = dict(
    route="DUCA_COVERAGE_V1",
    allocation="monotone_submodular_temporal_facility_location",
    temporal_anchor_count=96,
    temporal_kernel="exp_absolute_distance",
    temporal_kernel_sigma=1.0 / 95.0,
    quality_normalization="valid_window_affine",
    quality_coverage_scale="K_over_M",
)

model = dict(
    frame_selector=dict(
        acquisition_policy="temporal_coverage",
    ),
)

workflow = dict(
    training_profile="duca_coverage_v1_candidate",
)

work_dir = "exps/thumos/adatad/duca_coverage_v1_candidate"

