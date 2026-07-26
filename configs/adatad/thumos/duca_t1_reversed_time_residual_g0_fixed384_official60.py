_base_ = ["./duca_t1_true_time_residual_g0_fixed384_official60.py"]


duca_t1_contract = dict(
    descriptor_mode="reversed",
    negative_control="same_descriptor_distribution_wrong_temporal_alignment",
)


model = dict(
    true_time_residual=dict(descriptor_mode="reversed"),
)


work_dir = "exps/thumos/adatad/duca_t1_reversed_time_residual_g0_fixed384_official60"
