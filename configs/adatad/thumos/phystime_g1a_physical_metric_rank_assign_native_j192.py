_base_ = ["./phystime_g1a_physical_metric_native_j192.py"]

model = dict(
    rpn_head=dict(
        physical_grid_actionformer=dict(
            assignment_positions_key="phystime_uniform_rank_timestamps_sec",
            assignment_count_keys=["phystime_native_valid_count"],
        )
    )
)

work_dir = "exps/thumos/adatad/phystime_g1a_physical_metric_rank_assign_native_j192"
