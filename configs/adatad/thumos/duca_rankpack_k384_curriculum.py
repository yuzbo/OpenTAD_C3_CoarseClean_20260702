_base_ = ["./duca_truetime_indirect_curriculum_k384_base.py"]

arm = "RANKPACK_K384"
physical_time = False
experiment_scope = dict(
    route_variant=arm,
    temporal_coordinate_contract="selected_rank_inside_heavy_encoder",
    detector_coordinate_contract="shared_dense_physical",
)
model = dict(backbone=dict(backbone=dict(physical_time=False)))

work_dir = "exps/thumos/adatad/duca_rankpack_k384_curriculum"
