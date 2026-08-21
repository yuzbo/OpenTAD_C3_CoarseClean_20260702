_base_ = ["./duca_truetime_indirect_curriculum_k384_base.py"]

arm = "TRUETIME_K384"
physical_time = True
experiment_scope = dict(
    route_variant=arm,
    temporal_coordinate_contract="true_time_dense_index",
    preserve_source_positions=True,
    spatial_patch_embedding_before_temporal_mixing=True,
    temporal_mixing_condition="physical_delta_t",
    propagate_observation_mask=True,
    physical_coordinate_assignment=True,
    physical_coordinate_regression=True,
    decode_before_nms=True,
    pre_nms_coordinate_space="true_time_dense_index",
)

model = dict(
    physical_grid_actionformer=dict(
        enabled=True,
        required=True,
        strict=True,
        coordinate_space="true_time_dense_index",
        selected_position_key="irregular_selected_positions",
        dense_valid_len_key="irregular_dense_valid_len",
        delta_t_key="irregular_delta_t",
    )
)
work_dir = "exps/thumos/adatad/duca_truetime_k384_curriculum"

