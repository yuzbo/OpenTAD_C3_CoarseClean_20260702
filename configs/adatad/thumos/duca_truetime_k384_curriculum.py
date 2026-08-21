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
    backbone=dict(
        backbone=dict(
            physical_time=True,
            physical_time_nominal_pair_gap=2.0,
            physical_time_nominal_tubelet_gap=4.0,
            physical_time_extent=768.0,
        ),
    ),
)
optimizer = dict(
    backbone=dict(
        lr=0,
        weight_decay=0,
        custom=[
            dict(name="adapter", lr=2e-4, weight_decay=0.05),
            dict(name="physical_time_embedding", lr=2e-4, weight_decay=0.05),
        ],
        exclude=["backbone"],
    ),
)
work_dir = "exps/thumos/adatad/duca_truetime_k384_curriculum"
