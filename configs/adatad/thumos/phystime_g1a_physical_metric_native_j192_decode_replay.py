_base_ = ["./phystime_g1a_physical_metric_native_j192_p0_replay.py"]

inference = dict(
    phystime_decode_replay_capture=dict(
        enabled=True,
        train_axis="physical_time_seconds",
        expected_native_coordinate_mode="physical_time_seconds",
        weights_source="must_be_overridden",
        max_in_memory_bytes=8589934592,
        artifact_filename="decode_replay_inputs.npz",
        manifest_filename="decode_replay_manifest.json",
    ),
)

work_dir = (
    "exps/thumos/adatad/"
    "phystime_g1a_physical_metric_native_j192_decode_replay"
)
