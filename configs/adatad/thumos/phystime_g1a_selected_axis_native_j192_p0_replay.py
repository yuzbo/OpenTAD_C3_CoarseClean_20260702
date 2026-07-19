_base_ = ["./phystime_g1a_selected_axis_native_j192.py"]

post_processing = dict(
    filter_invalid_proposals=True,
    proposal_min_duration=1.0e-6,
    round_before_cross_window_nms=False,
    round_after_cross_window_nms=False,
    segment_round_digits=2,
    score_round_digits=4,
    save_dict=True,
    save_pre_cross_window_detections=True,
    save_post_processing_audit=True,
)

work_dir = "exps/thumos/adatad/phystime_g1a_selected_axis_native_j192_p0_replay"
