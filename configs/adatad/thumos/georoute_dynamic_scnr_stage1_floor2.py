"""Matched two-native-cell ROI-floor sensitivity arm for dynamic SCNR."""

_base_ = ["./georoute_dynamic_scnr_stage1_base.py"]

model = dict(
    backbone=dict(
        custom=dict(
            georoute_roi_extent_floor_mode="native_cells",
            georoute_roi_extent_floor_cells=2,
        )
    )
)

georoute_protocol = dict(
    floor_sensitivity_study="scnr-geometry-floor-sensitivity-v1",
    floor_sensitivity_arm="native_2cell_sensitivity",
    floor_cells=2,
    main_method=False,
    matched_except_roi_floor=True,
)

work_dir = "exps/thumos/adatad/georoute_dynamic_scnr_stage1_floor2_unbound"
