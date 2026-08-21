_base_ = ["./georoute_official_postbackbone_bc_common_seed42_v001.py"]

official_bc_arm = "C"
model = dict(
    backbone=dict(
        custom=dict(georoute_postbackbone_selection="roi"),
    )
)
work_dir = "exps/thumos/adatad/georoute_official_c_roi_postbackbone_seed42_v001"
