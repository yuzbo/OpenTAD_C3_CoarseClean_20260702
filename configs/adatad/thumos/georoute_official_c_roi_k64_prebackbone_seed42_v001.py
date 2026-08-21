_base_ = ["./georoute_official_prebackbone_bc_common_seed42_v001.py"]

official_bc_arm = "C"
model = dict(
    backbone=dict(custom=dict(georoute_official_support="roi_k64"))
)
work_dir = "exps/thumos/adatad/georoute_official_c_roi_k64_prebackbone_seed42_v001"
