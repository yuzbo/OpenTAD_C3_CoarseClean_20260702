_base_ = ["./georoute_official_prebackbone_bc_common_seed42_v001.py"]

official_bc_arm = "B"
model = dict(
    backbone=dict(custom=dict(georoute_official_support="all_native"))
)
work_dir = "exps/thumos/adatad/georoute_official_b_alltoken_prebackbone_seed42_v001"
