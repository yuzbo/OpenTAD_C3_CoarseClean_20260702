_base_ = ["./georoute_official_postbackbone_bc_common_seed42_v001.py"]

official_bc_arm = "B"
model = dict(
    backbone=dict(
        custom=dict(georoute_postbackbone_selection="all"),
    )
)
work_dir = "exps/thumos/adatad/georoute_official_b_alltoken_postbackbone_seed42_v001"
