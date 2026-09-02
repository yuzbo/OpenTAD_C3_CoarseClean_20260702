_base_ = ["./duca_ctdp_geometry_g2.py"]
model = dict(backbone=dict(backbone=dict(ct_tubelet=True)))
work_dir = "exps/thumos/adatad/duca_ctdp_geometry_g3_seed3407"
