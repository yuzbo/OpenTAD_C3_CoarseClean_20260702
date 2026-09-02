_base_ = ["./duca_ctdp_geometry_g2.py"]
model = dict(backbone=dict(backbone=dict(amod_config=dict(enabled=True, capacity=0.5, amod_layers=[1,3,5,7,9,11], boundary_prior_scale=0.25))) )
work_dir = "exps/thumos/adatad/duca_ctdp_mechanism_m01_seed3407"
