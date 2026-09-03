_base_ = ["./duca_ctdp_geometry_g1.py"]
model = dict(rpn_head=dict(physical_grid_actionformer=dict(enabled=True, required=True, strict=True)))
work_dir = "exps/thumos/adatad/duca_ctdp_geometry_g3_seed3407"
