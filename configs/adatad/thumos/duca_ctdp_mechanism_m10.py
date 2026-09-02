_base_ = ["./duca_ctdp_geometry_g2.py"]
model = dict(rpn_head=dict(conv_cfg=dict(type="ContinuousTimeScaleAdaptiveConv1d", reference_spacing_mode="level_nominal")))
work_dir = "exps/thumos/adatad/duca_ctdp_mechanism_m10_seed3407"
