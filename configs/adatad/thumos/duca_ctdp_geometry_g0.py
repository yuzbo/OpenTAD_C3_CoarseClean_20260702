_base_ = ["./duca_ct_dual_phase_bamod_thumos.py"]
model = dict(frame_selector=dict(type="DualPhaseFrameSelector", total_budget=384, scaffold_budget=384, burst_budget=0, force_uniform=True), backbone=dict(backbone=dict(ct_tubelet=False, amod_config=dict(enabled=False))), rpn_head=dict(conv_cfg=dict(type="Conv", kernel_size=3, padding=1)))
work_dir = "exps/thumos/adatad/duca_ctdp_geometry_g0_seed3407"
