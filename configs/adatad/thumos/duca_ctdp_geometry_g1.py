_base_ = ["./duca_ct_dual_phase_bamod_thumos.py"]
model = dict(frame_selector=dict(remap_gt_to_selected_axis=True), backbone=dict(backbone=dict(ct_tubelet=False, amod_config=dict(enabled=False))), rpn_head=dict(conv_cfg=dict(type="Conv", kernel_size=3, padding=1), physical_grid_actionformer=dict(enabled=False, required=False)))
work_dir = "exps/thumos/adatad/duca_ctdp_geometry_g1_seed3407"
