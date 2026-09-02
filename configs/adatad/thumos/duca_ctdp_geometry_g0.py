_base_ = ["./duca_ct_dual_phase_bamod_thumos.py"]

# G0 is the Dual-Phase + CT-Tubelet geometry baseline.  B-AMoD and the
# physical-grid head are isolated in later arms rather than disabled here.
model = dict(
    frame_selector=dict(
        type="DualPhaseFrameSelector",
        total_budget=384,
        scaffold_budget=128,
        burst_budget=256,
        burst_radius=2,
    ),
    backbone=dict(
        backbone=dict(
            ct_tubelet=True,
            amod_config=dict(enabled=False),
        )
    ),
    rpn_head=dict(
        conv_cfg=dict(type="Conv", kernel_size=3, padding=1),
        physical_grid_actionformer=dict(enabled=False, required=False),
    ),
)
work_dir = "exps/thumos/adatad/duca_ctdp_geometry_g0_seed3407"
