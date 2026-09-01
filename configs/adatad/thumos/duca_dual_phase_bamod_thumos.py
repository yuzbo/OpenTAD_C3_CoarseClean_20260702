_base_ = ["./duca_ct_dual_phase_bamod_thumos.py"]

# Ablation Arm 2: Dual-Phase Selector + B-AMoD ViT-Adapter with Standard Conv1d in Detection Head (Ablating CT-Conv)
model = dict(
    rpn_head=dict(
        conv_cfg=None,  # standard nn.Conv1d
    ),
)

work_dir = "exps/thumos/adatad/duca_dual_phase_bamod_stdconv_seed3407"
