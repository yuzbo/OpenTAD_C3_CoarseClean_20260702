_base_ = ["./duca_ct_dual_phase_bamod_thumos.py"]

# Ablation Arm 4: Dual-Phase Selector + Dense ViT-Adapter + Standard Conv1d (Ablating BOTH CT-Conv and B-AMoD / Base Control)
model = dict(
    backbone=dict(
        backbone=dict(
            amod_config=dict(
                enabled=False,
            ),
        ),
    ),
    rpn_head=dict(
        conv_cfg=None,  # standard nn.Conv1d
    ),
)

work_dir = "exps/thumos/adatad/duca_dual_phase_densevit_stdconv_seed3407"
