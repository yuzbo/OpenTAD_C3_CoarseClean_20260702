_base_ = ["./duca_ct_dual_phase_bamod_thumos.py"]

# Ablation Arm 3: Dual-Phase Selector + Dense ViT-Adapter + CT-Conv1d (Ablating B-AMoD)
model = dict(
    backbone=dict(
        backbone=dict(
            amod_config=dict(
                enabled=False,
            ),
        ),
    ),
)

work_dir = "exps/thumos/adatad/duca_ct_dual_phase_densevit_seed3407"
