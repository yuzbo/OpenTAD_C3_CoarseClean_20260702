_base_ = ["./duca_ct_dual_phase_bamod_thumos.py"]

# CT-DP v1: calibrated level-nominal spacing.  The base configuration keeps
# the original absolute-spacing route as the v0 scientific control.
model = dict(
    rpn_head=dict(
        conv_cfg=dict(
            type="ContinuousTimeScaleAdaptiveConv1d",
            reference_spacing_mode="level_nominal",
        ),
    ),
)

work_dir = "exps/thumos/adatad/duca_ct_dual_phase_bamod_revised_seed3407"
