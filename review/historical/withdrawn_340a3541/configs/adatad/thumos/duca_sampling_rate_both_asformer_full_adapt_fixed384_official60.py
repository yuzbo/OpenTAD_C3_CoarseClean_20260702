_base_ = ["./duca_sampling_rate_both_fixed384_official60.py"]

duca_sampling_rate_contract = dict(
    variant="rate_plus_cls_reg_contribution_with_full_asformer_encoder_adaptation",
    contribution_components="both",
    asformer_last_layer_adapted=False,
    asformer_full_encoder_adapted=True,
)

model = dict(
    frame_selector=dict(
        actionness_source_cfg=dict(
            policy_hidden_gradient_scope="asformer_full_encoder",
        ),
        loss_weight_schedule=dict(
            asformer_adapt=dict(
                start=0.0,
                end=1.0,
                warmup_steps=1500,
                transition_steps=900,
            ),
        ),
    )
)

work_dir = "exps/thumos/adatad/duca_sampling_rate_both_asformer_full_adapt_fixed384_official60"
