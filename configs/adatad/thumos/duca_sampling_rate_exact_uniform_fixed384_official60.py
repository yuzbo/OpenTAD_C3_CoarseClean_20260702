_base_ = ["./duca_sampling_rate_fixed384_official60.py"]


# Strict matched control for the sampling-rate matrix.  The same pre-backbone
# wrapper and detector remain present, but alpha=0 makes the production decoder
# emit canonical endpoint-uniform observations and removes all learned utility
# feedback.  It is not a separate sampler.
duca_sampling_rate_contract = dict(
    variant="sampling_rate_exact_uniform_control",
    contribution_components="none",
    asformer_last_layer_adapted=False,
    asformer_full_encoder_adapted=False,
    force_exact_uniform_control=True,
)


model = dict(
    frame_selector=dict(
        training_uniform_companion_fraction=0.0,
        training_uniform_companion_normalize_learned_gradient=False,
        loss_weight_schedule=dict(
            detector_gradient=dict(
                start=0.0,
                end=0.0,
                warmup_steps=0,
                transition_steps=1,
            ),
            policy_alpha=dict(
                start=0.0,
                end=0.0,
                warmup_steps=0,
                transition_steps=1,
            ),
            detector_contribution=dict(
                start=0.0,
                end=0.0,
                warmup_steps=0,
                transition_steps=1,
            ),
        ),
    )
)


work_dir = "exps/thumos/adatad/duca_sampling_rate_exact_uniform_fixed384_official60"
