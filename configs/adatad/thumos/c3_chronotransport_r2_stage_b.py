_base_ = ["./c3_chronotransport_adatad_videomae_s_768x1_160_stage_b.py"]

model = dict(
    backbone=dict(
        backbone=dict(
            chronotransport=dict(
                # Neutralize the inherited legacy compatibility field.  The r2
                # runtime contract uses the two explicit age fields below.
                max_cache_age=None,
                hard_cache_validity_age=47,
                transport_age_embedding_cap=8,
                signal_dims=6,
                risk_hidden_dims=64,
                risk_quantile=0.9,
            )
        )
    )
)

chronotransport_r2 = dict(
    protocol_id="CT-P3R-3S-r2",
    stage="B",
    seeds=(3407, 3408, 3409),
    successful_updates=140,
    batch_size=1,
    world_size=1,
    fp32=True,
    shuffle=False,
    candidate_exposure="candidate=(p+5*b+seed_offset)%16",
)
