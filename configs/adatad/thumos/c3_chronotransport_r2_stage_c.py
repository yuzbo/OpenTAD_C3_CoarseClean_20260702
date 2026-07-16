_base_ = ["./c3_chronotransport_r2_stage_b.py"]

chronotransport_r2 = dict(
    protocol_id="CT-P3R-3S-r2",
    stage="C",
    seeds=(3407, 3408, 3409),
    successful_updates=4200,
    epochs=60,
    batch_size=2,
    world_size=1,
    amp_fp16=True,
    gradient_accumulation=False,
    max_overflow_retries=3,
)

