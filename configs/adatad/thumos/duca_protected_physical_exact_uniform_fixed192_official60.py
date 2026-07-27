_base_ = ["./duca_protected_physical_exact_uniform_fixed384_official60.py"]

selected_budget = 192
chunk_num = selected_budget // 16

model = dict(
    frame_selector=dict(
        budget=selected_budget,
    ),
    backbone=dict(
        backbone=dict(
            total_frames=selected_budget,
            with_cp=False,
        ),
        custom=dict(
            pre_processing_pipeline=[
                dict(
                    type="Rearrange",
                    keys=["frames"],
                    ops="b n c (t1 t) h w -> (b t1) n c t h w",
                    t1=chunk_num,
                ),
            ],
            post_processing_pipeline=[
                dict(
                    type="Reduce",
                    keys=["feats"],
                    ops="b n c t h w -> b c t",
                    reduction="mean",
                ),
                dict(
                    type="Rearrange",
                    keys=["feats"],
                    ops="(b t1) c t -> b c (t1 t)",
                    t1=chunk_num,
                ),
                dict(type="Interpolate", keys=["feats"], size=selected_budget),
            ],
        ),
    ),
    projection=dict(max_seq_len=selected_budget),
)

duca_variant_contract = dict(
    variant="exact_uniform",
    exact_budget=selected_budget,
    native_heavy_frames=True,
    pad_to_k384=False,
    empirically_supported=False,
)

work_dir = "exps/thumos/adatad/duca_protected_physical_exact_uniform_fixed192_official60"
