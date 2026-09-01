_base_ = ["./continuous_roi_s2_v3_d160_seed4407.py"]

seed = 4407

train_pipeline = [
    dict(type="PrepareVideoInfo", format="mp4"),
    dict(type="mmaction.DecordInit", num_threads=4),
    dict(type="LoadFrames", num_clips=1, method="random_trunc", trunc_len=768, trunc_thresh=0.75, crop_ratio=[0.9, 1.0], scale_factor=1),
    dict(type="mmaction.DecordDecode"),
    dict(type="BAFDRSourceViews", global_size=96, output_key="bafdr_inputs", required_source_height=180, required_source_width=320),
    dict(type="ConvertToTensor", keys=["gt_segments", "gt_labels"]),
    dict(type="Collect", inputs="bafdr_inputs", keys=["masks", "gt_segments", "gt_labels"]),
]
evaluation_pipeline = [
    dict(type="PrepareVideoInfo", format="mp4"),
    dict(type="mmaction.DecordInit", num_threads=4),
    dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=1),
    dict(type="mmaction.DecordDecode"),
    dict(type="BAFDRSourceViews", global_size=96, output_key="bafdr_inputs", required_source_height=180, required_source_width=320),
    dict(type="ConvertToTensor", keys=["imgs"]),
    dict(type="Collect", inputs="bafdr_inputs", keys=["masks"]),
]

dataset = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=evaluation_pipeline),
    test=dict(pipeline=evaluation_pipeline),
)

model = dict(
    backbone=dict(
        custom=dict(
            wrapper_type="bafdr_k16_shared_videomae",
            bafdr_global_key="global",
            bafdr_source_key="source",
            bafdr_global_size=96,
            bafdr_local_size=128,
            bafdr_chunk_num=48,
            bafdr_k_chunks=16,
            bafdr_tubelets_per_chunk=8,
            bafdr_output_length=768,
            bafdr_uniform_mode=False,
            bafdr_return_bundle=True,
        )
    ),
    projection=dict(
        type="BAFDRAsymmetricProjection",
        in_channels=384,
        out_channels=384,
        arch=(2, 2, 5),
        conv_cfg=dict(kernel_size=3, proj_pdrop=0.0),
        norm_cfg=dict(type="LN"),
        attn_cfg=dict(n_head=4, n_mha_win_size=19),
        use_abs_pe=True,
        max_seq_len=2304,
    ),
)

bafdr_protocol = dict(
    protocol="ZOOMTOKEN-BA-FDR-K16-FULLMATRIX-v001",
    arm="BAFDR-K16-NOKD",
    seed=4407,
    k_chunks=16,
    uniform_mode=False,
    asymmetric_projection=True,
    distillation=False,
)
work_dir = f"exps/thumos/adatad/bafdr_k16_nokd_seed4407"
