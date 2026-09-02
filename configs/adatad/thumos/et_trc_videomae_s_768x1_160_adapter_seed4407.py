import os

_base_ = ["./continuous_roi_s2_v3_d160_seed4407.py"]
_pretrain = os.environ.get(
    "ETTRC_PRETRAIN",
    "/data/run01/sczc063/yuzibo/pretrained/"
    "vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth",
)

model = dict(
    backbone=dict(
        type="mmaction.Recognizer3D",
        backbone=dict(
            type="ETTRCVisionTransformerAdapter",
            img_size=160,
            patch_size=16,
            embed_dims=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4,
            qkv_bias=True,
            num_frames=16,
            drop_path_rate=0.1,
            norm_cfg=dict(type="LN", eps=1e-6),
            return_feat_map=True,
            with_cp=True,
            total_frames=768,
            adapter_index=list(range(12)),
            adapter_cfg=dict(mlp_ratio=0.25, kernel_size=3, dilation=1),
            stride_k=4,
            # The implementation is fixed-stride Taylor carryover.  It is
            # intentionally not called event-triggered until a learned event
            # policy is implemented and independently evaluated.
            segment_size=8,
            enable_taylor=True,
            jacobian_rank=64,
        ),
        custom=dict(
            pretrain=_pretrain,
            pretrain_required=True,
        ),
    ),
)

optimizer = dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=0.05,
    paramwise=True,
    backbone=dict(
        lr=0,
        weight_decay=0,
        custom=[
            dict(name="adapter", lr=2e-4, weight_decay=0.05),
            dict(name="jacobian_approx", lr=2e-4, weight_decay=0.05),
        ],
        exclude=["backbone"],
    ),
)

work_dir = "exps/thumos/adatad/et_trc_videomae_s_768x1_160_adapter_seed4407"
