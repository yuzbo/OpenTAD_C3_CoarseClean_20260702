_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]

# Continuous-Time Scale-Adaptive & Dual-Phase B-AMoD configuration for THUMOS14
window_size = 768
scale_factor = 1
selected_budget = 384
scaffold_budget = 128
burst_budget = 256
chunk_num = selected_budget // 16  # 24 chunks of 16 frames

model = dict(
    type="ActionFormer",
    # 1. Dual-Phase Frame Selector: reduces 768 -> 384 frames before backbone
    frame_selector=dict(
        type="DualPhaseFrameSelector",
        total_budget=selected_budget,
        scaffold_budget=scaffold_budget,
        burst_budget=burst_budget,
        burst_radius=2,
    ),
    # 2. B-AMoD Vision Transformer Backbone (VideoMAE-S)
    backbone=dict(
        type="mmaction.Recognizer3D",
        backbone=dict(
            type="VisionTransformerAdapter",
            img_size=224,
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
            total_frames=16,
            adapter_index=list(range(12)),
            # B-AMoD (Boundary-Biased Attention routing for Mixture-of-Depths)
            amod_config=dict(
                enabled=True,
                capacity=0.5,
                amod_layers=[1, 3, 5, 7, 9, 11],
                boundary_prior_scale=0.25,
            ),
        ),
        custom=dict(
            pretrain="pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth",
            pre_processing_pipeline=[
                dict(type="Rearrange", keys=["frames"], ops="b n c (t1 t) h w -> (b t1) n c t h w", t1=chunk_num),
            ],
            post_processing_pipeline=[
                dict(type="Reduce", keys=["feats"], ops="b n c t h w -> b c t", reduction="mean"),
                dict(type="Rearrange", keys=["feats"], ops="(b t1) c t -> b c (t1 t)", t1=chunk_num),
                dict(type="Interpolate", keys=["feats"], size=selected_budget),
            ],
            norm_eval=False,
            freeze_backbone=False,
        ),
    ),
    # 3. Projection aligned with FPN/Head channels (384 -> 512)
    projection=dict(
        in_channels=384,
        out_channels=512,
        max_seq_len=selected_budget,
        attn_cfg=dict(n_mha_win_size=-1),
    ),
    # 4. Continuous-Time Scale-Adaptive Convolution in Detection Head
    rpn_head=dict(
        in_channels=512,
        feat_channels=512,
        conv_cfg=dict(type="ContinuousTimeScaleAdaptiveConv1d"),
    ),
)

work_dir = "exps/thumos/adatad/duca_ct_dual_phase_bamod_seed3407"

workflow = dict(
    logging_interval=50,
    checkpoint_interval=2,
    val_loss_interval=-1,
    val_eval_interval=2,
    val_start_epoch=1,
    end_epoch=60,
)

