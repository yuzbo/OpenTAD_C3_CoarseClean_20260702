# Copyright (c) OpenTAD. All rights reserved.
from pathlib import Path

def generate_configs():
    seeds = [4407, 4408, 4409]
    configs_dir = Path("configs/adatad/thumos")
    configs_dir.mkdir(parents=True, exist_ok=True)

    bafdr_solver_str = '''dict(
    train=dict(batch_size=2, num_workers=2),
    val=dict(batch_size=2, num_workers=2),
    test=dict(batch_size=2, num_workers=2),
)'''

    # 1. D160
    for s in seeds:
        content = f'''_base_ = ["./continuous_roi_s2_v3_d160_seed{s}.py"]

seed = {s}
solver = {bafdr_solver_str}

bafdr_protocol = dict(
    protocol="ZOOMTOKEN-BA-FDR-K16-FULLMATRIX-v001",
    arm="D160",
    seed={s},
)
work_dir = f"exps/thumos/adatad/bafdr_k16_d160_seed{s}"
'''
        (configs_dir / f"bafdr_k16_d160_seed{s}.py").write_text(content, encoding="utf-8")

    # 2. G96
    for s in seeds:
        content = f'''_base_ = ["./continuous_roi_s2_v3_g96_seed{s}.py"]

seed = {s}
solver = {bafdr_solver_str}

bafdr_protocol = dict(
    protocol="ZOOMTOKEN-BA-FDR-K16-FULLMATRIX-v001",
    arm="G96",
    seed={s},
)
work_dir = f"exps/thumos/adatad/bafdr_k16_g96_seed{s}"
'''
        (configs_dir / f"bafdr_k16_g96_seed{s}.py").write_text(content, encoding="utf-8")

    # 3. U128-ALL48-A0
    for s in seeds:
        content = f'''_base_ = ["./continuous_roi_s2_v3_u128_a0_seed{s}.py"]

seed = {s}
solver = {bafdr_solver_str}

bafdr_protocol = dict(
    protocol="ZOOMTOKEN-BA-FDR-K16-FULLMATRIX-v001",
    arm="U128-ALL48-A0",
    seed={s},
)
work_dir = f"exps/thumos/adatad/bafdr_k16_u128_all48_a0_seed{s}"
'''
        (configs_dir / f"bafdr_k16_u128_all48_a0_seed{s}.py").write_text(content, encoding="utf-8")

    # Pipeline generator helper
    def bafdr_pipeline_str(is_train=True):
        if is_train:
            return '''[
    dict(type="PrepareVideoInfo", format="mp4"),
    dict(type="mmaction.DecordInit", num_threads=4),
    dict(type="LoadFrames", num_clips=1, method="random_trunc", trunc_len=768, trunc_thresh=0.75, crop_ratio=[0.9, 1.0], scale_factor=1),
    dict(type="mmaction.DecordDecode"),
    dict(type="BAFDRSourceViews", global_size=96, output_key="bafdr_inputs", required_source_height=180, required_source_width=320),
    dict(type="ConvertToTensor", keys=["gt_segments", "gt_labels"]),
    dict(type="Collect", inputs="bafdr_inputs", keys=["masks", "gt_segments", "gt_labels"], meta_keys=["id", "fps", "duration", "video_name", "snippet_boundaries", "window_start_frame", "bafdr_geometry"]),
]'''
        else:
            return '''[
    dict(type="PrepareVideoInfo", format="mp4"),
    dict(type="mmaction.DecordInit", num_threads=4),
    dict(type="LoadFrames", num_clips=1, method="sliding_window"),
    dict(type="mmaction.DecordDecode"),
    dict(type="BAFDRSourceViews", global_size=96, output_key="bafdr_inputs", required_source_height=180, required_source_width=320),
    dict(type="ConvertToTensor", keys=["gt_segments", "gt_labels"]),
    dict(type="Collect", inputs="bafdr_inputs", keys=["masks", "gt_segments", "gt_labels"], meta_keys=["id", "fps", "duration", "video_name", "snippet_boundaries", "window_start_frame", "bafdr_geometry"]),
]'''

    bafdr_optimizer_str = '''dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=0.05,
    paramwise=True,
    backbone=dict(
        lr=0,
        weight_decay=0,
        custom=[
            dict(name="adapter", lr=2e-4, weight_decay=0.05),
            dict(name="router", lr=1e-4, weight_decay=0.01),
            dict(name="gamma", lr=1e-3, weight_decay=0.0),
            dict(name="proj_local", lr=1e-4, weight_decay=0.01),
            dict(name="proj_global", lr=1e-4, weight_decay=0.01),
        ],
        exclude=["backbone.model"],
    ),
)'''

    # 4. U16-UNIFORM-A0
    for s in seeds:
        content = f'''_base_ = ["./continuous_roi_s2_v3_d160_seed{s}.py"]

seed = {s}
solver = {bafdr_solver_str}

train_pipeline = {bafdr_pipeline_str(True)}
evaluation_pipeline = {bafdr_pipeline_str(False)}

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
            bafdr_uniform_mode=True,
            bafdr_return_bundle=True,
        )
    ),
    projection=dict(
        type="BAFDRAsymmetricProjection",
        in_channels=384,
        out_channels=512,
        arch=(2, 2, 5),
        conv_cfg=dict(kernel_size=3, proj_pdrop=0.0),
        norm_cfg=dict(type="LN"),
        attn_cfg=dict(n_head=4, n_mha_win_size=-1),
        use_abs_pe=True,
        max_seq_len=2304,
    ),
)

optimizer = {bafdr_optimizer_str}

bafdr_protocol = dict(
    protocol="ZOOMTOKEN-BA-FDR-K16-FULLMATRIX-v001",
    arm="U16-UNIFORM-A0",
    seed={s},
    k_chunks=16,
    uniform_mode=True,
    asymmetric_projection=True,
    distillation=False,
)
work_dir = f"exps/thumos/adatad/bafdr_k16_u16_uniform_a0_seed{s}"
'''
        (configs_dir / f"bafdr_k16_u16_uniform_a0_seed{s}.py").write_text(content, encoding="utf-8")

    # 5. BAFDR-K16-LATE
    for s in seeds:
        content = f'''_base_ = ["./continuous_roi_s2_v3_d160_seed{s}.py"]

seed = {s}
solver = {bafdr_solver_str}

train_pipeline = {bafdr_pipeline_str(True)}
evaluation_pipeline = {bafdr_pipeline_str(False)}

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
        type="BAFDRLateProjection",
        in_channels=384,
        out_channels=512,
        arch=(2, 2, 5),
        conv_cfg=dict(kernel_size=3, proj_pdrop=0.0),
        norm_cfg=dict(type="LN"),
        attn_cfg=dict(n_head=4, n_mha_win_size=-1),
        use_abs_pe=True,
        max_seq_len=2304,
    ),
)

optimizer = {bafdr_optimizer_str}

bafdr_protocol = dict(
    protocol="ZOOMTOKEN-BA-FDR-K16-FULLMATRIX-v001",
    arm="BAFDR-K16-LATE",
    seed={s},
    k_chunks=16,
    uniform_mode=False,
    asymmetric_projection=False,
    distillation=False,
)
work_dir = f"exps/thumos/adatad/bafdr_k16_late_seed{s}"
'''
        (configs_dir / f"bafdr_k16_late_seed{s}.py").write_text(content, encoding="utf-8")

    # 6. BAFDR-K16-NOKD
    for s in seeds:
        content = f'''_base_ = ["./continuous_roi_s2_v3_d160_seed{s}.py"]

seed = {s}
solver = {bafdr_solver_str}

train_pipeline = {bafdr_pipeline_str(True)}
evaluation_pipeline = {bafdr_pipeline_str(False)}

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
        out_channels=512,
        arch=(2, 2, 5),
        conv_cfg=dict(kernel_size=3, proj_pdrop=0.0),
        norm_cfg=dict(type="LN"),
        attn_cfg=dict(n_head=4, n_mha_win_size=-1),
        use_abs_pe=True,
        max_seq_len=2304,
    ),
)

optimizer = {bafdr_optimizer_str}

bafdr_protocol = dict(
    protocol="ZOOMTOKEN-BA-FDR-K16-FULLMATRIX-v001",
    arm="BAFDR-K16-NOKD",
    seed={s},
    k_chunks=16,
    uniform_mode=False,
    asymmetric_projection=True,
    distillation=False,
)
work_dir = f"exps/thumos/adatad/bafdr_k16_nokd_seed{s}"
'''
        (configs_dir / f"bafdr_k16_nokd_seed{s}.py").write_text(content, encoding="utf-8")

    # 7. BAFDR-K16-FULL
    for s in seeds:
        content = f'''_base_ = ["./continuous_roi_s2_v3_d160_seed{s}.py"]

seed = {s}
solver = {bafdr_solver_str}

train_pipeline = {bafdr_pipeline_str(True)}
evaluation_pipeline = {bafdr_pipeline_str(False)}

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
        out_channels=512,
        arch=(2, 2, 5),
        conv_cfg=dict(kernel_size=3, proj_pdrop=0.0),
        norm_cfg=dict(type="LN"),
        attn_cfg=dict(n_head=4, n_mha_win_size=-1),
        use_abs_pe=True,
        max_seq_len=2304,
    ),
)

optimizer = {bafdr_optimizer_str}

bafdr_protocol = dict(
    protocol="ZOOMTOKEN-BA-FDR-K16-FULLMATRIX-v001",
    arm="BAFDR-K16-FULL",
    seed={s},
    k_chunks=16,
    uniform_mode=False,
    asymmetric_projection=True,
    distillation=True,
    teacher_config=f"configs/adatad/thumos/bafdr_k16_d160_seed{s}.py",
    teacher_checkpoint=f"exps/thumos/adatad/bafdr_k16_d160_seed{s}/checkpoint/epoch_59.pth",
)
work_dir = f"exps/thumos/adatad/bafdr_k16_full_seed{s}"
'''
        (configs_dir / f"bafdr_k16_full_seed{s}.py").write_text(content, encoding="utf-8")

    print(f"Successfully generated 21 config files for BA-FDR in {configs_dir}")

if __name__ == "__main__":
    generate_configs()
