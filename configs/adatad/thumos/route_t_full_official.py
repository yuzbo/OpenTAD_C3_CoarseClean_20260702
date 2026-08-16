import os


_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]

route_t_arm = os.environ.get("SPARSEHEAD_ROUTE_T_ARM", "risk")
route_t_seed = int(os.environ.get("SPARSEHEAD_ROUTE_T_SEED", "3407"))
valid_arms = {
    "dense",
    "risk",
    "generic",
    "shuffled_risk",
    "similarity",
    "random",
    "uniform",
    "bypass",
}
if route_t_arm not in valid_arms:
    raise ValueError(f"unknown SPARSEHEAD_ROUTE_T_ARM={route_t_arm}")

annotation_path = os.environ.get(
    "SPARSEHEAD_ANNOTATION",
    "/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json",
)
class_map = os.environ.get(
    "SPARSEHEAD_CLASS_MAP",
    "/data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt",
)
data_path = os.environ.get(
    "SPARSEHEAD_VIDEO_ROOT",
    "/data/run01/sczc063/yuzibo/thumos14/raw_data/video",
)
pretrained = os.environ.get(
    "SPARSEHEAD_PRETRAINED",
    "/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth",
)
run_root = os.environ.get(
    "SPARSEHEAD_RUN_ROOT",
    "/data/run01/sczc063/yuzibo/sparsehead_route_t_full_official",
)

dataset = dict(
    train=dict(ann_file=annotation_path, class_map=class_map, data_path=data_path),
    val=dict(ann_file=annotation_path, class_map=class_map, data_path=data_path),
    test=dict(ann_file=annotation_path, class_map=class_map, data_path=data_path),
)
evaluation = dict(ground_truth_filename=annotation_path)

model = dict(
    backbone=dict(
        backbone=dict(
            measure_preserving_coarsen_route=dict(
                enabled=route_t_arm != "dense",
                arm="risk" if route_t_arm == "dense" else route_t_arm,
                run_seed=route_t_seed,
                temperature=1.0,
                expected_tubelets=8,
            ),
        ),
        custom=dict(pretrain=pretrained),
    ),
)

optimizer = dict(
    backbone=dict(
        lr=0,
        weight_decay=0,
        custom=[
            dict(name="adapter", lr=2e-4, weight_decay=0.05),
            dict(name="measure_preserving_coarsen_route", lr=2e-4, weight_decay=0.05),
        ],
        exclude=["backbone"],
    ),
)

workflow = dict(end_epoch=60)
work_dir = os.path.join(run_root, f"{route_t_arm}_seed{route_t_seed}")

