_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]

import os


yuzibo_root = os.environ.get("YUZIBO_ROOT", os.path.expanduser("~/run/yuzibo"))
thumos14_root = os.path.join(yuzibo_root, "thumos14")
annotation_path = os.environ.get(
    "THUMOS14_ANNOTATION_PATH",
    os.path.join(thumos14_root, "annotations", "thumos_14_anno.json"),
)
class_map = os.environ.get(
    "THUMOS14_CLASS_MAP",
    os.path.join(thumos14_root, "annotations", "category_idx.txt"),
)
train_data_path = os.environ.get("THUMOS14_TRAIN_DATA_PATH", os.path.join(thumos14_root, "train"))
test_data_path = os.environ.get("THUMOS14_TEST_DATA_PATH", os.path.join(thumos14_root, "test"))

teacher_route = dict(
    route="c3_dense_adatad_teacher",
    purpose="train full dense AdaTAD teacher for train-only detector utility export",
    base_config="./e2e_thumos_videomae_s_768x1_160_adapter.py",
    claim_lock="teacher checkpoint only; no sparse acquisition claim without downstream mAP",
)

dataset = dict(
    train=dict(
        ann_file=annotation_path,
        class_map=class_map,
        data_path=train_data_path,
    ),
    val=dict(
        ann_file=annotation_path,
        class_map=class_map,
        data_path=test_data_path,
    ),
    test=dict(
        ann_file=annotation_path,
        class_map=class_map,
        data_path=test_data_path,
    ),
)

workflow = dict(
    logging_interval=50,
    checkpoint_interval=10,
    val_loss_interval=-1,
    val_eval_interval=10,
    val_eval_interval_anchor_epoch=10,
    val_start_epoch=9,
    end_epoch=60,
)

work_dir = "exps/thumos/adatad/c3_dense_adatad_teacher_full_train"
