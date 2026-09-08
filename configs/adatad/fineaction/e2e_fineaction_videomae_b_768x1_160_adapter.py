"""Official AdaTAD-B architecture adapted to FineAction; see the protocol note."""
_base_ = ['../thumos/e2e_thumos_videomae_b_768x1_160_adapter.py']

annotation_path = 'data/fineaction/annotations/native_metadata.json'
class_map = 'data/fineaction/annotations/category_idx.txt'
data_path = 'data/fineaction/videos'
dataset = dict(
    train=dict(type='AnetPaddingDataset', ann_file=annotation_path,
               class_map=class_map, data_path=data_path, block_list=None,
               filter_gt=True, fps=-1),
    val=dict(type='AnetSlidingDataset', ann_file=annotation_path,
             class_map=class_map, data_path=data_path, block_list=None,
             filter_gt=False, test_mode=True, fps=-1, window_overlap_ratio=0.5),
    test=dict(type='AnetSlidingDataset', ann_file=annotation_path,
              class_map=class_map, data_path=data_path, block_list=None,
              filter_gt=False, test_mode=True, fps=-1, window_overlap_ratio=0.5),
)
model = dict(rpn_head=dict(num_classes=106, loss_normalizer=100.0))
solver = dict(train=dict(batch_size=2, num_workers=4),
              val=dict(batch_size=1, num_workers=2),
              test=dict(batch_size=1, num_workers=2))
evaluation = dict(
    _delete_=True, type='mAP', subset='validation',
    ground_truth_filename='data/fineaction/annotations/annotations_gt.json',
    tiou_thresholds=[0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],
)
post_processing = dict(save_dict=True)
workflow = dict(logging_interval=50, checkpoint_interval=1,
                val_loss_interval=-1, val_eval_interval=5, val_start_epoch=0,
                end_epoch=60)
work_dir = 'exps/fineaction/adatad_b_768_160_seed0'
