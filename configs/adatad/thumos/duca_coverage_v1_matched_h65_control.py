import os


_base_ = ["./duca_sampling_rate_curriculum_stage2_joint384.py"]


duca_thumos14_root = os.environ.get("DUCA_THUMOS14_ROOT", "data/thumos-14")
duca_thumos14_annotation = os.path.join(
    duca_thumos14_root, "annotations", "thumos_14_anno.json"
)
duca_thumos14_class_map = os.path.join(
    duca_thumos14_root, "annotations", "category_idx.txt"
)
duca_thumos14_video = os.path.join(duca_thumos14_root, "raw_data", "video")


# This arm is the matched H65 allocation control for Coverage-v1.  The complete
# Stage-1 EMA model is loaded by the inherited Stage-2 contract, while the
# scout and priority-producing adapter are frozen for the full comparison.
duca_coverage_v1_contract = dict(
    route="DUCA_COVERAGE_V1_MATCHED_H65_CONTROL",
    scientific_variable="allocation_rule_only",
    priority="frozen_h65_pre_allocation_center_scores",
    allocation="h65_budget_calibrated_sampling_rate",
    exact_budget=384,
    temporal_anchor_count=None,
    scout_trainable=False,
    terminal_model_rule="epoch_59_state_dict_ema",
    seed=3407,
)

model = dict(
    frame_selector=dict(
        freeze_priority_path=True,
        allow_frozen_coarse_probe=True,
        actionness_source_cfg=dict(
            frozen=True,
            trainable=False,
        ),
    ),
)

dataset = dict(
    train=dict(
        ann_file=duca_thumos14_annotation,
        class_map=duca_thumos14_class_map,
        data_path=duca_thumos14_video,
    ),
    val=dict(
        ann_file=duca_thumos14_annotation,
        class_map=duca_thumos14_class_map,
        data_path=duca_thumos14_video,
    ),
    test=dict(
        ann_file=duca_thumos14_annotation,
        class_map=duca_thumos14_class_map,
        data_path=duca_thumos14_video,
    ),
)

evaluation = dict(ground_truth_filename=duca_thumos14_annotation)

workflow = dict(
    # Enable the generic successful-update contract without restoring the
    # legacy selected-axis variant binder disabled by the Stage-2 recipe.
    formal_successful_update_contract=(
        os.environ.get("DUCA_COVERAGE_PRE_RUN", "0") != "1"
    ),
    training_profile="duca_coverage_v1_matched_h65_control",
    checkpoint_interval=5,
    primary_checkpoint_epoch=59,
    primary_checkpoint_state_key="state_dict_ema",
    checkpoint_criterion="terminal_epoch_59_state_dict_ema",
)

work_dir = "exps/thumos/adatad/duca_coverage_v1_matched_h65_control"
