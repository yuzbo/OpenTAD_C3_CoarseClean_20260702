_base_ = ["./duca_trainfree_fixed384_official60_base.py"]


duca_trainfree_contract = dict(
    evidence="frozen_imagenet_classifier_confidence_rank",
    allocation="global_exact_k_max_hole",
)


model = dict(
    frame_selector=dict(
        actionness_source_cfg=dict(
            source_name="frozen_imagenet_mobilenetv3_semantic_saliency",
            train_free_evidence_mode="frozen_semantic_saliency",
        ),
    ),
)


work_dir = "exps/thumos/adatad/duca_trainfree_mobilenet_semantic_fixed384_official60"
