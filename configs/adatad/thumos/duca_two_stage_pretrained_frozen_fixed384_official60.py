_base_ = ["./duca_two_stage_pretrained_joint_fixed384_official60.py"]


duca_transition_only_contract = dict(
    route="DUCA_TWO_STAGE_PRETRAINED_FROZEN_FIXED384_OFFICIAL60",
    coarse_probe_training="frozen_for_all_official60_updates",
    frozen_components=["spatial_stem", "official_asformer", "action_head"],
    trainable_selector_components=["transition_scorer"],
    actionness_loss_weight_after_initialization=0.0,
)


model = dict(
    frame_selector=dict(
        allow_frozen_coarse_probe=True,
        actionness_source_cfg=dict(frozen=True, trainable=False),
        loss_weights=dict(actionness=0.0),
        loss_weight_schedule=dict(
            actionness=dict(start=0.0, end=0.0),
        ),
    ),
)


work_dir = "exps/thumos/adatad/duca_two_stage_pretrained_frozen_fixed384_official60"
