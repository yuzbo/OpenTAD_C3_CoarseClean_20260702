_base_ = ["./duca_sampling_rate_both_asformer_full_adapt_fixed384_official60.py"]

import os


def _required(name):
    value = os.environ.get(name, "")
    if not value:
        raise ValueError(f"{name} is required for DUCA rate curriculum stage 2")
    return value


duca_stage1_checkpoint = _required("DUCA_STAGE1_CHECKPOINT")
duca_stage1_checkpoint_sha256 = _required("DUCA_STAGE1_CHECKPOINT_SHA256")
duca_stage1_checkpoint_epoch = int(_required("DUCA_STAGE1_CHECKPOINT_EPOCH"))

# Stage 2 has a single, fresh 6,000-step optimizer schedule.  The first half
# retains coarse supervision while smoothly turning on learned sampling and
# detector feedback; the second half is TAD-led but never drops semantic and
# transition supervision to zero.
duca_stage2_half_steps = 3000

duca_sampling_rate_contract = dict(
    route="DUCA_RATE_CURRICULUM_STAGE2_JOINT384",
    task="offline_temporal_action_detection",
    stage="low_lr_joint_rate_adaptation_then_tad_led_joint_training",
    pre_backbone_plugin=True,
    stage1_initialization="full_uniform_k384_ema_model",
    optimizer_scheduler_amp_state_reset=True,
    detector_gradient="density_transport_st",
    final_loss_emphasis=dict(
        detector=1.0,
        actionness=0.25,
        transition=0.10,
        transition_boundary=0.25,
    ),
    paper_claim_allowed=False,
)

model = dict(
    frame_selector=dict(
        coarse_trunk_lr=1.0e-5,
        action_head_lr=2.0e-5,
        transition_scorer_lr=5.0e-5,
        loss_weights=dict(
            actionness=1.0,
            transition=0.50,
            transition_boundary=2.0,
        ),
        loss_weight_schedule=dict(
            _delete_=True,
            type="progressive_joint",
            shape="cosine",
            warmup_steps=0,
            transition_steps=duca_stage2_half_steps,
            actionness=dict(
                start=1.0,
                end=0.25,
                warmup_steps=0,
                transition_steps=duca_stage2_half_steps,
            ),
            transition=dict(
                start=0.50,
                end=0.10,
                warmup_steps=0,
                transition_steps=duca_stage2_half_steps,
            ),
            transition_boundary=dict(
                start=2.0,
                end=0.25,
                warmup_steps=0,
                transition_steps=duca_stage2_half_steps,
            ),
            policy_alpha=dict(
                start=0.0,
                end=1.0,
                warmup_steps=0,
                transition_steps=duca_stage2_half_steps,
            ),
            detector_gradient=dict(
                start=0.0,
                end=0.25,
                warmup_steps=1000,
                transition_steps=2000,
            ),
            detector_contribution=dict(
                start=0.0,
                end=1.0,
                warmup_steps=1000,
                transition_steps=2000,
            ),
            asformer_adapt=dict(
                start=0.0,
                end=1.0,
                warmup_steps=0,
                transition_steps=duca_stage2_half_steps,
            ),
        ),
        actionness_source_cfg=dict(
            policy_hidden_gradient_scope="asformer_full_encoder",
        ),
    ),
)

workflow = dict(
    # This curriculum candidate uses the same full THUMOS training and
    # validation protocol as the official-60 arms, but its phase boundary is
    # deliberately outside the frozen selected-axis evidence runtime.  It
    # becomes paper-comparable only after the measured model result is sealed.
    formal_protocol="",
    # This is a new curriculum candidate, not one of the sealed legacy P0
    # variants.  Leaving the inherited P0 contract enabled routes it through
    # the legacy variant binder before model initialization.
    formal_successful_update_contract=False,
    training_profile="duca_rate_curriculum_stage2_joint384",
    model_initialization=dict(
        enabled=True,
        checkpoint_path=duca_stage1_checkpoint,
        checkpoint_sha256=duca_stage1_checkpoint_sha256,
        state_key="state_dict_ema",
        expected_checkpoint_epoch=duca_stage1_checkpoint_epoch,
        reset_state_keys=["frame_selector._loss_weight_schedule_step"],
    ),
)

work_dir = "exps/thumos/adatad/duca_sampling_rate_curriculum_stage2_joint384"
