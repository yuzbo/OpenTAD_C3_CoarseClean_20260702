_base_ = ["./duca_protected_physical_fixed384_official60_base.py"]

import os


evaluation_block_list = os.environ.get(
    "DUCA_RIME_PHASE1_EVAL_BLOCK_LIST",
    "",
).strip()
if not evaluation_block_list:
    raise RuntimeError("Phase-1 probe cost control requires its frozen block list")

dataset = dict(
    val=None,
    test=dict(
        subset_name="training",
        block_list=evaluation_block_list,
        test_mode=True,
        window_size=768,
    ),
)
evaluation = dict(
    subset="training",
    blocked_videos=evaluation_block_list,
)
model = dict(
    frame_selector=dict(
        arm="probe_uniform",
        detector_bridge_gradient_scale=0.0,
        uniform_companion_fraction=0.0,
    ),
    backbone=dict(
        backbone=dict(with_cp=False),
    ),
)
solver = dict(
    test=dict(batch_size=1, num_workers=0),
    static_graph=False,
)
post_processing = dict(save_dict=False)

duca_variant_contract = dict(
    variant="probe_uniform",
    coarse_probe_executed=True,
    probe_output_used_for_selection=False,
    learned_selector=False,
    detector_gradient_bridge=False,
)
duca_rime_phase1_cost_contract = dict(
    contract="duca_rime_phase1_probe_uniform_cost_v1",
    coarse_probe_executed=True,
    probe_output_used_for_selection=False,
    selection_policy="exact_uniform",
    checkpoint_drop_prefixes=(
        "frame_selector._loss_weight_schedule_step",
        "frame_selector.adapter.transition_scorer.",
    ),
    paired_checkpoint_identity_required=True,
    accuracy_claim_allowed=False,
    uses_official_final=False,
)

work_dir = "exps/thumos/adatad/duca_rime_phase1_probe_uniform_cost"

del evaluation_block_list
