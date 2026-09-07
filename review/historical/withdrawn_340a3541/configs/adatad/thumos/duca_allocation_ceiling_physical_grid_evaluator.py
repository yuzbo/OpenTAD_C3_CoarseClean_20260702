_base_ = [
    "./duca_cellcf_transition_beta0_fixed384_official_adatad_backend_full_train.py"
]


allocation_ceiling_evaluator_contract = dict(
    task="offline_temporal_action_detection",
    execution="frozen_checkpoint_read_only_candidate_loss",
    trains_model=False,
    selector_execution=False,
    physical_grid_actionformer=True,
    dense_axis_gt=True,
    selected_axis_gt_remap=False,
    metric_claim_allowed=False,
    paper_claim_allowed=False,
)


model = dict(
    rpn_head=dict(
        physical_grid_actionformer=dict(
            enabled=True,
            required=True,
            strict=True,
            eps=1.0e-6,
            diagnostic=dict(
                emit_score_iou_entry=True,
                emit_proposal_cap_entry=True,
                emit_selected_vs_physical_axis_entry=True,
            ),
        ),
        assignment_debug=dict(enabled=True),
    ),
)


inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(save_dict=False)
work_dir = "exps/thumos/adatad/duca_allocation_ceiling_physical_grid_evaluator"
