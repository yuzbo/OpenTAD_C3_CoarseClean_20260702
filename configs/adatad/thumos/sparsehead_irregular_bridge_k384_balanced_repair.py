"""Diagnostic SparseHead repair candidate with balanced hard assignment."""

_base_ = ["./sparsehead_irregular_bridge_k384_baseline.py"]

candidate_name = "sparsehead_irregular_bridge_k384_balanced_repair"

model = dict(
    rpn_head=dict(
        allow_center_fallback_inside_gt=True,
        hard_min_points_per_gt=4,
        hard_min_points_per_level=2,
        hard_max_points_per_gt=10,
        route_contract=dict(
            route_label="SPARSEHEAD_ASSIGNMENT_REPAIR_PENDING_GATE",
            compatibility="irregular_geometry_repair_candidate",
            dense_equivalent_claim_allowed=False,
            allow_legacy_full_cell_span=False,
            allow_center_fallback_inside_gt=True,
            gt_axis="native",
            proposal_axis="native",
            nms_axis="native",
            postprocess_axis="native",
            eval_axis="seconds",
            expected_axis_contract=dict(
                gt_axis="native",
                proposal_axis="native",
                nms_axis="native",
                postprocess_axis="native",
            ),
            diagnostic_only=True,
            primary_result_allowed=False,
        ),
    ),
)

work_dir = "exps/thumos/adatad/sparsehead_irregular_bridge_k384_balanced_repair"
