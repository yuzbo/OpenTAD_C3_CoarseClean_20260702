"""Static, no-execution protocol packet for the DUCA dynamic-B pilot."""

protocol_id = "DUCA_DYNAMIC_B_N16R4_F1_F2_V001"
dataset = "THUMOS14"
backend = "OpenTAD-AdaTAD"
arms = ("dense", "uniform_k384", "dynamic_A", "dynamic_B", "k_shuffle", "no_risk")
shared = dict(detector="AdaTAD", loss="shared_detector_loss", nms="shared_nms", evaluator="THUMOS14_evaluator")
firewall = dict(fit="train_only", cal="calibration_only", hold="held_out_only", cross_split_access=False)
dynamic_policy = dict(dynamic_outer_k_required=True, batch_independent_per_video_window=True)
dynamic_A = dict(freedom_ceiling_only=True, claim_bearing=False)
dynamic_B = dict(claim_route="DUCA_HIERARCHICAL_DYNAMIC_PHYSICAL_ACQUISITION_B-v001", bounded_monotone_local_exact_k=True)
f1 = dict(name="F1", positions="endpoint_inclusive_integer_half_up_uniform")
f2 = dict(name="F2", positions="nonce_derived_canonical_row_order_fisher_yates")
cost_requirement = "matched_realized_mean_full_stack_cost_across_all_arms"
pilot = dict(repeats=4, windows=16, realized_cost_required=True, no_efficacy_claim=True)
