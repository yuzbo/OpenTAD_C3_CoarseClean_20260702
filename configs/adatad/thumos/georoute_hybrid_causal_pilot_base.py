"""Frozen support-only base for the GeoRoute Hybrid causal pilot.

Arm-specific binders must overwrite the structured route mode and the explicit
context/ROI/residual quotas.  This source file deliberately contains no
official-test population and no representation side channel.
"""

_base_ = ["./georoute_adatad_development_base.py"]

model = dict(
    backbone=dict(
        custom=dict(
            georoute_tokens_per_tubelet=64,
            georoute_context_tokens=8,
            georoute_structured_context_tokens=8,
            georoute_structured_roi_tokens=28,
            georoute_structured_residual_tokens=28,
            georoute_route_mode="structured_hybrid",
            georoute_policy_estimator="score_function",
            georoute_policy_temperature=0.7,
            georoute_score_function_weight=1.0,
            georoute_score_function_baseline_momentum=0.95,
            georoute_score_function_temporal_reduction="mean",
            georoute_route_study_seed=5227,
            georoute_random_seed=5227,
            georoute_roi_temperature=0.25,
            georoute_geometry_stride_tubelets=1,
            georoute_geometry_temporal_shift_tubelets=0,
            georoute_absolute_position_enabled=True,
            georoute_absolute_coordinates_enabled=False,
            georoute_roi_relative_coordinates_enabled=False,
            georoute_geometry_projection_enabled=False,
            georoute_geometry_side_channel=False,
            georoute_diagnostic_telemetry_enabled=True,
            georoute_pooling_mode="uniform_selected",
            georoute_geometry_smoothness_weight=0.0,
            georoute_area_prior_weight=0.0,
            georoute_min_roi_extent=0.20,
            georoute_max_roi_extent=1.00,
            georoute_max_batch_size=1,
        )
    )
)

solver = dict(
    # OpenTAD interprets these as job-global values and divides by world size.
    # The frozen world-size-two recipe is therefore local batch one.
    train=dict(batch_size=2, num_workers=2),
    val=dict(batch_size=2, num_workers=2),
    test=dict(batch_size=2, num_workers=2),
    amp=True,
    fp16_compress=False,
    static_graph=True,
    ema=True,
)

scheduler = dict(max_epoch=20)

workflow = dict(
    checkpoint_policy="final_only",
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=20,
    end_epoch=20,
    require_successful_update_hook=True,
    schedule_and_ema_on_success_only=True,
    max_amp_retries_per_batch=8,
    fail_on_skipped_update=True,
)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(save_dict=False)
georoute_development_profile = dict(enabled=True)
georoute_diagnostic_telemetry = dict(enabled=True)

georoute_hybrid_causal_protocol = dict(
    schema_version="georoute_hybrid_causal_pilot_config_v1",
    study_id="georoute_hybrid_causal_pilot_v1",
    stage="exploratory_development_only",
    exact_k=64,
    detector_risk_keys=["cls_loss", "reg_loss"],
    support_only=True,
    local_batch=1,
    global_batch=2,
    fp16_compress=False,
    official_test_open_allowed=False,
    partial_survivor_inference_allowed=False,
    paper_claim_allowed=False,
)
