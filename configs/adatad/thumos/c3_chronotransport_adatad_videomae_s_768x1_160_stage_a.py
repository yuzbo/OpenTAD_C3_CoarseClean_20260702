"""ChronoTransport Stage-A / precheck config for THUMOS14 AdaTAD VideoMAE-S.

This config keeps the detector's external 768-point temporal grid unchanged.
Internally, 768 frames become 48 independent 16-frame VideoMAE clips and each
clip yields eight tubelets, i.e. 384 internal temporal tubelet points before the
existing post-processing interpolation back to 768.

The default mode is forced dense. Learned scheduling is fail-closed unless a
measured cost table and an explicitly calibrated risk checkpoint are provided.
"""

_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]

import json
import math
import os
import pathlib


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name, default)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a float, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _load_measured_cost(path: str) -> dict | None:
    if not path:
        return None
    cost_path = pathlib.Path(path).expanduser().resolve()
    payload = json.loads(cost_path.read_text(encoding="utf-8"))
    required = ("recompute", "transport", "hold")
    for key in required:
        values = payload.get(key)
        if not isinstance(values, list) or len(values) != 3:
            raise ValueError(f"measured cost {key!r} must contain three layer-group values")
        if any((not math.isfinite(float(value))) or float(value) < 0.0 for value in values):
            raise ValueError(f"measured cost {key!r} must be finite and non-negative")
    overhead = float(payload.get("scheduler_overhead", 0.0))
    if not math.isfinite(overhead) or overhead < 0.0:
        raise ValueError("scheduler_overhead must be finite and non-negative")
    return {
        "recompute": tuple(float(value) for value in payload["recompute"]),
        "transport": tuple(float(value) for value in payload["transport"]),
        "hold": tuple(float(value) for value in payload["hold"]),
        "scheduler_overhead": overhead,
    }


window_size = 768
videomae_clip_frames = 16
videomae_tubelet_size = 2
chunk_num = window_size // videomae_clip_frames
internal_tubelet_points = chunk_num * (videomae_clip_frames // videomae_tubelet_size)
layer_groups = [(0, 4), (4, 8), (8, 12)]

if window_size != 768 or chunk_num != 48 or internal_tubelet_points != 384:
    raise AssertionError("ChronoTransport THUMOS contract requires 768 / 48 / 384 geometry")

chronotransport_mode = os.environ.get("CHRONOTRANSPORT_MODE", "dense").strip()
valid_modes = {
    "disabled",
    "dense",
    "periodic2_transport",
    "periodic4_transport",
    "periodic8_transport",
    "periodic2_hold",
    "hold_only",
    "transport_only",
    "layer_only_early_recompute",
    "layer_only_late_recompute",
    "joint_progressive_transport",
    "learned",
}
if chronotransport_mode not in valid_modes:
    raise ValueError(f"unsupported CHRONOTRANSPORT_MODE={chronotransport_mode!r}")

chronotransport_enabled = chronotransport_mode != "disabled"
chronotransport_forced_schedule = (
    None if chronotransport_mode in {"disabled", "learned"} else chronotransport_mode
)
chronotransport_cost_path = os.environ.get("CHRONOTRANSPORT_COST_JSON", "").strip()
chronotransport_measured_cost = _load_measured_cost(chronotransport_cost_path)
chronotransport_risk_ready = _env_bool("CHRONOTRANSPORT_RISK_READY", False)
chronotransport_allow_unmeasured_debug = _env_bool(
    "CHRONOTRANSPORT_ALLOW_UNMEASURED_DEBUG",
    False,
)
chronotransport_max_cache_age = _env_int("CHRONOTRANSPORT_MAX_CACHE_AGE", 8)
chronotransport_risk_quantile = _env_float("CHRONOTRANSPORT_RISK_QUANTILE", 0.9)
chronotransport_risk_epsilon = _env_float("CHRONOTRANSPORT_RISK_EPSILON", 1.0)

if chronotransport_max_cache_age <= 0:
    raise ValueError("CHRONOTRANSPORT_MAX_CACHE_AGE must be positive")
if not 0.0 < chronotransport_risk_quantile < 1.0:
    raise ValueError("CHRONOTRANSPORT_RISK_QUANTILE must lie in (0, 1)")
if chronotransport_risk_epsilon < 0.0:
    raise ValueError("CHRONOTRANSPORT_RISK_EPSILON must be non-negative")

chronotransport_runtime_cfg = dict(
    enabled=chronotransport_enabled,
    layer_groups=layer_groups,
    signal_dims=6,
    risk_hidden_dims=64,
    transport_bottleneck_dims=64,
    risk_quantile=chronotransport_risk_quantile,
    risk_epsilon=chronotransport_risk_epsilon,
    max_cache_age=chronotransport_max_cache_age,
    forced_schedule=chronotransport_forced_schedule,
    cache_detach=True,
    profile_sync_cuda=_env_bool("CHRONOTRANSPORT_PROFILE_SYNC_CUDA", True),
    measured_cost=chronotransport_measured_cost,
    allow_unmeasured_cost_for_debug=chronotransport_allow_unmeasured_debug,
    risk_ready=chronotransport_risk_ready,
    require_checkpoint_for_dynamic=True,
    # Allows a dense AdaTAD checkpoint to initialize Stage-A forced baselines.
    # Formal learned mode must disable this and load a ChronoTransport checkpoint.
    allow_legacy_checkpoint=(chronotransport_mode != "learned"),
)

chronotransport_contract = dict(
    schema_version="chronotransport_adatad_contract_v1",
    review_base_commit="c26b349ee27b6e427fa5cbff8c011778c2684b17",
    route="CHRONOTRANSPORT_ADATAD_VIDEOMAE_S_STAGE_A",
    implementation_stage="A_counterfactual_runtime_and_precheck",
    main_method_candidate=True,
    diagnostic_only=True,
    precheck_only=True,
    official_adatad_backend=True,
    changes_decoder=False,
    changes_projection=False,
    changes_head=False,
    changes_gt_assignment=False,
    changes_nms=False,
    external_dense_grid_points=window_size,
    videomae_chunks=chunk_num,
    clip_frames=videomae_clip_frames,
    tubelets_per_chunk=videomae_clip_frames // videomae_tubelet_size,
    internal_tubelet_points=internal_tubelet_points,
    layer_groups=layer_groups,
    action_space=("RECOMPUTE", "TRANSPORT", "HOLD"),
    heavy_attention_mlp_is_gated=True,
    patch_embed_remains_dense=True,
    adatad_adapter_innovation_remains_dense=True,
    neck_head_remain_dense=True,
    decode_cost_saving_claim=False,
    cache_reset_per_window=True,
    cross_window_cache_reuse=False,
    first_chunk_forced_recompute=True,
    no_test_gt_decision=True,
    no_test_teacher_decision=True,
    raw_prediction_cache_forbidden=True,
    deploy_visible_signals_only=True,
    tubelet_packed_runtime_mutually_exclusive=True,
    legacy_dense_checkpoint_allowed=(chronotransport_mode != "learned"),
    dynamic_mode_requires_chronotransport_checkpoint=True,
    mode=chronotransport_mode,
    forced_schedule=chronotransport_forced_schedule,
    measured_cost_available=chronotransport_measured_cost is not None,
    risk_ready=chronotransport_risk_ready,
    learned_mode_expected_fail_closed=(
        chronotransport_mode == "learned"
        and (chronotransport_measured_cost is None or not chronotransport_risk_ready)
    ),
    stage_b_detector_regret_training_complete=False,
    stage_c_joint_adapter_training_complete=False,
    three_seed_kill_gate_passed=False,
    deploy_claim_allowed=False,
    metric_claim_allowed=False,
    latency_claim_allowed=False,
    paper_claim_allowed=False,
    kill_gate=dict(
        min_p50_latency_saving_fraction=0.15,
        max_map_07_drop_absolute=1.5,
        max_shortest_duration_quartile_drop_absolute=1.5,
        max_overhead_fraction_of_recompute_saving=0.40,
        require_periodic_baseline_separation_in_three_seed_ci=True,
        require_positive_risk_regret_correlation=True,
    ),
)

model = dict(
    backbone=dict(
        backbone=dict(
            # Existing packed-tubelet routing and ChronoTransport cannot be active
            # simultaneously because both own transformer-block execution.
            tubelet_packed_runtime_route=None,
            chronotransport=chronotransport_runtime_cfg,
        )
    )
)

# Stage-A smoke/evaluation should not increase loader pressure while profiling.
solver = dict(
    val=dict(batch_size=1, num_workers=1),
    test=dict(batch_size=1, num_workers=1),
)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)

_yuzibo_root = pathlib.Path(os.environ.get("YUZIBO_ROOT", "/data/run01/sczc063/yuzibo")).expanduser()
work_dir = str(
    _yuzibo_root
    / "exps"
    / "thumos"
    / "chronotransport"
    / f"stage_a_{chronotransport_mode}"
)
