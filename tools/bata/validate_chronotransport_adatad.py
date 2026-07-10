from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG_DEFAULT = (
    "configs/adatad/thumos/"
    "c3_chronotransport_adatad_videomae_s_768x1_160_stage_a.py"
)
OFFICIAL_BASE_CONFIG = (
    "configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py"
)
ALLOWED_OUTPUT_ROOT = Path("/data/run01/sczc063/yuzibo")
REVIEW_BASE_COMMIT = "c26b349ee27b6e427fa5cbff8c011778c2684b17"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _as_plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _as_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_plain(item) for item in value]
    return value


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _validate_layer_groups(groups: Sequence[Sequence[int]], depth: int = 12) -> list[list[int]]:
    normalized = [[int(group[0]), int(group[1])] for group in groups]
    _require(normalized == [[0, 4], [4, 8], [8, 12]], "default layer groups must be 0:4/4:8/8:12")
    cursor = 0
    for start, end in normalized:
        _require(start == cursor and end > start, "layer groups must be contiguous and positive")
        cursor = end
    _require(cursor == int(depth), "layer groups must cover all transformer blocks")
    return normalized


def _validate_measured_cost(cost: Any, num_groups: int) -> dict[str, Any] | None:
    if cost is None:
        return None
    normalized: dict[str, Any] = {}
    for key in ("recompute", "transport", "hold"):
        values = list(cost[key])
        _require(len(values) == num_groups, f"{key} cost must have one value per layer group")
        parsed = [float(value) for value in values]
        _require(all(math.isfinite(value) and value >= 0.0 for value in parsed), f"{key} cost must be finite/non-negative")
        normalized[key] = parsed
    overhead = float(cost.get("scheduler_overhead", 0.0))
    _require(math.isfinite(overhead) and overhead >= 0.0, "scheduler overhead must be finite/non-negative")
    normalized["scheduler_overhead"] = overhead
    return normalized


def validate_config(
    config_path: str = CONFIG_DEFAULT,
    *,
    require_measured_cost: bool = False,
    require_risk_ready: bool = False,
) -> dict[str, Any]:
    config_path = str(config_path)
    cfg = Config.fromfile(config_path)
    official = Config.fromfile(str(ROOT / OFFICIAL_BASE_CONFIG))

    contract = cfg.chronotransport_contract
    backbone_cfg = cfg.model.backbone.backbone
    runtime_cfg = backbone_cfg.chronotransport
    groups = _validate_layer_groups(runtime_cfg.layer_groups, depth=int(backbone_cfg.depth))
    measured_cost = _validate_measured_cost(runtime_cfg.get("measured_cost", None), len(groups))
    nonlinear_cost = runtime_cfg.get("nonlinear_cost_entries", None)

    _require(contract.schema_version == "chronotransport_adatad_contract_v1", "unexpected contract schema")
    _require(contract.review_base_commit == REVIEW_BASE_COMMIT, "review base commit must be fixed")
    _require(contract.official_adatad_backend is True, "official AdaTAD backend must remain enabled")
    _require(contract.main_method_candidate is True, "ChronoTransport must be declared as a main-method candidate")
    _require(contract.diagnostic_only is True, "Stage-A config must remain diagnostic_only")
    _require(contract.precheck_only is True, "Stage-A config must remain precheck_only")
    _require(contract.changes_decoder is False, "Stage A must not modify decoder")
    _require(contract.changes_projection is False, "Stage A must not modify projection")
    _require(contract.changes_head is False, "Stage A must not modify ActionFormer head")
    _require(contract.changes_gt_assignment is False, "Stage A must not modify GT assignment")
    _require(contract.changes_nms is False, "Stage A must not modify NMS")

    _require(int(cfg.window_size) == 768, "external detector grid must remain 768")
    _require(int(cfg.chunk_num) == 48, "VideoMAE must receive 48 clips")
    _require(int(cfg.internal_tubelet_points) == 384, "internal temporal lattice must be 384 tubelets")
    _require(int(contract.external_dense_grid_points) == 768, "contract must expose 768 output points")
    _require(int(contract.videomae_chunks) == 48, "contract must expose 48 clips")
    _require(int(contract.internal_tubelet_points) == 384, "contract must expose 384 internal tubelets")

    _require(cfg.model.type == "ActionFormer", "official detector must stay ActionFormer")
    _require(cfg.model.get("frame_selector", None) is None, "ChronoTransport must not enable pre-backbone frame selection")
    _require(_as_plain(cfg.model.rpn_head) == _as_plain(official.model.rpn_head), "ActionFormer head must match official base")
    _require(int(cfg.model.projection.max_seq_len) == 768, "projection must keep the 768-point grid")
    _require(int(backbone_cfg.total_frames) == 768, "VideoMAE total_frames must stay 768")
    _require(int(backbone_cfg.num_frames) == 16, "VideoMAE clip length must stay 16")
    _require(int(backbone_cfg.depth) == 12, "VideoMAE-S depth must stay 12")
    _require(list(backbone_cfg.adapter_index) == list(range(12)), "all official AdaTAD adapters must remain present")

    packed_cfg = backbone_cfg.get("tubelet_packed_runtime_route", None)
    _require(
        packed_cfg is None or not bool(packed_cfg.get("enabled", False)),
        "packed tubelet routing and ChronoTransport are mutually exclusive",
    )
    _require(bool(runtime_cfg.enabled) == (str(contract.mode) != "disabled"), "runtime enabled flag disagrees with mode")
    _require(runtime_cfg.get("cache_detach", None) is True, "Stage-A cache must detach across chunks")
    _require(
        runtime_cfg.get("require_checkpoint_for_dynamic", None) is True,
        "dynamic scheduling must require a ChronoTransport checkpoint",
    )
    _require(int(runtime_cfg.max_cache_age) > 0, "max cache age must be positive")
    _require(0.0 < float(runtime_cfg.risk_quantile) < 1.0, "risk quantile must lie in (0,1)")
    _require(float(runtime_cfg.risk_epsilon) >= 0.0, "risk epsilon must be non-negative")

    _require(contract.patch_embed_remains_dense is True, "patch embedding must remain dense")
    _require(contract.adatad_adapter_innovation_remains_dense is True, "AdaTAD adapter path must remain dense")
    _require(contract.neck_head_remain_dense is True, "neck/head must remain dense")
    _require(contract.decode_cost_saving_claim is False, "Stage A must not claim decode saving")
    _require(contract.cache_reset_per_window is True, "cache must reset per window")
    _require(contract.cross_window_cache_reuse is False, "cross-window cache reuse must not be claimed")
    _require(contract.no_test_gt_decision is True, "test GT must not enter scheduling")
    _require(contract.no_test_teacher_decision is True, "test teacher must not enter scheduling")
    _require(contract.raw_prediction_cache_forbidden is True, "raw-prediction cache must be forbidden")
    _require(contract.deploy_visible_signals_only is True, "scheduler signals must be deploy-visible")

    _require(cfg.inference.load_from_raw_predictions is False, "raw prediction loading must be disabled")
    _require(cfg.inference.save_raw_prediction is False, "raw prediction saving must be disabled")
    _require(_is_within(Path(cfg.work_dir), ALLOWED_OUTPUT_ROOT), "work_dir must stay under /data/run01/sczc063/yuzibo")

    mode = str(contract.mode)
    risk_ready = bool(runtime_cfg.get("risk_ready", False))
    if mode == "learned" and (measured_cost is None or not risk_ready):
        _require(contract.learned_mode_expected_fail_closed is True, "unready learned mode must declare dense fail-closed")
    if require_measured_cost:
        _require(measured_cost is not None, "formal dynamic run requires a measured cost table")
        _require(
            isinstance(nonlinear_cost, Mapping) and len(nonlinear_cost) > 0,
            "formal dynamic run requires schedule-shape p50/p95 cost lookup",
        )
    if require_risk_ready:
        _require(risk_ready, "formal dynamic run requires an explicitly calibrated risk checkpoint")
        _require(
            runtime_cfg.get("allow_legacy_checkpoint", True) is False,
            "formal dynamic run must reject a legacy dense-only checkpoint",
        )
        _require(
            runtime_cfg.get("require_checkpoint_for_dynamic", False) is True,
            "formal dynamic run must keep checkpoint readiness guard enabled",
        )

    _require(contract.stage_b_detector_regret_training_complete is False, "Stage B must not be falsely marked complete")
    _require(contract.stage_c_joint_adapter_training_complete is False, "Stage C must not be falsely marked complete")
    _require(contract.three_seed_kill_gate_passed is False, "three-seed kill gate must remain false before experiments")
    for claim in ("deploy_claim_allowed", "metric_claim_allowed", "latency_claim_allowed", "paper_claim_allowed"):
        _require(getattr(contract, claim) is False, f"{claim} must remain false in Stage A")

    required_files = (
        ROOT / "opentad/models/chronotransport/actions.py",
        ROOT / "opentad/models/chronotransport/cache.py",
        ROOT / "opentad/models/chronotransport/transport.py",
        ROOT / "opentad/models/chronotransport/risk.py",
        ROOT / "opentad/models/chronotransport/scheduler.py",
        ROOT / "opentad/models/chronotransport/losses.py",
        ROOT / "opentad/models/chronotransport/profiler.py",
        ROOT / "opentad/models/chronotransport/runtime.py",
        ROOT / "tests/test_chronotransport_core.py",
        ROOT / "tests/test_chronotransport_repository_contract.py",
        ROOT / "tests/test_chronotransport_vit_adapter_integration.py",
        ROOT / "scripts/run_chronotransport_adatad_gpu1.sh",
    )
    missing = [str(path.relative_to(ROOT)) for path in required_files if not path.exists()]
    _require(not missing, f"missing ChronoTransport production/test files: {missing}")

    vit_text = (ROOT / "opentad/models/backbones/vit_adapter.py").read_text(encoding="utf-8")
    _require("ChronoTransportRuntime" in vit_text, "VisionTransformerAdapter lacks ChronoTransport import")
    _require("latest_chronotransport_summary" in vit_text, "VisionTransformerAdapter lacks runtime summary surface")
    _require("tubelet_packed_runtime_route and ChronoTransport" in vit_text, "mutual exclusion guard is missing")

    return {
        "status": "PASS",
        "config": config_path,
        "review_base_commit": REVIEW_BASE_COMMIT,
        "mode": mode,
        "external_dense_grid_points": 768,
        "videomae_chunks": 48,
        "internal_tubelet_points": 384,
        "layer_groups": groups,
        "measured_cost": measured_cost,
        "risk_ready": risk_ready,
        "learned_mode_expected_fail_closed": bool(contract.learned_mode_expected_fail_closed),
        "work_dir": str(cfg.work_dir),
        "claim_flags": {
            "deploy": bool(contract.deploy_claim_allowed),
            "metric": bool(contract.metric_claim_allowed),
            "latency": bool(contract.latency_claim_allowed),
            "paper": bool(contract.paper_claim_allowed),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ChronoTransport AdaTAD Stage-A contract")
    parser.add_argument("--config", default=CONFIG_DEFAULT)
    parser.add_argument("--require-measured-cost", action="store_true")
    parser.add_argument("--require-risk-ready", action="store_true")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = validate_config(
        args.config,
        require_measured_cost=args.require_measured_cost,
        require_risk_ready=args.require_risk_ready,
    )
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.output:
        output = Path(args.output)
        _require(_is_within(output, ALLOWED_OUTPUT_ROOT), "validator output must stay under allowed root")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
