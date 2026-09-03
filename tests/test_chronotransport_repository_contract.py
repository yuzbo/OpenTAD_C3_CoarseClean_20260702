from __future__ import annotations

import os
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/adatad/thumos/c3_chronotransport_adatad_videomae_s_768x1_160_stage_a.py"


def _clean_chronotransport_env(monkeypatch) -> None:
    for key in tuple(os.environ):
        if key.startswith("CHRONOTRANSPORT_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("YUZIBO_ROOT", "/data/run01/sczc063/yuzibo")


def test_stage_a_config_declares_real_48x8_internal_geometry(monkeypatch) -> None:
    _clean_chronotransport_env(monkeypatch)
    namespace = runpy.run_path(str(CONFIG))
    assert namespace["window_size"] == 768
    assert namespace["chunk_num"] == 48
    assert namespace["internal_tubelet_points"] == 384
    assert namespace["layer_groups"] == [(0, 4), (4, 8), (8, 12)]
    contract = namespace["chronotransport_contract"]
    assert contract["external_dense_grid_points"] == 768
    assert contract["internal_tubelet_points"] == 384
    assert contract["patch_embed_remains_dense"] is True
    assert contract["adatad_adapter_innovation_remains_dense"] is True
    assert contract["decode_cost_saving_claim"] is False


def test_stage_a_config_loads_with_repository_mmengine_config_parser(monkeypatch) -> None:
    _clean_chronotransport_env(monkeypatch)
    from mmengine.config import Config

    cfg = Config.fromfile(str(CONFIG))
    assert cfg.window_size == 768
    assert cfg.chronotransport_contract.internal_tubelet_points == 384


def test_default_config_is_forced_dense_and_all_claims_are_locked(monkeypatch) -> None:
    _clean_chronotransport_env(monkeypatch)
    namespace = runpy.run_path(str(CONFIG))
    runtime = namespace["chronotransport_runtime_cfg"]
    contract = namespace["chronotransport_contract"]
    assert runtime["enabled"] is True
    assert runtime["forced_schedule"] == "dense"
    assert runtime["risk_ready"] is False
    assert runtime["require_checkpoint_for_dynamic"] is True
    assert runtime["allow_legacy_checkpoint"] is True
    assert contract["diagnostic_only"] is True
    assert contract["precheck_only"] is True
    for key in (
        "deploy_claim_allowed",
        "metric_claim_allowed",
        "latency_claim_allowed",
        "paper_claim_allowed",
    ):
        assert contract[key] is False


def test_unready_learned_mode_is_explicitly_dense_fail_closed(monkeypatch) -> None:
    _clean_chronotransport_env(monkeypatch)
    monkeypatch.setenv("CHRONOTRANSPORT_MODE", "learned")
    namespace = runpy.run_path(str(CONFIG))
    runtime = namespace["chronotransport_runtime_cfg"]
    contract = namespace["chronotransport_contract"]
    assert runtime["forced_schedule"] is None
    assert runtime["measured_cost"] is None
    assert runtime["risk_ready"] is False
    assert runtime["allow_legacy_checkpoint"] is False
    assert contract["learned_mode_expected_fail_closed"] is True
    assert contract["legacy_dense_checkpoint_allowed"] is False


def test_vit_adapter_owns_one_opt_in_runtime_and_mutual_exclusion_guard() -> None:
    text = (ROOT / "opentad/models/backbones/vit_adapter.py").read_text(encoding="utf-8")
    assert "from ..chronotransport import ChronoTransportRuntime" in text
    assert "chronotransport: Optional[Dict] = None" in text
    assert "tubelet_packed_runtime_route and ChronoTransport are mutually exclusive" in text
    assert "latest_chronotransport_summary" in text
    assert "_chronotransport_load_state_dict_post_hook" in text
    assert "chronotransport_allow_legacy_checkpoint" in text


def test_launcher_is_precheck_first_gpu1_and_allocation_guarded() -> None:
    text = (ROOT / "scripts/run_chronotransport_adatad_gpu1.sh").read_text(encoding="utf-8")
    assert 'PRECHECK_ONLY="${PRECHECK_ONLY:-1}"' in text
    assert '[[ "${CUDA_VISIBLE_DEVICES}" == "1" ]]' in text
    assert "SLURM_JOB_ID" in text
    assert "CHRONOTRANSPORT_PROTECTED_ALLOCATION" in text
    assert "/data/run01/sczc063/yuzibo" in text
    assert "Stage B/C remain gated" in text
    assert "http_proxy" not in text
    assert "https_proxy" not in text


def test_validator_checks_claim_locks_and_no_shortcut_contracts() -> None:
    text = (ROOT / "tools/bata/validate_chronotransport_adatad.py").read_text(encoding="utf-8")
    for token in (
        "no_test_gt_decision",
        "no_test_teacher_decision",
        "raw_prediction_cache_forbidden",
        "require_measured_cost",
        "require_risk_ready",
        "three_seed_kill_gate_passed",
    ):
        assert token in text


def test_expected_production_and_tdd_files_exist() -> None:
    expected = [
        "actions.py",
        "cache.py",
        "transport.py",
        "risk.py",
        "scheduler.py",
        "losses.py",
        "profiler.py",
        "runtime.py",
    ]
    package = ROOT / "opentad/models/chronotransport"
    assert all((package / name).is_file() for name in expected)
    assert (ROOT / "tests/test_chronotransport_core.py").is_file()
    assert (ROOT / "docs/methods/2026-07-10-chronotransport-implementation-plan.md").is_file()


def test_paired_replay_stage_b_and_cost_lookup_files_exist() -> None:
    expected = (
        "opentad/models/chronotransport/replay.py",
        "opentad/models/chronotransport/training.py",
        "opentad/models/chronotransport/cost_lookup.py",
        "tools/bata/run_chronotransport_paired_replay.py",
        "tools/bata/train_chronotransport_stage_b.py",
        "tools/bata/profile_chronotransport_schedules.py",
        "configs/adatad/thumos/c3_chronotransport_adatad_videomae_s_768x1_160_stage_b.py",
        "configs/adatad/thumos/c3_chronotransport_adatad_videomae_s_768x1_160_stage_c.py",
    )
    assert all((ROOT / path).is_file() for path in expected)


def test_paired_replay_source_forbids_deployment_prediction_payloads() -> None:
    path = ROOT / "opentad/models/chronotransport/replay.py"
    text = path.read_text(encoding="utf-8")
    assert "raw_predictions" not in text
    assert "full_token_state" not in text
    assert "RNGSnapshot" in text
    assert "nonnegative_detector_regret" in text


def test_learned_cost_lookup_is_schedule_shape_keyed() -> None:
    text = (ROOT / "opentad/models/chronotransport/cost_lookup.py").read_text(encoding="utf-8")
    for token in (
        "hardware",
        "precision",
        "batch_size",
        "candidate_schedule",
        "selected_rows_per_group",
        "p50",
        "p95",
    ):
        assert token in text
