from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from mmengine.config import Config

from tools.bata.duca_boundary_burst_hard_swap_alignment import (
    SHARD_SCHEMA,
    _validate_shard,
)
from tools.bata.duca_protected_physical_training import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "adatad" / "thumos"


@pytest.fixture(autouse=True)
def _config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DUCA_CELLCF_TRAINING_PROFILE", "official60")
    monkeypatch.setenv("DUCA_FRONTEND_CHECKPOINT", "frontend.pth")
    monkeypatch.setenv("DUCA_FRONTEND_CHECKPOINT_SHA256", "a" * 64)
    monkeypatch.setenv("DUCA_FRONTEND_CHECKPOINT_EPOCH", "19")
    monkeypatch.setenv("DUCA_FRONTEND_TRAIN_BLOCK_LIST", "train.txt")


@pytest.mark.parametrize(
    ("name", "radius", "quota", "companion"),
    (
        ("duca_boundary_burst_g1_protected_fixed384_official60.py", 2, 3.0, 0.0),
        ("duca_boundary_burst_g2_uni_companion_fixed384_official60.py", 2, 3.0, 0.5),
        ("duca_boundary_burst_r4q5_g1_protected_fixed384_official60.py", 4, 5.0, 0.0),
        ("duca_boundary_burst_r4q5_g2_uni_companion_fixed384_official60.py", 4, 5.0, 0.5),
    ),
)
def test_r4_configs_are_real_official_adatad_training_arms(
    name: str, radius: int, quota: float, companion: float
) -> None:
    cfg = Config.fromfile(str(CONFIG_ROOT / name))
    selector = cfg.model.frame_selector

    assert cfg.model.type == "ActionFormer"
    assert cfg.model.rpn_head.type == "ActionFormerHead"
    assert selector.type == "DucaOnlineFrameSelector"
    assert selector.detector_gradient_mode == "protected_structured_transport"
    assert int(selector.budget) == 384
    assert int(selector.dense_window_size) == 768
    assert int(selector.max_unselected_hole) == 2
    assert int(selector.transition_boundary_radius) == radius
    assert float(selector.boundary_burst_quota) == quota
    assert float(selector.training_uniform_companion_fraction) == companion
    assert float(selector.loss_weight_schedule.detector_gradient.end) == 0.25
    assert float(selector.policy_hidden_gradient_scale) == 0.0
    assert selector.actionness_source_cfg.frozen is True
    assert selector.actionness_source_cfg.trainable is False
    assert cfg.duca_transition_only_contract.coarse_probe_training in {
        "frozen_for_all_official60_updates",
        "p0_pretrained_then_frozen",
    }
    if companion:
        assert selector.training_uniform_companion_normalize_learned_gradient is True
        assert (
            cfg.duca_transition_only_contract.detector_gradient_updates
            == "transition_scorer_only_on_learned_rows"
        )
        assert cfg.duca_transition_only_contract.inference_uses_learned_policy_only is True
        assert cfg.duca_transition_only_contract.inference_extra_companion_cost is False
    else:
        assert selector.training_uniform_companion_normalize_learned_gradient is False
        assert (
            cfg.duca_transition_only_contract.detector_gradient_updates
            == "transition_scorer_only"
        )
    assert int(cfg.workflow.expected_successful_optimizer_updates) == 6000
    assert int(cfg.workflow.primary_checkpoint_epoch) == 59
    assert cfg.duca_transition_only_contract.hard_swap_alignment_required is True


def _shard_case() -> tuple[dict, dict]:
    windows = []
    rows = []
    for window_index in range(16):
        video_id = f"video_{window_index:02d}"
        window_start = window_index * 10
        base_positions = list(range(0, 768, 2))
        windows.append(
            {
                "video_id": video_id,
                "window_start": window_start,
                "duration_stratum": "short",
                "base_selected_positions": base_positions,
                "base_selected_positions_sha256": canonical_sha256(base_positions),
            }
        )
        for swap_index in range(12):
            predicted = float(swap_index - 5.5)
            removed = 2 * swap_index
            incoming = removed + 1
            candidate = sorted(
                (set(base_positions) - {removed}) | {incoming}
            )
            rows.append(
                {
                    "video_id": video_id,
                    "window_start": window_start,
                    "duration_stratum": "short",
                    "removed": removed,
                    "incoming": incoming,
                    "quartile": swap_index // 3,
                    "predicted_delta": predicted,
                    "actual_delta": predicted * 0.1,
                    "predicted_utility": -predicted,
                    "detector_utility": -predicted * 0.1,
                    "legal_one_swap": True,
                    "hard_symmetric_difference_count": 2,
                    "base_selected_count": 384,
                    "candidate_selected_count": 384,
                    "base_positions_sha256": canonical_sha256(base_positions),
                    "candidate_positions_sha256": canonical_sha256(candidate),
                    "hard_forward_equal": True,
                }
            )
    context = {
        "git_commit": "a" * 40,
        "git_tree": "b" * 40,
        "context_sha256": "c" * 64,
        "alignment_model": {"config_sha256": "d" * 64},
        "population": {
            "config_sha256": "e" * 64,
            "windows": windows,
        },
        "selected_g0": {
            "checkpoint": {
                "path": "/sealed/g0.pth",
                "sha256": "f" * 64,
                "epoch": 59,
                "state_key": "state_dict_ema",
            }
        },
    }
    payload = {
        "schema": SHARD_SCHEMA,
        "ok": True,
        "stratum": "short",
        "runtime": {"git_commit": "a" * 40, "git_tree": "b" * 40},
        "alignment_context_sha256": "1" * 64,
        "alignment_context_self_sha256": "c" * 64,
        "config_sha256": "d" * 64,
        "population_config_sha256": "e" * 64,
        "selected_g0_checkpoint": context["selected_g0"]["checkpoint"],
        "optimizer_step": 0,
        "loss_normalizer_frozen": True,
        "train_split_only": True,
        "test_loader_built": False,
        "checkpoint_written": False,
        "hard_swap_semantics": {
            "type": "actual_hard_selected_position_one_swap",
            "exact_k_preserved": True,
            "physical_cap_preserved": True,
        },
        "windows": windows,
        "rows": rows,
        "row_sha256": canonical_sha256(rows),
    }
    return context, payload


def test_shard_gate_accepts_only_actual_exact_k_one_swaps() -> None:
    context, payload = _shard_case()
    rows, windows, swaps = _validate_shard(
        payload=payload,
        stratum="short",
        context=context,
        context_sha256="1" * 64,
    )

    assert len(rows) == 192
    assert len(windows) == 16
    assert len(swaps) == 192
    assert Counter(row["quartile"] for row in rows) == Counter({0: 48, 1: 48, 2: 48, 3: 48})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("legal_one_swap", False, "actual legal hard RGB swap"),
        ("hard_symmetric_difference_count", 4, "actual legal hard RGB swap"),
        ("candidate_selected_count", 383, "actual legal hard RGB swap"),
        ("hard_forward_equal", False, "actual legal hard RGB swap"),
        ("detector_utility", 99.0, "signed utility alias drift"),
    ),
)
def test_shard_gate_fails_closed_on_proxy_or_malformed_swaps(
    field: str, value, message: str
) -> None:
    context, payload = _shard_case()
    payload["rows"][0][field] = value
    payload["row_sha256"] = canonical_sha256(payload["rows"])

    with pytest.raises(RuntimeError, match=message):
        _validate_shard(
            payload=payload,
            stratum="short",
            context=context,
            context_sha256="1" * 64,
        )


def test_r4_slurm_runner_executes_real_backend_gates_and_all_swap_shards() -> None:
    source = (
        ROOT / "scripts" / "run_duca_boundary_burst_hard_swap_alignment_gpu1.sh"
    ).read_text(encoding="utf-8")

    assert "run_duca_protected_e2e_exact_full_model_gate.py" in source
    assert "duca_boundary_burst_hard_swap_alignment" in source
    assert "for stratum in short medium long" in source
    assert "--g1-gate" in source and "--g2-gate" in source
    assert "production_feedback_unlocked" not in source


def test_r4_worker_uses_real_rgb_swap_and_frozen_official_detector_loss() -> None:
    source = (
        ROOT / "tools" / "bata" / "run_duca_protected_physical_p3_shard.py"
    ).read_text(encoding="utf-8")

    assert "model._duca_gather_raw(batch[\"inputs\"], positions)" in source
    assert "selector._remap_train_targets_to_selected_axis" in source
    assert "model._duca_detector_objective(losses)" in source
    assert "model.load_state_dict(checkpoint[state_key], strict=True)" in source
    assert "actual_delta = candidate_loss - base_loss" in source
    assert '"detector_utility": -actual_delta' in source
    assert '"optimizer_step": 0' in source
    assert "_parameter_versions(model) == parameter_versions" in source


def test_r4_single_sbatch_runs_alignment_then_exact_g1_g2() -> None:
    source = (
        ROOT / "scripts" / "run_duca_boundary_burst_r4_gpu1.sbatch"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --gpus=1" in source
    assert "run_duca_boundary_burst_hard_swap_alignment_gpu1.sh" in source
    assert source.index("run_stage G1") < source.index("run_stage G2")
    assert "validate_alignment_artifact" in source
    assert "run_duca_two_stage_curriculum_variant_gpu1.sh" in source
    assert "jobs.complete" not in source
    assert "journal" not in source
    assert "sbatch " not in source
