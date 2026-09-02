from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
from mmengine.config import Config

from tools.bata.h65_pro_hard_one_swap_diagnostic import summarize_one_swap


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "adatad" / "thumos" / "h65_pro"
DOC_DIR = ROOT / "docs" / "experiments" / "h65_pro_fullmatrix_20260902"
MATRIX = DOC_DIR / "03_EXPERIMENT_MATRIX.csv"
_TORCH_LOADABLE: bool | None = None


def _rows() -> list[dict[str, str]]:
    with MATRIX.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _require_torch():
    global _TORCH_LOADABLE
    if _TORCH_LOADABLE is None:
        probe = subprocess.run(
            [sys.executable, "-c", "import torch"],
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _TORCH_LOADABLE = probe.returncode == 0
        if not _TORCH_LOADABLE:
            detail = (probe.stderr or probe.stdout).strip().splitlines()[-1:]
            pytest.skip("PyTorch is not loadable in this environment: " + (" ".join(detail) or str(probe.returncode)))
    if not _TORCH_LOADABLE:
        pytest.skip("PyTorch is not loadable in this environment")
    import torch

    return torch


def test_h65_pro_matrix_validator_and_smoke_subset_pass() -> None:
    proc = subprocess.run(
        [sys.executable, "tools/bata/validate_h65_pro_fullmatrix.py"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    assert "PASS H65-Pro fullmatrix" in proc.stdout
    rows = _rows()
    assert len(rows) == 28
    assert len({(row["experiment_id"], row["seed"], row["config"]) for row in rows}) == 28
    submitter = (ROOT / "tools/experiments/submit_h65_pro_fullmatrix.sh").read_text(encoding="utf-8")
    for experiment_id in ("REF-U384", "F01", "F02", "F03", "F05", "F09", "F13", "F16"):
        assert experiment_id in submitter


def test_h65_pro_train_precheck_allows_canonical_seed_suffixed_rows() -> None:
    text = (ROOT / "tools/experiments/run_h65_pro_train.sbatch").read_text(encoding="utf-8")
    assert 'canonical_prefix = experiment_id.split("-S", 1)[0]' in text
    assert 'config_experiment_id != canonical_prefix' in text
    assert 'not experiment_id.endswith(f"-S{seed}")' in text


def test_h65_pro_generated_configs_encode_requested_factor_contracts() -> None:
    f16 = Config.fromfile(str(CONFIG_DIR / "h65_pro_f16.py"))
    selector = f16.model.frame_selector
    assert selector.acquisition_policy == "semantic_phase_sampling"
    assert selector.semantic_phase_sigma == 2.0
    assert selector.semantic_phase_scaffold_budget == 128
    assert selector.semantic_phase_onset_budget == 64
    assert selector.semantic_phase_offset_budget == 64
    assert selector.semantic_phase_core_budget == 128
    assert f16.model.rpn_head.conv_cfg.type == "ContinuousTimeScaleAdaptiveConv1d"
    assert f16.model.rpn_head.conv_cfg.local_ref_delta_t == 2.0
    assert f16.model.rpn_head.conv_cfg.context_ref_delta_t == 2.0
    assert f16.model.rpn_head.conv_cfg.context_level_base == 2.0
    assert selector.detector_contribution_mode == "signed_removal_utility"
    assert selector.loss_weight_schedule.shape == "cosine"
    assert selector.loss_weight_schedule.policy_alpha.warmup_steps == 1500
    assert selector.loss_weight_schedule.policy_alpha.transition_steps == 2000
    amod = f16.model.backbone.backbone.amod_config
    assert amod.enabled is True
    assert list(amod.amod_layers) == [1, 3, 5, 7, 9, 11]
    assert amod.capacity_schedule.start_capacity == 1.0
    assert amod.capacity_schedule.end_capacity == 0.5

    f01 = Config.fromfile(str(CONFIG_DIR / "h65_pro_f01.py"))
    assert f01.model.frame_selector.acquisition_policy == "budget_calibrated_sampling_rate"
    assert f01.model.rpn_head.conv_cfg is None
    assert f01.model.backbone.backbone.amod_config.enabled is False
    assert f01.model.frame_selector.detector_contribution_mode == "abs_grad_times_input"


def test_semantic_phase_sampler_returns_sorted_unique_exact_k_and_uniform_alpha_zero() -> None:
    torch = _require_torch()
    from opentad.models.duca.acquisition import DucaAcquisitionAdapter
    from opentad.models.duca.structured_selection import exact_uniform_positions

    adapter = DucaAcquisitionAdapter(
        feature_dim=3,
        budget=16,
        acquisition_policy="semantic_phase_sampling",
        semantic_phase_scaffold_budget=4,
        semantic_phase_onset_budget=3,
        semantic_phase_offset_budget=3,
        semantic_phase_core_budget=6,
        hard_max_gap_repair=False,
        fail_on_infeasible_max_gap=False,
    )
    logits = torch.zeros(1, 32)
    logits[0, 8:13] = torch.tensor([-2.0, -0.5, 1.0, 2.5, 3.5])
    logits[0, 18:23] = torch.tensor([3.0, 2.0, 0.5, -1.0, -2.0])
    valid_mask = torch.ones(1, 32, dtype=torch.bool)
    budgets = torch.tensor([16])

    decoded = adapter._decode_semantic_phase_sampling(
        logits,
        None,
        valid_mask,
        budgets,
        stable_selection=False,
        policy_mix_alpha=1.0,
    )
    selected = decoded["selected_positions"][0]
    assert selected.numel() == 16
    assert torch.equal(selected, selected.sort().values)
    assert selected.unique().numel() == 16
    diagnostics = decoded["semantic_phase_diagnostics"][0]
    assert diagnostics["selected_by_group"]["scaffold"]
    assert diagnostics["selected_by_group"]["onset"]
    assert diagnostics["selected_by_group"]["offset"]
    assert diagnostics["selected_by_group"]["core"]

    uniform = adapter._decode_semantic_phase_sampling(
        logits,
        None,
        valid_mask,
        budgets,
        stable_selection=False,
        policy_mix_alpha=0.0,
    )
    assert torch.equal(uniform["selected_positions"][0], exact_uniform_positions(32, 16))


def test_ctconv_eta_zero_is_exact_conv1d_with_real_temporal_positions() -> None:
    torch = _require_torch()
    import torch.nn.functional as F
    from opentad.models.bricks.scale_adaptive_conv1d import ContinuousTimeScaleAdaptiveConv1d

    torch.manual_seed(7)
    layer = ContinuousTimeScaleAdaptiveConv1d(2, 3, kernel_size=3, padding=1, bias=True)
    x = torch.randn(2, 2, 12)
    temporal_positions = torch.tensor(
        [
            [0.0, 1.0, 2.0, 4.0, 7.0, 8.0, 10.0, 13.0, 14.0, 16.0, 17.0, 19.0],
            [0.0, 2.0, 3.0, 5.0, 6.0, 9.0, 11.0, 12.0, 15.0, 17.0, 18.0, 21.0],
        ]
    )
    expected = F.conv1d(x, layer.weight, layer.bias, padding=1)
    observed = layer(x, temporal_positions=temporal_positions, level_index=2)
    assert torch.allclose(observed, expected, atol=1.0e-6)
    assert layer.last_diagnostics["enabled"] is True
    assert layer.last_diagnostics["context_ref_delta_t"] == 8.0


def test_signed_taylor_detector_contribution_is_detached_first_order() -> None:
    torch = _require_torch()
    from opentad.models.selectors.duca_online_frame_selector import DucaOnlineFrameSelector

    selected = torch.tensor([[[1.0, -2.0, 3.0]]], requires_grad=True)
    objective = -(selected * selected).sum()
    signed = DucaOnlineFrameSelector._selected_detector_contribution(
        selected,
        objective,
        mode="signed_removal_utility",
    )
    assert signed.shape == (1, 3)
    assert signed.requires_grad is False
    assert torch.all(signed >= 0.0)
    assert torch.allclose(signed, torch.tensor([[2.0, 8.0, 18.0]]))

    absolute = DucaOnlineFrameSelector._selected_detector_contribution(
        selected,
        objective,
        mode="abs_grad_times_input",
    )
    assert torch.allclose(absolute, signed)


def test_vit_adapter_source_contains_topk_mod_identity_bypass_and_successful_update_schedule() -> None:
    text = (ROOT / "opentad/models/backbones/vit_adapter.py").read_text(encoding="utf-8")
    assert "def forward_amod" in text
    assert "torch.topk" in text
    assert "unselected_identity" in text
    assert "amod_successful_optimizer_updates" in text
    assert "def after_optimizer_step" in text
    assert "0.5 * (1.0 + math.cos(math.pi * progress))" in text


def test_selected_axis_terminal_binding_allows_distinct_afterok_eval_job_id() -> None:
    text = (ROOT / "tools/bata/duca_selected_axis_training.py").read_text(encoding="utf-8")
    terminal = text[text.index("def validate_terminal_checkpoint_binding") :]
    slurm_check = terminal.index('if key == "slurm_job_id":')
    next_binding_check = terminal.index("if audit.get(key) != expected:", slurm_check)
    assert "continue" in terminal[slurm_check:next_binding_check]


def test_h65_pro_hard_one_swap_diagnostic_summarizes_offline_records(tmp_path: Path) -> None:
    records = [
        {"video_id": "v1", "selector_delta": 0.4, "baseline_score": 1.0, "candidate_score": 1.3},
        {"video_id": "v1", "selector_delta": -0.2, "baseline_score": 1.0, "candidate_score": 0.9},
        {"video_id": "v2", "selector_delta": 0.1, "observed_delta": -0.1},
    ]
    summary = summarize_one_swap(records)
    assert summary["schema_version"] == "h65_pro_hard_one_swap_diagnostic_v1"
    assert summary["contract"]["training_path"] is False
    assert summary["record_count"] == 3
    assert summary["video_count"] == 2
    assert summary["positive_prediction_count"] == 2
    assert summary["sign_match_rate"] == 2 / 3
    assert summary["top_predicted_swap_observed_improvement_rate"] == 0.5

    input_jsonl = tmp_path / "one_swap.jsonl"
    output_json = tmp_path / "summary.json"
    input_jsonl.write_text("\n".join(json.dumps(item) for item in records), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            "tools/bata/h65_pro_hard_one_swap_diagnostic.py",
            "--input-jsonl",
            str(input_jsonl),
            "--output-json",
            str(output_json),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    assert "H65-Pro hard one-swap diagnostic" in proc.stdout
    assert json.loads(output_json.read_text(encoding="utf-8"))["record_count"] == 3
