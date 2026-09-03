from __future__ import annotations

import csv
import json
import subprocess
import sys
import types
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
    by_id = {row["experiment_id"]: row for row in rows}
    assert by_id["REF-D768"]["variant"] == "h65_pro_ref_d768"
    assert by_id["REF-MNV3FC384"]["config"].endswith("h65_pro_ref_mnv3fc384.py")
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


def test_h65_pro_reference_configs_encode_dense_and_mnv3fc_contracts() -> None:
    from tools.bata.duca_selected_axis_training import formal_training_contract

    ref_d768 = Config.fromfile(str(CONFIG_DIR / "h65_pro_ref_d768.py"))
    assert ref_d768.h65_pro_experiment_id == "REF-D768"
    assert ref_d768.model.frame_selector is None
    assert ref_d768.model.projection.max_seq_len == 768
    assert ref_d768.workflow.formal_protocol == "h65_pro_dense_reference_official60_v1"
    assert ref_d768.workflow.formal_successful_update_contract is True
    assert ref_d768.workflow.selector_schedule_required is False
    assert ref_d768.workflow.expected_successful_optimizer_updates == 6000
    assert ref_d768.workflow.intermediate_validation_selects_checkpoint is False
    ref_d768_contract = formal_training_contract(ref_d768)
    assert ref_d768_contract["h65_pro_dense_reference"] is True
    assert ref_d768_contract["selector_schedule_required"] is False

    ref_mnv = Config.fromfile(str(CONFIG_DIR / "h65_pro_ref_mnv3fc384.py"))
    assert ref_mnv.h65_pro_experiment_id == "REF-MNV3FC384"
    assert ref_mnv.h65_pro_factor_policy.reference == "frozen_mobilenetv3_feature_change"
    selector = ref_mnv.model.frame_selector
    assert selector.parameter_free_selector is True
    assert selector.actionness_source_cfg.train_free_evidence_mode == "frozen_feature_change"
    assert Path(selector.actionness_source_cfg.mobilenet_weights_path).name == (
        "mobilenet_v3_small-047dcff4.pth"
    )


def test_every_h65_pro_config_satisfies_its_runtime_training_contract() -> None:
    from tools.bata.duca_selected_axis_training import formal_training_contract

    for row in _rows():
        cfg = Config.fromfile(str(ROOT / row["config"]))
        contract = formal_training_contract(cfg)
        assert contract is not None, row["experiment_id"]
        assert contract["expected_successful_optimizer_updates"] == 6000
        assert cfg.workflow.get("intermediate_validation_selects_checkpoint", False) is False


def test_actionformer_uses_meta_aware_backbone_helper_for_train_and_test() -> None:
    torch = _require_torch()
    from opentad.models.detectors.actionformer import ActionFormer

    model = ActionFormer.__new__(ActionFormer)
    torch.nn.Module.__init__(model)
    model.backbone = torch.nn.Identity()
    model.frame_selector = None
    model.token_compressor = None
    model.true_time_residual = None
    model.selector_train_only_skip_detector = False

    def raise_if_helper_called(self, inputs, masks, metas):
        raise RuntimeError("helper_called")

    model._call_backbone_forward = types.MethodType(raise_if_helper_called, model)
    model._capture_protected_detector_rng = types.MethodType(lambda self, inputs: None, model)
    model._restore_protected_detector_rng = types.MethodType(lambda self, state: None, model)
    model._reject_pc_ot_mras_value_targets_in_forward_test = types.MethodType(lambda self, metas: None, model)

    with pytest.raises(RuntimeError, match="helper_called"):
        ActionFormer.forward_train(
            model,
            torch.zeros(1, 1, 4),
            torch.ones(1, 4, dtype=torch.bool),
            [{}],
            [],
            [],
            _duca_skip_frame_selector=True,
        )
    with pytest.raises(RuntimeError, match="helper_called"):
        ActionFormer.forward_test(
            model,
            torch.zeros(1, 1, 4),
            torch.ones(1, 4, dtype=torch.bool),
            [{}],
        )


def test_ctconv_parameters_are_covered_by_actionformer_optimizer_groups() -> None:
    torch = _require_torch()
    from opentad.models.bricks.scale_adaptive_conv1d import ContinuousTimeScaleAdaptiveConv1d
    from opentad.models.detectors.actionformer import ActionFormer

    model = ActionFormer.__new__(ActionFormer)
    torch.nn.Module.__init__(model)
    model.rpn_head = torch.nn.Module()
    model.rpn_head.ct = ContinuousTimeScaleAdaptiveConv1d(2, 3, kernel_size=3, padding=1, bias=True)

    optim_groups = ActionFormer.get_optim_groups(model, {"lr": 1.0e-3, "weight_decay": 0.05})
    decay_param_ids = {
        id(param)
        for group in optim_groups
        if group["weight_decay"] == 0.05
        for param in group["params"]
    }
    no_decay_param_ids = {
        id(param)
        for group in optim_groups
        if group["weight_decay"] == 0.0
        for param in group["params"]
    }
    assert id(model.rpn_head.ct.weight) in decay_param_ids
    assert id(model.rpn_head.ct.bias) in no_decay_param_ids
    assert id(model.rpn_head.ct.eta) in no_decay_param_ids


def test_semantic_phase_centered_derivative_respects_valid_tail_boundary() -> None:
    torch = _require_torch()
    from opentad.models.duca.acquisition import DucaAcquisitionAdapter

    values = torch.tensor([[0.0, 10.0, 12.0, 999.0, 999.0]])
    valid = torch.tensor([[True, True, True, False, False]])
    derivative = DucaAcquisitionAdapter._centered_derivative(values, valid)
    assert torch.allclose(derivative, torch.tensor([[10.0, 6.0, 2.0, 0.0, 0.0]]))


def test_anchor_free_head_temporal_positions_are_batch_invariant_for_short_windows() -> None:
    torch = _require_torch()
    from opentad.models.dense_heads.anchor_free_head import AnchorFreeHead

    head = AnchorFreeHead.__new__(AnchorFreeHead)
    short_meta = {
        "selected_axis_to_true_time_dense_index": [0, 4, 8],
        "selected_valid_len": 3,
    }
    long_meta = {
        "selected_axis_to_true_time_dense_index": [0, 5, 10, 15, 20],
        "selected_valid_len": 5,
    }

    solo = head._temporal_positions_from_metas(
        [short_meta],
        torch.device("cpu"),
        torch.float32,
        target_len=4,
    )
    batched = head._temporal_positions_from_metas(
        [short_meta, long_meta],
        torch.device("cpu"),
        torch.float32,
        target_len=4,
    )
    assert torch.equal(solo[0], torch.tensor([0.0, 4.0, 8.0, 8.0]))
    assert torch.equal(batched[0], solo[0])
    assert torch.equal(batched[1], torch.tensor([0.0, 5.0, 10.0, 15.0]))


def test_global_rank_clip_coordinates_supports_mixed_short_and_full_windows() -> None:
    torch = _require_torch()
    from opentad.models.utils.temporal_grid import global_rank_clip_coordinates

    positions = torch.tensor(
        [
            [0, 1, 2, 3, 4, 4, 4, 4],
            [0, 1, 3, 5, 6, 8, 10, 11],
        ]
    )
    coords = global_rank_clip_coordinates(
        positions,
        torch.tensor([5, 12]),
        k=8,
        clip_len=4,
        tubelet_size=2,
    )

    assert coords["actual"].shape == (2, 2, 2)
    assert torch.equal(coords["actual"][0], coords["canonical"][0])
    assert torch.equal(coords["irregular_selected_positions"], positions)


def test_frozen_mobilenet_feature_change_has_no_uninitialized_output_head() -> None:
    _require_torch()
    from tools.bata.train_lowres_action_probe import C3MobileNetV3ActionProbe

    probe = C3MobileNetV3ActionProbe(
        pretrained=False,
        freeze_backbone=True,
        preserve_pretrained_classifier=True,
    )
    assert probe.output_head is None
    assert all(not parameter.requires_grad for parameter in probe.module.parameters())


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

    selected = torch.tensor([[[1.0, 2.0, 3.0]]], requires_grad=True)
    objective = (selected * torch.tensor([[[2.0, -3.0, 4.0]]])).sum()
    signed = DucaOnlineFrameSelector._selected_detector_contribution(
        selected,
        objective,
        mode="signed_removal_utility",
    )
    assert signed.shape == (1, 3)
    assert signed.requires_grad is False
    assert torch.all(signed >= 0.0)
    assert torch.allclose(signed, torch.tensor([[0.0, 6.0, 0.0]]))

    absolute = DucaOnlineFrameSelector._selected_detector_contribution(
        selected,
        objective,
        mode="abs_grad_times_input",
    )
    assert torch.allclose(absolute, torch.tensor([[2.0, 6.0, 12.0]]))
    assert not torch.equal(absolute, signed)


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


def test_h65_pro_slurm_contract_is_fail_closed_and_collision_resistant() -> None:
    submitter = (ROOT / "tools/experiments/submit_h65_pro_fullmatrix.sh").read_text(encoding="utf-8")
    train = (ROOT / "tools/experiments/run_h65_pro_train.sbatch").read_text(encoding="utf-8")
    eval_script = (ROOT / "tools/experiments/run_h65_pro_eval.sbatch").read_text(encoding="utf-8")

    assert '[[ -n "${H65_PRO_EXPECTED_COMMIT:-}" ]]' in submitter
    assert 'COMMIT="${H65_PRO_EXPECTED_COMMIT:-$(git rev-parse HEAD)}"' not in submitter
    assert "EVAL_SUBMIT_FAILED_TRAIN_CANCEL_REQUESTED" in submitter
    assert 'scancel "$train_dependency"' in submitter
    assert "wait_for_submission_slots" in submitter
    assert "retrying in 60 seconds" in submitter
    assert "--export=ALL,CUDA_VISIBLE_DEVICES=1" not in submitter
    assert 'H65_PRO_WORK_ROOT="${H65_PRO_WORK_ROOT:-$YUZIBO_ROOT/experiments/' in submitter
    assert '[[ ! -s "$REGISTRY" ]]' in submitter
    assert 'H65_PRO_WORK_ROOT="$H65_PRO_WORK_ROOT"' in submitter

    for script in (train, eval_script):
        assert '[[ -n "${H65_PRO_EXPECTED_COMMIT:-}" ]]' in script
        assert 'EXPECTED_COMMIT="${H65_PRO_EXPECTED_COMMIT:-$(git rev-parse HEAD)}"' not in script
        assert 'slurm_job_id="${SLURM_JOB_ID:-0}"' in script
        assert "MASTER_PORT=$((30000 + (slurm_job_id % 20000)))" in script
        assert 'export DUCA_SELECTED_OPT_VARIANT="$VARIANT"' in script
        assert '[[ -n "${SLURM_JOB_ID:-}" ]]' in script
        assert "torch.cuda.device_count()" in script
        assert "direct H65-Pro launch requires physical GPU1" in script
        assert 'H65_PRO_WORK_ROOT:-$YUZIBO_ROOT/experiments/' in script
        assert "must stay outside the clean source checkout" in script


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
