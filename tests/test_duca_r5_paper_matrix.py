from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from mmengine.config import Config

from tools.bata.duca_r5_paper_matrix import (
    BACKENDS,
    BUDGETS,
    SEEDS,
    generate_matrix,
)
from tools.bata import duca_selected_axis_training as formal_training


ROOT = Path(__file__).resolve().parents[1]
UNIFORM = (
    ROOT
    / "configs/adatad/thumos/duca_two_stage_exact_uniform_fixed384_official60.py"
)
LEARNED = (
    ROOT
    / "configs/adatad/thumos/duca_boundary_burst_g1_protected_fixed384_official60.py"
)


def _generate(tmp_path: Path) -> tuple[dict, Path]:
    output = tmp_path / "r5"
    summary = generate_matrix(
        repo_root=ROOT,
        output_dir=output,
        uniform_config=UNIFORM,
        learned_config=LEARNED,
    )
    return summary, output


def test_generator_writes_only_explicit_configs_jobs_and_index(tmp_path: Path) -> None:
    summary, output = _generate(tmp_path)
    assert summary["cell_count"] == len(BACKENDS) * 2 * len(BUDGETS) * len(SEEDS) == 24
    assert summary["seeds"] == [3407, 5801, 8123]
    assert summary["budgets"] == [384, 256]
    assert summary["learned_variant"] == "boundary_burst_r2q3_g1"
    assert len(list((output / "configs").glob("*.py"))) == 24
    assert len(list((output / "jobs").glob("*.sbatch"))) == 30
    assert len((output / "cells.tsv").read_text(encoding="utf-8").splitlines()) == 25
    assert len((output / "costs.tsv").read_text(encoding="utf-8").splitlines()) == 5
    assert summary["cost_count"] == 4
    assert summary["git_commit"]
    assert (output / "matrix_summary.json.sha256").is_file()
    gate = (output / "jobs/temporalmaxer_one_step.sbatch").read_text(
        encoding="utf-8"
    )
    assert "run_duca_temporalmaxer_one_step" in gate
    assert "sbatch " not in gate
    generated_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [*(output / "configs").glob("*.py"), *(output / "jobs").glob("*.sbatch")]
    ).lower()
    assert "ledger" not in generated_text
    assert "manifest" not in generated_text


def test_generated_temporalmaxer_config_resolves_real_k256_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DUCA_FRONTEND_CHECKPOINT", "/tmp/frontend.pth")
    monkeypatch.setenv("DUCA_FRONTEND_CHECKPOINT_SHA256", "0" * 64)
    monkeypatch.setenv("DUCA_FRONTEND_CHECKPOINT_EPOCH", "6")
    _, output = _generate(tmp_path)
    cfg = Config.fromfile(
        str(output / "configs/temporalmaxer_learned_k256_s5801.py")
    )
    assert cfg.r5_cell == {
        "backend": "temporalmaxer",
        "arm": "learned",
        "budget": 256,
        "seed": 5801,
        "source_config": str(LEARNED).replace("\\", "/"),
        "live_duca_to_videomae": True,
        "detector_type": "TemporalMaxer",
        "paper_claim_allowed": False,
    }
    assert cfg.model.type == "TemporalMaxer"
    assert cfg.model.frame_selector.budget == 256
    assert cfg.model.frame_selector.temporal_sampling_contract.hard_budget == 256
    assert cfg.model.backbone.backbone.total_frames == 256
    assert cfg.model.backbone.custom.pre_processing_pipeline[0].t1 == 16
    assert cfg.model.backbone.custom.post_processing_pipeline[-1].size == 256
    assert cfg.model.projection.type == "TemporalMaxerProj"
    assert cfg.model.projection.in_channels == 384
    assert cfg.model.rpn_head.type == "TemporalMaxerHead"
    assert cfg.workflow.formal_protocol == "duca_r5_mechanism_matrix_v1"
    assert cfg.workflow.formal_successful_update_contract is True


def test_generated_actionformer_cell_keeps_official_head(tmp_path: Path) -> None:
    _, output = _generate(tmp_path)
    cfg = Config.fromfile(
        str(output / "configs/actionformer_uniform_k384_s8123.py")
    )
    assert cfg.model.type == "ActionFormer"
    assert cfg.model.projection.type == "Conv1DTransformerProj"
    assert cfg.model.projection.max_seq_len == 384
    assert cfg.model.rpn_head.type == "ActionFormerHead"
    assert cfg.r5_cell.seed == 8123
    contract = formal_training.formal_training_contract(cfg)
    assert contract is not None
    assert contract["formal_protocol"] == formal_training.R5_FORMAL_PROTOCOL
    assert contract["expected_successful_optimizer_updates"] == 6000


@pytest.mark.skipif(os.name == "nt", reason="selection validator imports Windows torch")
def test_r5_runtime_binding_reopens_matrix_gate_and_learned_frontend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools.bata import duca_boundary_burst_hard_swap_alignment as alignment
    from tools.bata import select_duca_boundary_burst_candidates as selection

    monkeypatch.setenv("DUCA_FRONTEND_CHECKPOINT", str(tmp_path / "frontend.pth"))
    monkeypatch.setenv("DUCA_FRONTEND_CHECKPOINT_SHA256", "0" * 64)
    monkeypatch.setenv("DUCA_FRONTEND_CHECKPOINT_EPOCH", "4")
    summary, output = _generate(tmp_path)
    config = output / "configs/actionformer_learned_k256_s5801.py"
    frontend = tmp_path / "frontend.pth"
    pretrain = tmp_path / "pretrain.pth"
    annotation = tmp_path / "annotation.json"
    class_map = tmp_path / "class_map.txt"
    frontend.write_bytes(b"frontend")
    pretrain.write_bytes(b"pretrain")
    annotation.write_text("{}\n", encoding="utf-8")
    class_map.write_text("action\n", encoding="utf-8")
    frontend_sha = hashlib.sha256(frontend.read_bytes()).hexdigest()
    learned_variant = summary["learned_variant"]
    decision = {
        "family_routing": {"selected_p0_variant": "boundary_burst_r2q3"},
        "winners": {
            "boundary_burst_r2q3": {
                "checkpoint_path": str(frontend.resolve()),
                "checkpoint_sha256": frontend_sha,
                "epoch_one_based": 5,
            }
        },
    }
    monkeypatch.setattr(selection, "validate_frontend_decision", lambda **_: decision)
    monkeypatch.setattr(
        alignment,
        "validate_alignment_artifact",
        lambda **_: {"path": "/alignment.json", "sha256": "a" * 64},
    )
    gate = {
        "ok": True,
        "task": "offline_temporal_action_detection",
        "git_commit": summary["git_commit"],
        "detector_type": "TemporalMaxer",
        "forward_backward_optimizer_step_completed": True,
        "config": summary["gate_config"],
        "config_sha256": summary["gate_config_sha256"],
        "pretrain_path": str(pretrain.resolve()),
        "pretrain_sha256": hashlib.sha256(pretrain.read_bytes()).hexdigest(),
        "selector_initialization": {
            "checkpoint_sha256": frontend_sha,
            "checkpoint_epoch": 4,
        },
    }
    gate_path = output / "temporalmaxer_one_step.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    matrix_path = output / "matrix_summary.json"
    monkeypatch.setenv("R5_MATRIX_SUMMARY", str(matrix_path))
    monkeypatch.setenv(
        "R5_MATRIX_SUMMARY_SHA256", hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    )
    monkeypatch.setenv("R5_MECHANISM_GATE_JSON", str(gate_path))
    monkeypatch.setenv(
        "R5_MECHANISM_GATE_SHA256", hashlib.sha256(gate_path.read_bytes()).hexdigest()
    )
    monkeypatch.setenv("R5_FRONTEND_DECISION", "/decision.json")
    monkeypatch.setenv("R5_FRONTEND_DECISION_SHA256", "b" * 64)
    monkeypatch.setenv("R5_ALIGNMENT_JSON", "/alignment.json")
    monkeypatch.setenv("R5_ALIGNMENT_SHA256", "c" * 64)
    cfg = Config.fromfile(str(config))
    cell = cfg.r5_cell.to_dict()
    bindings = formal_training.build_runtime_bindings(
        git_commit=summary["git_commit"],
        variant="actionformer_learned_k256_s5801",
        seed=5801,
        slurm_job_id="7",
        source_config_path=config,
        source_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        resolved_config_sha256="d" * 64,
        runtime_config_sha256="e" * 64,
        evaluation_annotation_path=annotation,
        evaluation_class_map_path=class_map,
        evaluation_config={
            "type": "mAP",
            "ground_truth_filename": str(annotation),
            "subset": "validation",
            "tiou_thresholds": [0.3, 0.4, 0.5, 0.6, 0.7],
        },
        runtime_pretrain_path=pretrain,
        selector_initialization={
            "enabled": True,
            "checkpoint_path": str(frontend),
            "checkpoint_sha256": frontend_sha,
            "expected_checkpoint_epoch": 4,
            "state_key": "state_dict_ema",
        },
        formal_protocol=formal_training.R5_FORMAL_PROTOCOL,
        r5_cell=cell,
    )
    assert bindings["variant"] == "actionformer_learned_k256_s5801"
    assert bindings["seed"] == 5801
    assert bindings["mechanism_gate_sha256"] == hashlib.sha256(
        gate_path.read_bytes()
    ).hexdigest()
    assert bindings["selector_initialization_contract"]["learned_variant"] == learned_variant


def test_generator_rejects_source_config_outside_repo(tmp_path: Path) -> None:
    outside = tmp_path / "outside.py"
    outside.write_text("model = {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inside the repository"):
        generate_matrix(
            repo_root=ROOT,
            output_dir=tmp_path / "output",
            uniform_config=outside,
            learned_config=LEARNED,
        )


def test_real_one_step_runner_uses_production_data_model_and_optimizer() -> None:
    source = (
        ROOT / "tools/bata/run_duca_temporalmaxer_one_step.py"
    ).read_text(encoding="utf-8")
    for required in (
        "build_dataset(cfg.dataset.train",
        "build_dataloader(",
        "build_detector(copy.deepcopy(cfg.model))",
        "initialize_frame_selector_from_checkpoint(",
        "prepare_optimizer_parameter_freezing(",
        "build_optimizer(",
        "torch.autograd.grad(",
        "scaler.scale(total).backward()",
        "selected_axis_summary",
    ):
        assert required in source
    assert "Dummy" not in source


def test_launcher_submits_gate_then_real_matrix_when_requested() -> None:
    source = (ROOT / "scripts/launch_duca_r5_paper_matrix.sh").read_text(
        encoding="utf-8"
    )
    assert 'R5_SUBMIT="${R5_SUBMIT:-0}"' in source
    assert 'R5_UPSTREAM_DEPENDENCY="${R5_UPSTREAM_DEPENDENCY:-}"' in source
    assert 'sbatch --parsable --clusters="${TARGET_CLUSTER}"' in source
    assert '--dependency="${R5_UPSTREAM_DEPENDENCY}"' in source
    assert '--dependency="afterok:${gate_job}"' in source
    assert '"${OUTPUT_DIR}/costs.tsv"' in source
    assert '"${OUTPUT_DIR}/jobs/aggregate.sbatch"' in source
    assert '>> "${OUTPUT_DIR}/jobs.tsv"' in source
    assert 'jobs.tsv.sha256' in source


def test_generated_jobs_bind_frontend_alignment_and_terminal_map(
    tmp_path: Path,
) -> None:
    _, output = _generate(tmp_path)
    learned = (output / "jobs/actionformer_learned_k384_s3407.sbatch").read_text(
        encoding="utf-8"
    )
    uniform = (output / "jobs/actionformer_uniform_k384_s3407.sbatch").read_text(
        encoding="utf-8"
    )
    gate = (output / "jobs/temporalmaxer_one_step.sbatch").read_text(
        encoding="utf-8"
    )
    for text in (learned, gate):
        assert "validate_frontend_decision" in text
        assert "validate_alignment_artifact" in text
        assert "DUCA_BOUNDARY_BURST_ALIGNMENT_JSON" in text
        assert "R5_FRONTEND_DECISION_SHA256_FILE" in text
        assert "R5_ALIGNMENT_SHA256_FILE" in text
    assert "R5_MATRIX_SUMMARY_SHA256" in learned
    assert "R5_MECHANISM_GATE_SHA256" in learned
    assert "tools/test.py" in learned
    assert "--checkpoint-state-key state_dict_ema" in learned
    assert "--expected-checkpoint-epoch 59" in learned
    assert "terminal_evaluation.json" in learned
    assert "DUCA_SELECTED_OPT_VARIANT=actionformer_learned_k384_s3407" in learned
    assert "validate_frontend_decision" not in uniform
    assert "tools/test.py" in uniform
    learned_cost = (
        output / "jobs/cost_actionformer_learned_k384_s3407.sbatch"
    ).read_text(encoding="utf-8")
    aggregate = (output / "jobs/aggregate.sbatch").read_text(encoding="utf-8")
    assert "run_duca_full_stack_cost_profile_gpu1.sh" in learned_cost
    assert "PROFILE_CHECKPOINT" in learned_cost
    assert "aggregate_duca_r5_paper_matrix" in aggregate
    assert "temporalmaxer_one_step.json.sha256" in gate


@pytest.mark.skipif(os.name == "nt", reason="Windows torch/c10.dll is unstable")
def test_temporalmaxer_live_rgb_backward_and_optimizer_coverage() -> None:
    try:
        import torch
        import torch.nn as nn
        from opentad.models.detectors.temporalmaxer import TemporalMaxer
        from opentad.models.utils.truetime_geometry import SELECTED_AXIS
    except BaseException as exc:  # pragma: no cover - dependency guard.
        pytest.skip(f"PyTorch/OpenTAD is unavailable: {exc}")

    class LiveSelector(nn.Module):
        selector_variant = "transition_only"
        transition_scorer_lr = 2e-4
        coarse_trunk_lr = 1e-5
        action_head_lr = 2e-5

        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.spatial = nn.Conv2d(1, 1, 1)
            self.norm = nn.BatchNorm2d(1)
            self.embedding = nn.Embedding(4, 2)

        @staticmethod
        def _metas(metas):
            output = []
            for meta in metas:
                updated = dict(meta)
                updated.update(
                    selected_axis_to_true_time_dense_index=[0, 2, 5, 7],
                    detector_prediction_inverse_map_required=True,
                    detector_output_coordinate_space=SELECTED_AXIS,
                    gt_remapped_to_selected_axis=True,
                    gt_coordinate_space=SELECTED_AXIS,
                )
                output.append(updated)
            return output

        def forward_train(self, inputs, masks, metas, gt_segments, gt_labels, **kwargs):
            selected = inputs[:, :, :, :4] * self.scale
            return {
                "inputs": selected,
                "masks": masks[:, :4],
                "metas": self._metas(metas),
                "gt_segments": [
                    torch.tensor([[0.0, 3.0]], device=inputs.device)
                    for _ in gt_segments
                ],
                "gt_labels": gt_labels,
                "losses": {"policy_loss": 0.01 * self.scale.square()},
            }

        def forward_test(self, inputs, masks, metas, **kwargs):
            return {
                "inputs": inputs[:, :, :, :4] * self.scale,
                "masks": masks[:, :4],
                "metas": self._metas(metas),
            }

    class VideoBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, clips):
            self.calls += 1
            return clips.mean(dim=(1, 4, 5))

    class Projection(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(1, 2, 1)

        def forward(self, features, masks):
            return self.conv(features), masks

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(2, 1, 1)
            self.segments = None

        def forward_train(
            self, features, masks, gt_segments, gt_labels, metas=None, **kwargs
        ):
            self.segments = gt_segments
            return {"detector_loss": self.conv(features).square().mean()}

        def forward_test(self, features, masks, metas=None):
            batch = features.shape[0]
            return (
                [features.new_tensor([[0.0, 3.0]]) for _ in range(batch)],
                [features.new_ones((1, 1)) for _ in range(batch)],
            )

    model = TemporalMaxer.__new__(TemporalMaxer)
    nn.Module.__init__(model)
    model.frame_selector = LiveSelector()
    model.backbone = VideoBackbone()
    model.projection = Projection()
    model.rpn_head = Head()

    inputs = torch.randn(1, 1, 1, 8, 2, 2)
    masks = torch.ones(1, 8, dtype=torch.bool)
    losses = model(
        inputs,
        masks,
        [{"video_name": "sample"}],
        gt_segments=[torch.tensor([[0.0, 7.0]])],
        gt_labels=[torch.tensor([1])],
        return_loss=True,
    )
    detector_to_selector = torch.autograd.grad(
        losses["detector_loss"], model.frame_selector.scale, retain_graph=True
    )[0]
    assert detector_to_selector.abs().item() > 0
    assert model.backbone.calls == 1
    assert model.rpn_head.segments[0].tolist() == [[0.0, 3.0]]
    assert model._last_selected_axis_training_summary["gt_coordinate_space"] == SELECTED_AXIS

    groups = model.get_optim_groups({"lr": 1e-4, "weight_decay": 0.05})
    covered = [parameter for group in groups for parameter in group["params"]]
    covered_ids = {id(parameter) for parameter in covered}
    expected_ids = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("backbone")
    }
    assert len(covered) == len(covered_ids)
    assert covered_ids == expected_ids
    for module_type in (nn.Conv2d, nn.BatchNorm2d, nn.Embedding):
        typed_ids = {
            id(parameter)
            for module in model.modules()
            if isinstance(module, module_type)
            for parameter in module.parameters(recurse=False)
            if parameter.requires_grad
        }
        assert typed_ids <= covered_ids

    optimizer = torch.optim.AdamW(groups, lr=1e-4, weight_decay=0.05)
    before = model.frame_selector.scale.detach().clone()
    losses["cost"].backward()
    assert model.rpn_head.conv.weight.grad is not None
    assert model.frame_selector.scale.grad is not None
    optimizer.step()
    assert not torch.equal(before, model.frame_selector.scale.detach())

    with torch.no_grad():
        model.forward_test(inputs, masks, metas=[{"video_name": "sample"}])
    assert model._last_forward_test_metas[0][
        "selected_axis_to_true_time_dense_index"
    ] == [0, 2, 5, 7]


@pytest.mark.skipif(os.name == "nt", reason="Windows torch/c10.dll is unstable")
def test_temporalmaxer_preserves_feature_backbone_mask_signature() -> None:
    try:
        import torch
        import torch.nn as nn
        from opentad.models.detectors.temporalmaxer import TemporalMaxer
    except BaseException as exc:  # pragma: no cover - dependency guard.
        pytest.skip(f"PyTorch/OpenTAD is unavailable: {exc}")

    class FeatureBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.received_masks = False

        def forward(self, features, masks):
            self.received_masks = masks is not None
            return features

    class Head(nn.Module):
        def forward_train(self, features, masks, gt_segments, gt_labels, **kwargs):
            return {"detector_loss": features.square().mean()}

    model = TemporalMaxer.__new__(TemporalMaxer)
    nn.Module.__init__(model)
    model.backbone = FeatureBackbone()
    model.rpn_head = Head()
    inputs = torch.randn(1, 2, 8, requires_grad=True)
    losses = model(
        inputs,
        torch.ones(1, 8, dtype=torch.bool),
        [{}],
        gt_segments=[torch.tensor([[0.0, 7.0]])],
        gt_labels=[torch.tensor([1])],
        return_loss=True,
    )
    assert model.backbone.received_masks is True
    assert losses["cost"].requires_grad
