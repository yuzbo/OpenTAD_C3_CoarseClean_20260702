from __future__ import annotations

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


ROOT = Path(__file__).resolve().parents[1]
UNIFORM = (
    ROOT
    / "configs/adatad/thumos/duca_two_stage_exact_uniform_fixed384_official60.py"
)
LEARNED = (
    ROOT
    / "configs/adatad/thumos/duca_global_curriculum_g1_protected_fixed384_official60.py"
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
    assert len(list((output / "configs").glob("*.py"))) == 24
    assert len(list((output / "jobs").glob("*.sbatch"))) == 25
    assert len((output / "cells.tsv").read_text(encoding="utf-8").splitlines()) == 25
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
    assert cfg.workflow.formal_successful_update_contract is False


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


def test_launcher_only_generates_and_never_submits() -> None:
    source = (ROOT / "scripts/launch_duca_r5_paper_matrix.sh").read_text(
        encoding="utf-8"
    )
    assert "Generated only; no jobs were submitted." in source
    assert "duca_r5_paper_matrix submit" not in source
    assert "ALLOW_DUCA_R5_SUBMIT" not in source


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
