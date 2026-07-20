from pathlib import Path

import pytest

from tools.bata.run_continuous_roi_s2_one_step_gate import (
    AUDITED_SOURCE_PATHS,
    CONFIGS,
    _parameter_component,
    audit_optimizer_coverage,
)


def test_gate_audits_its_own_executable_surface():
    required = {
        "scripts/run_continuous_roi_s2_cuda_gate_slurm.sh",
        "tools/bata/run_continuous_roi_s2_one_step_gate.py",
        "opentad/models/backbones/continuous_roi_wrapper.py",
        "opentad/models/detectors/actionformer.py",
        "opentad/cores/optimizer.py",
    }
    assert required.issubset(AUDITED_SOURCE_PATHS)
    assert Path(CONFIGS["U128"]).name == (
        "continuous_roi_s2_u128_videomae_s_768x1_adapter.py"
    )


@pytest.mark.parametrize(
    "name,expected",
    [
        ("backbone.model.backbone.blocks.0.adapter.gamma", "shared_adapter"),
        ("backbone.fusion.alpha.weight", "fusion"),
        ("backbone.global_aux_head.weight", "global_aux_head"),
        ("backbone.local_aux_head.weight", "local_aux_head"),
        ("projection.embed.0.weight", "projection"),
        ("rpn_head.cls_head.cls.weight", "rpn_head"),
    ],
)
def test_gate_parameter_components_are_fail_closed(name, expected):
    assert _parameter_component(name) == expected


def test_optimizer_coverage_rejects_duplicate_or_missing_parameters():
    parameter = type("Parameter", (), {"requires_grad": True})()
    model = type(
        "Model",
        (),
        {
            "named_parameters": lambda self: [
                ("projection.weight", parameter),
            ]
        },
    )()
    optimizer = type(
        "Optimizer",
        (),
        {
            "param_groups": [
                {"params": [parameter, parameter]}
            ]
        },
    )()
    with pytest.raises(RuntimeError, match="optimizer coverage"):
        audit_optimizer_coverage(model, optimizer)


def test_gate_launcher_uses_slurm_logical_cuda_zero_without_override():
    launcher = Path(
        "scripts/run_continuous_roi_s2_cuda_gate_slurm.sh"
    ).read_text(encoding="utf-8")
    assert "srun --exact" in launcher
    assert "--gpus=1" in launcher
    assert "--cpus-per-task=5" in launcher
    assert "--mem=96000M" in launcher
    assert "--device cuda:0" in launcher
    assert "CONTINUOUS_ROI_S2_SOURCE_ROOT" in launcher
    assert 'dirname "${BASH_SOURCE[0]}"' not in launcher
    assert 'cd "${ROOT}"' in launcher
    assert "export CUDA_VISIBLE_DEVICES=" not in launcher
    assert "official" not in launcher.lower()
