from types import SimpleNamespace

from pathlib import Path

import pytest

from tools.bata import run_continuous_roi_s2_one_step_gate as gate_module
from tools.bata.run_continuous_roi_s2_one_step_gate import (
    AUDITED_SOURCE_PATHS,
    CONFIGS,
    _parameter_component,
    audit_cuda_device_identity,
    audit_optimizer_coverage,
)


def test_gate_audits_its_own_executable_surface():
    required = {
        "scripts/run_continuous_roi_s2_cuda_gate_slurm.sh",
        "tools/bata/run_continuous_roi_s2_one_step_gate.py",
        "opentad/models/backbones/continuous_roi_wrapper.py",
        "opentad/models/detectors/actionformer.py",
        "opentad/cores/optimizer.py",
        "tools/bata/profile_spatial_zoom_s1.py",
        "tools/bata/continuous_roi_s2_training.py",
        "tools/bata/continuous_roi_s2_runtime_gate.py",
        "tools/bata/deploy_continuous_roi_s2_training_matrix.py",
        "tools/bata/precheck_continuous_roi_s2_training_runtime.py",
        "scripts/run_continuous_roi_s2_train_slurm.sh",
        "tools/train.py",
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


def test_gate_uses_backward_compatible_with_official_reentrant_checkpointing():
    source = Path(
        "tools/bata/run_continuous_roi_s2_one_step_gate.py"
    ).read_text(encoding="utf-8")
    assert "detector_gradients = torch.autograd.grad(" not in source
    assert "detector_cost.backward()" in source
    assert 'losses["cost"].backward()' in source


def test_gate_binds_logical_cuda_zero_to_slurm_visible_uuid(monkeypatch):
    expected_uuid = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    monkeypatch.setattr(
        gate_module,
        "_cuda_driver_device_uuid_hex",
        lambda ordinal: "aaaaaaaabbbbccccddddeeeeeeeeeeee",
    )
    monkeypatch.setattr(
        gate_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=f"{expected_uuid}, 00000000:01:00.0, NVIDIA A100\n",
            stderr="",
        ),
    )
    torch_module = SimpleNamespace(
        cuda=SimpleNamespace(
            get_device_properties=lambda device: SimpleNamespace(name="NVIDIA A100")
        )
    )
    device = SimpleNamespace(index=0)

    identity = audit_cuda_device_identity(torch_module, device)

    assert identity["cuda_visible_device_uuid"] == expected_uuid
    assert (
        identity["cuda_runtime_device_uuid_hex"]
        == "aaaaaaaabbbbccccddddeeeeeeeeeeee"
    )
    assert identity["logical_device"] == "cuda:0"


def test_gate_rejects_logical_and_visible_uuid_mismatch(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        gate_module,
        "_cuda_driver_device_uuid_hex",
        lambda ordinal: "11111111222233334444555555555555",
    )
    monkeypatch.setattr(
        gate_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee, "
                "00000000:01:00.0, NVIDIA A100\n"
            ),
            stderr="",
        ),
    )
    torch_module = SimpleNamespace(
        cuda=SimpleNamespace(
            get_device_properties=lambda device: SimpleNamespace(name="NVIDIA A100")
        )
    )

    with pytest.raises(RuntimeError, match="UUID differs"):
        audit_cuda_device_identity(torch_module, SimpleNamespace(index=0))
