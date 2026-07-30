import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_actionformer_official_matched_pair_n16r4.sbatch"


def test_official_matched_pair_launcher_is_syntax_valid_and_fail_closed():
    subprocess.run(
        ["bash", "-n", LAUNCHER.relative_to(ROOT).as_posix()],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    source = LAUNCHER.read_text(encoding="utf-8")
    assert source.index("source /etc/profile") < source.index("set -u")
    assert "CUDA_VISIBLE_DEVICES=" not in source
    assert 'test "$SCREENING_SEED" = "1234567891"' in source
    assert 'test "$EXPECTED_TERMINAL_EPOCH" = "35"' in source
    assert source.count("run_arm ") == 2
    assert 'run_arm dense "configs/thumos_i3d.yaml"' in source
    assert (
        'run_arm sparse "configs/thumos_i3d_sparsehead_k384_uniform.yaml"'
        in source
    )
    assert source.count("-epoch 35") == 2
    assert "--saveonly" in source
    assert "evaluate_actionformer_raw_predictions.py" in source
    assert "validate_attestation_snapshot" in source
    assert "validate_attestation_live(payload)" not in source
    assert ': "${ACTIONFORMER_PYTHON_ENV:?' in source
    assert ': "${ACTIONFORMER_ENVIRONMENT_RECEIPT:?' in source
    assert "EXPECTED_ENVIRONMENT_RECEIPT_SHA256" in source
    assert ': "${ACTIONFORMER_NMS_EXTENSION:?' in source
    assert "EXPECTED_ACTIONFORMER_NMS_EXTENSION_SHA256" in source
    assert 'assert tensorboard.__version__ == "2.20.0"' in source
    assert 'assert numpy.__version__ == "1.23.5"' in source
    assert 'module_path = Path(nms_1d_cpu.__file__).resolve()' in source
    assert "assert module_path == expected_module_path" in source
    environment_probe = source[
        source.index('ACTIONFORMER_PYTHON_ENV="$ACTIONFORMER_PYTHON_ENV"') :
        source.index('STAGE="source_identity"')
    ]
    assert environment_probe.index("import torch") < environment_probe.index(
        "import nms_1d_cpu"
    )
    assert "indices = nms_1d_cpu.softnms(" in source
    assert '"nms_softnms_7arg_probe": True' in source
    assert '"official_nms_extension": receipt(' in source
    assert '"paper_main_table_eligible": False' in source
    assert '"end_to_end_cost_claim_allowed": False' in source
    assert "requires_preregistered_multiseed" in source


def test_launcher_never_resumes_or_changes_the_official_seed():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "--resume" not in source
    assert "1234567891" in source
    assert "epoch_035.pth.tar" in source
    assert "state_dict_ema" in source
    assert "optimizer_epochs" in source
    assert "warmup_epochs" in source


def test_launcher_rejects_the_shadowing_open_tad_softnms_abi():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "ACTIONFORMER_NMS_EXTENSION is outside the pinned Python environment" in source
    assert 'test -f "$ACTIONFORMER_NMS_EXTENSION"' in source
    assert (
        'sha256sum "$ACTIONFORMER_NMS_EXTENSION"'
        in source
    )
    probe = source[source.index("indices = nms_1d_cpu.softnms(") :]
    probe = probe[: probe.index("assert indices.numel()")]
    assert "t1" not in probe
    assert "t2" not in probe
    assert probe.count(",") == 7
