import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    ROOT / "scripts" / "run_actionformer_official_s0_attribution_n16r4.sbatch"
)


def test_s0_attribution_launcher_is_syntax_valid_and_no_retraining():
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
    assert "train.py" not in source
    assert "--resume" not in source
    assert "torch.optim" not in source
    assert source.count("run_cell ") == 2
    assert "full_train_k384_eval" in source
    assert "selected_train_dense_eval" in source
    assert "configs/thumos_i3d.yaml" in source
    assert "configs/thumos_i3d_sparsehead_k384_uniform.yaml" in source
    assert source.count("--saveonly") == 1
    assert "evaluate_actionformer_raw_predictions.py" in source
    assert "validate_actionformer_s0_screening.py" in source
    assert "analyze_actionformer_matched_predictions.py" in source
    assert "state_dict_ema" in source
    assert "epoch_035.pth.tar" in source
    assert "ATTRIBUTION_COMPLETE.json" in source
    assert "NEGATIVE_DIAGNOSTICS.json" in source
    assert "S0_NEGATIVE_ANALYSIS_SUITE_COMPLETE.json" in source
    assert '"new_training": False' in source
    assert '"paper_main_table_eligible": False' in source
    assert '"independent_seed_count_increment": 0' in source


def test_s0_attribution_never_overwrites_parent_predictions_or_checkpoints():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert 'WORK_ROOT="$RUN_ROOT/work"' in source
    assert 'ln -s "$source_checkpoint" "$checkpoint_link"' in source
    assert 'raw_predictions="$ckpt_dir/eval_results.pkl"' in source
    assert 'test ! -e "$RUN_ROOT/ATTRIBUTION_COMPLETE.json"' in source
    assert 'test -f "$PARENT_PAIR_ROOT/MATCHED_PAIR_COMPLETE.json"' in source
    assert "os.replace(temporary, output)" in source


def test_s0_attribution_binds_official_runtime_and_import_order():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert ': "${ACTIONFORMER_PYTHON_ENV:?' in source
    assert ': "${ACTIONFORMER_NMS_EXTENSION:?' in source
    assert "EXPECTED_ACTIONFORMER_NMS_EXTENSION_SHA256" in source
    environment_probe = source[
        source.index('ACTIONFORMER_PYTHON_ENV="$ACTIONFORMER_PYTHON_ENV"') :
        source.index('STAGE="source_identity"')
    ]
    assert environment_probe.index("import torch") < environment_probe.index(
        "import nms_1d_cpu"
    )
    assert "indices = nms_1d_cpu.softnms(" in environment_probe
