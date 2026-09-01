import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_duca_independent_official60_gpu1.sh"
SUBMIT = ROOT / "scripts/submit_duca_independent_official60_suite.sh"


def test_independent_suite_has_no_inter_job_dependencies() -> None:
    source = SUBMIT.read_text(encoding="utf-8")
    assert "--dependency" not in source
    assert '"dependency": "none"' in source
    assert '"inter_job_dependencies": False' in source
    for variant in (
        "two_stage_exact_uniform",
        "gaussian_matched_g0",
        "boundary_burst_r2q3_g0",
        "boundary_burst_r4q5_g0",
    ):
        assert variant in source


def test_each_arm_is_self_contained_and_uses_terminal_official_map() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    submit_source = SUBMIT.read_text(encoding="utf-8")
    assert 'P0_CHECKPOINT="${P0_WORK}/gpu1_id0/checkpoint/epoch_19.pth"' in source
    assert ': > "${EMPTY_BLOCK_LIST}"' in source
    assert "run_duca_protected_e2e_exact_full_model_gate.py" in source
    assert 'CHECKPOINT="${WORK_DIR}/gpu1_id0/checkpoint/epoch_59.pth"' in source
    assert "tools/test.py" in source
    assert "--checkpoint-state-key state_dict_ema" in source
    assert 'evaluation_config.get("subset") != "validation"' in source
    assert '"official_validation_comparable": True' in source
    assert "require_absent" not in source
    assert "require_absent" not in submit_source


def test_protocol_audit_classifies_internal_and_official_metrics(tmp_path: Path) -> None:
    annotation = tmp_path / "annotation.json"
    class_map = tmp_path / "class_map.txt"
    annotation.write_text("{}\n", encoding="utf-8")
    class_map.write_text("action\n", encoding="utf-8")
    output = tmp_path / "audit.json"
    env = os.environ.copy()
    env.update(
        {
            "YUZIBO_ROOT": str(tmp_path),
            "THUMOS14_ANNOTATION_PATH": str(annotation),
            "THUMOS14_CLASS_MAP": str(class_map),
            "THUMOS14_TRAIN_DATA_PATH": str(tmp_path / "train"),
            "THUMOS14_TEST_DATA_PATH": str(tmp_path / "test"),
        }
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.bata.audit_duca_map_protocols",
            "--output-json",
            str(output),
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    classes = {row["stage"]: row for row in payload["experiment_classes"]}
    assert classes["R0_holdout_replay"]["official_validation_comparable"] is False
    assert (
        classes["R3_official60_terminal_evaluation"][
            "official_validation_comparable"
        ]
        is True
    )
    assert classes["R5_terminal_matrix_cells"]["official_validation_comparable"] is None
    assert (
        classes["R5_terminal_matrix_cells"][
            "protocol_eligible_for_official_validation"
        ]
        is True
    )
    assert payload["bootstrap_contract"]["may_block_official60_training"] is False
    assert all(
        row["official_validation_comparable"]
        for row in payload["formal_official60_arms"]
    )
