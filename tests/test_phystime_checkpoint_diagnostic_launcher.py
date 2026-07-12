from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_phystime_checkpoint_full_eval_gpu1.sh"


def test_checkpoint_diagnostic_launcher_is_read_only_and_exports_predictions():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "SLURM_JOB_ID" in text
    assert "tools/test.py" in text
    assert "--checkpoint" in text
    assert "post_processing.save_dict=True" in text
    assert "model.backbone.custom.pretrain=" in text
    assert "result_detection.json" in text
    assert "CHECKPOINT_DIAGNOSTIC_COMPLETE" in text
    assert "tools/train.py" not in text
