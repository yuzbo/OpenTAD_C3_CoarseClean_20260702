from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_duca_native_tubelet_coreset_n16r4.sbatch"


def test_launcher_binds_the_two_arms_and_terminal_ema_evaluation() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "duca_native_tubelet_uniform_reconstruct_fixed384_official60.py" in text
    assert "duca_native_tubelet_coreset_fixed384_official60.py" in text
    assert "workflow.max_train_iters=2" in text
    assert '--resume "$PRE_RUN_CHECKPOINT"' in text
    assert "successful_optimizer_updates\"]) == 4" in text
    assert "successful_optimizer_updates\"]) == 6000" in text
    assert "epoch_59.pth" in text
    assert "--checkpoint-state-key state_dict_ema" in text
    assert "--expected-checkpoint-epoch 59" in text
    assert "DUCA_STAGE1_CHECKPOINT_EPOCH=29" in text
    assert "CUDA_VISIBLE_DEVICES=" not in text
