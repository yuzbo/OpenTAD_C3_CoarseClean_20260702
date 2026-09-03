from pathlib import Path
from mmengine.config import Config


def test_et_trc_pair_uses_torchrun_for_local_rank_contract():
    script = Path(__file__).resolve().parents[1] / "scripts" / "submit_zoomtoken_et_trc_pair_n16r4.sh"
    text = script.read_text(encoding="utf-8")
    assert "torchrun --standalone --nproc_per_node=2 tools/train.py" in text
    assert "; python tools/train.py" not in text
    assert "source /etc/profile; set -euo pipefail" in text
    assert "workflow.max_train_iters=1" in text
    assert '--dependency="afterok:' in text
    assert "retrying in 60 seconds" in text
    assert "git branch --show-current" not in text


def test_et_trc_configs_bind_global_batch_for_world2():
    root = Path(__file__).resolve().parents[1]
    for name in (
        "et_trc_videomae_s_768x1_160_adapter_seed4407.py",
        "et_trc_videomae_s_768x1_160_adapter_off_seed4407.py",
    ):
        cfg = Config.fromfile(str(root / "configs" / "adatad" / "thumos" / name))
        assert cfg.solver.train.batch_size == 2
        assert cfg.solver.val.batch_size == 2
        assert cfg.solver.test.batch_size == 2
