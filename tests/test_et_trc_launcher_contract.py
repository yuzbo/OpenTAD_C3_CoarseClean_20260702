from pathlib import Path


def test_et_trc_pair_uses_torchrun_for_local_rank_contract():
    script = Path(__file__).resolve().parents[1] / "scripts" / "submit_zoomtoken_et_trc_pair_n16r4.sh"
    text = script.read_text(encoding="utf-8")
    assert "torchrun --standalone --nproc_per_node=2 tools/train.py" in text
    assert "; python tools/train.py" not in text
    assert "source /etc/profile; set -euo pipefail" in text
