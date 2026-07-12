from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "bata" / "run_phystime_attention_diagnostic.py"


def test_attention_diagnostic_uses_real_checkpoint_and_never_trains():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'checkpoint.get("state_dict_ema"' in text
    assert "build_dataset" in text
    assert "summarize_attention_rows" in text
    assert "query_embedding" in text
    assert "key_proj" in text
    assert "relative_time_mlp" in text
    assert '"phystime_attention_checkpoint_diagnostic_v1"' in text
    assert "optimizer" not in text.lower()
    assert "backward(" not in text
