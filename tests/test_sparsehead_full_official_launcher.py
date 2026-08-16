from pathlib import Path


def test_launcher_trains_and_evaluates_with_legal_memory():
    launcher = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_sparsehead_route_t_full_official_n16r4.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --mem=" not in launcher
    assert 'tools/train.py "$config"' in launcher
    assert 'tools/test.py "$config" --checkpoint "$checkpoint"' in launcher
    assert "epoch_59.pth" in launcher
