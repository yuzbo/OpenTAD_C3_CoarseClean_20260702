from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_x3d_interval_grid_covers_xs_and_s_intervals():
    script = (ROOT / "scripts" / "run_duca_trainfree_x3d_interval_grid_gpu0.sh").read_text(encoding="utf-8")
    assert 'PROVIDERS="${PROVIDERS:-x3d_xs x3d_s}"' in script
    assert 'FRAME_INTERVALS="${FRAME_INTERVALS:-1 2 4}"' in script
    assert "clip_frames=4" in script
    assert "clip_frames=13" in script
    assert "t${clip_frames}x${frame_interval}" in script


def test_single_x3d_launcher_logs_clip_interval_settings():
    script = (ROOT / "scripts" / "run_duca_trainfree_x3d_actionness_selection_gpu0.sh").read_text(encoding="utf-8")
    assert "clip_frames=${CLIP_FRAMES}" in script
    assert "frame_interval=${FRAME_INTERVAL}" in script
