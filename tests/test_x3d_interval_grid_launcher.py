from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]


def test_x3d_interval_grid_covers_xs_and_s_intervals():
    script = (ROOT / "scripts" / "run_duca_trainfree_x3d_interval_grid_gpu0.sh").read_text(encoding="utf-8")
    assert 'PROVIDERS="${PROVIDERS:-x3d_xs x3d_s}"' in script
    assert 'FRAME_INTERVALS="${FRAME_INTERVALS:-1 2 4}"' in script
    assert "clip_frames=4" in script
    assert "clip_frames=13" in script
    assert "x3d_xs|efficient_x3d_xs)" in script
    assert "x3d_s|efficient_x3d_s)" in script
    assert "t${clip_frames}x${frame_interval}" in script
    assert "tools/bata/summarize_trainfree_x3d_interval_grid.py" in script


def test_single_x3d_launcher_logs_clip_interval_settings():
    script = (ROOT / "scripts" / "run_duca_trainfree_x3d_actionness_selection_gpu0.sh").read_text(encoding="utf-8")
    assert "clip_frames=${CLIP_FRAMES}" in script
    assert "frame_interval=${FRAME_INTERVAL}" in script
    assert "tools/bata/eval_zero_shot_actionness.py" in script
    assert "_coarse_eval.summary.json" in script
    assert "--source-mode manual_jsonl" in script


def test_x3d_grid_summary_collects_coarse_and_selection_metrics(tmp_path):
    from tools.bata import summarize_trainfree_x3d_interval_grid as summarize

    manifest = tmp_path / "manifest.tsv"
    cell_root = tmp_path / "x3d_xs_t4x1"
    cell_root.mkdir()
    manifest.write_text(
        "provider\tclip_frames\tframe_interval\tcrop_size\tbatch_size\tout_root\tstatus\n"
        f"x3d_xs\t4\t1\t160\t16\t{cell_root}\tcomplete\n",
        encoding="utf-8",
    )
    (cell_root / "x3d_xs_validation_actionness.summary.json").write_text(
        json.dumps({"row_count": 8, "video_count": 1, "source_provenance": {"provider": "x3d_xs"}}),
        encoding="utf-8",
    )
    (cell_root / "x3d_xs_validation_coarse_eval.summary.json").write_text(
        json.dumps({"metrics": {"auroc": 0.75, "auprc": 0.8, "recall_at_k": {"384": 1.0}}}),
        encoding="utf-8",
    )
    (cell_root / "x3d_xs_validation_selection.summary.json").write_text(
        json.dumps(
            {
                "baseline_summaries": {
                    "manual": {
                        "mean_action_touched_recall": 0.9,
                        "mean_boundary_radius_recall": 0.5,
                        "mean_p95_hole": 4.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    summary = summarize.summarize_grid(
        manifest_tsv=manifest,
        summary_json=tmp_path / "grid.summary.json",
        summary_tsv=tmp_path / "grid.summary.tsv",
        subset="validation",
    )

    assert summary["decision"] == summarize.READY
    assert summary["row_count"] == 1
    row = summary["rows"][0]
    assert row["provider"] == "x3d_xs"
    assert row["clip_frames"] == 4
    assert row["uses_original_x3d_clip_window"] is True
    assert row["coarse_auroc"] == 0.75
    assert row["selection_manual_mean_boundary_radius_recall"] == 0.5
