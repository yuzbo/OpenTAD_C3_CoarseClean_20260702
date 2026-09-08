"""Synthetic plot fixtures only; never emit a scientific result."""
import json

import matplotlib.pyplot as plt
import pytest

from geosparse_ext.figures import plot_pareto, plot_risks, plot_training
from test_geosparse_figures import identity


class Capture:
    def __init__(self):
        self.figures = {}

    def save(self, name, fig, caption, rows=()):
        self.figures[name] = (fig, caption, rows)
        plt.close(fig)

    def wait(self, *args, **kwargs):
        pass


@pytest.mark.parametrize("complete", [False, True])
def test_incomplete_counts_keep_scatter_but_withhold_model_frontier(tmp_path, complete):
    evaluations, exports = [], []
    # Lower-bound A=4, B=6 would rank A ahead by cost; actual missing
    # operations could put A=10 beyond B=8. Never construct that frontier.
    for route, cost, score in (("A", 4., .60), ("B", 6., .65)):
        job = dict(job_id=route, route=route, families=["F01"], dataset="fixture", model=dict(axis="ST", budget=.5, route=route))
        checkpoint = identity(route, 39)
        training = dict(source_commit="fixture", checkpoint_identity=checkpoint)
        metrics = dict(official=dict(average_mAP=score), selected_checkpoint_epoch=39, checkpoint_identity=checkpoint)
        evaluations.append((tmp_path, dict(seed=0), job, training, metrics))
        receipt = dict(source_train_id=route, is_final_checkpoint=True, full_split=True,
                       selected_checkpoint_epoch=39, model_source_commit="fixture", checkpoint_identity=checkpoint)
        exports.append((tmp_path, receipt, [dict(heavy_macs=cost * 1e9, model_macs_counted=cost * 1e9,
                                                  mac_count_complete=complete or route == "B")]))
    report = Capture()
    plot_pareto(report, {}, evaluations, exports, single_seed=True)
    fig, caption, rows = report.figures["feasibility_pareto_fixture"]
    assert len(rows) == 2  # retain measured accuracy and Heavy cost
    assert any(line.get_linestyle() == "--" for line in fig.axes[0].lines)
    assert any(line.get_linestyle() == "--" for line in fig.axes[1].lines) == complete
    assert all(row["model_frontier_eligible"] == complete for row in rows)
    if not complete:
        assert any(line.get_marker() == ">" for line in fig.axes[1].lines)
        assert "frontier is withheld" in caption


@pytest.mark.parametrize("mismatch", [False, True])
def test_risk_cost_join_requires_same_checkpoint_bytes(tmp_path, mismatch):
    checkpoint = identity("A", 39)
    risk = dict(gt_count=2, matches=1, misses=1, short_count=1, short_matches=1,
                capped_start_sum_normalized=1., capped_end_sum_normalized=1.)
    metrics = dict(selected_checkpoint_epoch=39, checkpoint_identity=checkpoint,
                   duration_slices=dict(strata={"short_q1": dict(class_aware_recall=.5)}),
                   risk=dict(per_video={"video": risk}))
    receipt = dict(source_train_id="A", is_final_checkpoint=True, full_split=True,
                   selected_checkpoint_epoch=39, model_source_commit="fixture", checkpoint_identity=dict(checkpoint))
    if mismatch:
        receipt["checkpoint_identity"]["checkpoint_sha256"] = "different-weights-same-epoch"
    exports = [(tmp_path, receipt, [dict(video_id="video", model_macs_counted=1e9)])]
    evaluations = [(tmp_path, dict(source_train_id="A", seed=0), dict(route="A"), dict(source_commit="fixture"), metrics)]
    report = Capture()
    if mismatch:
        with pytest.raises(ValueError, match="checkpoint identity"):
            plot_risks(report, evaluations, exports)
        assert "cost_risk_A" not in report.figures
    else:
        plot_risks(report, evaluations, exports)
        assert report.figures["cost_risk_A"][2][0]["recall"] == .5


def test_training_plot_does_not_count_replayed_steps_twice(tmp_path):
    rows = [dict(epoch=0, step=step, losses=dict(cls_loss=loss, reg_loss=1.), seconds=2., successful_update=ok)
            for step, loss, ok in [(0, 9., True), (1, 8., False), (0, 7., True), (1, 6., True)]]
    (tmp_path / "train.log").write_text("\n".join(map(json.dumps, rows)))
    report = Capture()
    plot_training(report, {"A": (tmp_path, dict(kind="train", job_id="A", route="A", seed=0), {})})
    _, caption, output = report.figures["training_A"]
    assert [r["cls_loss"] for r in output] == [7., 6.]
    assert [r["update_attempt"] for r in output] == [1, 2]
    assert "2 superseded epoch/step rows excluded" in caption
