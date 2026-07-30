import importlib.util
import os
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "tools" / "bata" / "audit_actionformer_s0_assignment_support.py"
)
SPEC = importlib.util.spec_from_file_location("s0_assignment_support", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def points():
    return torch.tensor(
        [
            [0.5, 0.0, 4.0, 1.0],
            [1.5, 0.0, 4.0, 1.0],
            [2.5, 0.0, 4.0, 1.0],
            [3.5, 0.0, 4.0, 1.0],
        ]
    )


def test_assignment_matrices_match_center_radius_and_shortest_gt():
    result = audit.build_assignment_matrices(
        points(),
        torch.tensor([[0.0, 4.0], [1.0, 2.0]]),
        "none",
        1.5,
    )
    assert result["candidate"].shape == (4, 2)
    assert result["candidate"][:, 0].any()
    assert result["candidate"][:, 1].any()
    assert result["assigned_gt"].tolist() == [0, 1, 0, 0]


def test_analyze_sample_exposes_selected_candidate_and_assignment_loss():
    level_points = [points()]
    dense = [torch.ones(1, 1, 4, dtype=torch.bool)]
    selected = [
        torch.tensor([[[True, False, False, True]]], dtype=torch.bool)
    ]
    gt_segments = torch.tensor([[1.0, 2.0], [2.0, 3.0]])
    gt_labels = torch.tensor([0, 1])
    row = audit.analyze_sample(
        points=level_points,
        dense_masks=dense,
        selected_masks=selected,
        sample_index=0,
        gt_segments=gt_segments,
        gt_labels=gt_labels,
        sample_meta={
            "video_id": "synthetic",
            "feats": torch.zeros(4, 4),
            "feat_stride": 4,
            "fps": 4.0,
        },
        center_sample="radius",
        center_radius=1.5,
    )
    assert row["dense_valid_query_count"] == 4
    assert row["selected_query_count"] == 2
    assert row["dense_positive_count"] == 2
    assert row["selected_positive_count"] == 0
    assert all(gt["dense_candidate_count"] > 0 for gt in row["per_gt"])
    assert all(gt["selected_candidate_count"] == 0 for gt in row["per_gt"])


def test_aggregate_rows_separates_duration_and_fpn_level():
    row = {
        "dense_valid_query_count": 4,
        "selected_query_count": 2,
        "dense_positive_count": 2,
        "selected_positive_count": 1,
        "per_level": [
            {
                "level": 0,
                "valid_query_count": 4,
                "selected_query_count": 2,
                "dense_positive_count": 2,
                "selected_positive_count": 1,
                "max_selected_center_gap_feature_grid": 3.0,
            }
        ],
        "per_gt": [
            {
                "duration_bucket": "1_2s",
                "dense_candidate_count": 2,
                "selected_candidate_count": 1,
                "dense_assignment_count": 1,
                "selected_assignment_count": 1,
            },
            {
                "duration_bucket": "4_8s",
                "dense_candidate_count": 2,
                "selected_candidate_count": 0,
                "dense_assignment_count": 1,
                "selected_assignment_count": 0,
            },
        ],
    }
    summary = audit.aggregate_rows([row])
    assert summary["selected_positive_retention"] == pytest.approx(0.5)
    assert summary["selected_gt_without_candidate"] == 1
    assert summary["duration_buckets"]["4_8s"]["selected_gt_without_assignment"] == 1
    assert summary["per_level"]["0"]["selected_positive_retention"] == pytest.approx(
        0.5
    )


def test_source_forbids_loss_backward_optimizer_and_test_gt():
    source = MODULE_PATH.read_text(encoding="utf-8")
    for token in (
        ".backward(",
        "optimizer.step(",
        "make_optimizer(",
        "train_one_epoch(",
        "val_split",
    ):
        assert token not in source
    assert '"test_gt_used": False' in source
    assert '"paper_main_table_eligible": False' in source
    assert '"primary_result_allowed": False' in source


def test_atomic_write_refuses_overwrite(tmp_path):
    path = tmp_path / "artifact.json"
    audit.atomic_write_json(path, {"value": 1})
    with pytest.raises(audit.AuditError, match="output already exists"):
        audit.atomic_write_json(path, {"value": 2})


def test_formal_main_requires_slurm(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        audit,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {"seed": 1234567891, "num_windows": 64},
        )(),
    )
    with pytest.raises(audit.AuditError, match="Slurm CUDA allocation"):
        audit.main()
