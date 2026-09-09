import datetime as dt
import json
from types import SimpleNamespace

import pytest

from tools.bata import update_experiment_catalog as catalog


@pytest.mark.parametrize("age,status", [(20, "ACTIVE"), (4 * 3600, "STALE")])
def test_readable_old_receipt_is_not_an_active_supervisor(monkeypatch, age, status):
    checked = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=age)
    payload = dict(checked_at=checked.isoformat(), dispatcher=dict(status="BLOCKED", mode="plan"))
    monkeypatch.delenv("DUCA_CATALOG_DISABLE_REMOTE", raising=False)
    monkeypatch.setattr(catalog.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=0, stdout=json.dumps(payload), stderr=""))
    result = catalog.remote_receipt()
    assert result["status"] == status
    assert result["dispatcher_status"] == "BLOCKED"
    assert result["receipt_age_seconds"] >= age


def test_receipt_without_timestamp_cannot_claim_activity(monkeypatch):
    monkeypatch.delenv("DUCA_CATALOG_DISABLE_REMOTE", raising=False)
    monkeypatch.setattr(catalog.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=0, stdout="{}", stderr=""))
    assert catalog.remote_receipt()["status"] == "INVALID_RECEIPT"


def test_hung_receipt_query_is_bounded_and_not_reported_active(monkeypatch):
    monkeypatch.delenv("DUCA_CATALOG_DISABLE_REMOTE", raising=False)

    def timed_out(*args, **kwargs):
        assert kwargs["timeout"] == 45
        raise catalog.subprocess.TimeoutExpired(cmd=["ssh"], timeout=45)

    monkeypatch.setattr(catalog.subprocess, "run", timed_out)
    result = catalog.remote_receipt()
    assert result["status"] == "UNAVAILABLE"
    assert "45" in result["reason"]


def test_unpublished_local_repair_does_not_link_to_a_github_commit(monkeypatch):
    monkeypatch.setattr(catalog, "git_head", lambda path: "local-only")
    monkeypatch.setattr(catalog, "clean_tree", lambda path: True)
    monkeypatch.setattr(catalog, "remote_receipt", lambda: {"status": "NOT_QUERIED"})
    payload = catalog.catalog()
    entry = dict(payload["entries"][0], github_commit=None, sha="unpublished-local-sha")
    payload["entries"] = [entry]
    rendered = catalog.md_text(payload)
    assert "/commit/unpublished-local-sha" not in rendered
    assert payload["repository"] in rendered
    assert "unpublished-local-sha" in rendered


def test_user_stop_preserves_results_without_contacting_remote(monkeypatch):
    monkeypatch.setattr(catalog, "git_head", lambda path: "unchanged")
    monkeypatch.setattr(catalog, "clean_tree", lambda path: True)

    def unexpected_remote_query():
        pytest.fail("a stopped catalog must not poll the remote supervisor")

    monkeypatch.setattr(catalog, "remote_receipt", unexpected_remote_query)
    originals = {entry["internal_id"]: entry for entry in catalog.route_entries()}
    payload = catalog.catalog()
    assert payload["execution_control"]["status"] == "USER_STOPPED"
    assert payload["remote_supervisor"]["status"] == "USER_STOPPED"
    for entry in payload["entries"]:
        original = originals[entry["internal_id"]]
        assert entry["final_result"] == original["final_result"]
        assert entry["deployment_status"] == original["deployment_status"]
        if entry["category"] in {"related_task_repair", "historical_exact_route"}:
            assert entry["next_action"] == original["next_action"]
            assert "execution_status" not in entry
        else:
            assert entry["execution_status"] == "USER_STOPPED"
            assert entry["next_action_before_user_stop"] == original["next_action"]
            assert "未经" in entry["next_action"]
    assert "用户已于2026-09-09终止" in catalog.md_text(payload)


def test_cleanup_marks_old_weights_unavailable_but_keeps_scores(monkeypatch):
    monkeypatch.setattr(catalog, "git_head", lambda path: "unchanged")
    monkeypatch.setattr(catalog, "clean_tree", lambda path: True)
    originals = {entry["internal_id"]: entry for entry in catalog.route_entries()}
    payload = catalog.catalog()
    selected = [entry for entry in payload["entries"] if entry.get("best_test_checkpoint")]
    assert len(selected) == 5
    for entry in selected:
        assert entry["best_test_checkpoint"]["weight_file_available"] is False
        assert entry["best_test_checkpoint"]["average_mAP"] == originals[entry["internal_id"]]["best_test_checkpoint"]["average_mAP"]
        assert sum(row["checkpoint_weight_available"] for row in entry["periodic_results"]) == 1
        assert entry["terminal_training_evaluation"]["checkpoint_weight_available"] is True
    assert payload["checkpoint_cleanup"]["deleted_files"] == 803
    assert payload["checkpoint_cleanup"]["retained_files"] == 101
