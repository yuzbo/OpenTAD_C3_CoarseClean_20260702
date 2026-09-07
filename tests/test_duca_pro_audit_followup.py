from tools.bata import update_experiment_catalog as catalog
from tools.bata.inspect_official_adatad_history import differences


def test_official_recipe_comparison_distinguishes_output_path_and_training_changes():
    official = dict(work_dir="official", scheduler=dict(max_epoch=100),
                    model=dict(backbone=dict(total_frames=768)))
    recorded = dict(work_dir="run", scheduler=dict(max_epoch=100),
                    model=dict(backbone=dict(total_frames=768)))
    assert differences(official, recorded) == [
        dict(field="work_dir", official="official", recorded="run")]
    recorded["scheduler"]["max_epoch"] = 60
    recorded["model"]["backbone"]["total_frames"] = 16
    assert {row["field"] for row in differences(official, recorded)} == {
        "work_dir", "scheduler.max_epoch", "model.backbone.total_frames"}


def test_repair_identity_is_not_reassigned_to_historical_results(monkeypatch):
    monkeypatch.setattr(catalog, "git_head", lambda path: "local-head")
    monkeypatch.setattr(catalog, "clean_tree", lambda path: True)
    monkeypatch.setattr(catalog, "remote_receipt", lambda: dict(status="NOT_QUERIED"))
    payload = catalog.catalog()
    entries = {row["internal_id"]: row for row in payload["entries"]}
    for route in ("H65_PRO_ACTIVE", "CT_DP_BAMOD_ACTIVE", "EVIDENCE_ACTIVE"):
        entry = entries[route]
        repair = entry["latest_repair"]
        assert repair["sha"] not in entry["sha"]
        assert len(repair["sha"]) == 40
        assert repair["branch"].startswith("codex/")
    assert "f2068e18" in entries["H65_PRO_ACTIVE"]["sha"]
    assert "64.2265" in entries["H65_PRO_ACTIVE"]["final_result"]
    rendered = catalog.md_text(payload)
    for route in ("H65_PRO_ACTIVE", "CT_DP_BAMOD_ACTIVE", "EVIDENCE_ACTIVE"):
        assert f"/commit/{entries[route]['latest_repair']['sha']}" in rendered
    assert payload["pro_audit_followup"] in rendered
    assert "不能追溯套用6000成功更新规则" in payload["result_policy"]
