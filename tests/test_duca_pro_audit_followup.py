from tools.bata import update_experiment_catalog as catalog
from tools.bata.inspect_official_adatad_history import differences
import json


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


def test_terminal_observation_preserves_metric_scale_and_conditional_steps():
    evidence = json.loads((catalog.AUDIT / "16_HEARTBEAT_TERMINAL_EVIDENCE_20260908_0424.json").read_text(encoding="utf-8"))
    a1 = next(row for row in evidence["evidence"] if row["arm"] == "A1")
    entry = next(row for row in catalog.route_entries() if row["internal_id"] == "EVIDENCE_ACTIVE")
    assert f"{100 * a1['metrics']['average_mAP']:.4f}%" in entry["final_result"]
    assert entry["observed_successful_updates"]["A1"] == a1["update_audit"]["successful_optimizer_updates"] == 6000
    assert entry["conditional_time_parameter_updates"]["A1"] == 794
    assert a1["optimizer_groups"][1]["step_histogram"] == {"794.0": 4}
    assert "不作完整机制验收" in entry["result_status"]


def test_ct_terminal_metrics_keep_training_identity_and_fraction_scale():
    evidence = json.loads((catalog.AUDIT / "17_HEARTBEAT_EVIDENCE_20260908_0634.json").read_text(encoding="utf-8"))
    observed = evidence["ct_g2_g3_official_results"]
    entry = next(row for row in catalog.route_entries() if row["internal_id"] == "CT_DP_BAMOD_ACTIVE")
    result = entry["official_terminal_results"]
    assert result["training_sha"] == observed["training_sha"]
    assert result["evaluator_sha"] == observed["evaluator_sha"]
    assert result["training_sha"] != entry["latest_repair"]["sha"]
    assert result["metric_scale"] == "fraction_0_to_1"
    assert result["successful_optimizer_updates"] == result["scheduler_updates"] == result["ema_updates"] == 6000
    for arm in ("G2", "G3"):
        assert result[arm]["average_mAP"] == observed[arm]["metrics"]["average_mAP"]
        assert 0 < result[arm]["average_mAP"] < 1
        assert f"{100 * result[arm]['average_mAP']:.4f}%" in entry["final_result"]
        assert result[arm]["receipt_sha256"] == observed[arm]["receipt_sha256"]
        assert "COMPLETED" in entry["new_evaluation_states"][arm]
    assert evidence["actions"]["new_slurm_jobs"] == []


def test_coordinate_evaluator_does_not_reassign_old_terminal_results(monkeypatch):
    monkeypatch.setattr(catalog, "git_head", lambda path: "local-head")
    monkeypatch.setattr(catalog, "clean_tree", lambda path: True)
    monkeypatch.setattr(catalog, "remote_receipt", lambda: dict(status="NOT_QUERIED"))
    payload = catalog.catalog()
    entry = next(row for row in payload["entries"] if row["internal_id"] == "CT_DP_BAMOD_ACTIVE")
    evaluator = entry["coordinate_evaluator"]
    assert evaluator["training_sha"] == entry["latest_repair"]["sha"]
    assert evaluator["training_sha"] != entry["official_terminal_results"]["training_sha"]
    assert evaluator["sha"] != entry["official_terminal_results"]["evaluator_sha"]
    assert evaluator["precheck_launch"]["PRECHECK_ONLY"] == "1"
    assert evaluator["precheck_launch"]["CTDP_TRAIN_ROOT"].endswith("ctdp_coordinate_fe1c53db/precheck")
    assert evaluator["github_commit"] in catalog.md_text(payload)
