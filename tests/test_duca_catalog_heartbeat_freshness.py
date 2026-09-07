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
