"""Replay a real gateway failure in an isolated monitoring fixture; no SSH."""
import contextlib
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shutil
import subprocess
from unittest.mock import patch

import refresh_audit_status as refresh
import export_audit_disposition as disposition


def main():
    source = Path(__file__).resolve().parent
    original = json.loads((source / 'audit_status.json').read_text(encoding='utf-8'))
    latest = json.loads((source / 'audit_deployment.latest.json').read_text(encoding='utf-8'))
    original_delivery = Path(latest['local_delivery'])
    fixture = source / ('monitor_replay_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    fixture.mkdir()
    delivery = fixture / 'delivery'
    delivery.mkdir()
    for name in ['old_to_repaired_job_ids.json'] + [f'bindings.{phase}.{cluster}.json'
            for phase in ('primary', 'secondary') for cluster in ('N16R4', 'A100')]:
        shutil.copy2(original_delivery / name, delivery / name)
    latest['local_delivery'] = str(delivery)
    (fixture / 'audit_deployment.latest.json').write_text(json.dumps(latest), encoding='utf-8')
    (fixture / 'audit_status.json').write_text(json.dumps(original), encoding='utf-8')
    (fixture / 'IS_MOCK_MONITOR_TEST.json').write_text(json.dumps({'is_mock': True,
        'purpose': 'monitoring failure/recovery replay; no scientific results'}), encoding='utf-8')
    refresh.HERE = disposition.HERE = fixture
    refresh.VIS = fixture / 'visualizations'
    refresh.VIS.mkdir()
    mode = 'failure'
    calls = []

    def fake_run(args, *, input, capture_output, timeout):
        assert capture_output and timeout == 120
        cluster = 'N16R4' if any('BSCC-N16R4' in str(x) for x in args) else 'A100'
        assert cluster == 'N16R4' or any('GUANGZHOUXY-A100' in str(x) for x in args)
        phase = 'secondary' if b'audit_repair_b70ae056_secondary/' in input else 'primary'
        key = phase + '.' + cluster
        calls.append(key)
        if cluster == 'N16R4' and mode == 'failure':
            return subprocess.CompletedProcess(args, 1, stdout=b'', stderr=b'channel 0: open failed: connect failed')
        if cluster == 'N16R4' and mode == 'timeout':
            raise subprocess.TimeoutExpired(args, timeout)
        return subprocess.CompletedProcess(args, 0,
            stdout=json.dumps(original['states'][key]).encode(), stderr=b'')

    def execute():
        calls.clear()
        output = io.StringIO()
        with patch.object(refresh.subprocess, 'run', side_effect=fake_run), contextlib.redirect_stdout(output):
            refresh.main()
        assert sorted(calls) == sorted(original['states'])
        return json.loads(output.getvalue()), json.loads((fixture / 'audit_status.json').read_text(encoding='utf-8'))

    rows, first = execute()
    assert set(first['query_errors']) == {'primary.N16R4', 'secondary.N16R4'}
    n16 = [r for r in rows['rows'] if r[0] == 'N16R4']
    a100 = [r for r in rows['rows'] if r[0] == 'A100']
    assert n16 and all(r[3] == 'OBSERVATION_UNAVAILABLE' for r in n16)
    assert a100 and all(r[3] != 'OBSERVATION_UNAVAILABLE' for r in a100)
    for key in first['query_errors']:
        assert first['states'][key]['observed_at_utc'] == original['states'][key].get('observed_at_utc', original['observed_at_utc'])
    with contextlib.redirect_stdout(io.StringIO()):
        disposition.main()
    with (delivery / 'job_disposition_1545.csv').open(encoding='utf-8-sig', newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 1545
    assert len([r for r in csv_rows if r['kind'] == 'train' and r['seed'] == '0']) == 204
    observed_n16 = [r for r in csv_rows if r['cluster'] == 'N16R4' and not r['blocking_reasons']]
    assert observed_n16 and all(r['execution_status'] == 'OBSERVATION_UNAVAILABLE' and r['observation_error'] for r in observed_n16)
    assert all(r['execution_status'] != 'OBSERVATION_UNAVAILABLE' for r in csv_rows if r['cluster'] == 'A100')

    mode = 'timeout'
    _, repeated = execute()
    for key in first['query_errors']:
        assert repeated['states'][key]['observed_at_utc'] == first['states'][key]['observed_at_utc']
        assert repeated['states'][key]['query_attempt_at_utc'] > first['states'][key]['query_attempt_at_utc']

    mode = 'recovery'
    recovered_rows, recovered = execute()
    assert not recovered['query_errors']
    assert all(r[3] != 'OBSERVATION_UNAVAILABLE' for r in recovered_rows['rows'])
    assert all(not h.get('observation_error') for h in recovered['states'].values())
    for key in first['query_errors']:
        assert recovered['states'][key]['observed_at_utc'] > repeated['states'][key]['observed_at_utc']
    with contextlib.redirect_stdout(io.StringIO()):
        disposition.main()
    with (delivery / 'job_disposition_1545.csv').open(encoding='utf-8-sig', newline='') as stream:
        assert all(r['execution_status'] != 'OBSERVATION_UNAVAILABLE' and not r['observation_error'] for r in csv.DictReader(stream))
    print(json.dumps({'status': 'PASS', 'is_mock': True, 'cases': [
        'partial failure preserves successful host and last observation',
        'CSV retains registry counts and marks affected observations unavailable',
        'repeated timeout does not advance successful observation timestamp',
        'recovery clears stale errors in dashboard and CSV'], 'fixture': str(fixture)}, indent=2))


if __name__ == '__main__':
    main()
