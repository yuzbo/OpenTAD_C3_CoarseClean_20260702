"""Copy completed validation receipts without running inference or training.

Example: python collect_audit_validation_receipts.py --n16 tr-0afe8e4fc09b:25
Remote predictions remain in their original run directories.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path

from deploy_corrected import ssh

HERE = Path(__file__).resolve().parent
REMOTE = r'''
from pathlib import Path
import json
root=Path(ROOT);rows=[]
for jid,epoch in REQUESTS:
 run=root/'runs'/jid;metric=run/'intermediate_eval'/('epoch_%03d'%epoch)/'metrics.json'
 prediction=metric.parent/'predictions.json';best=run/'best.json'
 rows.append(dict(job_id=jid,completed_epochs=epoch,
  validation=json.loads(metric.read_text()) if metric.is_file() else None,
  best=json.loads(best.read_text()) if best.is_file() else None,
  source=json.loads((run/'source_commits.json').read_text()),
  raw_predictions=dict(path=str(prediction),exists=prediction.is_file(),bytes=prediction.stat().st_size if prediction.is_file() else None)))
print(json.dumps(rows))
'''


def collect(cluster, target, requests, delivery, canonical):
    binding = json.loads((delivery / ('bindings.primary.' + cluster + '.json')).read_text())
    requests = [(text.split(':')[0], int(text.split(':')[1])) for text in requests]
    if any(jid not in binding['assigned_training_ids'] for jid, _ in requests):
        raise ValueError('Requested run is not assigned to this primary cluster')
    code = REMOTE.replace('ROOT', repr(binding['work_root'])).replace('REQUESTS', repr(requests))
    rows = json.loads(ssh(target, 'python3 -', code.encode()))
    for row in rows:
        validation = row['validation']
        if not validation or validation['status'] != 'completed_validation':
            print(json.dumps(dict(job_id=row['job_id'], completed_epochs=row['completed_epochs'],
                                  status='NOT_COMPLETED', receipt=validation)), flush=True)
            continue
        assert validation['is_mock'] is False and validation['weights'] == 'ema'
        assert validation['source_train_id'] == row['job_id']
        assert validation['completed_epochs'] == row['completed_epochs']
        assert validation['videos'] == canonical and validation['windows'] == 792
        assert validation['provenance'] == row['source']
        assert row['source']['source_commit'] == binding['source_commit']
        assert row['raw_predictions']['bytes'] > 0
        best = row['best']
        if best:
            assert best['provenance'] == row['source'] and best['source_train_id'] == row['job_id']
            if best['completed_epochs'] == row['completed_epochs']:
                assert best['checkpoint_epoch'] == row['completed_epochs'] - 1
                assert best['score'] == validation['metrics']['average_mAP']
        row['observed_at_utc'] = datetime.now(timezone.utc).isoformat()
        out = delivery / 'observed_metrics' / row['job_id']
        out.mkdir(exist_ok=True)
        (out / ('epoch_%03d.receipt.json' % row['completed_epochs'])).write_text(
            json.dumps(row, indent=2), encoding='utf-8')
        print(json.dumps(dict(job_id=row['job_id'], completed_epochs=row['completed_epochs'],
                              videos=len(validation['videos']), windows=validation['windows'],
                              metrics=validation['metrics'], seconds=validation['seconds'],
                              raw_predictions=row['raw_predictions'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n16', nargs='+', default=[])
    parser.add_argument('--a100', nargs='+', default=[])
    args = parser.parse_args()
    latest = json.loads((HERE / 'audit_deployment.latest.json').read_text())
    delivery = Path(latest['local_delivery'])
    reference = delivery / 'observed_metrics/tr-0afe8e4fc09b/epoch_010.receipt.json'
    canonical = json.loads(reference.read_text(encoding='utf-8'))['validation']['videos']
    tasks = [(cluster, target, requests) for cluster, target, requests in
             [('N16R4', 'source', args.n16), ('A100', 'destination', args.a100)] if requests]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(collect, cluster, target, requests, delivery, canonical)
                   for cluster, target, requests in tasks]
        for future in futures:
            future.result()


if __name__ == '__main__':
    main()
