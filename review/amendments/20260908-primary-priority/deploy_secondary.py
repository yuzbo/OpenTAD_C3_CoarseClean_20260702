"""Deploy the first supplementary cohort, preserving all primary run sources."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from deploy_corrected import ssh, put, SOURCE

HERE = Path(__file__).resolve().parent
MODEL = '902fa05b5c64452ff1c94b82cabce801943d3484'
ASSIGNMENTS = {'N16R4': ['tr-f90ea0ae2ce4'], 'A100': ['tr-c2d93c4c3979', 'tr-bacdbb058eca', 'tr-6f26bfd4ff96', 'tr-067daa13036b']}

SETUP = r'''
from pathlib import Path
import fcntl,json,os,shutil,socket,subprocess,time
root=Path(ROOT);control=root/'control';b=json.loads((control/'bindings.json').read_text());repo=Path(b['repo_root'])
lock=open(control/'setup.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==b['source_commit']
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
jobs={j['job_id']:j for j in [json.loads(line) for line in (control/'experiments.jsonl').read_text().splitlines()]}
old=json.loads(Path(b['primary_binding_paths'][0]).read_text());oldroot=Path(old['work_root']);oldcontrol=oldroot.parent.parent/'control'
oldjobs={j['job_id']:j for j in [json.loads(line) for line in (oldcontrol/'experiments.jsonl').read_text().splitlines()]}
reused=[]
for jid in b['assigned_training_ids']:
 if jid not in oldjobs:continue
 assert oldjobs[jid]['model']==jobs[jid]['model'] and oldjobs[jid]['dataset']==jobs[jid]['dataset']
 assert jid not in old['assigned_training_ids'], 'primary coordinator still owns supplementary job'
 run=oldroot/'runs'/jid
 assert not (run/'progress.json').exists() and not (run/'result.json').exists() and not list(run.glob('checkpoint/*.pth')), 'do not repeat prior training'
 for source in (Path(old['capability_dir'])/jid).glob('*.json'):
  cap=json.loads(source.read_text())
  assert cap['commit']==b['source_commit'] and cap['ready'] and jid in cap['supported_variants']
  evidence=Path(cap['test_receipt']);check=json.loads(evidence.read_text())
  assert check['source_commit']==b['source_commit'] and check['routes'][jid]['status']=='PASS'
  target=Path(b['capability_dir'])/jid/source.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
  reused.append(dict(job_id=jid,capability=source.stem,test_receipt=str(evidence)))
receipt=control/'coordinator.json'
if not receipt.exists():
 env=dict(os.environ,PYTHONPATH=str(repo),PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 log=open(control/'coordinator.log','ab',buffering=0)
 p=subprocess.Popen([b['entrypoint'][0],str(control/'secondary_queue.py'),'--manifest',str(control/'experiments.jsonl'),'--bindings',str(control/'bindings.json')],cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 receipt.write_text(json.dumps(dict(pid=p.pid,host=socket.gethostname(),source_commit=b['source_commit'],started_at=time.time(),assigned_training_ids=b['assigned_training_ids']),indent=2))
 time.sleep(3)
 assert p.poll() is None,(control/'coordinator.log').read_text()[-3500:]
print(json.dumps(dict(cluster=b['cluster'],coordinator=json.loads(receipt.read_text()),reused_capabilities=reused,work_root=b['work_root'])))
'''


def main():
    assert subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() == MODEL
    assert not subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain'], text=True).strip()
    registered = [json.loads(line) for line in (HERE / 'experiments.corrected.full.jsonl').read_text().splitlines()]
    selected = {jid for ids in ASSIGNMENTS.values() for jid in ids}
    jobs = [row for row in registered if row['seed'] == 0 and row['kind'] in {'train', 'evaluate', 'benchmark'} and row.get('source_train_id', row['job_id']) in selected]
    assert len(jobs) == 15 and len({j['job_id'] for j in jobs}) == 15
    for row in jobs:
        row['execution_phase'] = 'secondary-after-main-deployment-seed0'
        row['protocol_amendment'] = 'official-full-data-20260908'
        if row['kind'] == 'train':
            assert row['epochs'] == 60 and row['model']['source_resolution'] == 160 and row['model']['query_length'] == 768
            row['checkpoints'] = list(range(4, 60, 5))
            row['training_validation'] = dict(subset='validation', interval_epochs=5, checkpoint_selection='best_average_mAP', weights='ema', tie_break='earlier_checkpoint')
        if row['kind'] == 'evaluate':
            row.pop('checkpoint_epoch', None)
            row['checkpoint_selection'] = 'best_full_validation_average_mAP'
    manifest = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in jobs).encode()
    path = HERE / 'experiments.secondary.first_wave.jsonl'
    if path.exists():
        assert path.read_bytes() == manifest
    else:
        path.write_bytes(manifest)
    def deploy(cluster, target):
        primary = [json.loads((HERE / ('bindings.' + phase + '.' + cluster + '.json')).read_text()) for phase in ['corrected', 'dynamic']]
        assert not selected.intersection({jid for b in primary for jid in b['assigned_training_ids']})
        b = dict(primary[0])
        root = Path(b['work_root']).parent.parent.parent.as_posix() + '/secondary_seed0_' + MODEL[:8]
        b.update(work_root=root + '/runs/' + MODEL[:8], protocol_root=root + '/protocol', capability_dir=root + '/capabilities/' + MODEL[:8],
                 assigned_training_ids=ASSIGNMENTS[cluster], priority_training_ids=ASSIGNMENTS[cluster], max_concurrent_jobs=1 if cluster == 'N16R4' else 2,
                 execution_phase='secondary-after-main-deployment-seed0',
                 primary_binding_paths=[Path(p['work_root']).parent.parent.as_posix() + '/control/bindings.json' for p in primary])
        local = HERE / ('bindings.secondary.' + cluster + '.json')
        if local.exists():
            assert json.loads(local.read_text()) == b
        else:
            local.write_text(json.dumps(b, indent=2), encoding='utf-8')
        put(target, root + '/control/bindings.json', local.read_bytes())
        put(target, root + '/control/experiments.jsonl', manifest)
        put(target, root + '/control/secondary_queue.py', (HERE / 'secondary_queue.py').read_bytes())
        return cluster, json.loads(ssh(target, 'python3 -', SETUP.replace('ROOT', repr(root)).encode()))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = dict(pool.map(lambda item: deploy(*item), [('N16R4', 'source'), ('A100', 'destination')]))
    results['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
    (HERE / 'secondary_deployment.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
