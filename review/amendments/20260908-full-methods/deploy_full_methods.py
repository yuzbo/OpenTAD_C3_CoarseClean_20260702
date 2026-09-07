"""User phase: keep A/B/C fixed .5; add their full dynamic methods; defer ablations."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from deploy_corrected import ssh, put, SETUP, SOURCE

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'deployment_a100'))
from transfer_assets import command
MODEL = '902fa05b5c64452ff1c94b82cabce801943d3484'
DYNAMIC = {'N16R4': ['tr-32754eef72d1'], 'A100': ['tr-8a922c4e6ca7', 'tr-72ba800e6f08']}
FIXED = {'N16R4': ['tr-c43d3e9cad58'], 'A100': ['tr-be89cd6cb2ca', 'tr-a45686a40afa']}

SHRINK = r'''
from pathlib import Path
import fcntl,json,os,signal,socket,subprocess,time
root=Path(ROOT);control=root/'control'
operation=control/'full_method_phase.json'
if operation.exists():
 print(operation.read_text());raise SystemExit(0)
receipt=control/'coordinator.json'
row=json.loads(receipt.read_text())
if socket.gethostname()!=row['host']:raise SystemExit(93)
lock=open(control/'phase_change.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
proc=Path('/proc')/str(row['pid'])/'cmdline'
cmd=proc.read_bytes().replace(b'\x00',b' ').decode() if proc.exists() else ''
if cmd:
 assert 'geosparse_ext.slurm_queue' in cmd and str(control/'bindings.json') in cmd,cmd
 assert (Path('/proc')/str(row['pid'])).stat().st_uid==os.getuid()
 os.kill(row['pid'],signal.SIGTERM)
 for _ in range(50):
  if not proc.exists():break
  time.sleep(.1)
 else:raise RuntimeError('Owned coordinator did not exit')
row.update(stopped_at=time.time(),reason='User defers extra controls; fixed .5 A/B/C remain active')
archive=control/('coordinator.before_full_methods.'+str(row['pid'])+'.json')
archive.write_text(json.dumps(row,indent=2));receipt.unlink()
binding_path=control/'bindings.json';b=json.loads(binding_path.read_text())
assert b['source_commit']=='902fa05b5c64452ff1c94b82cabce801943d3484' and b['cluster']=='A100'
state_path=Path(b['work_root'])/'slurm_state.json';state=json.loads(state_path.read_text())
assert not any(x.get('status')=='SUBMITTING' for x in state['jobs'].values())
(control/'bindings.before_full_methods.json').write_bytes(binding_path.read_bytes())
(control/'state.before_full_methods.json').write_bytes(state_path.read_bytes())
deferred=[]
for jid in ['tr-c2d93c4c3979','tr-bacdbb058eca']:
 item=state['jobs'][jid];sid=item.get('slurm_id')
 current=subprocess.run(['squeue','-h','-j',str(sid),'-o','%i|%j|%T'],capture_output=True,text=True)
 lines=[x.split('|') for x in current.stdout.splitlines() if x.startswith(str(sid)+'|')]
 if lines:
  assert lines[0][1]==jid,(jid,lines)
  subprocess.run(['scancel',str(sid)],check=True)
 run=Path(b['work_root'])/'runs'/jid
 progress=json.loads((run/'progress.json').read_text()) if (run/'progress.json').exists() else None
 details=dict(job_id=jid,slurm_id=sid,observed_slurm=lines,progress=progress,
   reason='DEFERRED_USER_PHASE: complete fixed .5 and dynamic A/B/C methods first; not a scientific failure',deferred_at=time.time())
 deferred.append(details)
 item.update(status='DEFERRED_USER_PHASE',reason=details['reason'])
 if item.get('attempts'):item['attempts'][-1].update(status='DEFERRED_USER_PHASE',finished_at=time.time(),reason=details['reason'])
b.update(assigned_training_ids=['tr-be89cd6cb2ca','tr-a45686a40afa'],priority_training_ids=['tr-be89cd6cb2ca','tr-a45686a40afa'],max_concurrent_jobs=2,execution_phase='full-methods-first-fixed50-plus-dynamic')
binding_path.write_text(json.dumps(b,indent=2));state_path.write_text(json.dumps(state,indent=2))
env=dict(os.environ,PYTHONPATH=b['repo_root'],PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
log=open(control/'coordinator.log','ab',buffering=0)
p=subprocess.Popen([b['entrypoint'][0],'-m','geosparse_ext.slurm_queue','--manifest',str(control/'experiments.jsonl'),'--bindings',str(binding_path),'--execute','--watch'],cwd=b['repo_root'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
current=dict(pid=p.pid,host=socket.gethostname(),source_commit=b['source_commit'],started_at=time.time(),assigned_training_ids=b['assigned_training_ids'])
receipt.write_text(json.dumps(current,indent=2));time.sleep(3)
assert p.poll() is None,(control/'coordinator.log').read_text()[-2500:]
result=dict(deferred=deferred,coordinator=current,assigned_training_ids=b['assigned_training_ids'],official_jobs_untouched=True)
operation.write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''


def shrink_controls():
    local = HERE / 'bindings.corrected.A100.json'
    b = json.loads(local.read_text())
    root = Path(b['work_root']).parent.parent.as_posix()
    for attempt in range(24):
        r = subprocess.run(command('destination', 'python3 -'), input=SHRINK.replace('ROOT', repr(root)).encode(), capture_output=True)
        if r.returncode == 93:
            continue
        if r.returncode:
            raise RuntimeError(r.stderr.decode(errors='replace') + r.stdout.decode(errors='replace'))
        result = json.loads(r.stdout)
        if not (HERE / 'bindings.fixed50.before_full_methods.A100.json').exists():
            (HERE / 'bindings.fixed50.before_full_methods.A100.json').write_bytes(local.read_bytes())
        b.update(assigned_training_ids=FIXED['A100'], priority_training_ids=FIXED['A100'], max_concurrent_jobs=2,
                 execution_phase='full-methods-first-fixed50-plus-dynamic')
        local.write_text(json.dumps(b, indent=2), encoding='utf-8')
        return result
    raise RuntimeError('Could not reach owned coordinator host; no unrelated process stopped')


def main():
    commit = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == MODEL and not subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain'], text=True).strip()
    record = HERE / 'full_methods_phase_operations.json'
    operations = json.loads(record.read_text()) if record.exists() else {}
    if 'controls_deferred' not in operations:
        operations['controls_deferred'] = shrink_controls()
        record.write_text(json.dumps(operations, indent=2), encoding='utf-8')
    spec = importlib.util.spec_from_file_location('matrix', SOURCE / 'geosparse_ext/matrix.py')
    matrix = importlib.util.module_from_spec(spec);spec.loader.exec_module(matrix)
    all_jobs, _ = matrix.compile_all()
    ids = {jid for values in DYNAMIC.values() for jid in values}
    jobs = [j for j in all_jobs if j['seed'] == 0 and j['kind'] in {'train','evaluate','benchmark'} and j.get('source_train_id',j['job_id']) in ids]
    assert len(jobs) == 9
    for j in jobs:
        j['execution_phase'] = 'full-methods-first-fixed50-plus-dynamic'
        if j['kind'] == 'train':
            assert j['model']['budget'] == .5 and j['model']['budget_mode'] == 'dynamic' and j['model']['estimator'] == 'pg_acquisition'
            j['checkpoints'] = list(range(4,60,5))
            j['training_validation'] = dict(subset='validation',interval_epochs=5,checkpoint_selection='best_average_mAP',weights='ema',tie_break='earlier_checkpoint')
        if j['kind'] == 'evaluate':
            j.pop('checkpoint_epoch',None)
            j['checkpoint_selection'] = 'best_full_validation_average_mAP'
    matrix.validate(jobs)
    manifest = ''.join(json.dumps(j,ensure_ascii=False,sort_keys=True)+'\n' for j in jobs).encode()
    (HERE/'experiments.full_methods.dynamic.jsonl').write_bytes(manifest)
    # Run the extra real-batch budget-head/utility witness before the existing
    # exact-config native GPU certification; a failure cannot publish readiness.
    fragment="'exec '+shlex.join(argv)"
    assert SETUP.count(fragment)==1
    dynamic_setup=SETUP.replace(fragment,"shlex.join([py,str(control/'full_method_witness.py'),'--manifest',str(control/'experiments.jsonl'),'--bindings',str(control/'bindings.json'),'--job-id',jid,'--output',str(out/'dynamic_policy_check.json')]),'exec '+shlex.join(argv)")
    def deploy(host, target):
        b=json.loads((HERE/('bindings.corrected.'+host+'.json')).read_text())
        root=Path(b['work_root']).parent.parent.parent.as_posix()+'/dynamic_full_'+commit[:8]
        b.update(work_root=root+'/runs/'+commit[:8],protocol_root=root+'/protocol',capability_dir=root+'/capabilities/'+commit[:8],
                 active_seeds=[0],assigned_training_ids=DYNAMIC[host],priority_training_ids=DYNAMIC[host],max_concurrent_jobs=len(DYNAMIC[host]),
                 execution_phase='full-methods-first-fixed50-plus-dynamic')
        local=HERE/('bindings.dynamic.'+host+'.json')
        if local.exists():assert json.loads(local.read_text())==b
        else:local.write_text(json.dumps(b,indent=2),encoding='utf-8')
        put(target,root+'/control/bindings.json',local.read_bytes())
        put(target,root+'/control/experiments.jsonl',manifest)
        put(target,root+'/control/full_method_witness.py',(HERE/'full_method_witness.py').read_bytes())
        out=ssh(target,'python3 -',dynamic_setup.replace('ROOT',repr(root)).encode()).decode()
        result=json.loads(out.strip().splitlines()[-1])
        return host,result
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda item:deploy(*item),[('N16R4','source'),('A100','destination')]))
    operations.update(dynamic=dict(results),fixed_assignments=FIXED,dynamic_assignments=DYNAMIC,updated_at_utc=datetime.now(timezone.utc).isoformat())
    record.write_text(json.dumps(operations,indent=2),encoding='utf-8')
    print(json.dumps(operations,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
