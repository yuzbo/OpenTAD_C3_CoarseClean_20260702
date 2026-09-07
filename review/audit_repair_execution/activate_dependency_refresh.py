"""Refresh only supplementary coordinators; leave all training processes intact."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path, PurePosixPath
from deploy_corrected import ssh, put

HERE = Path(__file__).resolve().parent
REVISION = 'backfill-1h-primary-resume-deps-v2'
CODE = r'''
from pathlib import Path
import json,os,signal,socket,subprocess,time
name=BINDING;ctl=Path(name).parent;b=json.loads(Path(name).read_text())
assert b['primary_binding_paths'], 'only supplementary coordinators may be restarted here'
receipt=ctl/'coordinator.json';co=json.loads(receipt.read_text())
rows=[]
if co['host']==socket.gethostname():
 pid=co['pid'];proc=Path('/proc')/str(pid)
 alive=False
 if proc.exists():
  cmd=(proc/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
  assert proc.stat().st_uid==os.getuid() and str(ctl/'audit_queue.py') in cmd and name in cmd,cmd
  alive=True
 if alive and co.get('supervisor_revision')==REVISION:
  rows.append(dict(binding=name,status='ALREADY_ACTIVE',coordinator=co))
 else:
  subprocess.run(['squeue','-h','--me','--states=PENDING','-o','%i'],check=True,stdout=subprocess.DEVNULL)
  subprocess.run([b['entrypoint'][0],'-m','py_compile',str(ctl/'audit_queue.py')],check=True)
  if alive:
   os.kill(pid,signal.SIGTERM);time.sleep(.5)
  receipt.replace(ctl/('coordinator.before_dependency_refresh_'+str(pid)+'.json'))
  env=dict(os.environ,PYTHONPATH=b['repo_root'],PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
  log=open(ctl/'coordinator.log','ab',buffering=0)
  p=subprocess.Popen([b['entrypoint'][0],str(ctl/'audit_queue.py'),'--manifest',str(ctl/'experiments.jsonl'),'--bindings',name],cwd=b['repo_root'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  new=dict(pid=p.pid,host=socket.gethostname(),source_commit=b['source_commit'],started_at=time.time(),assigned_training_ids=b['assigned_training_ids'],supervisor_revision=REVISION)
  receipt.write_text(json.dumps(new,indent=2));time.sleep(2)
  assert p.poll() is None,(ctl/'coordinator.log').read_text()[-4000:]
  entry=dict(binding=name,status='ACTIVE',coordinator=new)
  (ctl/'dependency_refresh_amendment.json').write_text(json.dumps(entry,indent=2));rows.append(entry)
print(json.dumps(rows))
'''


def activate(cluster, target, delivery):
    binding = json.loads((delivery / ('bindings.secondary.' + cluster + '.json')).read_text())
    control = str(PurePosixPath(binding['work_root']).parent.parent / 'control')
    path = control + '/bindings.json'
    put(target, control + '/audit_queue.py', (HERE / 'audit_queue.py').read_bytes())
    for _ in range(24):
        code = CODE.replace('BINDING', repr(path)).replace('REVISION', repr(REVISION))
        rows = json.loads(ssh(target, 'python3 -', code.encode()))
        if rows:
            (HERE / ('audit_dependency_refresh.' + cluster + '.json')).write_text(json.dumps(rows, indent=2))
            return cluster, rows
    raise RuntimeError('recorded coordinator login host not reached: ' + cluster)


if __name__ == '__main__':
    delivery = Path(json.loads((HERE / 'audit_deployment.latest.json').read_text())['local_delivery'])
    with ThreadPoolExecutor(max_workers=2) as pool:
        result = dict(pool.map(lambda args: activate(*args, delivery), [('N16R4', 'source'), ('A100', 'destination')]))
    print(json.dumps(result, indent=2))
