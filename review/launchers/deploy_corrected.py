"""Deploy a unique seed0 decision cohort on the two already-authorized clusters."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SOURCE = Path('E:/DeskTop/TAD/GeoSparse_Official_20260908')
sys.path.insert(0, str(HERE.parent / 'deployment_a100'))
from transfer_assets import command


def ssh(target, script, data=None):
    result = subprocess.run(command(target, script), input=data, capture_output=True)
    if result.returncode:
        raise RuntimeError(target + ': ' + result.stderr.decode(errors='replace') + result.stdout.decode(errors='replace'))
    return result.stdout


def put(target, path, content):
    code = 'from pathlib import Path; import sys; p=Path(' + repr(path) + '); p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+".new"); t.write_bytes(sys.stdin.buffer.read()); t.replace(p)'
    ssh(target, 'python3 -c ' + shlex.quote(code), content)


SETUP = r'''
from pathlib import Path
import fcntl,json,os,shlex,socket,subprocess,time
root=Path(ROOT)
control=root/'control'
lock=open(control/'setup.lock','a+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
b=json.loads((control/'bindings.json').read_text())
repo=Path(b['repo_root']); py=b['entrypoint'][0]
repo.parent.mkdir(parents=True,exist_ok=True)
if not (repo/'.git').exists():
 subprocess.run(['git','clone','-q',str(control/'source.bundle'),str(repo)],check=True)
 subprocess.run(['git','-C',str(repo),'checkout','--detach','-q',b['source_commit']],check=True)
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==b['source_commit']
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
subprocess.run(['git','-C',str(repo),'diff','--exit-code','346d09d19e2091372cec48172dbe40f7b28bdee6','--','opentad','configs','tools/train.py','tools/test.py'],check=True)
Path(b['work_root']).mkdir(parents=True,exist_ok=True)
Path(b['capability_dir']).mkdir(parents=True,exist_ok=True)
env=dict(os.environ,PYTHONPATH=str(repo),PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
receipt=control/'coordinator.json'
state=Path(b['work_root'])/'slurm_state.json'
if not receipt.exists():
 log=open(control/'coordinator.log','ab',buffering=0)
 p=subprocess.Popen([py,'-m','geosparse_ext.slurm_queue','--manifest',str(control/'experiments.jsonl'),'--bindings',str(control/'bindings.json'),'--execute','--watch'],cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 receipt.write_text(json.dumps(dict(pid=p.pid,host=socket.gethostname(),source_commit=b['source_commit'],started_at=time.time()),indent=2))
 time.sleep(3)
 assert p.poll() is None,(control/'coordinator.log').read_text()[-3000:]
nodes=[]
if b['cluster']=='N16R4':
 import re
 text=subprocess.check_output(['scontrol','show','nodes','-d'],text=True)
 for block in text.split('NodeName=')[1:]:
  match=re.search(r'GresUsed=.*?IDX:([0-9,-]+)\)',block)
  if not match or 'State=MIXED ' not in block:continue
  used=set()
  for item in match[1].split(','):
   bounds=item.split('-');used.update(range(int(bounds[0]),int(bounds[-1])+1))
  if 0 in used and 1 not in used and int(re.search(r'CPUTot=(\d+)',block)[1])-int(re.search(r'CPUAlloc=(\d+)',block)[1])>=8:nodes.append(block.split()[0])
submissions_path=control/'precheck_submissions.json'
submissions=json.loads(submissions_path.read_text()) if submissions_path.exists() else {}
for jid in b['assigned_training_ids']:
 if jid in submissions:continue
 out=root/'prechecks'/jid;out.mkdir(parents=True,exist_ok=True)
 script=out/'run.sbatch'
 if b['cluster']=='A100':
  resource=['#SBATCH --partition=a100x','#SBATCH --account=pxyai','#SBATCH --qos=normal','#SBATCH --cpus-per-task=12','#SBATCH --gres=gpu:a100:1','#SBATCH --mem-per-gpu=120G']
  setup=['source /etc/profile','set -eo pipefail','module load CUDA/11.8','source '+str(Path(py).parent/'activate'),'set -u']
  node=[]
 else:
  if not nodes:
   print('RESOURCE_WAIT no eligible physical GPU1 node for '+jid,flush=True);continue
  node=['--nodelist='+nodes.pop(0)]
  resource=['#SBATCH --partition=gpu','#SBATCH --cpus-per-task=8','#SBATCH --gres=gpu:1']
  setup=['source /etc/profile','set -euo pipefail','module load cuda/11.8','module load miniforge3/24.11','source '+str(Path(py).parent/'activate'),'[[ "${SLURM_JOB_GPUS:-}" == "1" && "${CUDA_VISIBLE_DEVICES:-}" == "0" ]] || exit 78']
 argv=[py,'-m','geosparse_ext.gpu_precheck','--manifest',str(control/'experiments.jsonl'),'--bindings',str(control/'bindings.json'),'--output',str(out),'--train-ids',jid,'--certify']
 lines=['#!/usr/bin/env bash','#SBATCH --nodes=1','#SBATCH --ntasks=1','#SBATCH --time=00:15:00',*resource,*setup,'export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2','cd '+shlex.quote(str(repo)),'exec '+shlex.join(argv)]
 script.write_text(chr(10).join(lines)+chr(10))
 result=subprocess.run(['sbatch','--parsable',*node,'--job-name=gs-check-'+jid,'--output='+str(out/'slurm-%j.out'),'--error='+str(out/'slurm-%j.err'),str(script)],capture_output=True,text=True)
 if result.returncode:
  print('SUBMISSION_BLOCKED '+jid+' '+result.stderr,flush=True);continue
 submissions[jid]=dict(slurm_id=result.stdout.strip().split(';')[0],source_commit=b['source_commit'],submitted_at=time.time())
 submissions_path.write_text(json.dumps(submissions,indent=2))
print(json.dumps(dict(cluster=b['cluster'],source_commit=b['source_commit'],coordinator=json.loads(receipt.read_text()),prechecks=submissions)))
'''


def main():
    commit = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    assert not subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain'], text=True).strip()
    bundle = HERE / ('geosparse_official_' + commit[:8] + '.bundle')
    subprocess.run(['git', '-C', str(SOURCE), 'bundle', 'create', str(bundle), 'HEAD'], check=True)
    spec = importlib.util.spec_from_file_location('corrected_matrix', SOURCE / 'geosparse_ext/matrix.py')
    matrix = importlib.util.module_from_spec(spec); spec.loader.exec_module(matrix)
    jobs, summary = matrix.compile_all()
    # The prior C/dense jobs were still pending (zero training epochs). Their
    # receipts are retained, while all writers now use the repaired snapshot.
    assignments = {'source': ['tr-c43d3e9cad58'],
                   'destination': ['tr-be89cd6cb2ca', 'tr-a45686a40afa', 'tr-c2d93c4c3979', 'tr-bacdbb058eca']}
    selected = set(sum(assignments.values(), []))
    focus = [j for j in jobs if j['seed'] == 0 and j.get('source_train_id', j['job_id']) in selected]
    for job in focus:
        job['protocol_amendment'] = 'official-full-data-20260908'
        if job['kind'] == 'train':
            job['checkpoints'] = list(range(4, 60, 5))
            job['training_validation'] = dict(subset='validation', interval_epochs=5, checkpoint_selection='best_average_mAP', weights='ema', tie_break='earlier_checkpoint')
    manifest = ''.join(json.dumps(j, ensure_ascii=False, sort_keys=True) + '\n' for j in focus).encode()
    (HERE / 'experiments.corrected.focus.jsonl').write_bytes(manifest)
    (HERE / 'matrix.corrected.summary.json').write_text(json.dumps(summary, indent=2))
    roots = {'source': '/data/run01/sczc063/yuzibo/geosparse_official_20260908/tooling_'+commit[:8],
             'destination': '/HOME/pxyai/pxyai_0057/HDD_POOL/yzb/geosparse_tad_20260907/corrected_20260908/tooling_'+commit[:8]}
    originals = {'source': '/data/run01/sczc063/yuzibo/geosparse_assets_20260907/bindings.d4e2cd8f.seed0.hosts.remote.json',
                 'destination': '/HOME/pxyai/pxyai_0057/HDD_POOL/yzb/geosparse_tad_20260907/control/bindings.a100.seed0.json'}
    def deploy(target):
        reader = 'from pathlib import Path; print(Path(' + repr(originals[target]) + ').read_text())'
        binding = json.loads(ssh(target, 'python3 -c ' + shlex.quote(reader)))
        root = roots[target]
        binding.update(repo_root=root+'/snapshots/'+commit[:8], source_commit=commit,
                       work_root=root+'/runs/'+commit[:8], protocol_root=root+'/protocol',
                       capability_dir=root+'/capabilities/'+commit[:8],
                       protocol_amendment='official-full-data-20260908', execution_phase='corrected-seed0-decision-cohort',
                       active_seeds=[0], assigned_training_ids=assignments[target], priority_training_ids=assignments[target],
                       max_concurrent_jobs=len(assignments[target]), cluster='N16R4' if target=='source' else 'A100',
                       runtime=dict(effective_batch=2, microbatch=2, evaluation_batch=1, num_workers=2))
        binding.pop('coordinator_repo', None)
        local = HERE / ('bindings.corrected.'+binding['cluster']+'.json')
        local.write_text(json.dumps(binding, indent=2, ensure_ascii=False))
        put(target, root+'/control/source.bundle', bundle.read_bytes())
        put(target, root+'/control/bindings.json', local.read_bytes())
        put(target, root+'/control/experiments.jsonl', manifest)
        result = ssh(target, 'python3 -', SETUP.replace('ROOT', repr(root)).encode())
        (HERE / ('deployment.corrected.'+binding['cluster']+'.log')).write_bytes(result)
        return target, result.decode(errors='replace')
    with ThreadPoolExecutor(max_workers=2) as pool:
        for target, output in pool.map(deploy, assignments):
            print(target, output, flush=True)


if __name__ == '__main__':
    main()
