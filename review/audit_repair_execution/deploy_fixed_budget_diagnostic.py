"""Submit one fixed-q diagnostic after the current owned benchmark releases GPU1."""
import argparse
import json
from pathlib import Path
import subprocess

from deploy_corrected import command


HERE = Path(__file__).resolve().parent
RESEARCH_SHA = "7ea094f40064dfd32259beb068548610514d1654"
TRAIN_ID = "tr-0afe8e4fc09b"
EVAL_ID = "ev-301df5cdf563"
DIAGNOSTIC_ID = "sci3-a-dynamic-best-fixed-q050-20260908"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    remote = r'''
import json,shlex,subprocess,time
from pathlib import Path
root=Path('/data/run01/sczc063/yuzibo/geosparse_official_20260908/audit_repair_b70ae056')
work=root/'runs/b70ae056'
train=work/'runs'/TRAIN_ID
reference=work/'runs'/EVAL_ID
research_root=root.parent/('sci3_evidence_'+RESEARCH_SHA[:8])
repo=research_root/'repo'
stage=root/'control'/DIAGNOSTIC_ID
def read(p): return json.loads(p.read_text())
def save(p,v):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(v,indent=2));temp.replace(p)
record=stage/'submission.json'
if record.exists():
    print(json.dumps(dict(status='ALREADY_RECORDED',submission=read(record))))
    raise SystemExit(0)
tests=read(research_root/'receipt.json')
if tests.get('source_commit')!=RESEARCH_SHA or tests.get('returncode')!=0 or tests.get('tests')!=['tests/test_sci3_fixed_budget_diagnostic.py']:
    raise RuntimeError('the exact research revision has not passed its focused CPU tests')
if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()!=RESEARCH_SHA:
    raise RuntimeError('research checkout is not the tested revision')
if subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip():
    raise RuntimeError('research checkout is dirty')
b=read(train/'bindings.json'); result=read(train/'result.json')
evaluation=read(reference/'result.json')
if result.get('status')!='completed' or result.get('completed_epochs')!=60 or not result.get('selection_complete') or not result.get('best_checkpoint_selection_complete'):
    raise RuntimeError('source training has incomplete final selection')
if evaluation.get('status')!='completed' or evaluation.get('is_mock') is not False or result['checkpoint_identity']!=evaluation.get('checkpoint_identity'):
    raise RuntimeError('reference evaluation does not use the final selected checkpoint')
state=read(work/'slurm_state.json')['jobs']['be-91499565831e']
benchmark_sid=str(state['slurm_id'])
if benchmark_sid!='1279571':
    raise RuntimeError('benchmark allocation changed; inspect its resource before scheduling')
queue=subprocess.run(['squeue','-h','-j',benchmark_sid,'-o','%T|%N'],capture_output=True,text=True,timeout=30,check=True)
if not queue.stdout.strip().startswith('RUNNING|g0087'):
    raise RuntimeError('expected owned benchmark is no longer running on g0087; resolve current resource')
argv=[b['entrypoint'][0],str(repo/'geosparse_research/fixed_budget_diagnostic.py'),
      '--training-run',str(train),'--reference-evaluation',str(reference)]
lines=['#!/usr/bin/env bash','#SBATCH --partition=gpu','#SBATCH --nodes=1','#SBATCH --ntasks=1',
       '#SBATCH --cpus-per-task=8','#SBATCH --gres=gpu:1','#SBATCH --nodelist=g0087',
       '#SBATCH --time=03:00:00','#SBATCH --nice=10000',
       'source /etc/profile','set -euo pipefail','module load cuda/11.8','module load miniforge3/24.11',
       'source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate',
       '[[ "${SLURM_JOB_GPUS:-}" == "1" && "${CUDA_VISIBLE_DEVICES:-}" == "0" ]] || exit 78',
       'export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONNOUSERSITE=1',
       'cd '+shlex.quote(b['repo_root']),
       shlex.join(argv+['--output',str(stage/'precheck'),'--precheck-only']),
       'exec '+shlex.join(argv+['--output',str(stage/'full')])]
script='\n'.join(lines)+'\n'
plan=dict(diagnostic_id=DIAGNOSTIC_ID,measurement_source_commit=RESEARCH_SHA,
    model_source_commit=result['source_commit'],source_train_id=TRAIN_ID,
    checkpoint_identity=result['checkpoint_identity'],reference_evaluation=EVAL_ID,
    measurement_override=dict(budget_mode='fixed',budget=.5),
    dependency_reason='resource serialization after the owned benchmark, independent of its scores or success',
    afterany_slurm_id=benchmark_sid,node='g0087',physical_gpu=1,container_cuda=0,
    release_policy='hold until the benchmark allocation has ended and fixed A has a real running allocation or a complete result; release the same ID',
    gpu_precheck_required=True,expected_test_videos=211,expected_test_windows=792,
    new_training=False,source_cpu_tests=tests,output=str(stage/'full'),sbatch=script)
if not EXECUTE:
    print(json.dumps(dict(status='PLAN_READY',**plan)));raise SystemExit(0)
stage.mkdir(parents=True,exist_ok=False)
path=stage/'run.sbatch';path.write_text(script)
save(stage/'amendment.json',plan)
submit=['sbatch','--parsable','--hold','--dependency=afterany:'+benchmark_sid,'--job-name=gs-a-fixedq-diagnostic',
        '--output='+str(stage/'slurm-%j.out'),'--error='+str(stage/'slurm-%j.err'),str(path)]
save(record,dict(status='SUBMITTING',created_at=time.time(),command=submit,**plan))
response=subprocess.run(submit,capture_output=True,text=True,timeout=30)
submission=dict(status='SUBMITTED' if response.returncode==0 else 'SUBMISSION_REJECTED',
    submitted_at=time.time(),stdout=response.stdout,stderr=response.stderr,**plan)
if response.returncode==0:
    sid=response.stdout.strip().split(';')[0]
    if not sid.isdigit():raise RuntimeError('ambiguous sbatch reply; reconcile SUBMITTING record before any retry')
    submission['slurm_id']=sid
save(record,submission)
print(json.dumps(submission))
if response.returncode:raise SystemExit(response.returncode)
'''.replace('RESEARCH_SHA',repr(RESEARCH_SHA)).replace('TRAIN_ID',repr(TRAIN_ID)).replace('EVAL_ID',repr(EVAL_ID)).replace('DIAGNOSTIC_ID',repr(DIAGNOSTIC_ID)).replace('EXECUTE',repr(args.execute))
    response = subprocess.run(command('source', 'python3 -'), input=remote.encode(), capture_output=True, timeout=110)
    destination = HERE / 'fixed_budget_diagnostic_20260908'
    destination.mkdir(exist_ok=True)
    mode = 'submission' if args.execute else 'plan'
    (destination / (mode + '.stdout.txt')).write_bytes(response.stdout)
    (destination / (mode + '.stderr.txt')).write_bytes(response.stderr)
    if response.stdout.strip():
        result = json.loads(response.stdout)
        (destination / (mode + '.json')).write_text(json.dumps(result, indent=2))
        print(json.dumps({key: result[key] for key in ('status','slurm_id','diagnostic_id','measurement_source_commit','model_source_commit','afterany_slurm_id','output') if key in result}))
    if response.returncode:
        print(response.stderr.decode(errors='replace'))
    response.check_returncode()


if __name__ == '__main__':
    main()
