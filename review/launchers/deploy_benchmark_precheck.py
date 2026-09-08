"""Stage and submit one owned hardware-path preflight after full training selection.

Uses the existing cluster/environment; it never edits the model or training queue.
Repeated calls report the recorded submission and do not submit another job.
"""
import argparse
import json
from pathlib import Path
import subprocess
from deploy_corrected import command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-id", required=True)
    parser.add_argument("--cluster", choices=["N16R4", "A100"], required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    deployment = json.loads((here / "audit_deployment.latest.json").read_text())["deployments"]["primary"][args.cluster]
    binding = json.loads((here / "audit_repair_b70ae056" / f"bindings.primary.{args.cluster}.json").read_text())
    if args.train_id not in binding["assigned_training_ids"]:
        raise ValueError("training is not assigned to the requested primary host")
    payload = dict(train_id=args.train_id, cluster=args.cluster, execute=args.execute,
                   control=deployment["control"], work_root=deployment["work_root"], files={
        name: (here / name).read_text(encoding="utf-8")
        for name in ["benchmark_capability_precheck.py", "test_benchmark_capability_precheck.py"]})
    payload["files"]["experiments.jsonl"] = (here / "audit_repair_b70ae056" / "experiments.primary.jsonl").read_text(encoding="utf-8")
    remote = r'''
import json,shlex,subprocess,time
from pathlib import Path
p=PAYLOAD
control=Path(p['control']); work=Path(p['work_root']); train=work/'runs'/p['train_id']
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)
result_path=train/'result.json'
if not result_path.is_file():
    print(json.dumps(dict(status='WAITING_SELECTION',train_id=p['train_id'])));raise SystemExit(0)
result=json.loads(result_path.read_text())
if (result.get('status')!='completed' or result.get('is_mock') is not False
    or result.get('completed_epochs')!=60 or not result.get('selection_complete')
    or not result.get('best_checkpoint_selection_complete')):
    print(json.dumps(dict(status='WAITING_SELECTION',train_id=p['train_id'])));raise SystemExit(0)
b=json.loads((train/'bindings.json').read_text())
stage=control/'benchmark_capability_v2_20260908'
out=stage/p['train_id']
record_path=out/'submission.json'
if record_path.exists():
    old=json.loads(record_path.read_text())
    # An expired completed Slurm ID was rejected before any job was created.
    # The complete training receipt above is the artifact prerequisite.
    if (p['execute'] and old.get('status')=='SUBMISSION_REJECTED' and not old.get('stdout')
        and 'Job dependency problem' in old.get('stderr','')):
        archive=out/'submission.rejected_dependency.json'
        if archive.exists():raise RuntimeError('dependency rejection already retained; inspect before retry')
        record_path.rename(archive)
    else:
        print(json.dumps(dict(status='ALREADY_RECORDED',submission=old)));raise SystemExit(0)
state=json.loads((work/'slurm_state.json').read_text())['jobs'][p['train_id']]
sid=str(state['slurm_id'])
argv=[b['entrypoint'][0],str(stage/'benchmark_capability_precheck.py'),'--manifest',str(stage/'experiments.jsonl'),
      '--bindings',str(train/'bindings.json'),'--train-id',p['train_id'],'--output',str(out),'--certify']
if p['cluster']=='N16R4':
    node=state['attempts'][-1]['node']
    if node not in json.loads((control/'gpu1_nodes.json').read_text())['nodes']:
        raise ValueError('no current authorized GPU1 node evidence for this allocation')
    lines=['#!/usr/bin/env bash','#SBATCH --partition=gpu','#SBATCH --nodes=1','#SBATCH --ntasks=1',
           '#SBATCH --cpus-per-task=8','#SBATCH --gres=gpu:1','#SBATCH --nodelist='+node,
           '#SBATCH --time=00:15:00','#SBATCH --nice=10000',
           'source /etc/profile','set -euo pipefail','module load cuda/11.8','module load miniforge3/24.11',
           'source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate',
           '[[ "${SLURM_JOB_GPUS:-}" == "1" && "${CUDA_VISIBLE_DEVICES:-}" == "0" ]] || exit 78']
else:
    lines=['#!/usr/bin/env bash','#SBATCH --partition=a100x','#SBATCH --account=pxyai','#SBATCH --qos=normal',
           '#SBATCH --nodes=1','#SBATCH --ntasks=1','#SBATCH --cpus-per-task=12','#SBATCH --gres=gpu:a100:1',
           '#SBATCH --mem-per-gpu=120G','#SBATCH --time=00:15:00','#SBATCH --nice=10000',
           'source /etc/profile','set -eo pipefail','module load CUDA/11.8',
           'source '+shlex.quote(str(Path(b['entrypoint'][0]).parent/'activate')),'set -u']
lines += ['export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONNOUSERSITE=1',
          'cd '+shlex.quote(b['repo_root']),'exec '+shlex.join(argv)]
script='\n'.join(lines)+'\n'
if not p['execute']:
    print(json.dumps(dict(status='PLAN_READY',command=argv,sbatch=script,source_commit=b['source_commit'])));raise SystemExit(0)
stage.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
for name,content in p['files'].items():
    path=stage/name
    if path.exists() and path.read_text()!=content:
        raise ValueError('immutable preflight source changed; retain original and use a new stage')
    path.write_text(content)
test=subprocess.run([b['entrypoint'][0],'-m','unittest','test_benchmark_capability_precheck','-v'],
                    cwd=stage,capture_output=True,text=True,timeout=30)
(out/'cpu_contract_tests.log').write_text(test.stdout+test.stderr)
if test.returncode:
    raise RuntimeError('remote capability contract tests failed: '+test.stderr)
path=out/'run.sbatch';path.write_text(script)
submit=['sbatch','--parsable','--job-name=gs-bench-check-'+p['train_id'],
        '--output='+str(out/'slurm-%j.out'),'--error='+str(out/'slurm-%j.err'),str(path)]
save(record_path,dict(status='SUBMITTING',command=submit,created_at=time.time(),source_commit=b['source_commit']))
run=subprocess.run(submit,capture_output=True,text=True,timeout=30)
record=dict(status='SUBMITTED' if run.returncode==0 else 'SUBMISSION_REJECTED',command=submit,
            source_commit=b['source_commit'],train_id=p['train_id'],source_train_slurm=sid,
            artifact_prerequisite='completed 60-epoch training with selection_complete; checked before submission and on GPU',
            stdout=run.stdout,stderr=run.stderr,submitted_at=time.time())
if run.returncode==0:
    new_id=run.stdout.strip().split(';')[0]
    if not new_id.isdigit():
        raise RuntimeError('ambiguous sbatch response; reconcile SUBMITTING record, never blind retry')
    record['slurm_id']=new_id
save(record_path,record)
print(json.dumps(record))
if run.returncode:raise SystemExit(run.returncode)
'''.replace("PAYLOAD", repr(payload))
    response = subprocess.run(command("source" if args.cluster == "N16R4" else "destination", "python3 -"),
                              input=remote.encode(), capture_output=True, timeout=110)
    print(response.stdout.decode(errors="replace"))
    print(response.stderr.decode(errors="replace"))
    response.check_returncode()


if __name__ == "__main__":
    main()
