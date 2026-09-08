"""One new frozen-C20 Scout audit in its own A100 allocation, no Heavy/train."""
from pathlib import Path
import json, subprocess
from deploy_corrected import command, put

HERE=Path(__file__).resolve().parent
ROOT='/HOME/pxyai/pxyai_0057/HDD_POOL/yzb/geosparse_tad_20260907/corrected_20260908/audit_repair_b70ae056'
STAGE=ROOT+'/control/c_dynamic_epoch020_budget_review_20260908'

def main():
 source=(HERE/'replay_validation_budgets.py').read_text(encoding='utf-8')
 old='if not os.environ.get("SLURM_JOB_ID") or not step_gpu.isdigit() or os.environ.get("CUDA_VISIBLE_DEVICES") != "0":'
 new='if not os.environ.get("SLURM_JOB_ID") or not step_gpu.isdigit() or not os.environ.get("CUDA_VISIBLE_DEVICES", "").isdigit():'
 assert source.count(old)==1
 source=source.replace(old,new)
 # Match production's A100 single-visible-device rule. Do not remap a GPU.
 source=source.replace('if binding["cluster"] == "N16R4" and step_gpu != "1":','if binding["cluster"] == "N16R4" and (step_gpu != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != "0"):')
 compile(source,'replay_validation_budgets.py','exec')
 put('destination',STAGE+'/replay_validation_budgets.py',source.encode())
 script=r'''
from pathlib import Path
import fcntl,json,shlex,subprocess,time
root=Path(ROOT);stage=Path(STAGE)
lock=open(stage/'submission.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX)
p=stage/'submission.json'
if p.exists():
 print(json.dumps(dict(status='EXISTING_SUBMISSION',submission=json.loads(p.read_text()))));raise SystemExit(0)
b=json.loads((root/'control/bindings.json').read_text());py=b['entrypoint'][0]
args=[py,str(stage/'replay_validation_budgets.py'),'--binding',str(root/'control/bindings.json'),'--cases','tr-5789d4417b4b:20','--output',str(stage/'output')]
lines=['#!/usr/bin/env bash','#SBATCH --partition=a100x','#SBATCH --account=pxyai','#SBATCH --qos=normal','#SBATCH --nodes=1','#SBATCH --ntasks=1','#SBATCH --cpus-per-task=4','#SBATCH --gres=gpu:a100:1','#SBATCH --mem-per-gpu=24G','#SBATCH --time=01:00:00','#SBATCH --job-name=gs-c20-budget-review','source /etc/profile','set -eo pipefail','module load CUDA/11.8','source '+shlex.quote(str(Path(py).parent/'activate')),'set -u','export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1','cd '+shlex.quote(b['repo_root']),'exec srun --ntasks=1 --gres=gpu:a100:1 '+shlex.join(args)]
(stage/'run.sbatch').write_text('\n'.join(lines)+'\n')
amendment=dict(case='tr-5789d4417b4b:20',model_source=b['source_commit'],purpose='Latest C EMA complete budget audit requested by user',scope='frozen Scout and original plan builder only; no Heavy/train; actual plans and analytical MAC',device_rule='A100 single numeric Slurm step and CUDA visible id, unchanged mapping; torch device_count must be1',duplicate_boundary='new E20 only; do not rerun completed E10 or other six cases')
(stage/'amendment.json').write_text(json.dumps(amendment,indent=2))
r=subprocess.run(['sbatch','--parsable','--output='+str(stage/'slurm-%j.out'),'--error='+str(stage/'slurm-%j.err'),str(stage/'run.sbatch')],capture_output=True,text=True,timeout=45)
if r.returncode:
 (stage/'submission_failed.json').write_text(json.dumps(dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2));print(json.dumps(dict(status='SUBMISSION_FAILED',stderr=r.stderr)));raise SystemExit(1)
receipt=dict(slurm_id=r.stdout.strip().split(';')[0],submitted_at=time.time(),stage=str(stage),case=amendment['case'],model_source=b['source_commit'])
p.write_text(json.dumps(receipt,indent=2));print(json.dumps(dict(status='SUBMITTED',submission=receipt)))
'''.replace('ROOT',repr(ROOT)).replace('STAGE',repr(STAGE))
 try:r=subprocess.run(command('destination','python3 -'),input=script.encode(),capture_output=True,timeout=70)
 except subprocess.TimeoutExpired:raise RuntimeError('Submission observation timed out; inspect stage/submission.json and squeue before any retry') from None
 if r.returncode:raise RuntimeError(r.stdout.decode(errors='replace')[-2000:]+r.stderr.decode(errors='replace')[-1000:])
 data=json.loads(r.stdout);out=HERE/'c20_budget_review_20260908';out.mkdir(exist_ok=True)
 (out/'submission.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
 (out/'replay_validation_budgets.py').write_text(source,encoding='utf-8')
 print(json.dumps(data))

if __name__=='__main__':main()
