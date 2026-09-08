"""Start bounded, low-memory read-only Scouts in existing owned allocations."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
from deploy_corrected import command

HERE=Path(__file__).resolve().parent
CASES={
    "N16R4":["tr-0afe8e4fc09b:40","tr-0afe8e4fc09b:30","tr-624db235c913:30"],
    "A100":["tr-1d7d835dcbe2:10","tr-5789d4417b4b:10","tr-712c0b6470f4:10","tr-7e78e2a74eee:10"],
}
ALLOCATION={"N16R4":"1279017","A100":"245124"}
def call(target,code,data=None,timeout=45):
    r=subprocess.run(command(target,code),input=data,capture_output=True,timeout=timeout)
    if r.returncode:
        raise RuntimeError(r.stderr.decode(errors="replace")+r.stdout.decode(errors="replace"))
    return r.stdout
def put(target,path,data):
    code="from pathlib import Path;import sys;p=Path("+repr(path)+");p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(sys.stdin.buffer.read())"
    call(target,"python3 -c "+shlex.quote(code),data)
def launch(cluster,target,delivery):
    binding=json.loads((delivery/f"bindings.primary.{cluster}.json").read_text())
    control=str(PurePosixPath(binding["work_root"]).parent.parent/"control")
    root=str(PurePosixPath(control).parent/"analysis"/"budget_replay_20260908")
    script=root+"/replay_validation_budgets.py"
    put(target,script,(HERE/"replay_validation_budgets.py").read_bytes())
    setup=["source /etc/profile","set -eo pipefail"]
    if cluster=="N16R4":
        setup+=["module load cuda/11.8","module load miniforge3/24.11"]
    else:
        setup+=["module load CUDA/11.8"]
    setup+=["source "+shlex.quote(str(PurePosixPath(binding["entrypoint"][0]).parent/"activate")),
            "export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1",
            "cd "+shlex.quote(binding["repo_root"])]
    argv=[binding["entrypoint"][0],script,"--binding",control+"/bindings.json","--cases",*CASES[cluster],"--output",root+"/results"]
    setup+=["exec nice -n 10 "+shlex.join(argv)]
    shell=root+"/run.sh"
    put(target,shell,("#!/usr/bin/env bash\n"+"\n".join(setup)+"\n").encode())
    args=["srun","--jobid="+ALLOCATION[cluster],"--overlap","--exact","--nodes=1","--ntasks=1",
          "--cpus-per-task=2","--gres=gpu:1","--time=00:35:00","--job-name=gs-budget-readonly",
          "--output="+root+"/srun.out","--error="+root+"/srun.err","bash",shell]
    remote=f'''
import json,os,socket,subprocess,time
from pathlib import Path
p=Path({root!r});receipt=p/'launch.json'
if receipt.exists():
 raise SystemExit('already launched: inspect receipt instead of duplicating')
info=subprocess.check_output(['squeue','-h','-j',{ALLOCATION[cluster]!r},'-o','%u|%T|%j'],text=True).strip().split('|')
if len(info)!=3 or info[1]!='RUNNING' or info[2] not in {repr([x.split(':')[0] for x in CASES[cluster]])}:
 raise SystemExit('allocation is no longer the expected running primary')
if info[0]!=subprocess.check_output(['id','-un'],text=True).strip():
 raise SystemExit('allocation owner differs')
subprocess.run(['bash','-n',{shell!r}],check=True)
log=open(p/'launcher.log','ab',buffering=0)
process=subprocess.Popen({args!r},stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
record=dict(pid=process.pid,host=socket.gethostname(),allocation={ALLOCATION[cluster]!r},
 root={root!r},cases={CASES[cluster]!r},command={args!r},started_at=time.time(),
 scope='existing allocation; read-only Scout and plan replay; no training/Heavy; 2GiB CUDA allocation cap')
receipt.write_text(json.dumps(record,indent=2))
print(json.dumps(record))
'''
    return cluster,json.loads(call(target,"python3 -",remote.encode()))
if __name__=="__main__":
    delivery=Path(json.loads((HERE/"audit_deployment.latest.json").read_text())["local_delivery"])
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(launch,c,t,delivery) for c,t in [("N16R4","source"),("A100","destination")]]
        results=dict(f.result() for f in futures)
    (HERE/"dynamic_budget_replay_launch.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    print(json.dumps(results,indent=2),flush=True)
