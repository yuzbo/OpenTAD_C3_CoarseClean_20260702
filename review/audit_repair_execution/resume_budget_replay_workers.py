"""Resume only our read-only diagnostic steps with two decoder workers."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from launch_budget_replay import HERE, call, put


def resume(cluster, target, record):
    root = record["root"]
    # Read and interrupt only the identified diagnostic step, never its parent job.
    code = f'''
import json,subprocess,time,shutil
from pathlib import Path
p=Path({root!r});a={record['allocation']!r}
if (p/'resume_workers2.json').exists():raise SystemExit('resume already recorded; do not duplicate')
steps=subprocess.check_output(['squeue','--steps','-h','-j',a,'-o','%i|%j'],text=True).splitlines()
matches=[r.split('|')[0] for r in steps if r.split('|')[1]=='gs-budget-readonly']
if len(matches)!=1 or not matches[0].startswith(a+'.'):raise SystemExit('expected own diagnostic step is absent or ambiguous')
step=matches[0]
shutil.copyfile(p/'replay_validation_budgets.py',p/'replay_validation_budgets.initial.py')
shutil.copyfile(p/'results/header.json',p/'results/header.initial.json')
subprocess.run(['scancel',step],check=True)
for _ in range(25):
 live=subprocess.check_output(['squeue','--steps','-h','-j',a,'-o','%i'],text=True).splitlines()
 if step not in live:break
 time.sleep(1)
else:raise SystemExit('old diagnostic step has not exited; no replacement started')
prefix={{}}
for f in (p/'results').glob('*.windows.jsonl'):
 data=[json.loads(line) for line in f.read_text().splitlines()]
 if [r['window_index'] for r in data]!=list(range(len(data))):raise SystemExit('non-contiguous prefix')
 prefix[f.name]=len(data)
result=dict(prior_step=step,prior_windows=prefix,reason='two workers for deterministic test decoding; preserve all completed windows',time=time.time())
(p/'workers2_stop.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
'''
    stopped = json.loads(call(target, "python3 -", code.encode(), timeout=60))
    put(target, root + "/replay_validation_budgets.py", (HERE / "replay_validation_budgets.py").read_bytes())
    args = [x for x in record["command"] if not x.startswith("--time=")]
    args[1:1] = ["--time=00:20:00", "--open-mode=append"]
    code = f'''
import json,subprocess,time,socket
from pathlib import Path
p=Path({root!r})
state=subprocess.check_output(['squeue','-h','-j',{record['allocation']!r},'-o','%T|%u|%j'],text=True).strip().split('|')
owner=subprocess.check_output(['id','-un'],text=True).strip()
if state!=['RUNNING',owner,{record['cases'][0].split(':')[0]!r}]:raise SystemExit('parent allocation changed')
log=open(p/'launcher.log','ab',buffering=0)
process=subprocess.Popen({args!r},stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
result=dict(pid=process.pid,host=socket.gethostname(),command={args!r},time=time.time(),stopped={stopped!r})
(p/'resume_workers2.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
'''
    return cluster, json.loads(call(target, "python3 -", code.encode()))


if __name__ == "__main__":
    launches = json.loads((HERE / "dynamic_budget_replay_launch.json").read_text())
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(resume, c, t, launches[c]) for c,t in [("N16R4","source"),("A100","destination")]]
        results = dict(f.result() for f in futures)
    (HERE / "dynamic_budget_replay_workers2.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)
