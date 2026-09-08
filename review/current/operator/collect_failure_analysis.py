"""Read-only, compact summaries of the actual frozen-M training logs."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import subprocess
from deploy_corrected import command

HERE = Path(__file__).resolve().parent
OUT = HERE / 'failure_analysis_20260908'
REMOTE = r'''
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
import json, math, statistics
root=Path(ROOT)
def quantile(a,q):
 a=sorted(a)
 if not a:return None
 x=(len(a)-1)*q;lo=int(x);hi=min(lo+1,len(a)-1)
 return a[lo]*(hi-x)+a[hi]*(x-lo) if lo!=hi else a[lo]
def stats(a):
 a=[float(x) for x in a if x is not None and math.isfinite(float(x))]
 return dict(n=len(a),mean=statistics.mean(a) if a else None,p50=quantile(a,.5),p95=quantile(a,.95),max=max(a) if a else None)
result=dict(observed_at_utc=datetime.now(timezone.utc).isoformat(),runs={})
for jid in JOBS:
 run=root/'runs/b70ae056/runs'/jid
 epochs=defaultdict(list);nmalformed=0
 p=run/'train.log'
 if not p.exists():
  result['runs'][jid]=dict(error='missing train.log',run=str(run));continue
 with p.open() as f:
  for line in f:
   try:r=json.loads(line)
   except json.JSONDecodeError:nmalformed+=1;continue
   if 'step' in r and 'losses' in r:epochs[r['epoch']].append(r)
 summary=[]
 for epoch,attempts in sorted(epochs.items()):
  # An epoch restart appends its replay; latest occurrence of each step is retained.
  last_by_step={r['step']:r for r in attempts}
  raw=[last_by_step[s] for s in sorted(last_by_step)]
  keys=set().union(*(r['losses'] for r in raw))
  good=[r for r in raw if r.get('successful_update')]
  norms=[r.get('gradient_norm') for r in raw]
  finite=[n for n in norms if n is not None and math.isfinite(n)]
  summary.append(dict(epoch_completed=epoch+1,rows=len(raw),raw_attempt_rows=len(attempts),superseded_attempt_rows=len(attempts)-len(raw),unique_steps=len(last_by_step),successful_updates=len(good),overflows=sum(r.get('gradient_status')=='AMP_overflow' for r in raw),losses={k:stats([r['losses'].get(k) for r in raw]) for k in sorted(keys)},task_loss=stats([r['losses'].get('cls_loss',0)+r['losses'].get('reg_loss',0) for r in raw]),gradient_norm=stats(norms),fraction_above_clip1=sum(n>1 for n in finite)/len(finite) if finite else None,clip_multiplier=stats([min(1,1/(n+1e-6)) for n in finite]),first_lr=raw[0].get('lr'),last_lr=raw[-1].get('lr')))
 trace_path=run/'cost_trace.jsonl';traces=defaultdict(lambda:dict(rows=0,requested=Counter(),probes=[]))
 if trace_path.exists():
  last_trace={}
  with trace_path.open() as f:
   for line in f:
    try:r=json.loads(line)
    except json.JSONDecodeError:continue
    last_trace[(r['epoch'],r['step'],r.get('microbatch',0))]=r
  for r in last_trace.values():
   t=traces[r['epoch']];t['rows']+=1;t['requested'].update(str(x) for x in r.get('requested_budget',[]))
   if r.get('acquisition') is not None:t['probes'].append(r['acquisition'])
 result['runs'][jid]=dict(run=str(run),log_bytes=p.stat().st_size,malformed_or_partial_lines=nmalformed,epochs=summary,trace_sampling='step0 or acquisition only; not a full training-budget distribution',trace_epochs={str(e+1):dict(rows=v['rows'],requested_budget_counts=dict(v['requested']),acquisition_count=len(v['probes']),acquisitions=v['probes']) for e,v in traces.items()})
print(json.dumps(result,allow_nan=False))
'''

TARGETS = {
    'source': ('/data/run01/sczc063/yuzibo/geosparse_official_20260908/audit_repair_b70ae056', ['tr-0afe8e4fc09b','tr-624db235c913']),
    'destination': ('/HOME/pxyai/pxyai_0057/HDD_POOL/yzb/geosparse_tad_20260907/corrected_20260908/audit_repair_b70ae056', ['tr-1d7d835dcbe2','tr-5789d4417b4b','tr-712c0b6470f4','tr-7e78e2a74eee']),
}

def collect(item):
    target,(root,jobs)=item
    script=REMOTE.replace('ROOT',repr(root)).replace('JOBS',repr(jobs))
    r=subprocess.run(command(target,'python3 -'),input=script.encode(),capture_output=True,timeout=110)
    if r.returncode:raise RuntimeError(target+': '+r.stderr.decode(errors='replace')[-2000:])
    data=json.loads(r.stdout)
    OUT.mkdir(exist_ok=True)
    (OUT/(target+'_training_summary.json')).write_text(json.dumps(data,indent=2),encoding='utf-8')
    for jid,run in data['runs'].items():
        if 'epochs' not in run:print(jid,run);continue
        selected=[e for e in run['epochs'] if e['epoch_completed'] in [5,6,7,10,15,20,30,40,50,60] or e is run['epochs'][-1]]
        print(json.dumps(dict(target=target,job_id=jid,epochs=[dict(epoch=e['epoch_completed'],task=e['task_loss']['mean'],grad_p50=e['gradient_norm']['p50'],grad_p95=e['gradient_norm']['p95'],clipped=e['fraction_above_clip1'],actor=e['losses'].get('actor_loss',{}).get('mean'),critic=e['losses'].get('critic_loss',{}).get('mean'),probe=e['losses'].get('acquisition_loss',{}).get('mean'),overflows=e['overflows']) for e in selected])))

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(collect,TARGETS.items()))
